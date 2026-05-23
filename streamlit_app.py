"""
Arabic Real-Time STT — Streamlit App
=====================================
The browser captures microphone audio via the Web Audio API (JavaScript),
encodes each chunk as a proper WAV blob, and POSTs it directly to the
FastAPI relay server via fetch().  The relay adds a WAV header if needed,
forwards the audio to faster-whisper-server, and returns the transcript.

Architecture:
  Browser JS  →  POST /v1/audio/transcriptions  →  FastAPI relay
                                                 →  faster-whisper-server
                                                 ←  {"text": "..."}
  Browser JS  →  postMessage("transcript", text)  →  Streamlit (via st.components)
"""

import time

import requests
import streamlit as st

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎙️ التفريغ الصوتي العربي",
    page_icon="🎙️",
    layout="wide",
)

# ─────────────────────────────────────────────
#  CSS — RTL + Arabic design
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
    direction: rtl;
}

.main { background: #0f1117; }

.title-box {
    text-align: center;
    padding: 2rem 1rem 1rem;
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%);
    border-radius: 16px;
    border: 1px solid #2a2f3e;
    margin-bottom: 1.5rem;
}
.title-box h1 { color: #e8f4fd; font-size: 2rem; margin: 0; }
.title-box p  { color: #8899aa; margin: 0.4rem 0 0; }

.status-pill {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 1rem;
}
.status-idle      { background:#1e2533; color:#6b7fa3; border:1px solid #2a3450; }
.status-recording { background:#1a2e1a; color:#4ade80; border:1px solid #166534;
                    animation: pulse 1.5s infinite; }
.status-error     { background:#2e1a1a; color:#f87171; border:1px solid #7f1d1d; }

@keyframes pulse {
    0%,100% { opacity:1; } 50% { opacity:.6; }
}

.transcript-box {
    background: #1a1f2e;
    border: 1px solid #2a3450;
    border-radius: 12px;
    padding: 1.5rem;
    min-height: 200px;
    font-size: 1.2rem;
    line-height: 2;
    color: #e8f4fd;
    direction: rtl;
    text-align: right;
    white-space: pre-wrap;
    word-break: break-word;
}

.interim { color: #6b9fd4; font-style: italic; }
.server-ok  { color: #4ade80; }
.server-err { color: #f87171; }

div[data-testid="stTextInput"] input {
    background: #1a1f2e !important;
    color: #e8f4fd !important;
    border: 1px solid #2a3450 !important;
    border-radius: 8px !important;
    direction: ltr !important;
}
div[data-testid="stButton"] button {
    border-radius: 10px !important;
    font-family: 'Cairo', sans-serif !important;
    font-weight: 600 !important;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session state init
# ─────────────────────────────────────────────
for key, default in {
    "transcript": "",
    "server_ok": None,
    "server_url": "http://localhost:2110",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def normalise_base_url(url: str) -> str:
    """Strip trailing slash and /v1/… suffix — return bare https://host."""
    url = url.strip().rstrip("/")
    # Drop any path component the user might have accidentally pasted
    for suffix in ["/v1/audio/transcriptions", "/v1", "/health"]:
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url


def check_server_health(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=6)
        return r.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────
#  Audio capture + transcription — pure JS
#
#  How it works:
#   1. getUserMedia → AudioContext (16 kHz, mono)
#   2. ScriptProcessor accumulates Float32 frames
#   3. Every INTERVAL_MS milliseconds the processor:
#        a. Converts Float32 → PCM-16 LE
#        b. Builds a proper WAV blob in JS (no server-side header needed)
#        c. POSTs the WAV blob to  <relay_url>/v1/audio/transcriptions
#        d. On success, calls window.parent.postMessage with the transcript
#   4. Streamlit's st.components.v1.html() can receive those postMessages
#      only in the SAME iframe — we work around this by writing results
#      into a hidden <textarea> that we poll via Streamlit component.
#
#  The cleanest cross-frame approach supported by Streamlit is to have the
#  JS write results into a dedicated Streamlit text_input via a hidden form
#  — but that requires a page reload.  Instead we embed the entire widget
#  (buttons + live transcript display) inside the component iframe so no
#  cross-frame communication is needed at all.
# ─────────────────────────────────────────────
def build_audio_widget(relay_url: str, language: str, beam_size: int) -> str:
    """Return the full HTML/JS audio-capture widget."""
    transcription_endpoint = f"{relay_url}/v1/audio/transcriptions"
    # Send every ~2 s of audio
    INTERVAL_MS = 2000

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ margin:0; background:transparent; font-family:'Cairo',sans-serif; }}

  #widget {{
    padding: 12px 16px;
    background: #1a1f2e;
    border: 1px solid #2a3450;
    border-radius: 12px;
  }}

  .btn-row {{ display:flex; gap:10px; justify-content:center; margin-bottom:12px; }}

  button {{
    padding: 9px 26px;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-family: 'Cairo', sans-serif;
    font-weight: 600;
    cursor: pointer;
    transition: opacity .2s;
  }}
  button:disabled {{ opacity:.4; cursor:default; }}

  #btn-start {{ background:#166534; color:#fff; }}
  #btn-stop  {{ background:#7f1d1d; color:#fff; }}

  #mic-status {{
    text-align:center;
    font-size:.85rem;
    color:#6b7fa3;
    margin-bottom: 10px;
  }}

  #live-transcript {{
    background:#0f1117;
    border:1px solid #2a3450;
    border-radius:8px;
    padding:12px;
    min-height:120px;
    max-height:300px;
    overflow-y:auto;
    font-size:1.05rem;
    line-height:1.9;
    color:#e8f4fd;
    direction:rtl;
    text-align:right;
    white-space:pre-wrap;
    word-break:break-word;
  }}

  #error-msg {{ color:#f87171; font-size:.85rem; text-align:center; margin-top:6px; }}

  .word-count {{ color:#6b7fa3; font-size:.78rem; text-align:left; margin-top:4px; }}

  .interim {{ color:#6b9fd4; font-style:italic; }}
</style>
</head>
<body>
<div id="widget">
  <div class="btn-row">
    <button id="btn-start" onclick="startRecording()">🎙️ ابدأ التسجيل</button>
    <button id="btn-stop"  onclick="stopRecording()" disabled>⏹ وقف</button>
  </div>
  <div id="mic-status">⏹ في الانتظار</div>
  <div id="live-transcript"><span style="color:#3a4a6a">النص سيظهر هنا أثناء الكلام…</span></div>
  <div id="error-msg"></div>
  <div class="word-count" id="word-count"></div>
</div>

<script>
// ── Config ───────────────────────────────────────────────────
const RELAY_URL    = "{transcription_endpoint}";
const LANGUAGE     = "{language}";
const BEAM_SIZE    = {beam_size};
const INTERVAL_MS  = {INTERVAL_MS};
const SAMPLE_RATE  = 16000;
const BUFFER_SIZE  = 4096;
const VAD_THRESHOLD = 0.02; // Increased from 0.005 to be more aggressive

// Common Whisper hallucinations in Arabic to filter out
const HALLUCINATIONS = [
    "اشتركوا في القناة", 
    "شكرا للمشاهدة", 
    "ترجمة نانسي قنقر", 
    "نانسي قنقر",
    "Subscribe",
    "Watching",
    "شكرا لك",
    "شكرا",
    "قناة",
    "المشاهدة",
    "سلام عليكم",
    "أحمد الله وبرك"
];

// ── State ────────────────────────────────────────────────────
let audioCtx    = null;
let processor   = null;
let sourceNode  = null;
let stream      = null;
let pcmBuffer   = [];   // Float32 samples accumulate here
let intervalId  = null;
let fullText    = "";
let isSending   = false;

const statusEl    = document.getElementById("mic-status");
const transcEl    = document.getElementById("live-transcript");
const errorEl     = document.getElementById("error-msg");
const wordCountEl = document.getElementById("word-count");
const btnStart    = document.getElementById("btn-start");
const btnStop     = document.getElementById("btn-stop");

// ── WAV builder ──────────────────────────────────────────────
function float32ToPcm16(f32arr) {{
    const buf = new ArrayBuffer(f32arr.length * 2);
    const view = new DataView(buf);
    for (let i = 0; i < f32arr.length; i++) {{
        let s = Math.max(-1, Math.min(1, f32arr[i]));
        view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }}
    return new Uint8Array(buf);
}}

function buildWav(pcm16Bytes) {{
    const numSamples   = pcm16Bytes.length / 2;
    const byteRate     = SAMPLE_RATE * 1 * 2;   // sampleRate * channels * bytesPerSample
    const blockAlign   = 1 * 2;
    const dataSize     = pcm16Bytes.length;
    const headerSize   = 44;
    const totalSize    = headerSize + dataSize;

    const buf  = new ArrayBuffer(totalSize);
    const view = new DataView(buf);
    let o = 0;
    const w = (s) => {{ for (let c of s) view.setUint8(o++, c.charCodeAt(0)); }};

    w("RIFF");
    view.setUint32(o, totalSize - 8, true); o += 4;
    w("WAVE");
    w("fmt ");
    view.setUint32(o, 16, true);            o += 4;   // PCM chunk size
    view.setUint16(o, 1, true);             o += 2;   // PCM format
    view.setUint16(o, 1, true);             o += 2;   // mono
    view.setUint32(o, SAMPLE_RATE, true);   o += 4;
    view.setUint32(o, byteRate, true);      o += 4;
    view.setUint16(o, blockAlign, true);    o += 2;
    view.setUint16(o, 16, true);            o += 2;   // bits per sample
    w("data");
    view.setUint32(o, dataSize, true);      o += 4;

    new Uint8Array(buf).set(pcm16Bytes, headerSize);
    return new Blob([buf], {{ type: "audio/wav" }});
}}

// ── Send a chunk to the relay ────────────────────────────────
async function sendChunk() {{
    if (isSending || pcmBuffer.length === 0) return;

    // ── Simple VAD (Voice Activity Detection) ────────────────
    let sum = 0;
    for (let i = 0; i < pcmBuffer.length; i++) sum += pcmBuffer[i] * pcmBuffer[i];
    const rms = Math.sqrt(sum / pcmBuffer.length);
    
    // If volume is below threshold, treat as silence and don't send
    if (rms < VAD_THRESHOLD) {{
        pcmBuffer = []; 
        statusEl.textContent = "💤 سكوت... (صوت منخفض)";
        statusEl.style.color = "#6b7fa3";
        return;
    }}

    statusEl.textContent = "🔴 يسجّل… (جاري المعالجة)";
    statusEl.style.color = "#4ade80";
    isSending = true;

    // Drain accumulated samples
    const samples = new Float32Array(pcmBuffer.splice(0, pcmBuffer.length));
    const pcm16   = float32ToPcm16(samples);
    const wavBlob = buildWav(pcm16);

    const fd = new FormData();
    fd.append("file",      wavBlob, "chunk.wav");
    fd.append("language",  LANGUAGE);
    fd.append("beam_size", BEAM_SIZE);
    fd.append("temperature", "0.0");
    fd.append("response_format", "json");

    try {{
        const resp = await fetch(RELAY_URL, {{ method:"POST", body:fd }});
        if (resp.ok) {{
            const json = await resp.json();
            let text = (json.text || "").trim();
            
            // ── Hallucination Filter ────────────────────────
            const cleanText = text.replace(/[.,\/#!$%\^&\*;:{{}}=\-_`~()]/g,"").trim();
            const isHallucination = HALLUCINATIONS.some(h => cleanText === h || (cleanText.includes(h) && cleanText.length < h.length + 5));
            
            // Check for repetition
            const lastLine = fullText.split("\\n").pop().replace(/[.,\/#!$%\^&\*;:{{}}=\-_`~()]/g,"").trim();
            const isRepetition = (cleanText === lastLine && cleanText.length > 0);

            if (text && !isHallucination && !isRepetition && text.length > 1) {{
                fullText += (fullText ? "\\n" : "") + text;
                transcEl.innerHTML = fullText;
                transcEl.scrollTop = transcEl.scrollHeight;
                const words = fullText.split(/\\s+/).filter(Boolean).length;
                wordCountEl.textContent = "عدد الكلمات: " + words;
            }}
            errorEl.textContent = "";
        }} else {{
            const errBody = await resp.text();
            errorEl.textContent = "⚠️ " + resp.status + ": " + errBody.substring(0, 120);
        }}
    }} catch(e) {{
        errorEl.textContent = "❌ تعذّر الاتصال بالسيرفر: " + e.message;
    }} finally {{
        isSending = false;
    }}
}}

// ── Recording control ────────────────────────────────────────
async function startRecording() {{
    errorEl.textContent = "";
    try {{
        stream   = await navigator.mediaDevices.getUserMedia({{ audio:true }});
        audioCtx = new AudioContext({{ sampleRate: SAMPLE_RATE }});
        sourceNode = audioCtx.createMediaStreamSource(stream);
        processor  = audioCtx.createScriptProcessor(BUFFER_SIZE, 1, 1);

        processor.onaudioprocess = (e) => {{
            const ch = e.inputBuffer.getChannelData(0);
            for (let i = 0; i < ch.length; i++) pcmBuffer.push(ch[i]);
        }};

        sourceNode.connect(processor);
        processor.connect(audioCtx.destination);

        // Fire send every INTERVAL_MS
        intervalId = setInterval(sendChunk, INTERVAL_MS);

        statusEl.textContent = "🔴 يسجّل…";
        statusEl.style.color = "#4ade80";
        btnStart.disabled = true;
        btnStop.disabled  = false;

    }} catch(err) {{
        errorEl.textContent = "❌ " + err.message;
    }}
}}

async function stopRecording() {{
    clearInterval(intervalId);

    // Final flush
    await sendChunk();

    if (processor)  {{ processor.disconnect(); processor = null; }}
    if (sourceNode) {{ sourceNode.disconnect(); sourceNode = null; }}
    if (audioCtx)   {{ await audioCtx.close(); audioCtx = null; }}
    if (stream)     {{ stream.getTracks().forEach(t => t.stop()); stream = null; }}
    pcmBuffer = [];

    statusEl.textContent = "⏹ متوقف";
    statusEl.style.color = "#6b7fa3";
    btnStart.disabled = false;
    btnStop.disabled  = true;
}}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="title-box">
  <h1>🎙️ التفريغ الصوتي العربي الفوري</h1>
  <p>مدعوم بـ faster-whisper large-v3-turbo عبر FastAPI</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ الإعدادات")

    server_url = st.text_input(
        "🌐 Relay URL (ngrok / local)",
        value=st.session_state.server_url,
        placeholder="https://xxxx-xx-xx-xxx-xx.ngrok-free.app",
        help="الـ URL الذي ظهر بعد تشغيل gpu_server_setup.py",
        key="server_url",
    )

    beam_size = st.slider("Beam Size", 1, 10, 3,
                          help="كبّر للدقة، صغّر للسرعة")

    language = st.selectbox("اللغة", ["ar", "en", "auto"], index=0)

    st.divider()

    if st.button("🔍 اختبر الاتصال"):
        if server_url:
            base = normalise_base_url(server_url)
            with st.spinner("جاري الاتصال …"):
                ok = check_server_health(base)
            st.session_state.server_ok = ok
        else:
            st.warning("أدخل الـ URL أولاً")

    if st.session_state.server_ok is True:
        st.markdown('<p class="server-ok">✅ السيرفر شغال</p>', unsafe_allow_html=True)
    elif st.session_state.server_ok is False:
        st.markdown('<p class="server-err">❌ تعذّر الاتصال بالسيرفر</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown("""
**خطوات التشغيل:**
1. شغّل `gpu_server_setup.py` على جهاز الـ GPU
2. انسخ الـ ngrok URL وضعه في الحقل أعلاه
3. اضغط **اختبر الاتصال** ← يجب أن يظهر ✅
4. اضغط **ابدأ التسجيل** في الويدجت
    """)

# ── Main area ─────────────────────────────────────────────────
if not server_url:
    st.info("👈 أدخل الـ Server URL في الإعدادات على اليسار ثم اختبر الاتصال.")
else:
    base_url = normalise_base_url(server_url)
    widget_html = build_audio_widget(base_url, language, beam_size)

    st.subheader("🎙️ التسجيل والتفريغ الفوري")
    st.caption(
        "يعمل التسجيل والتفريغ مباشرةً داخل الويدجت أدناه — لا حاجة لزر إضافي."
    )

    # Height ~520px to accommodate the live transcript inside the component
    st.components.v1.html(widget_html, height=520, scrolling=False)

    st.divider()

    # ── Persistent transcript download area ──────────────────
    st.subheader("📥 حفظ النص")
    st.caption(
        "النص يُعرض داخل الويدجت أعلاه مباشرة. "
        "لتحميله، انسخه يدوياً أو استخدم الزر أدناه بعد انتهاء الجلسة."
    )

    manual_text = st.text_area(
        "الصق النص هنا لتحميله:",
        height=150,
        placeholder="انسخ النص من الويدجت أعلاه والصقه هنا …",
    )

    if manual_text.strip():
        st.download_button(
            label="⬇️ تحميل النص",
            data=manual_text,
            file_name=f"transcript_{int(time.time())}.txt",
            mime="text/plain",
        )

"""
Arabic Real-Time STT — Streamlit App
=====================================
الـ Client بيفتح المايك في البراوزر عبر JavaScript،
يبعت الصوت WebSocket لـ faster-whisper-server عبر الـ ngrok URL.
"""

import asyncio
import json
import queue
import threading
import time
from typing import Optional

import numpy as np
import requests
import streamlit as st
import websockets

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
.status-recording { background:#1a2e1a; color:#4ade80; border:1px solid #166534; animation: pulse 1.5s infinite; }
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

/* Streamlit overrides */
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
    "interim": "",
    "is_recording": False,
    "ws_thread": None,
    "stop_event": None,
    "audio_queue": None,
    "server_ok": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def normalize_ws_url(url: str) -> str:
    """Convert https/http ngrok URL to wss/ws."""
    url = url.strip().rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[8:]
    elif url.startswith("http://"):
        url = "ws://" + url[7:]
    return url


def check_server_health(base_url: str) -> bool:
    """Ping /health on the REST side."""
    rest = base_url.replace("wss://", "https://").replace("ws://", "http://")
    rest = rest.split("/v1")[0]
    try:
        r = requests.get(f"{rest}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def build_ws_url(base_url: str) -> str:
    ws = normalize_ws_url(base_url)
    if not ws.endswith("/v1/audio/transcriptions"):
        ws = ws.rstrip("/") + "/v1/audio/transcriptions"
    return ws


# ─────────────────────────────────────────────
#  WebSocket worker (runs in background thread)
# ─────────────────────────────────────────────
def ws_worker(
    ws_url: str,
    audio_queue: queue.Queue,
    stop_event: threading.Event,
    transcript_queue: queue.Queue,
    beam_size: int,
    language: str,
):
    """
    Opens a WebSocket to faster-whisper-server, streams audio chunks,
    and puts transcript updates into transcript_queue.
    """

    async def run():
        init_msg = json.dumps({
            "language": language,
            "beam_size": beam_size,
            "temperature": 0.0,
            "response_format": "text",
            "vad_filter": True,
        })

        try:
            async with websockets.connect(
                ws_url,
                extra_headers={"ngrok-skip-browser-warning": "true"},
                ping_interval=20,
                ping_timeout=30,
                max_size=10 * 1024 * 1024,
            ) as ws:
                await ws.send(init_msg)
                transcript_queue.put(("status", "connected"))

                async def sender():
                    while not stop_event.is_set():
                        try:
                            chunk = audio_queue.get(timeout=0.1)
                            if chunk is None:
                                break
                            await ws.send(chunk)
                        except queue.Empty:
                            continue
                        except Exception as e:
                            transcript_queue.put(("error", str(e)))
                            break

                async def receiver():
                    async for msg in ws:
                        if stop_event.is_set():
                            break
                        try:
                            data = json.loads(msg) if msg.startswith("{") else None
                            if data:
                                text = data.get("text", "")
                                is_final = data.get("is_final", False)
                                transcript_queue.put(("final" if is_final else "interim", text))
                            else:
                                # plain text response_format
                                transcript_queue.put(("final", msg))
                        except Exception:
                            transcript_queue.put(("final", msg))

                await asyncio.gather(sender(), receiver())

        except Exception as e:
            transcript_queue.put(("error", str(e)))

    asyncio.run(run())


# ─────────────────────────────────────────────
#  JavaScript — capture mic in browser → send to Streamlit via component
# ─────────────────────────────────────────────
AUDIO_CAPTURE_JS = """
<script>
// ── Audio capture via MediaRecorder ──────────────────────────────
let mediaRecorder = null;
let audioContext  = null;
let stream        = null;

async function startRecording() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new AudioContext({ sampleRate: 16000 });

        const source   = audioContext.createMediaStreamSource(stream);
        const processor = audioContext.createScriptProcessor(4096, 1, 1);

        source.connect(processor);
        processor.connect(audioContext.destination);

        processor.onaudioprocess = (e) => {
            const f32 = e.inputBuffer.getChannelData(0);
            // Convert Float32 → PCM16
            const pcm16 = new Int16Array(f32.length);
            for (let i = 0; i < f32.length; i++) {
                let s = Math.max(-1, Math.min(1, f32[i]));
                pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            // Send to parent Streamlit frame via postMessage
            window.parent.postMessage({
                type: 'audio_chunk',
                data: Array.from(pcm16)
            }, '*');
        };

        document.getElementById('mic-status').textContent = '🔴 يسجّل...';
        document.getElementById('mic-status').style.color = '#4ade80';
        mediaRecorder = { processor, source };

    } catch(err) {
        document.getElementById('mic-status').textContent = '❌ ' + err.message;
    }
}

function stopRecording() {
    if (mediaRecorder) {
        mediaRecorder.processor.disconnect();
        mediaRecorder.source.disconnect();
        mediaRecorder = null;
    }
    if (audioContext)  { audioContext.close(); audioContext = null; }
    if (stream)        { stream.getTracks().forEach(t => t.stop()); stream = null; }
    document.getElementById('mic-status').textContent = '⏹ متوقف';
    document.getElementById('mic-status').style.color = '#6b7fa3';
    window.parent.postMessage({ type: 'recording_stopped' }, '*');
}
</script>

<div style="text-align:center; padding:1rem;">
  <button onclick="startRecording()"
    style="background:#166534;color:white;border:none;padding:10px 28px;
           border-radius:10px;font-size:1rem;cursor:pointer;margin-left:8px;
           font-family:Cairo,sans-serif;">
    🎙️ ابدأ التسجيل
  </button>
  <button onclick="stopRecording()"
    style="background:#7f1d1d;color:white;border:none;padding:10px 28px;
           border-radius:10px;font-size:1rem;cursor:pointer;
           font-family:Cairo,sans-serif;">
    ⏹ وقف
  </button>
  <p id="mic-status" style="color:#6b7fa3;margin-top:8px;font-family:Cairo,sans-serif;">
    ⏹ متوقف
  </p>
</div>
"""

# ─────────────────────────────────────────────
#  UI Layout
# ─────────────────────────────────────────────
st.markdown("""
<div class="title-box">
  <h1>🎙️ التفريغ الصوتي العربي الفوري</h1>
  <p>مدعوم بـ faster-whisper large-v3-turbo</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Settings ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ الإعدادات")

    server_url = st.text_input(
        "🌐 ngrok / Server URL",
        value=st.session_state.get("server_url", ""),
        placeholder="https://xxxx-xx-xx-xxx-xx.ngrok-free.app",
        help="الـ URL اللي ظهر لك بعد تشغيل gpu_server_setup.sh",
        key="server_url",
    )

    beam_size = st.slider("Beam Size", 1, 10, 3,
        help="كبّر للدقة، صغّر للسرعة")

    language = st.selectbox("اللغة", ["ar", "en", "auto"], index=0)

    st.divider()

    # Health check
    if st.button("🔍 اختبر الاتصال بالسيرفر"):
        if server_url:
            with st.spinner("جاري الاتصال..."):
                ok = check_server_health(server_url)
            st.session_state.server_ok = ok
        else:
            st.warning("أدخل الـ URL الأول")

    if st.session_state.server_ok is True:
        st.markdown('<p class="server-ok">✅ السيرفر شغال</p>', unsafe_allow_html=True)
    elif st.session_state.server_ok is False:
        st.markdown('<p class="server-err">❌ مش قادر يوصل للسيرفر</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown("""
**خطوات التشغيل:**
1. شغّل `gpu_server_setup.sh` على جهاز الـ GPU
2. انسخ الـ ngrok URL وحطه فوق
3. اضغط "اختبر الاتصال"
4. اضغط "ابدأ الجلسة" ثم سجّل
    """)

# ── Main area ──────────────────────────────────────────────
col_ctrl, col_transcript = st.columns([1, 2])

with col_ctrl:
    st.subheader("🎙️ التحكم")

    if not st.session_state.is_recording:
        if st.button("▶️ ابدأ الجلسة", type="primary"):
            if not server_url:
                st.error("أدخل الـ Server URL في الإعدادات")
            else:
                # init state
                st.session_state.is_recording  = True
                st.session_state.transcript    = ""
                st.session_state.interim       = ""
                st.session_state.stop_event    = threading.Event()
                st.session_state.audio_queue   = queue.Queue()
                st.session_state.tq            = queue.Queue()  # transcript updates

                ws_url = build_ws_url(server_url)
                t = threading.Thread(
                    target=ws_worker,
                    args=(
                        ws_url,
                        st.session_state.audio_queue,
                        st.session_state.stop_event,
                        st.session_state.tq,
                        beam_size,
                        language,
                    ),
                    daemon=True,
                )
                t.start()
                st.session_state.ws_thread = t
                st.rerun()
    else:
        if st.button("⏹ أوقف الجلسة", type="secondary"):
            st.session_state.stop_event.set()
            st.session_state.audio_queue.put(None)
            st.session_state.is_recording = False
            st.rerun()

    # Status pill
    if st.session_state.is_recording:
        st.markdown('<span class="status-pill status-recording">🔴 جلسة نشطة</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-idle">⏸ في الانتظار</span>',
                    unsafe_allow_html=True)

    # Mic capture widget (only shown during active session)
    if st.session_state.is_recording:
        st.components.v1.html(AUDIO_CAPTURE_JS, height=140)

        # Poll transcript queue and show updates
        if "tq" in st.session_state and st.session_state.tq:
            tq: queue.Queue = st.session_state.tq
            updated = False
            while not tq.empty():
                kind, text = tq.get_nowait()
                if kind == "final":
                    st.session_state.transcript += text + "\n"
                    st.session_state.interim = ""
                    updated = True
                elif kind == "interim":
                    st.session_state.interim = text
                    updated = True
                elif kind == "error":
                    st.error(f"خطأ: {text}")
                    st.session_state.is_recording = False
            if updated:
                st.rerun()

    # Copy / Clear buttons
    st.divider()
    if st.button("🗑️ مسح النص"):
        st.session_state.transcript = ""
        st.session_state.interim    = ""
        st.rerun()

with col_transcript:
    st.subheader("📝 النص المفرَّغ")

    display = st.session_state.transcript
    if st.session_state.interim:
        display += f'<span class="interim">{st.session_state.interim}</span>'

    st.markdown(
        f'<div class="transcript-box">{display if display else "<span style=\'color:#3a4a6a\'>النص سيظهر هنا أثناء الكلام...</span>"}</div>',
        unsafe_allow_html=True,
    )

    # Word count
    word_count = len(st.session_state.transcript.split()) if st.session_state.transcript.strip() else 0
    st.caption(f"عدد الكلمات: {word_count}")

    # Download
    if st.session_state.transcript.strip():
        st.download_button(
            label="⬇️ تحميل النص",
            data=st.session_state.transcript,
            file_name=f"transcript_{int(time.time())}.txt",
            mime="text/plain",
        )

# ── Auto-refresh while recording ──────────────────────────
if st.session_state.is_recording:
    time.sleep(0.5)
    st.rerun()

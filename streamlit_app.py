"""Giao diện Streamlit — client mỏng gọi FastAPI (mô hình chỉ nạp một lần ở backend)."""
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.request

import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Tự động bật FastAPI backend ngầm nếu chưa chạy
# FIX: khoá file chống spawn trùng nhiều uvicorn + chờ health-check thật sự
# ---------------------------------------------------------------------------
BACKEND_LOCAL = "http://127.0.0.1:8000"
BOOT_TIMEOUT = 240                      # cold start tải model ~1–2 phút
LOCK_FILE = "/tmp/ai_backend.lock"
BACKEND_LOG = "/tmp/ai_backend.log"


def _backend_alive(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{BACKEND_LOCAL}/api/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _spawn_backend() -> None:
    """Chỉ MỘT process Streamlit được spawn backend (khoá flock) → hết tải model trùng/OOM."""
    try:
        import fcntl
        lock = open(LOCK_FILE, "w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)   # process khác giữ khoá → bỏ qua
    except (BlockingIOError, OSError):
        return
    env = {**os.environ, "YOLO_CONFIG_DIR": "/tmp/Ultralytics"}
    try:
        hf_token = st.secrets.get("HF_TOKEN", "")
    except Exception:
        hf_token = ""
    if hf_token:
        env["HF_TOKEN"] = hf_token
    with open(BACKEND_LOG, "ab") as log:                   # log riêng, dễ debug
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app",
             "--host", "127.0.0.1", "--port", "8000"],
            stdout=log, stderr=log, env=env, start_new_session=True,
        )


def ensure_backend(max_wait: int = BOOT_TIMEOUT) -> bool:
    if _backend_alive():
        return True
    _spawn_backend()
    deadline = time.time() + max_wait                      # vòng chờ thay cho sleep(5)
    while time.time() < deadline:
        if _backend_alive():
            return True
        time.sleep(3)
    return False


st.set_page_config(page_title="AI Web Apps", page_icon="🤖", layout="wide")

API_URL = st.sidebar.text_input(
    "API URL", os.environ.get("API_URL", BACKEND_LOCAL)
).rstrip("/")

with st.status("Đang chờ backend tải mô hình (lần đầu có thể mất 1–2 phút)…") as boot:
    ready = ensure_backend()
    boot.update(label="Backend sẵn sàng" if ready else "Backend KHÔNG khởi động được",
                state="complete" if ready else "error")

if not ready:
    try:
        tail = open(BACKEND_LOG, "rb").read()[-2000:].decode("utf-8", "ignore")
    except OSError:
        tail = "(chưa có log)"
    st.error(f"Backend không lên sau {BOOT_TIMEOUT}s. Log cuối:\n```\n{tail}\n```")
    st.stop()

# ---------------------------------------------------------------------------
# Health + tiện ích gọi API
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def health(url: str):
    try:
        return requests.get(f"{url}/api/health", timeout=5).json()
    except Exception as exc:                               # FIX: bắt cả JSONDecodeError
        return {"status": "down", "error": str(exc), "models": {}}


h = health(API_URL)
if h.get("status") == "ok":                                # FIX: .get() chống KeyError
    st.sidebar.markdown(f"Backend: 🟢 {h.get('device', '')}")
    for name, ok in h.get("models", {}).items():
        st.sidebar.write(("✅ " if ok else "⛔ ") + name)
else:
    st.sidebar.markdown("Backend: 🔴 không kết nối")


def post(path: str, **kwargs):
    try:
        r = requests.post(f"{API_URL}{path}", timeout=120, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Không gọi được API: {exc}")
        return None
    if not r.ok:
        try:                                               # FIX: json() có thể ném ValueError
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        st.error(f"Lỗi {r.status_code}: {detail}")
        return None
    try:
        return r.json()
    except ValueError as exc:
        st.error(f"Phản hồi không phải JSON: {exc}")
        return None


def upload(label: str, key: str):
    f = st.file_uploader(label, type=["jpg", "jpeg", "png", "webp"], key=key)
    if f:
        st.image(f, caption="Ảnh đầu vào", width="stretch")
    return f


st.title("🤖 AI Web Apps")
st.caption("Phân loại ảnh · Phát hiện đối tượng · Tìm kiếm ảnh · Chatbot RAG — "
           "một backend FastAPI, hai giao diện Streamlit & React")

tab1, tab2, tab3, tab4 = st.tabs(["🌼 Phân loại", "🚗 Phát hiện", "🔎 Tìm ảnh", "💬 Chatbot"])

# ------------------------------- Tab 1: phân loại -------------------------
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        f = upload("Ảnh món ăn Việt Nam (Phở bò, Bánh mì, Bún chả, Gỏi cuốn, Bánh xèo)", "cls")
        top_k = st.slider("Top-k", 1, 5, 3)
        explain = st.checkbox("Giải thích bằng Grad-CAM (vùng mô hình chú ý)")
        endpoint = "/api/classify/explain" if explain else "/api/classify"
        if f and (res := post(endpoint, files={"file": f.getvalue()}, data={"top_k": top_k})):
            with c2:
                if not res["confident"]:
                    st.warning("Mô hình không chắc chắn — ảnh có thể không thuộc 5 món đã học.")
                for p in res["predictions"]:
                    st.progress(p["score"], text=f"{p['label']}: {p['score']:.1%}")
                if "overlay" in res:
                    heat = Image.open(io.BytesIO(base64.b64decode(res["overlay"].split(",", 1)[1])))
                    st.image(heat, width="stretch", caption=(
                        f"Grad-CAM cho nhãn “{res['target']}”: vùng đỏ/vàng ảnh hưởng nhiều nhất. "
                        "Chỉ hiển thị phần giữa ảnh mà mô hình nhìn."))
                st.caption(f"⏱ {res['latency_ms']} ms")

# ------------------------------- Tab 2: phát hiện -------------------------
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        f = upload("Ảnh bất kỳ (người, xe, động vật, đồ vật…)", "det")
        conf = st.slider("Ngưỡng tin cậy", 0.05, 0.95, 0.25, 0.05)
        if f and (res := post("/api/detect", files={"file": f.getvalue()}, data={"conf": conf})):
            with c2:
                img = Image.open(io.BytesIO(base64.b64decode(res["image"].split(",", 1)[1])))
                st.image(img, caption=f"{len(res['detections'])} đối tượng · {res['latency_ms']} ms",
                         width="stretch")
                st.write(res["summary"])
                st.dataframe(res["detections"], width="stretch")

# ------------------------------- Tab 3: tìm ảnh ---------------------------
with tab3:
    mode = st.radio("Tìm bằng", ["Câu mô tả (tiếng Anh)", "Ảnh mẫu"], horizontal=True)
    k = st.slider("Số kết quả", 4, 24, 8, 4)
    res = None
    if mode.startswith("Câu"):
        q = st.text_input("Ví dụ: a red flower, a dog on a sofa, people riding bikes",
                          "yellow sunflowers in a field")
        if q:
            res = post("/api/search/text", json={"query": q, "k": k})
    else:
        f = upload("Ảnh mẫu", "ret")
        if f:
            res = post("/api/search/image", files={"file": f.getvalue()}, data={"k": k})
    if res:
        cols = st.columns(4)
        for i, r in enumerate(res["results"]):
            # tải ảnh phía server Streamlit: trình duyệt có thể không truy cập trực tiếp API_URL
            img_bytes = requests.get(f"{API_URL}{r['url']}", timeout=30).content
            cols[i % 4].image(img_bytes, caption=f"{r['label']} · {r['score']:.3f}", width="stretch")

# ------------------------------- Tab 4: chatbot RAG -----------------------
with tab4:
    st.info("Trợ lý ShopLite trả lời dựa trên tài liệu chính sách (RAG). Thử: Đổi trả trong bao lâu?")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        st.chat_message(m["role"]).markdown(m["content"])

    if prompt := st.chat_input("Nhập câu hỏi…"):
        st.chat_message("user").markdown(prompt)
        sources = []

        def stream():
            with requests.post(f"{API_URL}/api/chat",
                               json={"message": prompt, "history": st.session_state.chat},
                               stream=True, timeout=300) as r:
                r.raise_for_status()
                r.encoding = "utf-8"
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        ev = json.loads(line[6:])
                    except ValueError:
                        continue
                    if ev["type"] == "sources":
                        sources.extend(ev["items"])
                    elif ev["type"] == "token":
                        yield ev["text"]

        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(stream())
            except requests.RequestException as exc:
                answer = f"Lỗi: {exc}"
                st.error(answer)
            with st.expander("Nguồn đã dùng"):
                for s in sources:
                    st.markdown(f"**{s['source']}** · điểm {s['score']}\n\n> {s['text'][:300]}…")
        st.session_state.chat += [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": answer}]
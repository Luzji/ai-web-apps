"""Giao diện Streamlit — client mỏng gọi FastAPI backend chạy trên laptop, expose qua ngrok."""
import base64
import io
import json
import os

import requests
import streamlit as st
from PIL import Image

# FIX NGROK: bản free chặn client không phải browser bằng trang HTML cảnh báo.
# Header này bỏ qua trang đó => API trả JSON đúng. BẮT BUỘC cho mọi request.
NGROK_HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "ai-web-apps-streamlit",
}


def _default_api_url() -> str:
    try:
        return str(st.secrets["API_URL"])
    except Exception:
        return os.environ.get("API_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="AI Web Apps", page_icon="🤖", layout="wide")

API_URL = st.sidebar.text_input("API URL (ngrok backend)", _default_api_url()).rstrip("/")


@st.cache_data(ttl=30, show_spinner=False)
def health(url: str):
    try:
        r = requests.get(f"{url}/api/health", timeout=8, headers=NGROK_HEADERS)
        return r.json()
    except Exception as exc:
        return {"status": "down", "error": f"{type(exc).__name__}: {exc}", "models": {}}


h = health(API_URL)
if h.get("status") == "ok":
    st.sidebar.markdown(f"Backend: 🟢 {h.get('device', '')}")
    for name, ok in h.get("models", {}).items():
        st.sidebar.write(("✅ " if ok else "⛔ ") + name)
else:
    st.sidebar.markdown("Backend: 🔴 không kết nối")
    st.sidebar.code(h.get("error", ""))


def post(path: str, **kwargs):
    kwargs.setdefault("headers", NGROK_HEADERS)
    try:
        r = requests.post(f"{API_URL}{path}", timeout=120, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Không gọi được API ({type(exc).__name__}): {exc}\n"
                 "Kiểm tra: laptop thức? ngrok + uvicorn chạy? URL đúng?")
        return None
    if not r.ok:
        st.error(f"Lỗi {r.status_code}: {r.text[:300]}")
        return None
    try:
        return r.json()
    except ValueError:
        st.error("API trả về không phải JSON — thường là trang cảnh báo ngrok.")
        return None


def fetch_bytes(url: str):
    try:
        r = requests.get(url, timeout=30, headers=NGROK_HEADERS)
        r.raise_for_status()
        return r.content
    except requests.RequestException as exc:
        st.warning(f"Không tải được ảnh {url}: {exc}")
        return None


def upload(label: str, key: str):
    f = st.file_uploader(label, type=["jpg", "jpeg", "png", "webp"], key=key)
    if f:
        st.image(f, caption="Ảnh đầu vào", width="stretch")
    return f


st.title("🤖 AI Web Apps")
st.caption("Phân loại · Phát hiện · Tìm ảnh · Chatbot RAG — backend FastAPI trên laptop qua ngrok")

tab1, tab2, tab3, tab4 = st.tabs(["🌼 Phân loại", "🚗 Phát hiện", "🔎 Tìm ảnh", "💬 Chatbot"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        f = upload("Ảnh món ăn Việt Nam (Phở bò, Bánh mì, Bún chả, Gỏi cuốn, Bánh xèo)", "cls")
        top_k = st.slider("Top-k", 1, 5, 3)
        explain = st.checkbox("Giải thích bằng Grad-CAM (vùng mô hình chú ý)")
        endpoint = "/api/classify/explain" if explain else "/api/classify"
        if f and (res := post(endpoint, files={"file": f.getvalue()}, data={"top_k": top_k})):
            with c2:
                if not res.get("confident", True):
                    st.warning("Mô hình không chắc chắn — ảnh có thể không thuộc 5 món đã học.")
                for p in res.get("predictions", []):
                    st.progress(p["score"], text=f"{p['label']}: {p['score']:.1%}")
                if "overlay" in res:
                    heat = Image.open(io.BytesIO(base64.b64decode(res["overlay"].split(",", 1)[1])))
                    st.image(heat, width="stretch", caption=(
                        f"Grad-CAM cho nhãn “{res.get('target')}”: vùng đỏ/vàng ảnh hưởng nhiều nhất."))
                st.caption(f"⏱ {res.get('latency_ms')} ms")

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
                st.write(res.get("summary", ""))
                st.dataframe(res["detections"], width="stretch")

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
        for i, r in enumerate(res.get("results", [])):
            img_bytes = fetch_bytes(f"{API_URL}{r['url']}")
            if img_bytes:
                cols[i % 4].image(img_bytes, caption=f"{r['label']} · {r['score']:.3f}",
                                  width="stretch")

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
                               headers=NGROK_HEADERS, stream=True, timeout=300) as r:
                r.raise_for_status()
                r.encoding = "utf-8"
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        ev = json.loads(line[6:])
                    except ValueError:
                        continue
                    if ev.get("type") == "sources":
                        sources.extend(ev.get("items", []))
                    elif ev.get("type") == "token":
                        yield ev.get("text", "")

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

# AI Web Apps — 4 ứng dụng AI trên một trang web

Bài tập nhóm học phần **Lập trình Web nâng cao (N05)**. Một backend **FastAPI** giữ 4 mô hình AI; hai giao diện web cùng gọi vào backend đó: **React** (một trang, 4 tab) và **Streamlit**.

| Thành viên | MSSV | Phụ trách |
|---|---|---|
| _(điền)_ | _(điền)_ | _(điền)_ |

## Liên kết sản phẩm

| Mục | Link |
|---|---|
| Giao diện web (React, Vercel/Netlify) | _(điền sau khi triển khai — xem [DEPLOY.md](DEPLOY.md))_ |
| Backend API (Hugging Face Spaces) | _(điền)_ · kiểm tra `/api/health` · tài liệu `/docs` |
| Giao diện Streamlit (tùy chọn) | _(điền)_ |
| Model card | [MODEL_CARD.md](MODEL_CARD.md) |

## 4 chức năng AI

| # | Chức năng | Mô hình | Dữ liệu | Chỉ số đánh giá |
|---|---|---|---|---|
| 1 | Nhận diện loài hoa (5 lớp) + **giải thích Grad-CAM** | ResNet-18 fine-tune | TF Flowers — 3.670 ảnh | Accuracy, F1, ma trận nhầm lẫn |
| 2 | Phát hiện đối tượng (80 lớp COCO) | YOLO11n | COCO128 | mAP50, mAP50-95 |
| 3 | Tìm kiếm ảnh (chữ → ảnh, ảnh → ảnh) | CLIP ViT-B/32 + FAISS | COCO128 + 500 ảnh hoa | Precision@5, Precision@10 |
| 4 | Chatbot chăm sóc khách hàng (RAG) | Qwen2.5-Instruct + MiniLM đa ngôn ngữ + FAISS | 6 tài liệu chính sách ShopLite (tiếng Việt) | Hit@1, Hit@3 |

## Kiến trúc

```
                ┌──────────────── Streamlit (cổng 8501) ─────────────┐
 Trình duyệt ──►│                                                     │──► FastAPI (cổng 8000) ──► core/ (4 mô hình, nạp 1 lần)
                └──────────── React build (phục vụ bởi FastAPI) ─────┘         /api/classify · /api/detect
                                                                                /api/search/* · /api/chat (SSE)
```

`core/` chỉ chứa suy luận (không biết gì về web) → `api/` bọc thành HTTP → `streamlit_app.py` và `web/` chỉ là giao diện.

```
├── config.py               # cấu hình tập trung, ghi đè bằng biến môi trường
├── core/                   # classifier.py · detector.py · retrieval.py · llm.py
├── api/main.py             # FastAPI (và phục vụ web/dist nếu đã build)
├── streamlit_app.py        # giao diện Streamlit
├── web/                    # giao diện React (Vite)
├── data/kb/                # kho tri thức cho chatbot (6 file Markdown)
├── scripts/prepare.py      # tải dữ liệu, huấn luyện, lập chỉ mục, đo chỉ số
├── scripts/smoke_test.py   # kiểm thử toàn bộ API đang chạy
├── scripts/benchmark.py    # đo độ trễ p50/p95 + RAM (locustfile.py: phương án dùng Locust)
├── scripts/fill_docs.py    # điền số đo thật vào README / MODEL_CARD
├── tests/test_api.py       # pytest, không cần GPU / mô hình thật (chạy trong CI)
├── MODEL_CARD.md           # model card: dữ liệu, chỉ số, giới hạn, rủi ro
├── DEPLOY.md               # hướng dẫn triển khai HF Spaces / Vercel / Streamlit Cloud
└── docs/screenshots/       # ảnh giao diện trong README
```

## Grad-CAM — giải thích dự đoán trên web

Ở tab **Phân loại ảnh** (React và Streamlit) tick **“Giải thích bằng Grad-CAM”** để xem mô hình nhìn vào đâu: vùng **đỏ/vàng** là vùng ảnh ảnh hưởng nhiều nhất đến nhãn được giải thích (mặc định là nhãn có xác suất cao nhất; API cho chọn nhãn khác qua tham số `target`).

- Cài đặt: `core/gradcam.py`, lấy đạo hàm của điểm số lớp theo bản đồ đặc trưng `layer4` (7×7) của ResNet-18, trung bình theo không gian làm trọng số kênh, ReLU, phóng lên và phủ lên ảnh. Không huấn luyện thêm.
- Ảnh hiển thị là phần **cắt giữa 224×224** vì mô hình chỉ nhìn phần đó.
- Cách đọc đúng: bản đồ cho biết mô hình **dựa vào đâu**, không chứng minh dự đoán **đúng**. Với ảnh ngoài 5 loài, bản đồ vẫn tô sáng một vùng nào đó — xem thêm [MODEL_CARD.md](MODEL_CARD.md).

## Cài đặt và chạy

Cần **Python 3.11+** và **Node 22** (Vite 8 yêu cầu Node ≥ 20.19). Lần đầu tải khoảng 2–4 GB (dữ liệu + trọng số mô hình).

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt

python scripts/prepare.py           # chạy 1 lần: tải dữ liệu, huấn luyện ResNet-18, lập chỉ mục, đo chỉ số

cd web && npm install && npm run build && cd ..
uvicorn api.main:app --port 8000    # React + API → http://localhost:8000   (Swagger: /docs)
streamlit run streamlit_app.py      # Streamlit   → http://localhost:8501   (mở terminal khác)
```

- Chờ log `Application startup complete` (nạp 4 mô hình mất từ vài chục giây đến vài phút), rồi kiểm tra `http://localhost:8000/api/health` trả `ok`.
- Không có GPU: `prepare.py` tự chuyển sang chế độ nhẹ (1 epoch, 800 ảnh, LLM 0.5B). Máy yếu hoặc mạng chậm: chạy `prepare.py` trên Google Colab rồi tải thư mục `artifacts/` và `data/gallery/` về đặt vào dự án.
- Phát triển React: `cd web && npm run dev` (cổng 5173, proxy `/api` → 8000).
- Kiểm thử: `python -m pytest -q` (38 ca, API với mô hình giả, không cần GPU — mỗi endpoint có ca thành công, ca 400 và ca 422; kèm 9 ca kiểm tra lõi Grad-CAM; tự chạy trên GitHub Actions mỗi lần push) và `python scripts/smoke_test.py` (API thật đang chạy).

## Ảnh giao diện

<!-- Chụp ảnh từ giao diện đang chạy, lưu vào docs/screenshots/ với đúng tên file bên dưới -->

**React — một trang web, 4 chức năng**

<p>
  <img src="docs/screenshots/react-classify.png" width="49%" alt="Phân loại ảnh">
  <img src="docs/screenshots/react-detect.png" width="49%" alt="Phát hiện đối tượng">
</p>
<p>
  <img src="docs/screenshots/react-search.png" width="49%" alt="Tìm kiếm ảnh">
  <img src="docs/screenshots/react-chat.png" width="49%" alt="Chatbot RAG">
</p>

**Streamlit**

<img src="docs/screenshots/streamlit.png" width="70%" alt="Giao diện Streamlit">

## API

| Method | Endpoint | Đầu vào | Đầu ra |
|---|---|---|---|
| GET | `/api/health` | — | trạng thái, thiết bị, mô hình nào đã nạp |
| POST | `/api/classify` | `file` (ảnh), `top_k` | `predictions`, `confident`, `latency_ms` |
| POST | `/api/classify/explain` | `file`, `top_k`, `target` (tùy chọn, tên lớp) | như `/api/classify` + `target`, `overlay` (ảnh JPEG base64 có bản đồ nhiệt Grad-CAM) |
| POST | `/api/detect` | `file`, `conf` | `detections`, `summary`, `image` (base64 đã vẽ hộp) |
| POST | `/api/search/text` | JSON `{query, k}` | `results` [{id, label, score, url}] |
| POST | `/api/search/image` | `file`, `k` | như trên |
| GET | `/api/gallery/{id}` | — | file ảnh trong kho |
| POST | `/api/chat` | JSON `{message, history}` | Server-Sent Events: `sources` → `token`… → `done` |
| POST | `/api/chat/sync` | như trên | `{answer, sources}` (không stream) |

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `ENABLED_MODELS` | `classifier,detector,retrieval,llm` | Mô hình được nạp (bớt đi nếu thiếu RAM) |
| `LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` (GPU) / `Qwen/Qwen2.5-0.5B-Instruct` (CPU) | Mô hình sinh của chatbot |
| `EMBED_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding cho RAG |
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | Tìm kiếm ảnh |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:8501` | Origin được gọi API |
| `API_URL` | `http://localhost:8000` | (Streamlit) địa chỉ backend |

## Chỉ số mô hình

Số đo thật do `prepare.py` ghi vào `artifacts/*/metrics.json` và `artifacts/rag_metrics.json`; chạy `python scripts/fill_docs.py` để bảng dưới tự cập nhật (đừng sửa tay giữa hai thẻ).

<!-- METRICS:START -->
_chưa có — chạy `python scripts/prepare.py` rồi `python scripts/fill_docs.py`_
<!-- METRICS:END -->

Ma trận nhầm lẫn của bộ phân loại: `artifacts/classifier/confusion_matrix.png`. Giới hạn và rủi ro: xem [MODEL_CARD.md](MODEL_CARD.md).

## Đo hiệu năng

Đo bằng `python scripts/benchmark.py --api <link backend>` (tuần tự, 1 người dùng, sau 3 lần khởi động), rồi `python scripts/fill_docs.py`.

<!-- BENCH:START -->
_chưa đo — chạy `python scripts/benchmark.py` rồi `python scripts/fill_docs.py`_
<!-- BENCH:END -->

**Phần cứng của backend:** _(điền: CPU/GPU, RAM, nơi chạy — ví dụ "HF Spaces CPU basic, 2 vCPU, 16 GB")_

## Docker (tùy chọn)

Chạy `prepare.py` trước để có `artifacts/` và `data/gallery/`, sau đó:

```bash
docker build -t ai-web-apps . && docker run -p 7860:7860 ai-web-apps   # mở http://localhost:7860
```

## Kết quả mô hình (đo một lần trên tập test)

### Ứng dụng 1 — Phân loại 5 món ăn Việt (500 ảnh = 5 lớp × 100)

| Cấu hình | Tập test | Test accuracy | Macro F1 | Ghi chú |
|---|---|---|---|---|
| v0 · ResNet-18, 1 epoch, CPU, chia tập chưa phân tầng | 23 ảnh | 0.609 | 0.598 | mốc "trước" — giữ làm bằng chứng cải tiến |
| v1 · ResNet-18, chia 80/10/10 phân tầng, huấn luyện GPU | 50 ảnh (10/lớp) | — | — | sẽ điền ở bước sau |
| v2 · MobileNetV3 cùng cách chia + xuất ONNX | 50 ảnh (10/lớp) | — | — | sẽ điền ở bước sau |

Ma trận nhầm lẫn bản v0:

![Ma trận nhầm lẫn v0](artifacts/classifier/confusion_matrix.png)

## Công cụ AI đã sử dụng

Theo yêu cầu của giảng viên, nhóm ghi rõ công cụ AI và phiên bản:

| Công cụ | Phiên bản | Dùng để làm gì |
|---|---|---|
| Claude (Anthropic) | Claude Sonnet 5.5 | Tách notebook của giảng viên thành cấu trúc repo; viết `core/gradcam.py` và endpoint `/api/classify/explain`, `scripts/prepare.py`, `smoke_test.py`, `benchmark.py`, `locustfile.py`, `fill_docs.py`, bộ test `tests/test_api.py`, CI, `MODEL_CARD.md`, `DEPLOY.md`, README |

Các file `config.py`, `core/`, `api/main.py`, `streamlit_app.py`, `web/`, `tests/` và `data/kb/` được trích nguyên văn từ notebook `AI_Web_Apps_Streamlit_React.ipynb` do giảng viên cung cấp (Phenikaa Applied AI Lab). Dữ liệu: TF Flowers, COCO128. Mô hình: ResNet-18 (torchvision), YOLO11n (Ultralytics), CLIP (OpenAI), MiniLM (sentence-transformers), Qwen2.5 (Alibaba) qua Hugging Face.

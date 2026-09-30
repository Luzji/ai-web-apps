# Hướng dẫn triển khai (C2 + C3)

Kiến trúc triển khai: **backend FastAPI + 4 mô hình** chạy trong một container (Hugging Face Spaces) → **giao diện** (React trên Vercel/Netlify, hoặc Streamlit Cloud) gọi vào backend đó qua HTTPS.

> Image Docker của repo này đã kèm sẵn bản build React, nên chỉ riêng backend (C2) cũng mở được trọn giao diện tại `https://<space>.hf.space/`. Bước C3 dùng để có giao diện tách riêng, đúng yêu cầu của môn.

## C2 — Backend lên Hugging Face Spaces (Docker)

Mô hình cần RAM: 4 mô hình cùng nạp chiếm khoảng 2–3 GB, vừa gói CPU miễn phí (16 GB RAM) của Spaces. Cấu hình sẵn dùng LLM 0.5B (`LLM_MODEL` trong Dockerfile).

1. Chạy `python scripts/prepare.py` trên máy/Colab để có `artifacts/` và `data/gallery/` (Dockerfile sao chép hai thư mục này vào image).
2. Tạo Space mới: huggingface.co → **New Space** → SDK **Docker** → *Blank*.
3. Clone repo của Space về, chép toàn bộ mã nguồn vào, kèm `artifacts/` và `data/gallery/`. Các file mô hình/chỉ mục bị `.gitignore` của GitHub bỏ qua, nhưng Space cần chúng, nên dùng **Git LFS** và **không** chép `.gitignore` sang:
   ```bash
   git lfs install
   git lfs track "*.pt" "*.faiss" "*.jpg" "*.png"
   ```
4. Sửa đầu file `README.md` **của Space** (không phải README trên GitHub) thành:
   ```yaml
   ---
   title: AI Web Apps
   sdk: docker
   app_port: 7860
   ---
   ```
5. Trong Space → **Settings → Variables**, đặt `CORS_ORIGINS` = địa chỉ giao diện sẽ gọi vào (ví dụ `https://ai-web-apps.vercel.app,https://<app>.streamlit.app`), cách nhau bằng dấu phẩy, không có "/" cuối.
6. `git push` → chờ Space build xong. **Kiểm tra:** `https://<space>.hf.space/api/health` trả `"status": "ok"` và cả 4 mô hình `true`.

Phương án thay thế — **Render**: New → Web Service → chọn repo → Runtime *Docker*; đặt biến `CORS_ORIGINS`. Gói miễn phí chỉ 512 MB RAM, **không đủ** cho 4 mô hình; đặt `ENABLED_MODELS=classifier,retrieval` hoặc dùng gói trả phí.

## C3 — Giao diện

### React → Vercel (hoặc Netlify)

1. Import repo GitHub vào Vercel, **Root Directory** = `web`, Framework = Vite (`npm run build`, output `dist`).
2. Environment Variables: `VITE_API_URL` = `https://<space>.hf.space` (xem `web/.env.example`). Biến này được nhúng lúc *build*, đổi xong phải deploy lại.
3. Deploy, mở link trên điện thoại, thử đủ 4 tab. Nếu Console báo CORS: thêm đúng origin của Vercel vào `CORS_ORIGINS` của backend (C2, bước 5).

### Streamlit → Streamlit Community Cloud (thay thế)

1. Streamlit Cloud cài `requirements.txt` nằm ở gốc repo — file này của dự án chứa cả torch/ultralytics (nặng, dễ lỗi). Vì vậy tạo nhánh riêng cho giao diện:
   ```bash
   git checkout -b streamlit-deploy
   cp requirements-streamlit.txt requirements.txt && git commit -am "streamlit: chỉ cài thư viện nhẹ" && git push -u origin streamlit-deploy
   ```
   Rồi share.streamlit.io → New app → chọn repo, nhánh `streamlit-deploy`, file chính `streamlit_app.py`.
2. Advanced settings → Secrets/Environment: `API_URL = "https://<space>.hf.space"`.
3. Streamlit gọi API từ server của nó nên **không** dính CORS.

## Kiểm tra sau khi triển khai

```bash
python scripts/smoke_test.py --api https://<space>.hf.space
python scripts/benchmark.py  --api https://<space>.hf.space --n 20 --chat-n 3
python scripts/fill_docs.py
```

Lưu ý khi đo: Space miễn phí ngủ sau một thời gian không dùng — lần gọi đầu sau khi ngủ chậm hơn nhiều; số đo qua mạng gồm cả độ trễ mạng. Ghi rõ điều này (và phần cứng của Space) khi điền README.

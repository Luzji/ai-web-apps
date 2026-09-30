# Model card — AI Web Apps (4 chức năng)

Bài tập nhóm học phần Lập trình Web nâng cao (N05). Đây là sản phẩm **học tập/demo**, không dùng cho quyết định thực tế.

## 1. Tổng quan

| # | Chức năng | Mô hình | Dữ liệu | Việc mô hình làm |
|---|---|---|---|---|
| 1 | Phân loại hoa | ResNet-18 (ImageNet) fine-tune | TF Flowers, 3.670 ảnh, 5 lớp; chia 80/10/10 phân tầng, seed 42 | Gán 1 trong 5 nhãn: daisy, dandelion, roses, sunflowers, tulips; kèm bản đồ nhiệt Grad-CAM cho biết vùng ảnh ảnh hưởng đến nhãn |
| 2 | Phát hiện đối tượng | YOLO11n pretrained, không huấn luyện thêm | COCO (80 lớp); đánh giá trên COCO128 | Vẽ hộp + nhãn cho vật thể thuộc 80 lớp COCO |
| 3 | Tìm kiếm ảnh | CLIP ViT-B/32 + FAISS `IndexFlatIP` | COCO128 + 500 ảnh hoa | Tìm ảnh giống nhất theo câu mô tả (tiếng Anh) hoặc ảnh mẫu |
| 4 | Chatbot chăm sóc khách hàng | Qwen2.5-Instruct (0.5B CPU / 1.5B GPU) + MiniLM đa ngôn ngữ + FAISS | 6 tài liệu chính sách của cửa hàng giả định **ShopLite** (tiếng Việt) | Trả lời dựa trên tài liệu, có ghi nguồn |

## 2. Chỉ số (đo trên tập kiểm tra, sinh tự động bởi `prepare.py`)

<!-- METRICS:START -->
_chưa có — chạy `python scripts/prepare.py` rồi `python scripts/fill_docs.py`_
<!-- METRICS:END -->

Độ trễ và RAM: xem mục **Đo hiệu năng** trong `README.md`.

## 3. Giới hạn đã biết

- **Phân loại:** chỉ biết 5 loài hoa. Ảnh ngoài phân phối (xe, người, hoa khác) vẫn bị gán một trong 5 nhãn; hệ thống chỉ cảnh báo "không chắc" khi điểm cao nhất < 0,5 — ngưỡng này **chưa được hiệu chỉnh**, nên vẫn có thể báo sai với độ tin cậy cao. Ảnh nhiều loài hoa trong một khung, hoặc chụp cận một bộ phận, dễ sai.
- **Grad-CAM:** chỉ cho thấy vùng ảnh ảnh hưởng đến điểm số của nhãn, **không** chứng minh mô hình hiểu đúng hoa. Với ảnh ngoài 5 loài, bản đồ vẫn tô sáng một vùng nên dễ tạo cảm giác tin cậy giả; mô hình cũng có thể dựa vào nền (cỏ, lá, chậu) thay vì bông hoa, và chỉ nhìn bản đồ thì khó phân biệt hai trường hợp này. Độ phân giải thô (lưới 7×7 phóng lên) nên ranh giới mờ. Chưa đánh giá định lượng độ trung thực của bản đồ (ví dụ deletion/insertion). Chỉ hiển thị phần cắt giữa 224×224 của ảnh.
- **Phát hiện:** mAP đo trên COCO128 (128 ảnh, một phần nằm trong dữ liệu huấn luyện gốc của YOLO) nên **lạc quan** hơn hiệu năng thật. Yếu với vật thể nhỏ, bị che khuất, ánh sáng kém; không có lớp ngoài 80 lớp COCO.
- **Tìm ảnh:** kho chỉ ~600 ảnh; truy vấn chữ **chỉ tiếng Anh** (CLIP gốc). Kết quả luôn trả đủ k ảnh dù không có ảnh nào thật sự khớp.
- **Chatbot:** LLM nhỏ (0.5B–1.5B) có thể diễn đạt sai hoặc bịa dù đã có tài liệu; chỉ biết 6 tài liệu ShopLite (cửa hàng **giả định**, hotline 1900 0000 là giả). Bộ đánh giá Hit@k chỉ đo bước *truy xuất* đúng tài liệu, **không** đo câu trả lời cuối có đúng hay không.
- Chưa đo công bằng theo điều kiện chụp/nguồn ảnh; chưa có bộ kiểm thử đối kháng ngoài vài ca prompt injection thử tay.

## 4. Rủi ro

| Rủi ro | Mức | Giảm thiểu hiện có |
|---|---|---|
| Người dùng tin câu trả lời sai của chatbot (chính sách, tiền, hạn đổi trả) | Trung bình | Trả lời kèm nguồn `[tên_file]`; prompt yêu cầu từ chối khi tài liệu không có; giao diện hiển thị đoạn tài liệu đã dùng |
| Prompt injection qua câu hỏi hoặc nội dung tài liệu | Trung bình | System prompt coi tài liệu là dữ liệu, không phải lệnh; giới hạn 1.000 ký tự/câu hỏi. Chưa có bộ test tự động |
| Ảnh riêng tư (có mặt người) bị tải lên | Thấp–Trung bình | Ảnh chỉ xử lý trong bộ nhớ, **không lưu**; giới hạn 8 MB; không ghi ảnh vào log. Không nên tải ảnh người khác khi chưa được phép |
| Lạm dụng/quá tải API công khai | Thấp | Giới hạn dung lượng và độ dài đầu vào; **chưa có** rate-limit hay xác thực |
| Thiên lệch dữ liệu (hoa chụp ngoài trời châu Âu/Mỹ; COCO thiên về bối cảnh phương Tây) | Thấp | Chỉ nêu rõ trong tài liệu này |

## 5. Cách dùng đúng / sai

**Đúng:** học tập, demo kỹ thuật, dùng Grad-CAM để phát hiện mô hình học nhầm nền, thử nghiệm kiến trúc `core/ ↔ api/ ↔ giao diện`; nhận diện 5 loài hoa trong ảnh rõ nét; tìm ảnh trong kho nhỏ bằng câu tiếng Anh ngắn; hỏi chatbot về 6 chủ đề trong `data/kb/` rồi **đối chiếu nguồn** được trích.

**Sai:** xem Grad-CAM là bằng chứng dự đoán đúng; chẩn đoán cây trồng hay nhận diện loài hoa độc/ăn được; giám sát, đếm người hay ra quyết định về con người bằng phát hiện đối tượng; xem chatbot như tư vấn pháp lý/tài chính hoặc thay cho nhân viên hỗ trợ thật; đưa dữ liệu cá nhân nhạy cảm vào ô chat; triển khai sản phẩm thật mà chưa có xác thực, rate-limit và bộ đánh giá riêng.

## 6. Nguồn và giấy phép

TF Flowers (Google, CC-BY) · COCO / COCO128 (Ultralytics) · ResNet-18 (torchvision) · YOLO11n (Ultralytics, AGPL-3.0) · CLIP (OpenAI) · paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers) · Qwen2.5-Instruct (Alibaba). Kiểm tra giấy phép từng thành phần trước khi dùng thương mại.

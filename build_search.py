import core.retrieval as ret
import config
import shutil
import time

print("1. Đang gom ảnh vào kho Gallery...")
GALLERY = config.DATA_DIR / "gallery"
GALLERY.mkdir(exist_ok=True)
items = []

FOOD_DIR = config.DATA_DIR / "vietnamese_foods"
classes = [d.name for d in FOOD_DIR.iterdir() if d.is_dir()]

# Copy ảnh sang thư mục gallery để làm kho tìm kiếm
for c in classes:
    files = sorted((FOOD_DIR / c).glob("*.jpg"))
    for p in files[:100]:  # Lấy tối đa 100 ảnh mỗi món
        dst = GALLERY / f"{c}_{p.name}"
        shutil.copy(p, dst)
        items.append({"path": str(dst.relative_to(config.ROOT)).replace("\\", "/"), "label": c, "source": "vietnamese_foods"})
        
print(f"2. Bắt đầu dùng AI CLIP để đọc và mã hóa {len(items)} ảnh...")
print("Quá trình này mất khoảng 1-2 phút, bạn kiên nhẫn chờ nhé...")
t0 = time.time()
encoder = ret.ClipEncoder()
ret.build_index(encoder, items)
print(f"\n=> XONG! Đã lập chỉ mục xong trong {time.time() - t0:.0f} giây.")
print("=> Tính năng Tìm Kiếm Ảnh đã sẵn sàng!")
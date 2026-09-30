from icrawler.builtin import BingImageCrawler
from pathlib import Path

FOOD_DIR = Path("data/vietnamese_foods")
FOOD_DIR.mkdir(parents=True, exist_ok=True)

# Để nguyên 5 món, món nào nãy lỡ tải được vài tấm nó sẽ tự động tải bù cho đủ
mon_an = ["Phở bò", "Bánh mì Việt Nam", "Bún chả", "Gỏi cuốn", "Bánh xèo"]

for mon in mon_an:
    print(f"\n--- Đang tải ảnh món: {mon} ---")
    save_dir = FOOD_DIR / mon
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Sử dụng BingImageCrawler thay vì Google/DuckDuckGo
    crawler = BingImageCrawler(storage={'root_dir': str(save_dir)})
    crawler.crawl(keyword=mon, max_num=120)

print("\nHoàn tất tải toàn bộ dữ liệu bằng Bing!")
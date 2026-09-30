"""Kiểm thử khói toàn bộ API đang chạy (tách từ notebook mục 6).

Chạy sau khi đã `python scripts/prepare.py` và bật server:
    python scripts/smoke_test.py [--api http://localhost:8000]
"""
import argparse
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8000")
    api = ap.parse_args().api.rstrip("/")

    flower_img = next((ROOT / "data/flowers/flower_photos/sunflowers").glob("*.jpg")).read_bytes()
    coco_img = sorted((ROOT / "data/coco128/images/train2017").glob("*.jpg"))[0].read_bytes()

    r = requests.post(f"{api}/api/classify", files={"file": flower_img}, data={"top_k": 3})
    r.raise_for_status()
    print("classify:", r.json()["predictions"][0], r.headers["X-Process-Time-ms"], "ms")

    r = requests.post(f"{api}/api/classify/explain", files={"file": flower_img}, data={"top_k": 3})
    r.raise_for_status()
    assert r.json()["overlay"].startswith("data:image/jpeg;base64,")
    print("classify/explain:", r.json()["target"], r.headers["X-Process-Time-ms"], "ms")

    r = requests.post(f"{api}/api/detect", files={"file": coco_img}, data={"conf": 0.3})
    r.raise_for_status()
    print("detect:", r.json()["summary"])

    r = requests.post(f"{api}/api/search/text", json={"query": "a red flower", "k": 3})
    r.raise_for_status()
    print("search/text:", [(x["label"], x["score"]) for x in r.json()["results"]])
    assert requests.get(api + r.json()["results"][0]["url"]).headers["content-type"].startswith("image/")

    r = requests.post(f"{api}/api/search/image", files={"file": flower_img}, data={"k": 3})
    r.raise_for_status()
    print("search/image:", [(x["label"], x["score"]) for x in r.json()["results"]])

    r = requests.post(f"{api}/api/chat/sync", json={"message": "Phí giao hàng cho đơn 200.000đ là bao nhiêu?"})
    r.raise_for_status()
    print("chat:", r.json()["answer"][:200])

    # Streaming: đọc các sự kiện SSE
    with requests.post(f"{api}/api/chat", json={"message": "Bảo hành đồ gia dụng bao lâu?"}, stream=True) as s:
        s.encoding = "utf-8"
        events = [json.loads(line[6:]) for line in s.iter_lines(decode_unicode=True) if line.startswith("data: ")]
    print("chat SSE:", [e["type"] for e in events][:5], "…", "".join(e.get("text", "") for e in events)[:120])

    # Các ca lỗi phải trả mã HTTP đúng
    assert requests.post(f"{api}/api/classify", files={"file": b"not an image"}).status_code == 400
    assert requests.post(f"{api}/api/classify/explain", files={"file": flower_img}, data={"target": "cactus"}).status_code == 422
    assert requests.post(f"{api}/api/search/text", json={"query": ""}).status_code == 422
    print("✅ Tất cả smoke test đạt")


if __name__ == "__main__":
    main()

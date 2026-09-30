"""Đo độ trễ p50/p95 của từng endpoint và RAM của backend (yêu cầu C5).

Chạy sau khi đã `python scripts/prepare.py` và bật server (uvicorn hoặc link đã triển khai):
    python scripts/benchmark.py                                   # http://localhost:8000
    python scripts/benchmark.py --api https://<space>.hf.space --n 30 --chat-n 5

Kết quả in ra bảng Markdown (dán vào README) và lưu vào docs/benchmark.json.
Đo độ trễ *tuần tự* (1 người dùng) sau vài lần khởi động; thêm --workers 4 để đo khi có 4 người dùng đồng thời.
Chỉ dùng thư viện đã có sẵn trong requirements (requests, pillow).
"""
import argparse
import io
import json
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sample_image() -> bytes:
    """Ưu tiên ảnh hoa thật (nếu đã chạy prepare.py); nếu không, tạo ảnh 640x480 để đo tốc độ."""
    found = next((ROOT / "data/flowers/flower_photos/sunflowers").glob("*.jpg"), None) \
        if (ROOT / "data/flowers/flower_photos/sunflowers").exists() else None
    if found:
        return found.read_bytes()
    buf = io.BytesIO()
    Image.effect_noise((640, 480), 64).convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


def run(name: str, call, n: int, workers: int) -> dict:
    for _ in range(min(3, n)):  # khởi động: lần đầu thường chậm hơn (cache, JIT)
        call()

    def timed(_):
        t0 = time.perf_counter()
        r = call()
        return (time.perf_counter() - t0) * 1000, r.status_code

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(timed, range(n)))
    wall = time.perf_counter() - t_start
    ms = [t for t, _ in results]
    errors = sum(1 for _, code in results if code != 200)
    row = {"endpoint": name, "n": n, "p50_ms": round(pct(ms, 0.50), 1), "p95_ms": round(pct(ms, 0.95), 1),
           "mean_ms": round(statistics.fmean(ms), 1), "rps": round(n / wall, 2), "errors": errors}
    print(f"  {name:<22} p50={row['p50_ms']:>8} ms  p95={row['p95_ms']:>8} ms  rps={row['rps']:<6} lỗi={errors}/{n}")
    return row


def hardware() -> dict:
    info = {"os": platform.platform(), "cpu": platform.processor() or platform.machine(), "python": platform.python_version()}
    try:
        import os
        import psutil
        info["cpu_cores"] = os.cpu_count()
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "không có (CPU)"
    except ImportError:
        pass
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=30, help="số request mỗi endpoint (classify/detect/search)")
    ap.add_argument("--chat-n", type=int, default=5, help="số request cho chatbot (LLM chậm hơn nhiều)")
    ap.add_argument("--workers", type=int, default=1, help="số người dùng đồng thời")
    args = ap.parse_args()
    api = args.api.rstrip("/")

    health = requests.get(f"{api}/api/health", timeout=30).json()
    print("Backend:", api, "| thiết bị:", health.get("device"), "| mô hình:", health.get("models"))
    img = sample_image()
    up = lambda: {"file": ("x.jpg", img, "image/jpeg")}

    cases = {
        "classify": (lambda: requests.post(f"{api}/api/classify", files=up(), data={"top_k": 3}, timeout=120), args.n),
        "classify/explain": (lambda: requests.post(f"{api}/api/classify/explain", files=up(), data={"top_k": 3}, timeout=120), args.n),
        "detect": (lambda: requests.post(f"{api}/api/detect", files=up(), data={"conf": 0.25}, timeout=120), args.n),
        "search/text": (lambda: requests.post(f"{api}/api/search/text", json={"query": "a red flower", "k": 8}, timeout=120), args.n),
        "search/image": (lambda: requests.post(f"{api}/api/search/image", files=up(), data={"k": 8}, timeout=120), args.n),
        "chat/sync": (lambda: requests.post(f"{api}/api/chat/sync", json={"message": "Phí giao hàng cho đơn 200.000đ là bao nhiêu?"}, timeout=300), args.chat_n),
    }
    rows = []
    for name, (call, n) in cases.items():
        if not health["models"].get({"classify": "classifier", "classify/explain": "classifier", "detect": "detector", "search/text": "retrieval",
                                     "search/image": "retrieval", "chat/sync": "llm"}[name]):
            print(f"  {name:<22} bỏ qua (mô hình chưa nạp)")
            continue
        rows.append(run(name, call, n, args.workers))

    ram = requests.get(f"{api}/api/health", timeout=30).json().get("ram_mb")  # đo SAU khi đã chạy đủ 4 mô hình
    result = {"api": api, "workers": args.workers, "device": health.get("device"), "ram_mb_after_load": ram,
              "hardware": hardware(), "rows": rows}
    out = ROOT / "docs" / "benchmark.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nRAM tiến trình backend: {ram} MB | phần cứng (máy chạy script này): {result['hardware']}")
    print("\n| Endpoint | n | p50 (ms) | p95 (ms) | req/s | lỗi |\n|---|---|---|---|---|---|")
    for r in rows:
        print(f"| `{r['endpoint']}` | {r['n']} | {r['p50_ms']} | {r['p95_ms']} | {r['rps']} | {r['errors']} |")
    print(f"\nĐã lưu {out.relative_to(ROOT)}. Ghi rõ phần cứng của MÁY CHẠY BACKEND vào README (nếu backend ở xa, phần cứng ở trên là của máy đo).")


if __name__ == "__main__":
    main()

"""Điền số đo THẬT vào README.md và MODEL_CARD.md từ artifacts/*.json và docs/benchmark.json.

    python scripts/fill_docs.py

Chỉ thay nội dung giữa cặp thẻ <!-- METRICS:START --> … <!-- METRICS:END --> và
<!-- BENCH:START --> … <!-- BENCH:END -->; phần còn lại của file giữ nguyên.
Chạy lại sau mỗi lần `prepare.py` hoặc `benchmark.py`.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def metrics_table() -> str:
    cls, det = load(ART / "classifier/metrics.json"), load(ART / "detector/metrics.json")
    ret, rag = load(ART / "retrieval/metrics.json"), load(ART / "rag_metrics.json")
    missing = "_chưa có — chạy `python scripts/prepare.py`_"
    rows = ["| Ứng dụng | Chỉ số | Kết quả |", "|---|---|---|"]
    rows.append("| 1. Phân loại (ResNet-18, TF Flowers) | Test accuracy · macro-F1 | " + (
        f"**{cls['test_accuracy']:.3f}** · **{cls['test_macro_f1']:.3f}** ({cls['epochs']} epoch)" if cls else missing) + " |")
    rows.append("| 2. Phát hiện (YOLO11n, COCO128) | mAP50 · mAP50-95 | " + (
        f"**{det['mAP50']:.3f}** · **{det['mAP50_95']:.3f}**" if det else missing) + " |")
    if ret:
        t10 = ret.get("text_to_image_precision@10", {})
        t10_avg = sum(t10.values()) / len(t10) if t10 else float("nan")
        cell = (f"P@5 ảnh→ảnh **{ret['image_to_image_precision@5']:.3f}** · "
                f"P@10 chữ→ảnh (TB 5 loài) **{t10_avg:.3f}** · kho {ret['gallery_size']} ảnh")
    else:
        cell = missing
    rows.append(f"| 3. Tìm ảnh (CLIP + FAISS) | Precision@k | {cell} |")
    rows.append("| 4. Chatbot RAG (MiniLM + " + (rag["llm"].split("/")[-1] if rag else "Qwen2.5") + ") | Hit@1 · Hit@3 | " + (
        f"**{rag['hit@1']:.3f}** · **{rag['hit@3']:.3f}** ({rag['n_questions']} câu hỏi)" if rag else missing) + " |")
    return "\n".join(rows)


def bench_table() -> str:
    b = load(ROOT / "docs/benchmark.json")
    if not b:
        return "_chưa đo — chạy `python scripts/benchmark.py` rồi `python scripts/fill_docs.py`_"
    hw = b["hardware"]
    hw_txt = ", ".join(f"{k}: {v}" for k, v in hw.items())
    lines = [f"Thiết bị backend: `{b['device']}` · số người dùng đồng thời: {b['workers']} · "
             f"RAM tiến trình backend sau khi nạp và chạy đủ mô hình: **{b['ram_mb_after_load']} MB**",
             f"Phần cứng máy chạy script đo: {hw_txt}", "",
             "| Endpoint | n | p50 (ms) | p95 (ms) | req/s | lỗi |", "|---|---|---|---|---|---|"]
    lines += [f"| `{r['endpoint']}` | {r['n']} | {r['p50_ms']} | {r['p95_ms']} | {r['rps']} | {r['errors']} |" for r in b["rows"]]
    return "\n".join(lines)


def replace_block(text: str, tag: str, body: str) -> tuple[str, bool]:
    pattern = re.compile(rf"(<!-- {tag}:START -->)(.*?)(<!-- {tag}:END -->)", re.S)
    if not pattern.search(text):
        return text, False
    return pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text), True


def main():
    blocks = {"METRICS": metrics_table(), "BENCH": bench_table()}
    for name in ("README.md", "MODEL_CARD.md"):
        path = ROOT / name
        if not path.exists():
            continue
        text, touched = path.read_text(encoding="utf-8"), []
        for tag, body in blocks.items():
            text, ok = replace_block(text, tag, body)
            if ok:
                touched.append(tag)
        path.write_text(text, encoding="utf-8")
        print(f"{name}: cập nhật {', '.join(touched) or 'không có thẻ nào'}")


if __name__ == "__main__":
    main()

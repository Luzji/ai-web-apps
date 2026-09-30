"""Chuẩn bị dữ liệu, huấn luyện và đánh giá cho cả 4 ứng dụng AI.

Tách từ các ô "chạy một lần" của notebook AI_Web_Apps_Streamlit_React.ipynb (mục 1-5)
để chạy được ngoài Colab. Chạy từ thư mục gốc dự án:

    python scripts/prepare.py                       # mọi bước còn thiếu
    python scripts/prepare.py classifier detector   # chỉ một số bước
    python scripts/prepare.py --force               # huấn luyện / lập chỉ mục lại

Các bước theo thứ tự: data -> classifier -> detector -> retrieval -> rag
Đầu ra: artifacts/classifier/ (model.pt, classes.json, metrics.json, confusion_matrix.png)
        artifacts/detector/ (yolo11n.pt, metrics.json)
        artifacts/retrieval/ (index.faiss, meta.json, metrics.json) + data/gallery/
        artifacts/rag_metrics.json
"""
import argparse
import json
import random
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để import config, core

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import config  # noqa: E402
from config import ART_DIR, DATA_DIR, DEVICE  # noqa: E402

SEED = 42
FAST = DEVICE == "cpu"  # CPU: ít epoch và ít ảnh train để chạy trong vài phút
FLOWERS_DIR = DATA_DIR / "flowers" / "flower_photos"
COCO_DIR = DATA_DIR / "coco128"

# Câu hỏi đánh giá bước truy xuất của chatbot: (câu hỏi, file tài liệu đúng)
EVAL_QA = [
    ("Tôi được đổi trả trong bao nhiêu ngày?", "doi_tra.md"),
    ("Mỹ phẩm đã mở nắp có trả lại được không?", "doi_tra.md"),
    ("Đơn bao nhiêu tiền thì được miễn phí giao hàng?", "giao_hang.md"),
    ("Giao hỏa tốc mất bao lâu?", "giao_hang.md"),
    ("Đơn 12 triệu có thanh toán khi nhận hàng được không?", "thanh_toan.md"),
    ("Trả góp 0% áp dụng cho đơn từ bao nhiêu?", "thanh_toan.md"),
    ("Tai nghe được bảo hành bao lâu?", "bao_hanh.md"),
    ("Quên mật khẩu thì làm sao?", "tai_khoan.md"),
    ("Hạng Vàng được giảm thêm bao nhiêu phần trăm?", "khach_hang_than_thiet.md"),
    ("Một điểm thưởng quy đổi được bao nhiêu tiền?", "khach_hang_than_thiet.md"),
]


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print("↓", url)
        urllib.request.urlretrieve(url, dest)
    return dest


def coco_images() -> list[Path]:
    return sorted((COCO_DIR / "images" / "train2017").glob("*.jpg"))


def flower_classes() -> list[str]:
    return sorted(d.name for d in FLOWERS_DIR.iterdir() if d.is_dir())


def done(folder: Path, *names: str, force: bool) -> bool:
    if not force and all((folder / n).exists() for n in names):
        print(f"đã có kết quả trong {folder.relative_to(config.ROOT)} — bỏ qua (thêm --force để làm lại)")
        return True
    return False


# ---------- Dữ liệu (notebook mục 1) ----------
def step_data(force: bool = False):
    if not FLOWERS_DIR.exists():
        tgz = download("https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
                       DATA_DIR / "flower_photos.tgz")
        with tarfile.open(tgz) as t:
            try:
                t.extractall(DATA_DIR / "flowers", filter="data")
            except TypeError:  # Python cũ chưa có tham số filter
                t.extractall(DATA_DIR / "flowers")
        tgz.unlink()
    (FLOWERS_DIR / "LICENSE.txt").unlink(missing_ok=True)
    print("Flowers:", {c: len(list((FLOWERS_DIR / c).glob("*.jpg"))) for c in flower_classes()})

    if not COCO_DIR.exists():  # COCO128: 128 ảnh COCO kèm nhãn YOLO
        z = download("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip", DATA_DIR / "coco128.zip")
        with zipfile.ZipFile(z) as f:
            f.extractall(DATA_DIR)
        z.unlink()
    print("COCO128:", len(coco_images()), "ảnh")


# ---------- Ứng dụng 1: huấn luyện + đánh giá bộ phân loại (notebook mục 2) ----------
def step_classifier(force: bool = False):
    out = ART_DIR / "classifier"
    if done(out, "model.pt", "classes.json", "metrics.json", force=force):
        return
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder

    import core.classifier as clf

    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    base = ImageFolder(FLOWERS_DIR)
    classes, targets = base.classes, np.array(base.targets)
    all_idx = np.arange(len(targets))
    train_idx, tmp_idx = train_test_split(all_idx, test_size=0.2, stratify=targets, random_state=SEED)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, stratify=targets[tmp_idx], random_state=SEED)
    if FAST:  # CPU: dùng 800 ảnh train để chạy trong vài phút
        train_idx = np.random.default_rng(SEED).choice(train_idx, size=min(800, len(train_idx)), replace=False)

    train_ds = Subset(ImageFolder(FLOWERS_DIR, transform=clf.TRAIN_TF), train_idx)
    val_ds = Subset(ImageFolder(FLOWERS_DIR, transform=clf.EVAL_TF), val_idx)
    test_ds = Subset(ImageFolder(FLOWERS_DIR, transform=clf.EVAL_TF), test_idx)
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=2, pin_memory=DEVICE == "cuda")
    train_dl, val_dl, test_dl = loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)

    # Lưu cách chia để mọi người tái lập đúng tập test
    (out / "split.json").write_text(json.dumps(
        {"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()}))
    print(f"Lớp: {classes}\ntrain={len(train_ds)} · val={len(val_ds)} · test={len(test_ds)}")

    epochs = 1 if FAST else 5
    model = clf.build_model(len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=epochs * len(train_dl))
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def run_epoch(dl, train: bool):
        model.train(train)
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.set_grad_enabled(train), torch.autocast(DEVICE, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        return loss_sum / total, correct / total

    best_acc, history = 0.0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_dl, True)
        va_loss, va_acc = run_epoch(val_dl, False)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        if va_acc > best_acc:  # chọn checkpoint theo tập validation, không nhìn tập test
            best_acc = va_acc
            torch.save(model.state_dict(), out / "model.pt")
        print(f"epoch {epoch}/{epochs} · train loss {tr_loss:.3f} acc {tr_acc:.3f} · "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} · {time.time() - t0:.0f}s")
    (out / "classes.json").write_text(json.dumps(classes))
    print("Val accuracy tốt nhất:", round(best_acc, 4))

    # Đánh giá MỘT LẦN trên tập test với checkpoint tốt nhất
    model.load_state_dict(torch.load(out / "model.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for x, y in test_dl:
            y_pred += model(x.to(DEVICE)).argmax(1).cpu().tolist()
            y_true += y.tolist()
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))
    metrics = {"test_accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
               "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
               "epochs": epochs, "history": history, "model": "resnet18-imagenet-finetune"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    try:  # lưu ma trận nhầm lẫn thành ảnh (dùng được trong README / báo cáo)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=classes).plot(cmap="Blues", xticks_rotation=30)
        plt.title(f"Ma trận nhầm lẫn — test accuracy {metrics['test_accuracy']:.3f}")
        plt.tight_layout()
        plt.savefig(out / "confusion_matrix.png", dpi=150)
        plt.close()
    except ImportError:
        print("Bỏ qua ảnh ma trận nhầm lẫn (chưa cài matplotlib)")
    print({k: v for k, v in metrics.items() if k != "history"})


# ---------- Ứng dụng 2: tải trọng số YOLO + đo mAP (notebook mục 3) ----------
def step_detector(force: bool = False):
    out = ART_DIR / "detector"
    if done(out, "yolo11n.pt", "metrics.json", force=force):
        return
    from ultralytics.utils.downloads import attempt_download_asset

    import core.detector as det

    out.mkdir(parents=True, exist_ok=True)
    attempt_download_asset(str(out / "yolo11n.pt"))  # tải trọng số vào artifacts/detector/
    detector = det.ObjectDetector()
    # mAP là chỉ số chuẩn của object detection (COCO128 có nhãn)
    val = detector.model.val(data="coco128.yaml", imgsz=640, batch=16, device=detector.device, plots=False, verbose=False)
    metrics = {"mAP50": float(val.box.map50), "mAP50_95": float(val.box.map), "dataset": "coco128", "model": "yolo11n"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)


# ---------- Ứng dụng 3: dựng kho ảnh + chỉ mục CLIP/FAISS + đo Precision@k (notebook mục 4) ----------
def step_retrieval(force: bool = False):
    out = ART_DIR / "retrieval"
    if done(out, "index.faiss", "meta.json", "metrics.json", force=force):
        return
    import core.detector as det
    import core.retrieval as ret

    root = config.ROOT
    gallery = DATA_DIR / "gallery"  # kho ảnh riêng: khi triển khai chỉ cần mang theo thư mục này
    gallery.mkdir(parents=True, exist_ok=True)
    classes = flower_classes()

    # meta.json lưu đường dẫn TƯƠNG ĐỐI (dạng posix) để index chạy được trên máy khác / Docker
    detector = det.ObjectDetector()
    items = []
    for p in coco_images():
        summary = detector.detect(Image.open(p), conf=0.4)[0]["summary"]  # dùng Ứng dụng 2 để gắn nhãn ảnh COCO
        label = ", ".join(sorted(summary, key=summary.get, reverse=True)[:2]) or "coco"
        dst = gallery / f"coco_{p.name}"
        shutil.copy(p, dst)
        items.append({"path": dst.relative_to(root).as_posix(), "label": label, "source": "coco128"})
    rng = random.Random(SEED)
    for c in classes:
        files = sorted((FLOWERS_DIR / c).glob("*.jpg"))
        for p in rng.sample(files, min(100, len(files))):
            dst = gallery / f"{c}_{p.name}"
            shutil.copy(p, dst)
            items.append({"path": dst.relative_to(root).as_posix(), "label": c, "source": "flowers"})
    del detector

    encoder = ret.ClipEncoder()
    t0 = time.time()
    ret.build_index(encoder, items)
    engine = ret.ImageSearch(encoder=encoder)
    print(f"Đã lập chỉ mục {engine.index.ntotal} ảnh trong {time.time() - t0:.0f}s")

    # Đánh giá định lượng trên phần ảnh hoa (có nhãn loài)
    flower_ids = [i for i, m in enumerate(engine.meta) if m["source"] == "flowers"]
    queries = random.Random(SEED).sample(flower_ids, 50)
    p_at_5 = []
    for qi in queries:  # ảnh -> ảnh: bỏ chính nó, xem 5 kết quả đầu có cùng loài không
        img = Image.open(config.resolve_path(engine.meta[qi]["path"]))
        res = [r for r in engine.search_image(img, k=6) if r["id"] != qi][:5]
        p_at_5.append(np.mean([r["label"] == engine.meta[qi]["label"] for r in res]))
    text_p10 = {}
    for c in classes:  # văn bản -> ảnh (zero-shot): "a photo of tulips" -> 10 kết quả đầu có đúng loài không
        res = engine.search_text(f"a photo of {c}", k=10)
        text_p10[c] = float(np.mean([r["label"] == c for r in res]))
    metrics = {"image_to_image_precision@5": float(np.mean(p_at_5)), "text_to_image_precision@10": text_p10,
               "gallery_size": engine.index.ntotal, "model": config.CLIP_MODEL}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


# ---------- Ứng dụng 4: đo bước truy xuất của chatbot RAG (notebook mục 5) ----------
def step_rag(force: bool = False):
    if done(ART_DIR, "rag_metrics.json", force=force):
        return
    import core.llm as llm_mod

    chunks = llm_mod.load_chunks()
    print(f"{len(chunks)} đoạn từ {len({c['source'] for c in chunks})} tài liệu")
    retriever = llm_mod.Retriever(chunks)
    hits1 = hits3 = 0
    for q, src in EVAL_QA:  # câu hỏi -> tài liệu đúng có nằm trong top-k không?
        found = [r["source"] for r in retriever.search(q, k=3)]
        hits1 += found[0] == src
        hits3 += src in found
        print(("✅" if src in found else "❌"), q, "→", found)
    metrics = {"hit@1": hits1 / len(EVAL_QA), "hit@3": hits3 / len(EVAL_QA), "n_questions": len(EVAL_QA),
               "embed_model": config.EMBED_MODEL, "llm": config.LLM_MODEL}
    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "rag_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(metrics)


STEPS = {"data": step_data, "classifier": step_classifier, "detector": step_detector,
         "retrieval": step_retrieval, "rag": step_rag}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", help="bước cần chạy (mặc định: tất cả): " + ", ".join(STEPS))
    ap.add_argument("--force", action="store_true", help="huấn luyện / lập chỉ mục lại dù đã có kết quả")
    args = ap.parse_args()
    unknown = [s for s in args.steps if s not in STEPS]
    if unknown:
        ap.error(f"bước không hợp lệ: {unknown}. Chọn trong: {list(STEPS)}")

    print(f"Thiết bị: {DEVICE} | FAST mode: {FAST} | LLM: {config.LLM_MODEL}")
    for name, fn in STEPS.items():  # luôn chạy theo thứ tự phụ thuộc
        if args.steps and name not in args.steps:
            continue
        print(f"\n=== {name} ===")
        t0 = time.time()
        fn(args.force)
        print(f"--- {name} xong sau {time.time() - t0:.0f}s")
    print("\n✅ Xong. Bật server: uvicorn api.main:app --port 8000")


if __name__ == "__main__":  # bắt buộc trên Windows: DataLoader(num_workers>0) tạo tiến trình con
    main()

import importlib
import numpy as np
import json
import torch
from torch import nn
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

import config
import core.classifier as clf

print("1. Đang chuẩn bị dữ liệu...")
FOOD_DIR = config.DATA_DIR / "vietnamese_foods" 
base = ImageFolder(FOOD_DIR)
classes, targets = base.classes, np.array(base.targets)
all_idx = np.arange(len(targets))

# Chia dữ liệu: 80% train, 10% val, 10% test
train_idx, tmp_idx = train_test_split(all_idx, test_size=0.2, stratify=targets, random_state=42)
val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, stratify=targets[tmp_idx], random_state=42)

train_ds = Subset(ImageFolder(FOOD_DIR, transform=clf.TRAIN_TF), train_idx)
val_ds = Subset(ImageFolder(FOOD_DIR, transform=clf.EVAL_TF), val_idx)
test_ds = Subset(ImageFolder(FOOD_DIR, transform=clf.EVAL_TF), test_idx)

# num_workers=0 để không lỗi đơ máy trên Windows
loader = lambda ds, shuffle: DataLoader(ds, batch_size=32, shuffle=shuffle, num_workers=0, pin_memory=config.DEVICE == "cuda")
train_dl, val_dl, test_dl = loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)

split_data = {"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()}

# ĐÃ FIX: Tự động tạo thư mục nếu chưa có
(config.ART_DIR / "classifier").mkdir(parents=True, exist_ok=True)
(config.ART_DIR / "classifier" / "split.json").write_text(json.dumps(split_data))

print(f"Lớp: {classes}\ntrain={len(train_ds)} · val={len(val_ds)} · test={len(test_ds)}")

print("\n2. Bắt đầu huấn luyện AI (sẽ mất vài phút)...")
EPOCHS = 5
model = clf.build_model(len(classes)).to(config.DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=EPOCHS * len(train_dl))
use_amp = config.DEVICE == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

def run_epoch(dl, train: bool):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in dl:
        x, y = x.to(config.DEVICE, non_blocking=True), y.to(config.DEVICE, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(config.DEVICE, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); scheduler.step()
        loss_sum += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item(); total += len(y)
    return loss_sum / total, correct / total

best_acc, history = 0.0, []
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_dl, True)
    va_loss, va_acc = run_epoch(val_dl, False)
    history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
    if va_acc > best_acc:
        best_acc = va_acc
        torch.save(model.state_dict(), config.ART_DIR / "classifier" / "model.pt")
    print(f"Epoch {epoch}/{EPOCHS} · train loss {tr_loss:.3f} acc {tr_acc:.3f} · val loss {va_loss:.3f} acc {va_acc:.3f} · {time.time()-t0:.0f}s")

(config.ART_DIR / "classifier" / "classes.json").write_text(json.dumps(classes))
print(f"\nHuấn luyện xong! Val accuracy tốt nhất: {best_acc:.4f}")

print("\n3. Đang xuất Báo cáo và Biểu đồ...")
model.load_state_dict(torch.load(config.ART_DIR / "classifier" / "model.pt", map_location=config.DEVICE, weights_only=True))
model.eval()
y_true, y_pred = [], []
with torch.inference_mode():
    for x, y in test_dl:
        y_pred += model(x.to(config.DEVICE)).argmax(1).cpu().tolist(); y_true += y.tolist()

print(classification_report(y_true, y_pred, target_names=classes, digits=3))
metrics = {"test_accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
           "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
           "epochs": EPOCHS, "history": history, "model": "resnet18-imagenet-finetune"}
(config.ART_DIR / "classifier" / "metrics.json").write_text(json.dumps(metrics, indent=2))

ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=classes).plot(cmap="Blues", xticks_rotation=30)
plt.title(f"Ma tran nham lan - test accuracy {metrics['test_accuracy']:.3f}")
plt.show()
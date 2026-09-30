"""Grad-CAM: tô sáng vùng ảnh khiến mô hình chọn một nhãn (Selvaraju et al., 2017).

Ý tưởng: lấy bản đồ đặc trưng A của tầng tích chập cuối (ResNet-18: layer4, 512 kênh x 7 x 7), tính đạo hàm
của điểm số lớp c theo A, lấy trung bình theo không gian làm trọng số kênh, rồi
    CAM = ReLU( tổng_kênh( trọng_số * A ) )
Chuẩn hoá về [0, 1], phóng to bằng nội suy và phủ lên ảnh. Chỉ dùng lại mô hình đã huấn luyện, không huấn luyện thêm.

Hai hàm `jet` và `overlay` chỉ dùng numpy/PIL nên test được mà không cần mô hình.
"""
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Hook được gắn tạm lên mô hình dùng chung → chỉ cho một request Grad-CAM chạy tại một thời điểm.
_LOCK = threading.Lock()


def grad_cam(model: torch.nn.Module, layer: torch.nn.Module, x: torch.Tensor, target_idx: int | None = None):
    """x: tensor (1, 3, H, W) đã chuẩn hoá. Trả về (xác suất mọi lớp, chỉ số lớp được giải thích, CAM numpy 2D trong [0, 1])."""
    store: dict[str, torch.Tensor] = {}
    with _LOCK:
        handle = layer.register_forward_hook(lambda _m, _i, out: store.__setitem__("acts", out))
        try:
            with torch.enable_grad():  # predict() chạy trong inference_mode, còn Grad-CAM cần đạo hàm
                logits = model(x.clone().requires_grad_(True))
                idx = int(logits[0].argmax()) if target_idx is None else int(target_idx)
                acts = store["acts"]
                (grads,) = torch.autograd.grad(logits[0, idx], acts)  # không ghi .grad vào tham số mô hình
        finally:
            handle.remove()  # luôn gỡ hook, kể cả khi lỗi
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts.detach()).sum(dim=1))[0]
    cam = cam - cam.min()
    peak = cam.max()
    cam = cam / peak if peak > 0 else torch.zeros_like(cam)  # ảnh không kích hoạt gì → bản đồ trống, không chia cho 0
    return logits.softmax(dim=-1)[0].detach().cpu(), idx, cam.cpu().numpy()


def jet(v: np.ndarray) -> np.ndarray:
    """Bảng màu 'jet' (xanh → vàng → đỏ) cho giá trị trong [0, 1]; trả mảng (..., 3) trong [0, 255]."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1) * 255


def overlay(cam: np.ndarray, image: Image.Image, max_alpha: float = 0.6) -> Image.Image:
    """Phủ bản đồ nhiệt lên ảnh. Độ trong suốt tỉ lệ với CAM: vùng ít quan trọng vẫn thấy rõ ảnh gốc."""
    image = image.convert("RGB")
    heat_small = Image.fromarray((np.clip(cam, 0, 1) * 255).astype("uint8"))
    c = np.asarray(heat_small.resize(image.size, Image.BILINEAR), dtype=np.float32) / 255.0
    base = np.asarray(image, dtype=np.float32)
    alpha = (max_alpha * c)[..., None]
    out = (1 - alpha) * base + alpha * jet(c)
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"))

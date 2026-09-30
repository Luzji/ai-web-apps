"""Ứng dụng 1 — Phân loại ảnh (ResNet-18 fine-tune trên bộ Flowers)."""
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

from config import ART_DIR, DEVICE
from core.gradcam import grad_cam, overlay

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
VIEW_TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])  # đúng vùng ảnh mà mô hình nhìn thấy
EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """ResNet-18 ImageNet, thay lớp cuối bằng num_classes đầu ra."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


class ImageClassifier:
    def __init__(self, model_dir: Path = ART_DIR / "classifier", min_confidence: float = 0.5):
        model_dir = Path(model_dir)
        self.classes: list[str] = json.loads((model_dir / "classes.json").read_text(encoding="utf-8"))
        self.model = build_model(len(self.classes), pretrained=False)
        state = torch.load(model_dir / "model.pt", map_location=DEVICE, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(DEVICE).eval()
        self.min_confidence = min_confidence

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        x = EVAL_TF(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        probs = self.model(x).softmax(dim=-1)[0]
        scores, idx = probs.topk(min(top_k, len(self.classes)))
        preds = [{"label": self.classes[i], "score": round(float(s), 4)} for s, i in zip(scores.tolist(), idx.tolist())]
        return {
            "predictions": preds,
            # Ảnh không thuộc 5 loài hoa vẫn bị gán nhãn: báo "không chắc" thay vì khẳng định sai
            "confident": preds[0]["score"] >= self.min_confidence,
        }

    def explain(self, image: Image.Image, target: str | None = None, top_k: int = 3, size: int = 448):
        """Grad-CAM: trả về (kết quả dự đoán + nhãn được giải thích, ảnh đã phủ bản đồ nhiệt).

        target=None → giải thích nhãn dự đoán cao nhất; target="tulips" → hỏi "vì sao có thể là tulips?".
        Ảnh phủ là vùng cắt giữa 224x224 (phóng lên `size`) vì mô hình chỉ nhìn phần này.
        """
        if target is not None and target not in self.classes:
            raise ValueError(f"target phải thuộc {self.classes}")
        rgb = image.convert("RGB")
        x = EVAL_TF(rgb).unsqueeze(0).to(DEVICE)
        probs, idx, cam = grad_cam(self.model, self.model.layer4, x, None if target is None else self.classes.index(target))
        scores, ids = probs.topk(min(top_k, len(self.classes)))
        preds = [{"label": self.classes[i], "score": round(float(s), 4)} for s, i in zip(scores.tolist(), ids.tolist())]
        result = {"predictions": preds, "confident": preds[0]["score"] >= self.min_confidence, "target": self.classes[idx]}
        return result, overlay(cam, VIEW_TF(rgb).resize((size, size), Image.BICUBIC))

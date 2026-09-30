"""Test lõi Grad-CAM với ResNet-18 khởi tạo ngẫu nhiên (không cần trọng số, dữ liệu hay mạng).
Cần torch + torchvision; thiếu thì bỏ qua (CI đã cài bản CPU)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image

from config import DEVICE
from core import gradcam
from core.classifier import ImageClassifier, build_model

CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


@pytest.fixture(scope="module")
def clf():
    torch.manual_seed(0)
    c = ImageClassifier.__new__(ImageClassifier)  # bỏ qua __init__ (không cần file model.pt)
    c.classes, c.min_confidence = CLASSES, 0.5
    c.model = build_model(len(CLASSES), pretrained=False).to(DEVICE).eval()
    return c


@pytest.fixture(scope="module")
def img():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))


def test_explain_shapes_and_target(clf, img):
    result, heat = clf.explain(img, top_k=3)
    assert heat.size == (448, 448) and heat.mode == "RGB"
    assert len(result["predictions"]) == 3 and result["target"] in CLASSES
    assert result["target"] == result["predictions"][0]["label"]  # mặc định giải thích nhãn cao nhất


def test_explain_explicit_target(clf, img):
    result, _ = clf.explain(img, target="tulips")
    assert result["target"] == "tulips"


def test_explain_rejects_unknown_target(clf, img):
    with pytest.raises(ValueError):
        clf.explain(img, target="cactus")


def test_explain_is_deterministic(clf, img):
    a = np.asarray(clf.explain(img)[1])
    b = np.asarray(clf.explain(img)[1])
    assert np.array_equal(a, b)


def test_explain_has_no_side_effects_on_model(clf, img):
    before = clf.predict(img, top_k=5)
    clf.explain(img)
    clf.explain(img, target="daisy")
    assert len(clf.model.layer4._forward_hooks) == 0            # hook đã được gỡ
    assert all(p.grad is None for p in clf.model.parameters())  # không ghi đạo hàm vào tham số
    assert clf.predict(img, top_k=5) == before                  # dự đoán thường không đổi


def test_hook_removed_even_when_forward_fails(clf):
    with pytest.raises(Exception):
        gradcam.grad_cam(clf.model, clf.model.layer4, torch.zeros(1, 1, 8, 8))  # sai số kênh → lỗi giữa chừng
    assert len(clf.model.layer4._forward_hooks) == 0


def test_cam_is_normalised(clf, img):
    x = torch.randn(1, 3, 224, 224).to(DEVICE)
    probs, idx, cam = gradcam.grad_cam(clf.model, clf.model.layer4, x)
    assert cam.shape == (7, 7) and 0.0 <= cam.min() and cam.max() <= 1.0
    assert abs(float(probs.sum()) - 1.0) < 1e-4 and 0 <= idx < len(CLASSES)


def test_overlay_keeps_image_where_cam_is_zero():
    base = Image.new("RGB", (64, 64), (100, 100, 100))
    out = np.asarray(gradcam.overlay(np.zeros((7, 7)), base))
    assert (out == 100).all()


def test_overlay_colours_hot_region_red():
    cam = np.zeros((7, 7)); cam[3, 3] = 1.0
    out = np.asarray(gradcam.overlay(cam, Image.new("RGB", (70, 70), (100, 100, 100))))
    r, g, b = out[35, 35]
    assert r > b and out[0, 0].tolist() == [100, 100, 100]

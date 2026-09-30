"""Test API không cần GPU hay tải mô hình: thay mô hình thật bằng bản giả (dependency injection).

Chạy: python -m pytest -q

Quy ước (yêu cầu C4): mỗi endpoint có ít nhất 1 ca thành công, 1 ca lỗi 4xx do dữ liệu sai
(400 với file không phải ảnh) và 1 ca 422 (thiếu/sai tham số). Endpoint nhận JSON không có
"file sai định dạng" nên ca 4xx của chúng là 422 (Pydantic) và 503 (mô hình chưa nạp).
"""
import io
import os

os.environ["ENABLED_MODELS"] = ""  # không nạp mô hình thật khi khởi động

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import main


class FakeClassifier:
    classes = ["daisy", "roses"]

    def predict(self, image, top_k=3):
        return {"predictions": [{"label": "roses", "score": 0.9}][:top_k], "confident": True}

    def explain(self, image, target=None, top_k=3):
        return {"predictions": [{"label": "roses", "score": 0.9}][:top_k], "confident": True,
                "target": target or "roses"}, image


class FakeDetector:
    def detect(self, image, conf=0.25):
        return {"detections": [{"label": "person", "score": 0.8, "box_xyxy": [0, 0, 10, 10]}],
                "summary": {"person": 1}}, image


class FakeSearch:
    def __init__(self, image_path: str):
        self.meta = [{"path": image_path, "label": "roses", "source": "flowers"}]

    def _hits(self, k):
        return [{"id": 0, "score": 0.9, **self.meta[0]}][:k]

    def search_text(self, query, k=8):
        return self._hits(k)

    def search_image(self, image, k=8):
        return self._hits(k)


class FakeBot:
    def stream(self, message, history=None):
        return [{"source": "doi_tra.md", "text": "7 ngày", "score": 0.9}], iter(["Được ", "7 ngày."])

    def answer(self, message, history=None):
        return {"answer": "Được 7 ngày.", "sources": []}


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, format="PNG")
    return buf.getvalue()


IMG = {"file": ("a.png", png_bytes(), "image/png")}
NOT_IMG = {"file": ("a.txt", b"hello", "text/plain")}


@pytest.fixture()
def client(tmp_path):
    gallery_img = tmp_path / "g0.png"
    gallery_img.write_bytes(png_bytes())
    with TestClient(main.app) as c:
        main.MODELS.update(classifier=FakeClassifier(), detector=FakeDetector(),
                           retrieval=FakeSearch(str(gallery_img)), llm=FakeBot())
        yield c
        main.MODELS.clear()


# ---------- /api/health ----------
def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and "ram_mb" in body


# ---------- /api/classify ----------
def test_classify_ok(client):
    r = client.post("/api/classify", files=IMG, data={"top_k": 1})
    assert r.status_code == 200
    assert r.json()["predictions"][0]["label"] == "roses" and "latency_ms" in r.json()


def test_classify_400_not_an_image(client):
    assert client.post("/api/classify", files=NOT_IMG).status_code == 400


def test_classify_422_missing_file(client):
    assert client.post("/api/classify", data={"top_k": 1}).status_code == 422


def test_classify_413_too_large(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 0)
    assert client.post("/api/classify", files=IMG).status_code == 413


# ---------- /api/classify/explain (Grad-CAM) ----------
def test_explain_ok_returns_overlay_and_target(client):
    r = client.post("/api/classify/explain", files=IMG, data={"top_k": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["overlay"].startswith("data:image/jpeg;base64,") and body["target"] == "roses"


def test_explain_ok_with_explicit_target(client):
    r = client.post("/api/classify/explain", files=IMG, data={"target": "daisy"})
    assert r.status_code == 200 and r.json()["target"] == "daisy"


def test_explain_400_not_an_image(client):
    assert client.post("/api/classify/explain", files=NOT_IMG).status_code == 400


def test_explain_422_missing_file(client):
    assert client.post("/api/classify/explain", data={"target": "daisy"}).status_code == 422


def test_explain_422_unknown_target(client):
    assert client.post("/api/classify/explain", files=IMG, data={"target": "cactus"}).status_code == 422


# ---------- /api/detect ----------
def test_detect_ok_returns_annotated_image(client):
    r = client.post("/api/detect", files=IMG, data={"conf": 0.3})
    assert r.status_code == 200
    assert r.json()["image"].startswith("data:image/jpeg;base64,") and r.json()["summary"] == {"person": 1}


def test_detect_400_not_an_image(client):
    assert client.post("/api/detect", files=NOT_IMG).status_code == 400


def test_detect_422_bad_conf(client):
    assert client.post("/api/detect", files=IMG, data={"conf": "abc"}).status_code == 422


# ---------- /api/search/text ----------
def test_search_text_ok_has_url_and_no_server_path(client):
    r = client.post("/api/search/text", json={"query": "a red rose", "k": 3})
    assert r.status_code == 200
    hit = r.json()["results"][0]
    assert hit["url"] == "/api/gallery/0" and "path" not in hit


def test_search_text_422_empty_query(client):
    assert client.post("/api/search/text", json={"query": ""}).status_code == 422


def test_search_text_422_k_out_of_range(client):
    assert client.post("/api/search/text", json={"query": "rose", "k": 999}).status_code == 422


def test_search_text_503_when_model_not_loaded(client):
    main.MODELS.pop("retrieval")
    assert client.post("/api/search/text", json={"query": "a dog"}).status_code == 503


# ---------- /api/search/image ----------
def test_search_image_ok(client):
    r = client.post("/api/search/image", files=IMG, data={"k": 2})
    assert r.status_code == 200 and r.json()["results"][0]["label"] == "roses"


def test_search_image_400_not_an_image(client):
    assert client.post("/api/search/image", files=NOT_IMG).status_code == 400


def test_search_image_422_missing_file(client):
    assert client.post("/api/search/image", data={"k": 2}).status_code == 422


# ---------- /api/gallery/{id} ----------
def test_gallery_ok_returns_image(client):
    r = client.get("/api/gallery/0")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/")


def test_gallery_404_unknown_id(client):
    assert client.get("/api/gallery/99").status_code == 404


def test_gallery_422_id_not_int(client):
    assert client.get("/api/gallery/abc").status_code == 422


# ---------- /api/chat (SSE) và /api/chat/sync ----------
def test_chat_stream_events(client):
    with client.stream("POST", "/api/chat", json={"message": "Đổi trả?"}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert '"type": "sources"' in body and '"type": "done"' in body and "7 ngày" in body


def test_chat_422_empty_message(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_chat_422_message_too_long(client):
    assert client.post("/api/chat", json={"message": "x" * 1001}).status_code == 422


def test_chat_sync_ok(client):
    r = client.post("/api/chat/sync", json={"message": "Đổi trả?"})
    assert r.status_code == 200 and r.json()["answer"]


def test_chat_sync_422_missing_message(client):
    assert client.post("/api/chat/sync", json={}).status_code == 422


def test_chat_503_when_model_not_loaded(client):
    main.MODELS.pop("llm")
    assert client.post("/api/chat/sync", json={"message": "xin chào"}).status_code == 503

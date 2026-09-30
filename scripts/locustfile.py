"""Tải thử bằng Locust (tùy chọn, thay cho scripts/benchmark.py):
    pip install locust
    locust -f scripts/locustfile.py --host http://localhost:8000 --headless -u 5 -r 1 -t 60s --csv docs/locust
Kết quả p50/p95 nằm trong docs/locust_stats.csv."""
import io

from locust import HttpUser, between, task
from PIL import Image

_buf = io.BytesIO()
Image.effect_noise((640, 480), 64).convert("RGB").save(_buf, format="JPEG")
IMG = _buf.getvalue()


class AppUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def classify(self):
        self.client.post("/api/classify", files={"file": ("x.jpg", IMG, "image/jpeg")}, data={"top_k": 3})

    @task(2)
    def detect(self):
        self.client.post("/api/detect", files={"file": ("x.jpg", IMG, "image/jpeg")})

    @task(2)
    def search_text(self):
        self.client.post("/api/search/text", json={"query": "a red flower", "k": 8})

    @task(1)
    def chat(self):
        self.client.post("/api/chat/sync", json={"message": "Đổi trả trong bao nhiêu ngày?"})

    @task(1)
    def health(self):
        self.client.get("/api/health")

"""Integration tests of the local photo API and static delivery."""
from fastapi.testclient import TestClient
from web.server import app, ROOT

client = TestClient(app)


def test_web_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["mode"] == "local"


def test_web_invalid_photo():
    response = client.post("/api/solve", files={"image": ("bad.png", b"not an image", "image/png")})
    assert response.status_code == 422
    assert "No se pudo leer" in response.json()["detail"]


def test_web_invalid_method():
    response = client.post("/api/solve", files={"image": ("bad.png", b"data")}, data={"method": "bad"})
    assert response.status_code == 422


def test_web_photo_and_cache():
    data = (ROOT / "prototipo_interactivo" / "sample_images" / "kenken_4x4.png").read_bytes()
    response = client.post("/api/solve", files={"image": ("board.png", data, "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "OPTIMAL"
    assert len(payload["grid"]) == 4
    assert client.get("/" + payload["image"]).headers["content-type"] == "image/png"
    second = client.post("/api/solve", files={"image": ("board.png", data, "image/png")}).json()
    assert second["cached"] is True
    assert second["grid"] == payload["grid"]

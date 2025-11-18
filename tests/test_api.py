from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_kmeans_run_endpoint() -> None:
    payload = {"m": 20, "n": 25, "k": 3, "seed": 2024}
    response = client.post("/kmeans/run", json=payload)

    assert response.status_code == 200
    body = response.json()

    assert len(body["neighborhoods"]) == payload["n"]
    assert len(body["hospitals"]) == payload["k"]
    assert len(body["assignments"]) == payload["n"]


def test_pretrained_endpoint() -> None:
    response = client.get("/kmeans/pretrained")

    assert response.status_code == 200
    body = response.json()
    assert "hospitals" in body and isinstance(body["hospitals"], list)

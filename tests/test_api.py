from fastapi.testclient import TestClient

from lreader_engine.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chapter_requires_images() -> None:
    response = client.post(
        "/v1/chapters",
        json={
            "source_language": "auto",
            "target_language": "ko",
            "images": [],
        },
    )

    assert response.status_code == 422

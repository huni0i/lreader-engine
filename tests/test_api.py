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


def test_fast_ocr_requires_source_language() -> None:
    response = client.post(
        "/v1/images/translate",
        params={
            "source_language": "auto",
            "target_language": "en",
        },
        files={"file": ("page.png", b"not-read", "image/png")},
    )

    assert response.status_code == 422

from pathlib import Path

from fastapi.testclient import TestClient

import lreader_engine.main as main
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


class FakeOcrEngine:
    def spot(self, image_path: Path) -> str:
        del image_path
        return "오늘은 어디로 갈까?"


def test_source_language_is_detected_from_first_image(monkeypatch) -> None:
    monkeypatch.setattr(main, "high_quality_ocr", lambda: FakeOcrEngine())

    assert main.resolve_source_language(Path("page.png"), "auto") == "ko"

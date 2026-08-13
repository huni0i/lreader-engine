from pathlib import Path

from PIL import Image, ImageDraw

from lreader_engine.bubble_detector import BubbleDetector
from lreader_engine.models import OcrRegion, Point


def test_detect_expands_text_region_to_white_bubble(tmp_path: Path) -> None:
    image_path = tmp_path / "bubble.png"
    image = Image.new("RGB", (200, 150), "black")
    ImageDraw.Draw(image).ellipse((30, 20, 170, 130), fill="white")
    image.save(image_path)
    text_polygon = [
        Point(x=75, y=60),
        Point(x=125, y=60),
        Point(x=125, y=85),
        Point(x=75, y=85),
    ]
    region = OcrRegion(
        polygon=text_polygon,
        text_polygons=[text_polygon],
        text="대사",
        confidence=0.95,
    )

    detected = BubbleDetector().detect(image_path, [region])[0]
    xs = [point.x for point in detected.polygon]
    ys = [point.y for point in detected.polygon]

    assert min(xs) <= 35
    assert max(xs) >= 165
    assert min(ys) <= 25
    assert max(ys) >= 125
    assert detected.text_polygons == [text_polygon]

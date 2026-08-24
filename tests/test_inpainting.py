from pathlib import Path

from PIL import Image

from lreader_engine.inpainting import InpaintingEngine
from lreader_engine.models import OcrRegion, Point


class FakeInpaintingModel:
    def __init__(self) -> None:
        self.mask: Image.Image | None = None

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        self.mask = mask
        return image


def test_erase_text_builds_expanded_mask_and_data_url(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    polygon = [
        Point(x=30, y=30),
        Point(x=70, y=30),
        Point(x=70, y=50),
        Point(x=30, y=50),
    ]
    region = OcrRegion(
        polygon=polygon,
        text_polygons=[polygon],
        text="원문",
        confidence=0.9,
    )
    fake_model = FakeInpaintingModel()
    engine = InpaintingEngine()
    engine.__dict__["model"] = fake_model

    result = engine.erase_text(image_path, [region], "lama")

    assert result is not None
    assert result.startswith("data:image/webp;base64,")
    assert fake_model.mask is not None
    assert fake_model.mask.getpixel((30, 30)) == 255
    assert fake_model.mask.getpixel((28, 30)) == 255


def test_erase_text_masks_outer_box_not_just_glyphs(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (120, 80), "white").save(image_path)
    outer = [
        Point(x=10, y=10),
        Point(x=110, y=10),
        Point(x=110, y=70),
        Point(x=10, y=70),
    ]
    inner = [
        Point(x=40, y=30),
        Point(x=80, y=30),
        Point(x=80, y=50),
        Point(x=40, y=50),
    ]
    region = OcrRegion(
        polygon=outer,
        text_polygons=[inner],
        text="키링 샀는데",
        confidence=0.8,
    )
    fake_model = FakeInpaintingModel()
    engine = InpaintingEngine()
    engine.__dict__["model"] = fake_model

    engine.erase_text(image_path, [region], "lama")

    assert fake_model.mask is not None
    assert fake_model.mask.getpixel((12, 12)) == 255
    assert fake_model.mask.getpixel((60, 40)) == 255

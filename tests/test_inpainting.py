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

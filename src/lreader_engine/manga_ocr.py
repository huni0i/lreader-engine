from functools import cached_property
from pathlib import Path

from manga_ocr import MangaOcr
from PIL import Image

from lreader_engine.models import OcrRegion


class MangaOcrEngine:
    @cached_property
    def model(self) -> MangaOcr:
        return MangaOcr()

    def recognize_region(
        self,
        image_path: str | Path,
        region: OcrRegion,
    ) -> str:
        image = Image.open(image_path).convert("RGB")
        xs = [point.x for point in region.polygon]
        ys = [point.y for point in region.polygon]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        horizontal_padding = max(24, int(width * 0.2))
        vertical_padding = max(18, int(height * 0.12))
        crop = image.crop(
            (
                max(0, int(min(xs)) - horizontal_padding),
                max(0, int(min(ys)) - vertical_padding),
                min(image.width, int(max(xs)) + horizontal_padding),
                min(image.height, int(max(ys)) + vertical_padding),
            )
        )
        return self.model(crop).strip()

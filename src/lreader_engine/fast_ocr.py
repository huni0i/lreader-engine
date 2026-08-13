from functools import cached_property
from pathlib import Path
from statistics import fmean

import easyocr

from lreader_engine.models import OcrRegion, Point, SourceLanguage


LANGUAGE_GROUPS: dict[SourceLanguage, list[str]] = {
    "auto": ["en"],
    "ja": ["ja", "en"],
    "en": ["en"],
    "zh": ["ch_sim", "en"],
    "ko": ["ko", "en"],
}


class FastOcrEngine:
    def __init__(self, source_language: SourceLanguage) -> None:
        if source_language == "auto":
            raise ValueError("Fast OCR requires an explicit source language")
        self.source_language = source_language

    @cached_property
    def reader(self) -> easyocr.Reader:
        cache_dir = Path(__file__).resolve().parents[2] / ".cache" / "easyocr"
        return easyocr.Reader(
            LANGUAGE_GROUPS[self.source_language],
            gpu=True,
            model_storage_directory=str(cache_dir / "models"),
            user_network_directory=str(cache_dir / "networks"),
            verbose=False,
        )

    def recognize(self, image_path: str | Path) -> list[OcrRegion]:
        results = self.reader.readtext(
            str(image_path),
            detail=1,
            paragraph=False,
            canvas_size=2560,
        )
        return [
            OcrRegion(
                polygon=[
                    Point(x=float(x), y=float(y)) for x, y in polygon
                ],
                text_polygons=[
                    [Point(x=float(x), y=float(y)) for x, y in polygon]
                ],
                text=text,
                confidence=float(confidence),
            )
            for polygon, text, confidence in results
        ]

    def recognize_blocks(self, image_path: str | Path) -> list[OcrRegion]:
        lines = sorted(
            (region for region in self.recognize(image_path) if region.confidence >= 0.2),
            key=lambda region: min(point.y for point in region.polygon),
        )
        blocks: list[list[OcrRegion]] = []

        for line in lines:
            if not blocks or not self._is_nearby(blocks[-1][-1], line):
                blocks.append([line])
            else:
                blocks[-1].append(line)

        return [self._merge_block(block) for block in blocks]

    @staticmethod
    def _bounds(region: OcrRegion) -> tuple[float, float, float, float]:
        xs = [point.x for point in region.polygon]
        ys = [point.y for point in region.polygon]
        return min(xs), min(ys), max(xs), max(ys)

    @classmethod
    def _is_nearby(cls, previous: OcrRegion, current: OcrRegion) -> bool:
        prev_left, prev_top, prev_right, prev_bottom = cls._bounds(previous)
        left, top, right, bottom = cls._bounds(current)
        average_height = ((prev_bottom - prev_top) + (bottom - top)) / 2
        vertical_gap = top - prev_bottom
        horizontal_overlap = min(prev_right, right) - max(prev_left, left)
        minimum_width = min(prev_right - prev_left, right - left)
        return (
            vertical_gap <= max(16, average_height * 0.6)
            and horizontal_overlap >= minimum_width * 0.35
        )

    @classmethod
    def _merge_block(cls, block: list[OcrRegion]) -> OcrRegion:
        bounds = [cls._bounds(region) for region in block]
        left = min(bound[0] for bound in bounds)
        top = min(bound[1] for bound in bounds)
        right = max(bound[2] for bound in bounds)
        bottom = max(bound[3] for bound in bounds)
        return OcrRegion(
            polygon=[
                Point(x=left, y=top),
                Point(x=right, y=top),
                Point(x=right, y=bottom),
                Point(x=left, y=bottom),
            ],
            text_polygons=[
                polygon
                for region in block
                for polygon in (region.text_polygons or [region.polygon])
            ],
            text=" ".join(region.text for region in block),
            confidence=fmean(region.confidence for region in block),
        )

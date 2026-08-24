from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path

from lreader_engine.detection_dataset import box_to_region, clamp_coord
from lreader_engine.eval import Box
from lreader_engine.models import OcrRegion


def default_weights_path() -> Path:
    configured = os.getenv("LREADER_YOLO_WEIGHTS")
    if configured:
        return Path(configured)
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "det-text"
        / "runs"
        / "text-detector"
        / "weights"
        / "best.pt"
    )


class YoloTextDetector:
    def __init__(self, weights: Path | None = None, confidence: float = 0.25) -> None:
        self.weights = weights or default_weights_path()
        self.confidence = confidence

    @cached_property
    def model(self):
        if not self.weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")
        from ultralytics import YOLO

        return YOLO(str(self.weights))

    def detect(self, image_path: str | Path) -> list[OcrRegion]:
        regions: list[OcrRegion] = []
        for result in self.model.predict(
            str(image_path),
            conf=self.confidence,
            verbose=False,
        ):
            if result.boxes is None:
                continue
            scores = result.boxes.conf.cpu().tolist()
            for xyxy, score in zip(result.boxes.xyxy.cpu().tolist(), scores, strict=True):
                box = Box(
                    left=clamp_coord(xyxy[0]),
                    top=clamp_coord(xyxy[1]),
                    right=clamp_coord(xyxy[2]),
                    bottom=clamp_coord(xyxy[3]),
                )
                regions.append(
                    box_to_region(box, text="", confidence=min(1.0, float(score)))
                )
        return regions

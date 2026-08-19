from pathlib import Path

import cv2
import numpy as np

from lreader_engine.models import OcrRegion, Point


class BubbleDetector:
    def detect(
        self,
        image_path: str | Path,
        regions: list[OcrRegion],
    ) -> list[OcrRegion]:
        image = cv2.imread(str(image_path))
        if image is None or not regions:
            return regions

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        white = cv2.inRange(gray, 205, 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        connected = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(connected)
        if count <= 1:
            return regions

        image_area = image.shape[0] * image.shape[1]
        return [
            self._attach_bubble(region, labels, stats, image_area)
            for region in regions
        ]

    @staticmethod
    def _attach_bubble(
        region: OcrRegion,
        labels: np.ndarray,
        stats: np.ndarray,
        image_area: int,
    ) -> OcrRegion:
        xs = [point.x for point in region.polygon]
        ys = [point.y for point in region.polygon]
        left = max(0, int(min(xs)))
        top = max(0, int(min(ys)))
        right = min(labels.shape[1], int(max(xs)) + 1)
        bottom = min(labels.shape[0], int(max(ys)) + 1)
        if right <= left or bottom <= top:
            return region

        candidates = labels[top:bottom, left:right]
        candidate_labels, frequencies = np.unique(candidates, return_counts=True)
        ordered = sorted(
            zip(candidate_labels, frequencies, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        text_area = max(1, (right - left) * (bottom - top))

        for label, _ in ordered:
            if label == 0:
                continue
            x, y, width, height, area = stats[label]
            if area < text_area * 1.2 or area > image_area * 0.6:
                continue
            return region.model_copy(
                update={
                    "polygon": [
                        Point(x=float(x), y=float(y)),
                        Point(x=float(x + width), y=float(y)),
                        Point(x=float(x + width), y=float(y + height)),
                        Point(x=float(x), y=float(y + height)),
                    ]
                }
            )

        return region

    def has_speech_bubbles(self, image_path: str | Path) -> bool:
        image = cv2.imread(str(image_path))
        if image is None:
            return False

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        white = cv2.inRange(gray, 205, 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        connected = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(connected)
        if count <= 1:
            return False

        image_area = image.shape[0] * image.shape[1]
        return any(
            image_area * 0.01 <= stats[label, cv2.CC_STAT_AREA] <= image_area * 0.6
            for label in range(1, count)
        )

import base64
from functools import cached_property
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from lreader_engine.device import resolve_torch_device
from lreader_engine.models import InpaintingMethod, OcrRegion


class InpaintingEngine:
    @cached_property
    def model(self):
        from simple_lama_inpainting import SimpleLama

        return SimpleLama(device=resolve_torch_device().type)

    def erase_text(
        self,
        image_path: str | Path,
        regions: list[OcrRegion],
        method: InpaintingMethod = "opencv",
    ) -> str | None:
        if not regions:
            return None

        image = Image.open(image_path).convert("RGB")
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)

        for region in regions:
            polygons = [region.polygon, *(region.text_polygons or [])]
            for polygon in polygons:
                if len(polygon) < 3:
                    continue
                draw.polygon([(point.x, point.y) for point in polygon], fill=255)

        radius = max(7, min(24, round(min(image.size) * 0.012)))
        mask = mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))
        inpainted = (
            self.model(image, mask)
            if method == "lama"
            else self._opencv_inpaint(image, mask)
        )

        output = BytesIO()
        inpainted.save(output, format="WEBP", quality=90, method=4)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/webp;base64,{encoded}"

    @staticmethod
    def _opencv_inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
        source = np.asarray(image)
        mask_array = np.asarray(mask)
        result = source.copy()
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask_array)

        for label in range(1, count):
            x, y, width, height, _ = stats[label]
            margin = max(16, round(max(width, height) * 0.25))
            left = max(0, x - margin)
            top = max(0, y - margin)
            right = min(source.shape[1], x + width + margin)
            bottom = min(source.shape[0], y + height + margin)
            patch = source[top:bottom, left:right]
            patch_mask = mask_array[top:bottom, left:right]
            result[top:bottom, left:right] = cv2.inpaint(
                patch,
                patch_mask,
                4,
                cv2.INPAINT_TELEA,
            )

        return Image.fromarray(result)

from functools import cached_property
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from lreader_engine.device import resolve_torch_device, resolve_torch_dtype
from lreader_engine.models import OcrRegion


class OcrEngine:
    model_id = "PaddlePaddle/PaddleOCR-VL-1.6"

    def __init__(self) -> None:
        self.device = resolve_torch_device()
        self.dtype = resolve_torch_dtype(self.device)

    @cached_property
    def processor(self):
        return AutoProcessor.from_pretrained(self.model_id)

    @cached_property
    def model(self):
        return (
            AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                dtype=self.dtype,
            )
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def spot(self, image_path: str | Path) -> str:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        if width < 1500 and height < 1500:
            image = image.resize(
                (width * 2, height * 2),
                Image.Resampling.LANCZOS,
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Spotting:"},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={
                "size": {
                    "shortest_edge": (
                        self.processor.image_processor.size.shortest_edge
                    ),
                    "longest_edge": 2048 * 28 * 28,
                }
            },
        ).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=1024)
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def recognize_region(
        self,
        image_path: str | Path,
        region: OcrRegion,
        padding: int = 24,
    ) -> str:
        image = Image.open(image_path).convert("RGB")
        xs = [point.x for point in region.polygon]
        ys = [point.y for point in region.polygon]
        crop = image.crop(
            (
                max(0, int(min(xs)) - padding),
                max(0, int(min(ys)) - padding),
                min(image.width, int(max(xs)) + padding),
                min(image.height, int(max(ys)) + padding),
            )
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop},
                    {"type": "text", "text": "OCR:"},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

from functools import cached_property
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lreader_engine.device import resolve_torch_device, resolve_torch_dtype
from lreader_engine.models import SourceLanguage, TargetLanguage


LANGUAGE_NAMES: dict[TargetLanguage, str] = {
    "ja": "Japanese",
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
}


def clean_translation(result: str) -> str:
    result = re.split(
        r"\n(?:Korean|English|Japanese|Chinese):",
        result,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    result = re.sub(
        r"^(?:Korean|English|Japanese|Chinese):\s*",
        "",
        result.strip(),
        flags=re.IGNORECASE,
    )
    return result.strip()


class TranslationEngine:
    model_id = "tencent/Hy-MT2-1.8B"

    def __init__(self) -> None:
        self.device = resolve_torch_device()
        self.dtype = resolve_torch_dtype(self.device)

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

    @cached_property
    def model(self):
        return (
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=self.dtype,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def translate(
        self,
        text: str,
        source_language: SourceLanguage,
        target_language: TargetLanguage,
    ) -> str:
        if source_language == "auto":
            raise ValueError("Translation requires an explicit source language")
        prompt = (
            f"Translate the following text into {LANGUAGE_NAMES[target_language]}. "
            "Only output the translated result without any explanation:\n\n"
            f"{text}"
        )
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.05,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        result = self.tokenizer.decode(generated, skip_special_tokens=True)
        return clean_translation(result)

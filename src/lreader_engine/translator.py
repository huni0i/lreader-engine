import os
import re
from functools import cached_property

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
    default_model_id = "tencent/Hy-MT2-1.8B"

    def __init__(self) -> None:
        self.model_id = os.getenv(
            "LREADER_TRANSLATION_MODEL",
            self.default_model_id,
        )
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
        if self.device.type == "cuda":
            return AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=self.dtype,
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval()

        return (
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
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
        source_name = LANGUAGE_NAMES[source_language]
        target_name = LANGUAGE_NAMES[target_language]
        prompt = (
            f"Translate this comic dialogue from {source_name} to {target_name}. "
            "Preserve the original meaning, tone, names, and sentence type. "
            "Only output the translated dialogue without any explanation:\n\n"
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

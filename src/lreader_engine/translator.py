import json
import logging
import os
import re
import time
from functools import cached_property

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lreader_engine.device import resolve_torch_device, resolve_torch_dtype
from lreader_engine.models import SourceLanguage, TargetLanguage


logger = logging.getLogger(__name__)

LANGUAGE_NAMES: dict[TargetLanguage, str] = {
    "ja": "Japanese",
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
}

SOURCE_TEXT_PATTERNS: dict[TargetLanguage, re.Pattern[str]] = {
    "ja": re.compile(r"[\u3040-\u30ff\u3400-\u9fff]"),
    "en": re.compile(r"[A-Za-z]"),
    "zh": re.compile(r"[\u3400-\u9fff]"),
    "ko": re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]"),
}


def contains_source_text(text: str, source_language: SourceLanguage) -> bool:
    if source_language == "auto":
        return bool(text.strip())
    return bool(SOURCE_TEXT_PATTERNS[source_language].search(text))


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


def parse_numbered_translations(result: str, count: int) -> list[str] | None:
    stripped = result.strip().strip("`").strip()
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, list) and len(decoded) == count:
        return [clean_translation(str(item)) for item in decoded]
    if isinstance(decoded, dict) and all(
        str(index) in decoded for index in range(count)
    ):
        return [
            clean_translation(str(decoded[str(index)])) for index in range(count)
        ]

    translations: dict[int, str] = {}
    for match in re.finditer(
        r"^\s*(?:[-*]\s*)?[\[(]?(\d+)[\]).:\-]\s*(.+?)\s*$",
        stripped,
        flags=re.MULTILINE,
    ):
        index = int(match.group(1))
        if 0 <= index < count:
            translations[index] = clean_translation(match.group(2))

    if len(translations) == count:
        return [translations[index] for index in range(count)]

    lines = [
        clean_translation(re.sub(r"^\s*[-*]\s*", "", line))
        for line in stripped.splitlines()
        if line.strip()
        and line.strip() not in {"```", "```json"}
        and not re.fullmatch(
            r"(?:Translations?|Korean|English|Japanese|Chinese):?",
            line.strip(),
            flags=re.IGNORECASE,
        )
    ]
    return lines if len(lines) == count else None


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
            max_new_tokens=max(12, min(64, len(text) * 2 + 8)),
            do_sample=False,
            repetition_penalty=1.05,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        result = self.tokenizer.decode(generated, skip_special_tokens=True)
        return clean_translation(result)

    @torch.inference_mode()
    def translate_many(
        self,
        texts: list[str],
        source_language: SourceLanguage,
        target_language: TargetLanguage,
    ) -> list[str]:
        if not texts:
            return []
        if len(texts) == 1:
            return [self.translate(texts[0], source_language, target_language)]
        if source_language == "auto":
            raise ValueError("Translation requires an explicit source language")

        source_name = LANGUAGE_NAMES[source_language]
        target_name = LANGUAGE_NAMES[target_language]
        numbered_texts = "\n".join(
            f"[{index}] {text}" for index, text in enumerate(texts)
        )
        prompt = (
            f"Translate each numbered comic dialogue from {source_name} "
            f"to {target_name}. Preserve meaning, tone, names, and sentence type. "
            "Use surrounding lines as context, but translate every item separately. "
            "Output exactly one translated line per item in the same [number] format. "
            "Do not add explanations or language labels.\n\n"
            f"{numbered_texts}"
        )
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        max_new_tokens = max(
            32,
            min(256, sum(len(text) for text in texts) * 2 + len(texts) * 10),
        )
        start = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        logger.info(
            "batch generate seconds=%.2f prompt_tokens=%d new_tokens=%d limit=%d",
            time.perf_counter() - start,
            inputs["input_ids"].shape[-1],
            generated.shape[-1],
            max_new_tokens,
        )
        result = self.tokenizer.decode(generated, skip_special_tokens=True)
        parsed = parse_numbered_translations(result, len(texts))
        if parsed is not None:
            return parsed

        logger.warning("Could not parse batch translation output: %r", result)
        return [
            self.translate(text, source_language, target_language) for text in texts
        ]

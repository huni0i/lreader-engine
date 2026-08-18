from functools import cached_property

from lreader_engine.models import SourceLanguage, TargetLanguage
from lreader_engine.translator import (
    LANGUAGE_NAMES,
    clean_translation,
    parse_numbered_translations,
)


class MlxTranslationEngine:
    model_id = "sahilchachra/hy-mt2-1.8b-4bit-mlx"

    @cached_property
    def runtime(self):
        from mlx_lm import load

        return load(self.model_id)

    def translate(
        self,
        text: str,
        source_language: SourceLanguage,
        target_language: TargetLanguage,
    ) -> str:
        if source_language == "auto":
            raise ValueError("MLX translation requires an explicit source language")
        target_name = LANGUAGE_NAMES[target_language]
        instruction = (
            f"Translate the following text into {target_name}. "
            "Only output the translated result without any explanation:\n\n"
            f"{text}"
        )
        model, tokenizer = self.runtime
        from mlx_lm import generate

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        result = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max(12, min(48, len(text) * 2)),
            verbose=False,
        )
        return clean_translation(result)

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
            raise ValueError("MLX translation requires an explicit source language")

        target_name = LANGUAGE_NAMES[target_language]
        numbered_texts = "\n".join(
            f"[{index}] {text}" for index, text in enumerate(texts)
        )
        instruction = (
            f"Translate each numbered comic dialogue into {target_name}. "
            "Use context but keep every item separate. Output exactly one line "
            "per item in the same [number] format without explanations.\n\n"
            f"{numbered_texts}"
        )
        model, tokenizer = self.runtime
        from mlx_lm import generate

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        result = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max(
                32,
                min(256, sum(len(text) for text in texts) * 2 + len(texts) * 10),
            ),
            verbose=False,
        )
        parsed = parse_numbered_translations(result, len(texts))
        if parsed is not None:
            return parsed
        return [
            self.translate(text, source_language, target_language) for text in texts
        ]

from functools import cached_property

from lreader_engine.models import SourceLanguage, TargetLanguage
from lreader_engine.translator import LANGUAGE_NAMES, clean_translation


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

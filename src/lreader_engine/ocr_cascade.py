from __future__ import annotations

from typing import Literal

from lreader_engine.models import OcrMode, SourceLanguage, TranslationQuality


PrimaryOcr = Literal["easy", "yolo", "spot"]


def select_primary_ocr(
    ocr_mode: OcrMode,
    source_language: SourceLanguage,
    quality: TranslationQuality,
    has_white_bubbles: bool,
) -> PrimaryOcr:
    """Pick a detector from page appearance. Translation is not involved."""
    if ocr_mode in {"easy", "yolo", "spot"}:
        return ocr_mode
    if source_language == "ja" and quality in {"ocr", "balanced"}:
        return "easy" if has_white_bubbles else "yolo"
    return "easy"


def should_fallback_to_spotting(
    ocr_mode: OcrMode,
    source_language: SourceLanguage,
    quality: TranslationQuality,
    *,
    used_spotting: bool,
    has_source_text: bool,
) -> bool:
    if used_spotting or ocr_mode != "route":
        return False
    if source_language != "ja" or quality not in {"ocr", "balanced"}:
        return False
    return not has_source_text

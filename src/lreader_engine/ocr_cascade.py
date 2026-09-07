from __future__ import annotations

from typing import Literal

from lreader_engine.models import OcrMode, SourceLanguage, TranslationQuality


PrimaryOcr = Literal["easy", "yolo", "spot"]


def select_primary_ocr(
    ocr_mode: OcrMode,
    source_language: SourceLanguage,
    quality: TranslationQuality,
) -> PrimaryOcr:
    """Pick a detector for route mode. Translation is not involved.

    Benchmarked on Manga109-s test (60 pages, book-level split): a comic-
    trained YOLO detector reaches recall=0.863/precision=0.937, against
    EasyOCR's recall=0.076/precision=0.140. Routing Japanese pages to
    EasyOCR by white-bubble appearance no longer earns its keep -- it only
    diverts the ~3% of pages with light backgrounds to the far weaker
    detector. Always prefer YOLO for Japanese ocr/balanced requests.
    """
    if ocr_mode in {"easy", "yolo", "spot"}:
        return ocr_mode
    if source_language == "ja" and quality in {"ocr", "balanced"}:
        return "yolo"
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


def should_rerecognize_region(
    source_language: SourceLanguage,
    quality: TranslationQuality,
    confidence: float,
) -> bool:
    del confidence
    if quality not in {"ocr", "balanced"}:
        return False
    # Japanese boxes often have empty/weak text and need manga-ocr.
    # Generative VL OCR must not "correct" Hangul: it turns 소컵 into 수컵
    # and 구운란 into 구운난 even when EasyOCR already read them correctly.
    return source_language == "ja"

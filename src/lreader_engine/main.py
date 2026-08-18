import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypeVar

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile

from lreader_engine.bubble_detector import BubbleDetector
from lreader_engine.device import resolve_torch_device
from lreader_engine.fast_ocr import FastOcrEngine
from lreader_engine.inpainting import InpaintingEngine
from lreader_engine.language_detector import detect_text_language
from lreader_engine.manga_ocr import MangaOcrEngine
from lreader_engine.mlx_translator import MlxTranslationEngine
from lreader_engine.models import (
    ChapterRequest,
    ImageTranslationResult,
    ImageUrlTranslationRequest,
    InpaintingMethod,
    SourceLanguage,
    TargetLanguage,
    TranslatedOcrRegion,
    TranslationQuality,
)
from lreader_engine.ocr import OcrEngine
from lreader_engine.translator import TranslationEngine, contains_source_text


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T")

app = FastAPI(title="Lreader local engine", version="0.1.0")


@contextmanager
def stage_timer(stage: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("stage=%s seconds=%.2f", stage, time.perf_counter() - start)


def timed(stage: str, run: Callable[[], T]) -> T:
    with stage_timer(stage):
        return run()


@lru_cache(maxsize=4)
def fast_ocr(source_language: SourceLanguage) -> FastOcrEngine:
    return FastOcrEngine(source_language)


@lru_cache(maxsize=1)
def bubble_detector() -> BubbleDetector:
    return BubbleDetector()


@lru_cache(maxsize=1)
def fast_translator() -> MlxTranslationEngine | TranslationEngine:
    if resolve_torch_device().type == "mps":
        return MlxTranslationEngine()
    return TranslationEngine()


@lru_cache(maxsize=1)
def balanced_translator() -> TranslationEngine:
    return TranslationEngine()


@lru_cache(maxsize=1)
def high_quality_ocr() -> OcrEngine:
    return OcrEngine()


@lru_cache(maxsize=1)
def manga_ocr() -> MangaOcrEngine:
    return MangaOcrEngine()


@lru_cache(maxsize=1)
def inpainter() -> InpaintingEngine:
    return InpaintingEngine()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def translate_path(
    image_path: Path,
    source_language: SourceLanguage,
    target_language: TargetLanguage,
    quality: TranslationQuality,
) -> list[TranslatedOcrRegion]:
    if source_language == "auto":
        raise HTTPException(
            status_code=422,
            detail="The fast OCR path requires an explicit source language.",
        )

    used_spotting = False
    if source_language == "ja" and quality in {"ocr", "balanced"}:
        detected_regions = timed(
            "ocr.spotting",
            lambda: high_quality_ocr().spot_regions(image_path),
        )
        used_spotting = bool(detected_regions)
        if not detected_regions:
            detected_regions = timed(
                "ocr.fast",
                lambda: fast_ocr(source_language).recognize_blocks(image_path),
            )
    else:
        detected_regions = timed(
            "ocr.fast",
            lambda: fast_ocr(source_language).recognize_blocks(image_path),
        )

    regions = timed(
        "bubble.detect",
        lambda: bubble_detector().detect(image_path, detected_regions),
    )
    if quality in {"ocr", "balanced"} and not used_spotting:
        recognizer = (
            manga_ocr().recognize_region
            if source_language == "ja"
            else high_quality_ocr().recognize_region
        )
        with stage_timer("ocr.recognize_regions"):
            regions = [
                region.model_copy(
                    update={
                        "text": recognizer(
                            image_path,
                            region,
                        )
                    }
                )
                if source_language == "ja" or region.confidence < 0.8
                else region
                for region in regions
            ]

    translation_engine = (
        balanced_translator() if quality == "balanced" else fast_translator()
    )
    translatable_regions = [
        region
        for region in regions
        if region.confidence >= 0.2
        and contains_source_text(region.text, source_language)
    ]
    logger.info(
        "translating regions=%d texts=%r",
        len(translatable_regions),
        [region.text for region in translatable_regions],
    )
    translations = (
        [region.text for region in translatable_regions]
        if source_language == target_language
        else timed(
            "translate.batch",
            lambda: translation_engine.translate_many(
                [region.text for region in translatable_regions],
                source_language,
                target_language,
            ),
        )
    )
    return [
        TranslatedOcrRegion(
            **region.model_dump(),
            translated_text=translated_text,
        )
        for region, translated_text in zip(
            translatable_regions,
            translations,
            strict=True,
        )
    ]


def resolve_source_language(
    image_path: Path,
    source_language: SourceLanguage,
) -> TargetLanguage:
    if source_language != "auto":
        return source_language

    detected = detect_text_language(high_quality_ocr().spot(image_path))
    if detected is None:
        raise HTTPException(
            status_code=422,
            detail="Could not automatically detect the source language.",
        )
    return detected


def translate_and_inpaint(
    image_path: Path,
    source_language: SourceLanguage,
    target_language: TargetLanguage,
    quality: TranslationQuality,
    inpaint: bool,
    inpaint_method: InpaintingMethod,
) -> ImageTranslationResult:
    resolved_source_language = resolve_source_language(
        image_path,
        source_language,
    )
    regions = translate_path(
        image_path,
        resolved_source_language,
        target_language,
        quality,
    )
    return ImageTranslationResult(
        source_language=resolved_source_language,
        regions=regions,
        inpainted_image=(
            inpainter().erase_text(image_path, regions, inpaint_method)
            if inpaint
            else None
        ),
    )


@app.post("/v1/images/translate")
async def translate_image(
    source_language: SourceLanguage,
    target_language: TargetLanguage,
    quality: TranslationQuality = "fast",
    inpaint: bool = True,
    inpaint_method: InpaintingMethod = "opencv",
    file: UploadFile = File(),
) -> ImageTranslationResult:
    suffix = Path(file.filename or "image.png").suffix or ".png"
    temp_path: Path | None = None

    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(await file.read())
            temp_path = Path(temporary.name)

        return translate_and_inpaint(
            temp_path,
            source_language,
            target_language,
            quality,
            inpaint,
            inpaint_method,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.post("/v1/images/translate-url")
async def translate_image_url(
    request: ImageUrlTranslationRequest,
) -> ImageTranslationResult:
    suffix = Path(request.url.path).suffix or ".jpg"
    temp_path: Path | None = None

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            response = await client.get(
                str(request.url),
                headers={
                    "Referer": str(request.referrer),
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 Chrome/139.0 Safari/537.36"
                    ),
                },
            )
            response.raise_for_status()

        with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(response.content)
            temp_path = Path(temporary.name)

        return translate_and_inpaint(
            temp_path,
            request.source_language,
            request.target_language,
            request.quality,
            request.inpaint,
            request.inpaint_method,
        )
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not download the source image: {error}",
        ) from error
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.post("/v1/chapters")
def translate_chapter(request: ChapterRequest) -> None:
    del request
    raise HTTPException(
        status_code=501,
        detail="The OCR pipeline is not connected yet.",
    )

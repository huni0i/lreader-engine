from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


SourceLanguage = Literal["auto", "ja", "en", "zh", "ko"]
TargetLanguage = Literal["ja", "en", "zh", "ko"]
TranslationQuality = Literal["fast", "ocr", "balanced"]
OcrMode = Literal["route", "easy", "spot", "yolo"]
InpaintingMethod = Literal["opencv", "lama"]


class ImageInput(BaseModel):
    index: int = Field(ge=0)
    url: HttpUrl
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)


class ChapterRequest(BaseModel):
    source_language: SourceLanguage = "auto"
    target_language: TargetLanguage
    images: list[ImageInput] = Field(min_length=1)


class ImageUrlTranslationRequest(BaseModel):
    url: HttpUrl
    referrer: HttpUrl
    source_language: SourceLanguage
    target_language: TargetLanguage
    quality: TranslationQuality = "balanced"
    ocr_mode: OcrMode = "route"
    skip_translate: bool = False
    inpaint: bool = True
    inpaint_method: InpaintingMethod = "opencv"


class BoundingBox(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class Point(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class OcrRegion(BaseModel):
    polygon: list[Point] = Field(min_length=4, max_length=4)
    text_polygons: list[list[Point]] = Field(default_factory=list)
    text: str
    confidence: float = Field(ge=0, le=1)


class TranslatedOcrRegion(OcrRegion):
    translated_text: str


class ImageTranslationResult(BaseModel):
    source_language: TargetLanguage
    regions: list[TranslatedOcrRegion]
    inpainted_image: str | None = None


class TextRegion(BaseModel):
    id: str
    image_index: int
    box: BoundingBox
    source_text: str
    translated_text: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    orientation: Literal["horizontal", "vertical", "unknown"] = "unknown"

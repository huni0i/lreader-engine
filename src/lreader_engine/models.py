from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


SourceLanguage = Literal["auto", "ja", "en", "zh", "ko"]
TargetLanguage = Literal["ja", "en", "zh", "ko"]


class ImageInput(BaseModel):
    index: int = Field(ge=0)
    url: HttpUrl
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)


class ChapterRequest(BaseModel):
    source_language: SourceLanguage = "auto"
    target_language: TargetLanguage
    images: list[ImageInput] = Field(min_length=1)


class BoundingBox(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class TextRegion(BaseModel):
    id: str
    image_index: int
    box: BoundingBox
    source_text: str
    translated_text: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    orientation: Literal["horizontal", "vertical", "unknown"] = "unknown"

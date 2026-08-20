from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


JAPANESE_LINES = [
    "ここは…",
    "どこだ？",
    "待って！",
    "行こう。",
    "嘘だろ。",
    "姉さま！",
    "目を覚ました。",
    "何があった？",
    "もう遅い。",
    "大丈夫だ。",
]

ENGLISH_LINES = [
    "Where is this?",
    "Wait!",
    "Let's go.",
    "No way.",
    "She woke up.",
    "What happened?",
    "We're late.",
    "It's okay.",
    "Look out!",
    "I don't know.",
]

JAPANESE_FONTS = [
    Path("/System/Library/Fonts/AppleGothic.ttf"),
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
]
ENGLISH_FONTS = [
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
]


def available_font(candidates: list[Path], size: int) -> ImageFont.FreeTypeFont:
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_vertical_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    origin: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: str,
) -> list[dict[str, int]]:
    x, y = origin
    boxes: list[dict[str, int]] = []
    for character in text:
        bbox = draw.textbbox((x, y), character, font=font)
        draw.text((x, y), character, font=font, fill=fill)
        boxes.append(
            {
                "left": bbox[0],
                "top": bbox[1],
                "right": bbox[2],
                "bottom": bbox[3],
            }
        )
        y = bbox[3] + 4
    return boxes


def render_page(
    language: str,
    page_index: int,
    output_dir: Path,
    rng: random.Random,
) -> dict:
    width, height = 720, 1024
    if language == "ja":
        image = Image.new("RGB", (width, height), (18, 12, 10))
        draw = ImageDraw.Draw(image)
        for _ in range(40):
            x = rng.randint(0, width)
            y = rng.randint(0, height)
            draw.ellipse((x, y, x + 3, y + 3), fill=(220, 200, 120))
        font = available_font(JAPANESE_FONTS, 42)
        line = JAPANESE_LINES[page_index % len(JAPANESE_LINES)]
        boxes = draw_vertical_text(draw, line, (320, 280), font, "white")
        style = "vertical_dark"
    else:
        image = Image.new("RGB", (width, height), (32, 36, 48))
        draw = ImageDraw.Draw(image)
        bubble = (80, 90, 640, 360)
        draw.rounded_rectangle(bubble, radius=36, fill="white", outline="#111827", width=4)
        font = available_font(ENGLISH_FONTS, 36)
        line = ENGLISH_LINES[page_index % len(ENGLISH_LINES)]
        draw.text((120, 180), line, font=font, fill="#111827")
        bbox = draw.textbbox((120, 180), line, font=font)
        boxes = [
            {
                "left": bbox[0],
                "top": bbox[1],
                "right": bbox[2],
                "bottom": bbox[3],
            }
        ]
        style = "english_bubble"

    filename = f"{language}_{page_index:03d}.png"
    image_path = output_dir / filename
    image.save(image_path)
    return {
        "file": filename,
        "language": language,
        "text": line,
        "style": style,
        "split": "train" if page_index % 5 else "test",
        "boxes": boxes,
    }


def generate_synthetic_split(
    output_dir: Path,
    pages_per_language: int = 40,
    seed: int = 42,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records = [
        render_page(language, index, output_dir, rng)
        for language in ("ja", "en")
        for index in range(pages_per_language)
    ]
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "lreader-synthetic-ja-en",
                "license": "CC0-1.0",
                "pages": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path

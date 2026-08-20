from __future__ import annotations

import json
from pathlib import Path

from lreader_engine.eval import Box, EvalPage


def load_synthetic_pages(root: Path) -> list[EvalPage]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    pages: list[EvalPage] = []
    for record in manifest["pages"]:
        pages.append(
            EvalPage(
                id=Path(record["file"]).stem,
                image_path=str(root / record["file"]),
                language=record["language"],
                split=record.get("split", "test"),
                source="synthetic-ja-en",
                boxes=[
                    Box(
                        left=float(box["left"]),
                        top=float(box["top"]),
                        right=float(box["right"]),
                        bottom=float(box["bottom"]),
                        text=record.get("text"),
                    )
                    for box in record["boxes"]
                ],
                expect_white_bubbles=record.get("style") == "english_bubble",
            )
        )
    return pages


def load_comix_pages(
    root: Path,
    *,
    split: str = "test",
    limit: int | None = None,
) -> list[EvalPage]:
    pages_dir = root / "pages"
    pages: list[EvalPage] = []
    for json_path in sorted(pages_dir.glob("*.json")):
        record = json.loads(json_path.read_text(encoding="utf-8"))
        if record.get("split") != split:
            continue
        image_name = record["image"]["file"]
        text_boxes = (
            record.get("detections", {}).get("fasterrcnn", {}).get("text", [])
        )
        pages.append(
            EvalPage(
                id=record["page_id"],
                image_path=str(pages_dir / image_name),
                language="en",
                split=split,
                source="comix-tiny",
                boxes=[
                    Box(
                        left=float(item["bbox"][0]),
                        top=float(item["bbox"][1]),
                        right=float(item["bbox"][2]),
                        bottom=float(item["bbox"][3]),
                    )
                    for item in text_boxes
                    if len(item.get("bbox") or []) == 4
                ],
                expect_white_bubbles=None,
            )
        )
        if limit is not None and len(pages) >= limit:
            break
    return pages

from __future__ import annotations

import csv
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from lreader_engine.eval import Box, EvalPage

MANGA109S_RELEASE = "Manga109s_released_2026_05_21"


def manga109s_root(data_root: Path) -> Path:
    direct = data_root / "manga109-s" / MANGA109S_RELEASE
    if direct.exists():
        return direct
    return data_root / MANGA109S_RELEASE


def list_manga109s_books(root: Path) -> list[str]:
    books_file = root / "books.txt"
    if not books_file.exists():
        return []
    return [
        line.strip()
        for line in books_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assign_book_splits(
    books: list[str],
    *,
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, str]:
    ordered = sorted(books)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    test_count = max(1, round(len(ordered) * test_ratio))
    val_count = max(1, round(len(ordered) * val_ratio))
    train_count = len(ordered) - test_count - val_count
    assignment: dict[str, str] = {}
    for index, book in enumerate(ordered):
        if index < train_count:
            assignment[book] = "train"
        elif index < train_count + val_count:
            assignment[book] = "val"
        else:
            assignment[book] = "test"
    return assignment


def _page_image_path(images_root: Path, book: str, index: int) -> Path | None:
    folder = images_root / book
    for name in (f"{index:03d}.jpg", f"{index:03d}.png", f"{index:03d}.jpeg"):
        path = folder / name
        if path.exists():
            return path
    return None


def load_manga109s_pages(
    root: Path,
    *,
    split: str | None = "test",
    assignment: dict[str, str] | None = None,
    books: list[str] | None = None,
    limit: int | None = None,
    min_text_boxes: int = 1,
) -> list[EvalPage]:
    book_names = books or list_manga109s_books(root)
    if assignment is None:
        assignment = assign_book_splits(book_names)
    images_root = root / "images"
    pages: list[EvalPage] = []
    for book in book_names:
        book_split = assignment.get(book)
        if split is not None and book_split != split:
            continue
        xml_path = root / "annotations" / f"{book}.xml"
        if not xml_path.exists():
            continue
        tree = ET.parse(xml_path)
        for page in tree.getroot().findall("pages/page"):
            index = int(page.attrib["index"])
            texts = page.findall("text")
            if len(texts) < min_text_boxes:
                continue
            image_path = _page_image_path(images_root, book, index)
            if image_path is None:
                continue
            pages.append(
                EvalPage(
                    id=f"{book}/{index:03d}",
                    image_path=str(image_path),
                    language="ja",
                    split=book_split or "test",
                    source="manga109-s",
                    boxes=[
                        Box(
                            left=float(node.attrib["xmin"]),
                            top=float(node.attrib["ymin"]),
                            right=float(node.attrib["xmax"]),
                            bottom=float(node.attrib["ymax"]),
                            text=(node.text or "").strip() or None,
                        )
                        for node in texts
                    ],
                )
            )
            if limit is not None and len(pages) >= limit:
                return pages
    return pages


def write_book_split_csv(
    path: Path,
    root: Path,
    assignment: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    for book, split in sorted(assignment.items()):
        xml_path = root / "annotations" / f"{book}.xml"
        page_count = 0
        text_pages = 0
        text_boxes = 0
        if xml_path.exists():
            tree = ET.parse(xml_path)
            for page in tree.getroot().findall("pages/page"):
                page_count += 1
                texts = page.findall("text")
                if texts:
                    text_pages += 1
                    text_boxes += len(texts)
        rows.append(
            {
                "book": book,
                "split": split,
                "pages": page_count,
                "text_pages": text_pages,
                "text_boxes": text_boxes,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["book", "split", "pages", "text_pages", "text_boxes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def read_book_split_csv(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["book"]: row["split"] for row in csv.DictReader(handle)}


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

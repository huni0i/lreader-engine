import json
from pathlib import Path

from PIL import Image

from lreader_engine.eval import Box, character_error_rate, match_boxes, matched_character_error_rates
from lreader_engine.eval_datasets import (
    assign_book_splits,
    load_comix_pages,
    load_manga109s_pages,
    load_synthetic_pages,
    write_book_split_csv,
)
from lreader_engine.synthetic_comics import generate_synthetic_split


def test_match_boxes_counts_an_overlapping_prediction() -> None:
    gold = Box(left=10, top=10, right=50, bottom=40, text="ここは")
    predicted = Box(left=12, top=11, right=48, bottom=38, text="ここは")

    metrics = match_boxes([predicted], [gold])

    assert metrics.matches == 1
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.mean_iou > 0.8


def test_character_error_rate_is_normalized_levenshtein() -> None:
    assert character_error_rate("abc", "abc") == 0.0
    assert character_error_rate("abx", "abc") == 1 / 3


def test_load_synthetic_pages_keeps_language_and_bubble_expectation(
    tmp_path: Path,
) -> None:
    generate_synthetic_split(tmp_path, pages_per_language=1)
    pages = load_synthetic_pages(tmp_path)

    japanese = next(page for page in pages if page.language == "ja")
    english = next(page for page in pages if page.language == "en")

    assert japanese.expect_white_bubbles is False
    assert english.expect_white_bubbles is True
    assert japanese.boxes
    assert english.boxes[0].text == "Where is this?"


def test_load_comix_pages_reads_fasterrcnn_text_boxes(tmp_path: Path) -> None:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    Image.new("RGB", (40, 60), "white").save(pages_dir / "c000_p000.jpg")
    (pages_dir / "c000_p000.json").write_text(
        json.dumps(
            {
                "page_id": "c000_p000",
                "split": "test",
                "image": {"file": "c000_p000.jpg"},
                "detections": {
                    "fasterrcnn": {
                        "text": [{"bbox": [2, 4, 20, 18]}, {"bbox": [5, 30, 30, 50]}]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    pages = load_comix_pages(tmp_path, split="test")

    assert len(pages) == 1
    assert pages[0].language == "en"
    assert len(pages[0].boxes) == 2
    assert pages[0].boxes[0].left == 2


def test_matched_cer_uses_iou_pairs() -> None:
    gold = Box(left=0, top=0, right=10, bottom=10, text="abc")
    predicted = Box(left=0, top=0, right=10, bottom=10, text="abx")

    assert matched_character_error_rates([predicted], [gold]) == [1 / 3]


def test_book_splits_are_disjoint_and_cover_every_book() -> None:
    books = [f"book{index:02d}" for index in range(20)]
    assignment = assign_book_splits(books, seed=42)

    assert set(assignment) == set(books)
    assert set(assignment.values()) == {"train", "val", "test"}
    assert sum(split == "test" for split in assignment.values()) == 3
    assert assign_book_splits(books, seed=42) == assignment


def test_load_manga109s_pages_reads_text_boxes(tmp_path: Path) -> None:
    images = tmp_path / "images" / "DemoBook"
    images.mkdir(parents=True)
    Image.new("RGB", (40, 60), "white").save(images / "002.jpg")
    (tmp_path / "books.txt").write_text("DemoBook\n", encoding="utf-8")
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "DemoBook.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<book title="DemoBook">
  <pages>
    <page index="0" width="40" height="60" />
    <page index="2" width="40" height="60">
      <text id="t1" xmin="4" ymin="5" xmax="18" ymax="20">あ</text>
      <text id="t2" xmin="10" ymin="30" xmax="30" ymax="50">い</text>
    </page>
  </pages>
</book>
""",
        encoding="utf-8",
    )

    pages = load_manga109s_pages(
        tmp_path,
        split=None,
        assignment={"DemoBook": "test"},
    )

    assert len(pages) == 1
    assert pages[0].id == "DemoBook/002"
    assert pages[0].language == "ja"
    assert [box.text for box in pages[0].boxes] == ["あ", "い"]


def test_write_book_split_csv_round_trips(tmp_path: Path) -> None:
    (tmp_path / "annotations").mkdir()
    csv_path = tmp_path / "split.csv"
    write_book_split_csv(csv_path, tmp_path, {"Demo": "test"})
    text = csv_path.read_text(encoding="utf-8")
    assert "Demo,test,0,0,0" in text


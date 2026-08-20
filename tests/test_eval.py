import json
from pathlib import Path

from PIL import Image

from lreader_engine.eval import Box, character_error_rate, match_boxes
from lreader_engine.eval_datasets import load_comix_pages, load_synthetic_pages
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

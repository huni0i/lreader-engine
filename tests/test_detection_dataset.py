from pathlib import Path

from PIL import Image

from lreader_engine.detection_dataset import box_to_region, write_yolo_split, yolo_label_lines
from lreader_engine.eval import Box, EvalPage


def test_yolo_labels_are_normalized(tmp_path: Path) -> None:
    image_path = tmp_path / "page.jpg"
    Image.new("RGB", (100, 50), "white").save(image_path)
    page = EvalPage(
        id="Demo/001",
        image_path=str(image_path),
        language="ja",
        split="train",
        source="manga109-s",
        boxes=[Box(left=10, top=5, right=30, bottom=25, text="あ")],
    )

    assert yolo_label_lines(page) == ["0 0.200000 0.300000 0.200000 0.400000"]


def test_write_yolo_split_uses_page_id(tmp_path: Path) -> None:
    image_path = tmp_path / "page.jpg"
    Image.new("RGB", (40, 60), "white").save(image_path)
    page = EvalPage(
        id="Demo/002",
        image_path=str(image_path),
        language="ja",
        split="train",
        source="manga109-s",
        boxes=[Box(left=4, top=5, right=18, bottom=20)],
    )
    write_yolo_split([page], tmp_path, "train")

    label = (tmp_path / "labels" / "train" / "Demo_002.txt").read_text(encoding="utf-8")
    assert label.startswith("0 ")
    assert (tmp_path / "images" / "train" / "Demo_002.jpg").exists()


def test_box_to_region_clamps_negative_coords() -> None:
    region = box_to_region(Box(left=-2, top=1, right=10, bottom=8), text="あ")
    assert region.polygon[0].x == 0.0
    assert region.text == "あ"

from __future__ import annotations

from pathlib import Path

from PIL import Image

from lreader_engine.eval import Box, EvalPage
from lreader_engine.models import OcrRegion, Point


def page_stem(page: EvalPage) -> str:
    return page.id.replace("/", "_")


def clamp_coord(value: float) -> float:
    return max(0.0, float(value))


def box_to_region(box: Box, text: str = "", confidence: float = 1.0) -> OcrRegion:
    return OcrRegion(
        polygon=[
            Point(x=clamp_coord(box.left), y=clamp_coord(box.top)),
            Point(x=clamp_coord(box.right), y=clamp_coord(box.top)),
            Point(x=clamp_coord(box.right), y=clamp_coord(box.bottom)),
            Point(x=clamp_coord(box.left), y=clamp_coord(box.bottom)),
        ],
        text=text,
        confidence=confidence,
    )


def yolo_label_lines(page: EvalPage) -> list[str]:
    with Image.open(page.image_path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        return []
    lines: list[str] = []
    for box in page.boxes:
        box_width = box.right - box.left
        box_height = box.bottom - box.top
        if box_width <= 0 or box_height <= 0:
            continue
        x_center = ((box.left + box.right) / 2) / width
        y_center = ((box.top + box.bottom) / 2) / height
        lines.append(
            "0 "
            f"{x_center:.6f} {y_center:.6f} "
            f"{box_width / width:.6f} {box_height / height:.6f}"
        )
    return lines


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        destination.write_bytes(source.read_bytes())


def write_yolo_split(pages: list[EvalPage], output_dir: Path, split: str) -> None:
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for page in pages:
        source = Path(page.image_path)
        stem = page_stem(page)
        _link_or_copy(source, image_dir / f"{stem}{source.suffix.lower()}")
        (label_dir / f"{stem}.txt").write_text(
            "\n".join(yolo_label_lines(page)) + ("\n" if page.boxes else ""),
            encoding="utf-8",
        )


def write_dataset_yaml(output_dir: Path) -> Path:
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(
        (
            f"path: {output_dir.resolve()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: text\n"
        ),
        encoding="utf-8",
    )
    return yaml_path

#!/usr/bin/env python3
"""Train a YOLO text detector on Manga109-s book-level splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lreader_engine.detection_dataset import write_dataset_yaml, write_yolo_split
from lreader_engine.eval import Box, match_boxes
from lreader_engine.eval_datasets import (
    load_manga109s_pages,
    manga109s_root,
    read_book_split_csv,
)


DEFAULT_SPLIT = ROOT / "evals" / "manga109s_book_split.csv"


def log(message: str) -> None:
    print(message, flush=True)


def load_split(data_root: Path, split_csv: Path, split: str, limit: int | None):
    root = manga109s_root(data_root)
    assignment = read_book_split_csv(split_csv) if split_csv.exists() else None
    return load_manga109s_pages(
        root,
        split=split,
        assignment=assignment,
        limit=limit,
    )


def resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict_boxes(model, image_path: str) -> list[Box]:
    boxes: list[Box] = []
    for result in model.predict(image_path, verbose=False):
        if result.boxes is None:
            continue
        for xyxy in result.boxes.xyxy.cpu().tolist():
            boxes.append(Box(left=xyxy[0], top=xyxy[1], right=xyxy[2], bottom=xyxy[3]))
    return boxes


def evaluate_pages(model, pages) -> dict:
    metrics = [
        match_boxes(predict_boxes(model, page.image_path), page.boxes) for page in pages
    ]
    predicted = sum(item.predicted for item in metrics)
    ground_truth = sum(item.ground_truth for item in metrics)
    matches = sum(item.matches for item in metrics)
    return {
        "pages": len(pages),
        "predicted": predicted,
        "ground_truth": ground_truth,
        "matches": matches,
        "precision": matches / predicted if predicted else 1.0,
        "recall": matches / ground_truth if ground_truth else 1.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "det-text")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_pages = load_split(args.data_root, args.split_csv, "train", args.train_limit)
    val_pages = load_split(args.data_root, args.split_csv, "val", args.val_limit)
    test_pages = load_split(args.data_root, args.split_csv, "test", args.eval_limit)
    if not train_pages or not val_pages:
        raise SystemExit("Manga109-s train/val pages missing")

    log(f"train={len(train_pages)} val={len(val_pages)} test_eval={len(test_pages)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_yolo_split(train_pages, args.output_dir, "train")
    write_yolo_split(val_pages, args.output_dir, "val")
    yaml_path = write_dataset_yaml(args.output_dir)
    if args.skip_train:
        log(f"dataset ready at {yaml_path}")
        return 0

    from ultralytics import YOLO

    device = resolve_device()
    log(f"training {args.model} on {device}")
    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(args.output_dir / "runs"),
        name="text-detector",
        exist_ok=True,
        workers=2,
        patience=5,
        plots=False,
    )
    best = args.output_dir / "runs" / "text-detector" / "weights" / "best.pt"
    trained = YOLO(str(best))
    report = {
        "device": device,
        "model": args.model,
        "train_pages": len(train_pages),
        "val_pages": len(val_pages),
        "weights": str(best),
        "val": evaluate_pages(trained, val_pages[: args.eval_limit]),
        "manga109s_test": evaluate_pages(trained, test_pages),
    }
    report_path = args.output_dir / "train_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(json.dumps(report, indent=2))
    log(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

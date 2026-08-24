#!/usr/bin/env python3
"""Score YOLO boxes + manga-ocr on Manga109-s, using the same metrics as the engine bench."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lreader_engine.detection_dataset import box_to_region
from lreader_engine.eval import Box, match_boxes, matched_character_error_rates, mean
from lreader_engine.eval_datasets import (
    load_manga109s_pages,
    manga109s_root,
    read_book_split_csv,
)
from lreader_engine.manga_ocr import MangaOcrEngine


DEFAULT_SPLIT = ROOT / "evals" / "manga109s_book_split.csv"
DEFAULT_WEIGHTS = ROOT / "data" / "det-text" / "runs" / "text-detector" / "weights" / "best.pt"


def log(message: str) -> None:
    print(message, flush=True)


def clamp(value: float) -> float:
    return max(0.0, float(value))


def predict_boxes(model, image_path: str, confidence: float) -> list[Box]:
    boxes: list[Box] = []
    for result in model.predict(image_path, conf=confidence, verbose=False):
        if result.boxes is None:
            continue
        for xyxy in result.boxes.xyxy.cpu().tolist():
            boxes.append(
                Box(
                    left=clamp(xyxy[0]),
                    top=clamp(xyxy[1]),
                    right=clamp(xyxy[2]),
                    bottom=clamp(xyxy[3]),
                )
            )
    return boxes


def summarize(rows: list[dict]) -> dict:
    detections = [row["detection"] for row in rows]
    predicted = sum(item["predicted"] for item in detections)
    ground_truth = sum(item["ground_truth"] for item in detections)
    matches = sum(item["matches"] for item in detections)
    cers = [cer for row in rows for cer in row["cers"]]
    return {
        "pages": len(rows),
        "predicted": predicted,
        "ground_truth": ground_truth,
        "matches": matches,
        "precision": matches / predicted if predicted else 1.0,
        "recall": matches / ground_truth if ground_truth else 1.0,
        "mean_iou": mean([item["mean_iou"] for item in detections if item["matches"]]),
        "cer": mean(cers) if cers else None,
        "latency_sec": mean([row["latency_sec"] for row in rows]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "benchmarks" / "yolo_manga_ocr.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.weights.exists():
        raise SystemExit(f"missing YOLO weights at {args.weights}")

    from ultralytics import YOLO

    root = manga109s_root(args.data_root)
    assignment = (
        read_book_split_csv(args.split_csv) if args.split_csv.exists() else None
    )
    pages = load_manga109s_pages(
        root,
        split=args.split,
        assignment=assignment,
        limit=args.limit,
    )
    if not pages:
        raise SystemExit("no Manga109-s pages loaded")

    detector = YOLO(str(args.weights))
    reader = MangaOcrEngine()
    rows: list[dict] = []
    log(f"pages={len(pages)} weights={args.weights}")

    for index, page in enumerate(pages, start=1):
        started = time.perf_counter()
        boxes = predict_boxes(detector, page.image_path, args.conf)
        predicted: list[Box] = []
        for box in boxes:
            text = reader.recognize_region(page.image_path, box_to_region(box))
            predicted.append(
                Box(
                    left=box.left,
                    top=box.top,
                    right=box.right,
                    bottom=box.bottom,
                    text=text,
                )
            )
        elapsed = time.perf_counter() - started
        metrics = match_boxes(predicted, page.boxes)
        cers = matched_character_error_rates(predicted, page.boxes)
        rows.append(
            {
                "id": page.id,
                "latency_sec": elapsed,
                "detection": {
                    "predicted": metrics.predicted,
                    "ground_truth": metrics.ground_truth,
                    "matches": metrics.matches,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "mean_iou": metrics.mean_iou,
                },
                "cers": cers,
            }
        )
        log(
            f"  {index}/{len(pages)} {page.id} "
            f"recall={metrics.recall:.3f} cer={mean(cers) if cers else None} "
            f"{elapsed:.1f}s"
        )

    summary = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "pages": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(
        f"recall={summary['recall']:.3f} precision={summary['precision']:.3f} "
        f"cer={summary['cer']} latency={summary['latency_sec']:.1f}s"
    )
    log(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare EasyOCR / spotting / routed OCR on Manga109-s test books."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx

from lreader_engine.bubble_detector import BubbleDetector
from lreader_engine.eval import (
    Box,
    box_from_polygon,
    match_boxes,
    matched_character_error_rates,
    mean,
)
from lreader_engine.eval_datasets import (
    load_manga109s_pages,
    manga109s_root,
    read_book_split_csv,
)


DEFAULT_ENGINE_URL = os.getenv("LREADER_ENGINE_URL", "http://100.107.63.5:8765")
DEFAULT_SPLIT = ROOT / "evals" / "manga109s_book_split.csv"
OCR_MODES = ("easy", "spot", "route")


def log(message: str) -> None:
    print(message, flush=True)


def engine_health(engine_url: str) -> bool:
    try:
        response = httpx.get(f"{engine_url.rstrip('/')}/health", timeout=5)
    except httpx.HTTPError:
        return False
    return response.status_code == 200 and response.json().get("status") == "ok"


def predict(
    engine_url: str,
    image_path: str,
    ocr_mode: str,
) -> tuple[list[Box], float]:
    started = time.perf_counter()
    with Path(image_path).open("rb") as handle:
        response = httpx.post(
            f"{engine_url.rstrip('/')}/v1/images/translate",
            params={
                "source_language": "ja",
                "target_language": "ko",
                "quality": "ocr",
                "inpaint": "false",
                "ocr_mode": ocr_mode,
                "skip_translate": "true",
            },
            files={"file": (Path(image_path).name, handle, "image/jpeg")},
            timeout=300,
        )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    boxes: list[Box] = []
    for region in response.json().get("regions") or []:
        polygon = region.get("polygon") or []
        if len(polygon) < 4:
            continue
        boxes.append(box_from_polygon(polygon, text=region.get("text")))
    return boxes, elapsed


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
    parser.add_argument("--engine-url", default=DEFAULT_ENGINE_URL)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(OCR_MODES),
        choices=OCR_MODES,
    )
    parser.add_argument("--skip-engine", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "benchmarks" / "manga109s.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    log(f"loaded pages={len(pages)} split={args.split} root={root}")
    if not pages:
        raise SystemExit("no Manga109-s pages loaded")

    detector = BubbleDetector()
    bubble_hits = 0
    for page in pages:
        if detector.has_speech_bubbles(page.image_path):
            bubble_hits += 1
    bubble = {
        "pages": len(pages),
        "opencv_white_bubble_rate": bubble_hits / len(pages),
    }
    log(
        "opencv white-bubble rate="
        f"{bubble['opencv_white_bubble_rate']:.3f} pages={len(pages)}"
    )

    reachable = False if args.skip_engine else engine_health(args.engine_url)
    modes: dict[str, dict] = {}
    if reachable:
        for mode in args.modes:
            log(f"mode={mode}")
            rows: list[dict] = []
            for index, page in enumerate(pages, start=1):
                log(f"  {index}/{len(pages)} {page.id}")
                predicted, latency = predict(args.engine_url, page.image_path, mode)
                metrics = match_boxes(predicted, page.boxes)
                rows.append(
                    {
                        "id": page.id,
                        "latency_sec": latency,
                        "detection": {
                            "predicted": metrics.predicted,
                            "ground_truth": metrics.ground_truth,
                            "matches": metrics.matches,
                            "precision": metrics.precision,
                            "recall": metrics.recall,
                            "mean_iou": metrics.mean_iou,
                        },
                        "cers": matched_character_error_rates(predicted, page.boxes),
                    }
                )
            modes[mode] = {"pages": rows, "summary": summarize(rows)}
            summary = modes[mode]["summary"]
            log(
                f"  recall={summary['recall']:.3f} "
                f"precision={summary['precision']:.3f} "
                f"cer={summary['cer']} "
                f"latency={summary['latency_sec']:.1f}s"
            )
    else:
        log(f"engine offline at {args.engine_url}; OCR modes skipped")

    report = {
        "engine": {"url": args.engine_url, "reachable": reachable},
        "split": args.split,
        "limit": args.limit,
        "bubble_probe": bubble,
        "modes": {
            mode: result["summary"]
            for mode, result in modes.items()
        },
        "pages": {
            mode: result["pages"]
            for mode, result in modes.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown = [
        "# Manga109-s OCR cascade",
        "",
        f"- split: `{args.split}`",
        f"- pages: {len(pages)}",
        f"- engine: `{args.engine_url}` reachable={reachable}",
        f"- OpenCV white-bubble rate: {bubble['opencv_white_bubble_rate']:.3f}",
        "",
    ]
    if modes:
        markdown.extend(
            [
                "| mode | recall@0.5 | precision | CER | latency (s) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for mode in args.modes:
            if mode not in modes:
                continue
            item = modes[mode]["summary"]
            cer = "n/a" if item["cer"] is None else f"{item['cer']:.3f}"
            markdown.append(
                f"| {mode} | {item['recall']:.3f} | {item['precision']:.3f} "
                f"| {cer} | {item['latency_sec']:.1f} |"
            )
        markdown.append("")
    args.output.with_suffix(".md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )
    log(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

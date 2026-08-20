#!/usr/bin/env python3
"""Benchmark Lreader stages against labeled Japanese/English pages."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx

from lreader_engine.bubble_detector import BubbleDetector
from lreader_engine.eval import (
    Box,
    DetectionMetrics,
    EvalPage,
    box_from_polygon,
    character_error_rate,
    match_boxes,
    mean,
)
from lreader_engine.eval_datasets import load_comix_pages, load_synthetic_pages


DEFAULT_ENGINE_URL = os.getenv("LREADER_ENGINE_URL", "http://100.107.63.5:8765")


def log(message: str) -> None:
    print(message, flush=True)


def load_pages(data_root: Path, comix_limit: int | None) -> list[EvalPage]:
    pages = load_synthetic_pages(data_root / "synthetic-ja-en")
    comix_root = data_root / "comix-tiny"
    if (comix_root / "pages").exists():
        pages.extend(
            load_comix_pages(comix_root, split="test", limit=comix_limit)
        )
    return pages


def engine_health(engine_url: str) -> bool:
    try:
        response = httpx.get(f"{engine_url.rstrip('/')}/health", timeout=3)
    except httpx.HTTPError:
        return False
    return response.status_code == 200 and response.json().get("status") == "ok"


def probe_bubbles(pages: list[EvalPage]) -> dict:
    detector = BubbleDetector()
    labeled = [page for page in pages if page.expect_white_bubbles is not None]
    correct = 0
    by_source: dict[str, list[int]] = defaultdict(list)
    for page in labeled:
        predicted = detector.has_speech_bubbles(page.image_path)
        hit = int(predicted is page.expect_white_bubbles)
        correct += hit
        by_source[page.source].append(hit)
    return {
        "pages": len(labeled),
        "accuracy": correct / len(labeled) if labeled else 0.0,
        "by_source": {
            source: sum(values) / len(values)
            for source, values in by_source.items()
        },
    }


def predict_engine_boxes(
    engine_url: str,
    page: EvalPage,
) -> list[Box]:
    source_language = page.language if page.language in {"ja", "en"} else "en"
    with Path(page.image_path).open("rb") as handle:
        response = httpx.post(
            f"{engine_url.rstrip('/')}/v1/images/translate",
            params={
                "source_language": source_language,
                "target_language": "ko",
                "quality": "ocr",
                "inpaint": "false",
            },
            files={"file": (Path(page.image_path).name, handle, "image/jpeg")},
            timeout=180,
        )
    response.raise_for_status()
    payload = response.json()
    boxes: list[Box] = []
    for region in payload.get("regions") or []:
        polygon = region.get("polygon") or []
        if len(polygon) < 4:
            continue
        boxes.append(
            box_from_polygon(
                polygon,
                text=region.get("text"),
            )
        )
    return boxes


def summarize_detection(metrics: list[DetectionMetrics]) -> dict:
    predicted = sum(item.predicted for item in metrics)
    ground_truth = sum(item.ground_truth for item in metrics)
    matches = sum(item.matches for item in metrics)
    return {
        "pages": len(metrics),
        "predicted": predicted,
        "ground_truth": ground_truth,
        "matches": matches,
        "precision": matches / predicted if predicted else 1.0,
        "recall": matches / ground_truth if ground_truth else 1.0,
        "mean_iou": mean([item.mean_iou for item in metrics if item.matches]),
    }


def run_engine_detection(
    engine_url: str,
    pages: list[EvalPage],
    limit: int,
) -> dict:
    selected = [page for page in pages if page.boxes][:limit]
    by_source: dict[str, list[DetectionMetrics]] = defaultdict(list)
    cers: list[float] = []
    for index, page in enumerate(selected, start=1):
        log(f"  engine {index}/{len(selected)} {page.id}")
        predicted = predict_engine_boxes(engine_url, page)
        metrics = match_boxes(predicted, page.boxes)
        by_source[page.source].append(metrics)
        gold_texts = [box.text for box in page.boxes if box.text]
        pred_texts = [box.text for box in predicted if box.text]
        if gold_texts and pred_texts:
            cers.append(character_error_rate("".join(pred_texts), "".join(gold_texts)))
    return {
        "pages": len(selected),
        "overall": summarize_detection(
            [item for group in by_source.values() for item in group]
        ),
        "by_source": {
            source: summarize_detection(group)
            for source, group in by_source.items()
        },
        "character_error_rate": mean(cers) if cers else None,
    }


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        "# Lreader benchmark",
        "",
        f"- engine: `{report['engine']['url']}`",
        f"- engine reachable: {report['engine']['reachable']}",
        f"- pages: {report['pages']}",
        f"- bubble probe accuracy: {report['bubble_probe']['accuracy']:.3f} "
        f"({report['bubble_probe']['pages']} labeled pages)",
    ]
    if report.get("ocr_detection"):
        overall = report["ocr_detection"]["overall"]
        markdown.append(
            f"- OCR box recall: {overall['recall']:.3f} / precision: {overall['precision']:.3f}"
        )
    else:
        markdown.append("- OCR detection: skipped (engine offline)")
    markdown.append("")
    path.with_suffix(".md").write_text("\n".join(markdown), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--engine-url", default=DEFAULT_ENGINE_URL)
    parser.add_argument("--comix-limit", type=int, default=None)
    parser.add_argument("--engine-limit", type=int, default=20)
    parser.add_argument("--skip-engine", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "benchmarks" / "report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pages = load_pages(args.data_root, args.comix_limit)
    log(f"loaded pages={len(pages)}")
    bubble = probe_bubbles(pages)
    log(f"bubble probe accuracy={bubble['accuracy']:.3f} pages={bubble['pages']}")

    reachable = False if args.skip_engine else engine_health(args.engine_url)
    ocr_detection = None
    if reachable:
        log("engine reachable; running OCR box matching")
        ocr_detection = run_engine_detection(
            args.engine_url,
            pages,
            args.engine_limit,
        )
    else:
        log(f"engine offline at {args.engine_url}; OCR stage skipped")

    report = {
        "pages": len(pages),
        "sources": sorted({page.source for page in pages}),
        "engine": {"url": args.engine_url, "reachable": reachable},
        "bubble_probe": bubble,
        "ocr_detection": ocr_detection,
    }
    write_report(args.output, report)
    log(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

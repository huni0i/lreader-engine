#!/usr/bin/env python3
"""Train the small appearance-conditioned OCR router on synthetic pages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lreader_engine.eval_datasets import load_synthetic_pages
from lreader_engine.ocr_router import train_router


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "synthetic-ja-en",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "experiments" / "ocr-router",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pages = load_synthetic_pages(args.data_root)
    print(f"pages={len(pages)}", flush=True)
    report = train_router(
        pages,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "baseline={:.3f} learned={:.3f} heatmap_iou={:.3f} device={}".format(
            report["baseline_opencv_test_accuracy"],
            report["learned_test"]["route_accuracy"],
            report["learned_test"]["heatmap_iou"],
            report["device"],
        ),
        flush=True,
    )
    print(f"wrote {args.output / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

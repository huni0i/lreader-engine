#!/usr/bin/env python3
"""Write a reproducible Manga109-s book-level train/val/test split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lreader_engine.eval_datasets import (
    assign_book_splits,
    list_manga109s_books,
    manga109s_root,
    write_book_split_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evals" / "manga109s_book_split.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = manga109s_root(args.data_root)
    books = list_manga109s_books(root)
    if not books:
        raise SystemExit(f"no Manga109-s books under {root}")
    assignment = assign_book_splits(books, seed=args.seed)
    write_book_split_csv(args.output, root, assignment)
    counts = Counter(assignment.values())
    summary = {
        "root": str(root),
        "seed": args.seed,
        "books": len(books),
        "splits": dict(counts),
        "csv": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"books={len(books)} train={counts['train']} "
        f"val={counts['val']} test={counts['test']}",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

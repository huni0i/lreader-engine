#!/usr/bin/env python3
"""Download Manga109-s after Hugging Face access is granted."""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "manga109-s"
REPO_ID = "hal-utokyo/Manga109-s"
ZIP_NAME = "Manga109s_released_2026_05_21.zip"
ZIP_BYTES = 3_295_300_228


def log(message: str) -> None:
    print(message, flush=True)


def has_images(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.jpg"))


def main() -> int:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import get_token, hf_hub_download

    if not get_token():
        log("Hugging Face login is missing. Run:")
        log("  /Users/huni0i/Projects/Lreader/server/.venv/bin/hf auth login")
        return 1

    TARGET.mkdir(parents=True, exist_ok=True)
    if has_images(TARGET) or has_images(TARGET / "images"):
        log(f"already present: {TARGET}")
        return 0

    zip_path = TARGET / ZIP_NAME
    free = shutil.disk_usage(ROOT).free
    if not zip_path.exists() or zip_path.stat().st_size < ZIP_BYTES * 0.99:
        if free < ZIP_BYTES + 800_000_000:
            log(
                f"Not enough disk to download the zip. Need ~4GB free, have {free / 1e9:.2f}GB."
            )
            return 1
        log(f"downloading {REPO_ID}")
        zip_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=ZIP_NAME,
                local_dir=str(TARGET),
            )
        )

    free = shutil.disk_usage(ROOT).free
    if free < ZIP_BYTES:
        log(
            f"Zip is downloaded but extract needs ~{ZIP_BYTES / 1e9:.1f}GB more free. "
            f"Have {free / 1e9:.2f}GB."
        )
        return 1

    log(f"extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(TARGET)
    zip_path.unlink(missing_ok=True)
    log(f"unpacked to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

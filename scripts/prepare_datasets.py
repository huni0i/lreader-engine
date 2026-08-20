#!/usr/bin/env python3
"""Collect Japanese and English comic research datasets that we can legally use."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
USER_AGENT = "Mozilla/5.0 (compatible; lreader-engine-dataset-prep/0.2; academic research)"
MAX_IA_BYTES = 100 * 1024 * 1024

CATALOG = [
    {
        "id": "synthetic-ja-en",
        "language": ["ja", "en"],
        "license": "CC0-1.0",
        "access": "generated",
        "notes": "Local synthetic vertical Japanese text and English bubbles with exact boxes.",
    },
    {
        "id": "dcm772-annotations",
        "language": ["en"],
        "license": "public-domain images; research annotations from Univ. La Rochelle",
        "access": "git",
        "source": "https://gitlab.univ-lr.fr/crigau02/dcm-dataset",
        "notes": "772 English golden-age pages. Images are not in the repo.",
    },
    {
        "id": "internet-archive-pd-comics",
        "language": ["en"],
        "license": "public-domain",
        "access": "internet-archive",
        "notes": "Public-domain English comic books that match DCM-style titles.",
    },
    {
        "id": "comics-cvpr2017",
        "language": ["en"],
        "license": "public-domain source comics; research annotations from UMD",
        "access": "http",
        "source": "https://obj.umiacs.umd.edu/comics/index.html",
        "notes": "OCR transcriptions and textbox boxes. Full page/panel image tars are skipped (100GB+).",
    },
    {
        "id": "comix-tiny",
        "language": ["en"],
        "license": "CC0-1.0; underlying scans are public domain",
        "access": "huggingface",
        "source": "https://huggingface.co/datasets/emanuelevivoli/comix-v0_1-pages-tiny",
        "notes": "CoMix tiny train/test/validation pages. Tars are extracted then deleted to save disk.",
    },
    {
        "id": "coo-annotations",
        "language": ["ja"],
        "license": "research annotations; images require Manga109",
        "access": "git",
        "source": "https://github.com/ku21fan/COO-Comic-Onomatopoeia",
        "notes": "Japanese onomatopoeia polygons. Pair with Manga109 images after access is granted.",
    },
    {
        "id": "manga109",
        "language": ["ja"],
        "license": "author-permitted academic use; no redistribution",
        "access": "gated",
        "apply_url": "https://huggingface.co/datasets/hal-utokyo/Manga109",
        "notes": "Apply with an academic use statement. Do not commit images.",
    },
    {
        "id": "manga109-s",
        "language": ["ja"],
        "license": "author-permitted commercial/research use; no redistribution",
        "access": "gated",
        "apply_url": "https://huggingface.co/datasets/hal-utokyo/Manga109-s",
        "notes": "87 volumes. Prefer this if a portfolio demo needs a commercial-safe Japanese set.",
    },
    {
        "id": "ebdtheque",
        "language": ["en", "ja", "fr"],
        "license": "research registration required",
        "access": "gated",
        "apply_url": "https://ebdtheque.univ-lr.fr/registration",
        "notes": "100 annotated pages with panels, balloons, and text lines.",
    },
]


IA_ITEMS = [
    "Animal_Comics_001",
    "Big_Shot_Comics_001",
    "DCMGoldenAge",
    "JoePalooka1937",
    "OutOfThisWorldAdventuresIssue1",
    "OutOfThisWorldAdventuresIssue2",
    "FamousAuthorsIllustrated00903",
    "gangsters-and-gun-molls-04-1952-06.-realistic-jvj-2by-nation-ia",
]

COMICS_FILES = [
    "COMICS_ocr_file.csv",
    "textboxes_annotations.zip",
    "predadpages.txt",
]

COMIX_TINY_FILES = [
    "README.md",
    "pages-test-00059.tar",
    "pages-test-00081.tar",
    "pages-test-00094.tar",
    "pages-train-00059.tar",
    "pages-train-00068.tar",
    "pages-train-00081.tar",
    "pages-train-00094.tar",
    "pages-train-00095.tar",
    "pages-validation-00068.tar",
    "pages-validation-00095.tar",
]


def log(message: str) -> None:
    print(message, flush=True)


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def download_file(url: str, dest: Path, *, max_bytes: int | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=600) as response:
        length_header = response.headers.get("Content-Length")
        if max_bytes is not None and length_header and int(length_header) > max_bytes:
            raise RuntimeError(
                f"{dest.name} is {length_header} bytes, over the {max_bytes} byte cap"
            )
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as handle:
            copied = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if max_bytes is not None and copied > max_bytes:
                    handle.close()
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"{dest.name} exceeded the {max_bytes} byte cap")
                handle.write(chunk)
        tmp.replace(dest)
    return dest


def choose_ia_file(files: list[dict]) -> dict | None:
    skip_tokens = ("jp2", "daisy", "_text.", "files.xml", ".xml", ".sqlite", ".torrent")
    candidates: list[dict] = []
    for item in files:
        name = str(item.get("name") or "")
        lower = name.lower()
        if not lower.endswith((".pdf", ".cbz", ".cbr", ".zip")):
            continue
        if any(token in lower for token in skip_tokens):
            continue
        size = int(item.get("size") or 0)
        if size <= 0 or size > MAX_IA_BYTES:
            continue
        candidates.append(item)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: int(item.get("size") or 0))[0]


def git_clone(url: str, dest: Path, *, sparse_paths: list[str] | None = None) -> Path:
    if dest.exists() and any(dest.iterdir()):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = ["git", "clone", "--depth", "1"]
    if sparse_paths:
        command.extend(["--filter=blob:none", "--sparse", url, str(dest)])
        subprocess.check_call(command)
        subprocess.check_call(["git", "-C", str(dest), "sparse-checkout", "set", *sparse_paths])
        return dest
    command.extend([url, str(dest)])
    subprocess.check_call(command)
    return dest


def write_catalog() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "catalog.json").write_text(
        json.dumps(CATALOG, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def generate_synthetic() -> Path:
    from lreader_engine.synthetic_comics import generate_synthetic_split

    output_dir = DATA_ROOT / "synthetic-ja-en"
    return generate_synthetic_split(output_dir, pages_per_language=80)


def fetch_dcm_annotations() -> Path:
    target = DATA_ROOT / "dcm772" / "repo"
    return git_clone("https://gitlab.univ-lr.fr/crigau02/dcm-dataset.git", target)


def fetch_internet_archive_item(identifier: str) -> Path | None:
    target = DATA_ROOT / "internet-archive-pd" / identifier
    if any(target.glob("*.pdf")) or any(target.glob("*.cbz")) or any(target.glob("*.cbr")):
        return target
    metadata = json.loads(
        urlopen(
            Request(
                f"https://archive.org/metadata/{identifier}",
                headers={"User-Agent": USER_AGENT},
            ),
            timeout=60,
        ).read().decode("utf-8")
    )
    chosen = choose_ia_file(metadata.get("files") or [])
    if chosen is None:
        write_text(
            target / "MISSING.txt",
            f"No public-domain comic file under {MAX_IA_BYTES} bytes for {identifier}\n",
        )
        return None
    filename = chosen["name"]
    download_file(
        f"https://archive.org/download/{identifier}/{quote(filename)}",
        target / Path(filename).name,
        max_bytes=MAX_IA_BYTES,
    )
    write_text(
        target / "SOURCE.txt",
        "\n".join(
            [
                f"identifier: {identifier}",
                f"file: {filename}",
                f"size: {chosen.get('size')}",
                "https://archive.org/details/" + identifier,
                "",
            ]
        ),
    )
    return target


def fetch_comics_annotations() -> Path:
    target = DATA_ROOT / "comics-cvpr2017"
    for name in COMICS_FILES:
        log(f"  downloading {name}")
        download_file(f"https://obj.umiacs.umd.edu/comics/{name}", target / name)
    zip_path = target / "textboxes_annotations.zip"
    extract_dir = target / "textboxes"
    if zip_path.exists() and not any(extract_dir.rglob("*")):
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
    if zip_path.exists() and any(extract_dir.rglob("*")):
        zip_path.unlink()
    write_text(
        target / "README.txt",
        "\n".join(
            [
                "COMICS (CVPR 2017) annotations from UMD.",
                "Source comics are public-domain Golden Age scans.",
                "Page and panel image tars were skipped because they are 100GB+.",
                "https://obj.umiacs.umd.edu/comics/index.html",
                "",
            ]
        ),
    )
    return target


def fetch_comix_tiny() -> Path:
    target = DATA_ROOT / "comix-tiny"
    pages_dir = target / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    base = "https://huggingface.co/datasets/emanuelevivoli/comix-v0_1-pages-tiny/resolve/main"
    for name in COMIX_TINY_FILES:
        dest = target / name
        if not name.endswith(".tar"):
            log(f"  downloading {name}")
            download_file(f"{base}/{name}", dest)
            continue
        marker = pages_dir / f".extracted-{name}"
        if marker.exists():
            dest.unlink(missing_ok=True)
            continue
        if not dest.exists() or dest.stat().st_size == 0:
            log(f"  downloading {name}")
            download_file(f"{base}/{name}", dest)
        log(f"  extracting {name}")
        with tarfile.open(dest) as archive:
            archive.extractall(pages_dir)
        marker.touch()
        dest.unlink()
    write_text(
        target / "SOURCE.txt",
        "CoMix v0.1 pages-tiny train/test/validation pages. Archive tars are deleted after extract.\n"
        "https://huggingface.co/datasets/emanuelevivoli/comix-v0_1-pages-tiny\n",
    )
    return target


def fetch_coo_annotations() -> Path:
    target = DATA_ROOT / "coo"
    git_clone(
        "https://github.com/ku21fan/COO-Comic-Onomatopoeia.git",
        target / "repo",
        sparse_paths=["COO-data"],
    )
    write_text(
        target / "APPLY.txt",
        "\n".join(
            [
                "COO annotations are Japanese onomatopoeia polygons on Manga109 pages.",
                "Images are not redistributed. After Manga109 access:",
                "  huggingface-cli download hal-utokyo/Manga109 --repo-type dataset --local-dir data/manga109",
                "Then copy or symlink data/manga109/images into data/coo/repo/COO-data/images.",
                "",
            ]
        ),
    )
    return target


def try_manga109() -> None:
    write_text(
        DATA_ROOT / "manga109" / "APPLY.txt",
        "\n".join(
            [
                "Manga109 is gated. Apply here, then download with your Hugging Face account:",
                "https://huggingface.co/datasets/hal-utokyo/Manga109",
                "https://huggingface.co/datasets/hal-utokyo/Manga109-s",
                "",
                "After access is granted:",
                "huggingface-cli login",
                "huggingface-cli download hal-utokyo/Manga109 --repo-type dataset --local-dir data/manga109",
                "",
                "Do not commit the images. Redistribution is forbidden.",
                "",
            ]
        ),
    )


def try_ebdtheque() -> None:
    write_text(
        DATA_ROOT / "ebdtheque" / "APPLY.txt",
        "Register at https://ebdtheque.univ-lr.fr/registration and extract the archive into data/ebdtheque.\n",
    )


def collect_status(results: dict[str, str]) -> Path:
    lines = ["# Dataset collection status", ""]
    for name, value in results.items():
        lines.append(f"- `{name}`: {value}")
    lines.append("")
    return write_text(DATA_ROOT / "STATUS.md", "\n".join(lines))


def run_step(name: str, callback, results: dict[str, str]) -> None:
    log(name)
    try:
        value = callback()
        results[name] = str(value)
        log(f"  {value}")
    except (HTTPError, URLError, RuntimeError, subprocess.CalledProcessError, OSError) as error:
        results[name] = f"failed: {error}"
        log(f"  failed: {error}")


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    write_catalog()
    results: dict[str, str] = {}
    run_step("synthetic-ja-en", generate_synthetic, results)
    run_step("dcm772-annotations", fetch_dcm_annotations, results)

    log("Fetching public-domain English comics from Internet Archive")
    ia_ok: list[str] = []
    for identifier in IA_ITEMS:
        try:
            path = fetch_internet_archive_item(identifier)
            log(f"  {identifier} -> {path}")
            if path is not None:
                ia_ok.append(identifier)
        except Exception as error:  # noqa: BLE001
            log(f"  {identifier} failed: {error}")
    results["internet-archive-pd"] = ", ".join(ia_ok) if ia_ok else "none"

    run_step("comics-cvpr2017", fetch_comics_annotations, results)
    run_step("comix-tiny", fetch_comix_tiny, results)
    run_step("coo-annotations", fetch_coo_annotations, results)
    try_manga109()
    try_ebdtheque()
    results["manga109"] = "apply at Hugging Face; see data/manga109/APPLY.txt"
    results["ebdtheque"] = "register; see data/ebdtheque/APPLY.txt"
    collect_status(results)
    log(f"Catalog written to {DATA_ROOT / 'catalog.json'}")
    log(f"Status written to {DATA_ROOT / 'STATUS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

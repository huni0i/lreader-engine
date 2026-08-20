import importlib.util
from pathlib import Path


def load_prepare_datasets():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_datasets.py"
    spec = importlib.util.spec_from_file_location("prepare_datasets", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_choose_ia_file_skips_derived_and_oversized_archives() -> None:
    choose_ia_file = load_prepare_datasets().choose_ia_file
    chosen = choose_ia_file(
        [
            {"name": "book_jp2.zip", "size": "1000"},
            {"name": "book_text.pdf", "size": "2000"},
            {"name": "book.pdf", "size": "3000"},
            {"name": "huge.pdf", "size": str(80 * 1024 * 1024)},
        ]
    )

    assert chosen is not None
    assert chosen["name"] == "book.pdf"

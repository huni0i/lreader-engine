from pathlib import Path

from lreader_engine.synthetic_comics import generate_synthetic_split


def test_generate_synthetic_split_writes_japanese_and_english_pages(tmp_path: Path) -> None:
    manifest_path = generate_synthetic_split(tmp_path, pages_per_language=2)
    manifest = manifest_path.read_text(encoding="utf-8")

    assert (tmp_path / "ja_000.png").exists()
    assert (tmp_path / "en_000.png").exists()
    assert '"language": "ja"' in manifest
    assert '"language": "en"' in manifest
    assert "CC0-1.0" in manifest

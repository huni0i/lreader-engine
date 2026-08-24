from lreader_engine.ocr_cascade import select_primary_ocr, should_fallback_to_spotting


def test_route_uses_easyocr_on_white_bubbles() -> None:
    assert select_primary_ocr("route", "ja", "ocr", True) == "easy"


def test_route_uses_yolo_on_japanese_manga_without_bubbles() -> None:
    assert select_primary_ocr("route", "ja", "ocr", False) == "yolo"


def test_route_keeps_easyocr_for_english() -> None:
    assert select_primary_ocr("route", "en", "ocr", False) == "easy"


def test_explicit_modes_are_not_overridden() -> None:
    assert select_primary_ocr("spot", "ja", "ocr", True) == "spot"
    assert select_primary_ocr("yolo", "ja", "ocr", True) == "yolo"
    assert select_primary_ocr("easy", "ja", "ocr", False) == "easy"


def test_spotting_fallback_only_after_route_finds_no_text() -> None:
    assert should_fallback_to_spotting(
        "route", "ja", "ocr", used_spotting=False, has_source_text=False
    )
    assert not should_fallback_to_spotting(
        "route", "ja", "ocr", used_spotting=False, has_source_text=True
    )
    assert not should_fallback_to_spotting(
        "yolo", "ja", "ocr", used_spotting=False, has_source_text=False
    )

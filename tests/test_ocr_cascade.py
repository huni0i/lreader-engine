from lreader_engine.ocr_cascade import (
    select_primary_ocr,
    should_fallback_to_spotting,
    should_rerecognize_region,
)


def test_route_uses_yolo_on_japanese_manga() -> None:
    assert select_primary_ocr("route", "ja", "ocr") == "yolo"


def test_route_uses_yolo_for_balanced_quality_too() -> None:
    assert select_primary_ocr("route", "ja", "balanced") == "yolo"


def test_route_keeps_easyocr_for_english() -> None:
    assert select_primary_ocr("route", "en", "ocr") == "easy"


def test_route_keeps_easyocr_for_fast_quality() -> None:
    assert select_primary_ocr("route", "ja", "fast") == "easy"


def test_explicit_modes_are_not_overridden() -> None:
    assert select_primary_ocr("spot", "ja", "ocr") == "spot"
    assert select_primary_ocr("yolo", "ja", "ocr") == "yolo"
    assert select_primary_ocr("easy", "ja", "ocr") == "easy"


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


def test_korean_easyocr_is_not_rewritten_by_vl_ocr() -> None:
    assert not should_rerecognize_region("ko", "ocr", 0.6)
    assert not should_rerecognize_region("ko", "fast", 0.4)
    assert not should_rerecognize_region("en", "balanced", 0.5)


def test_japanese_boxes_still_use_manga_ocr() -> None:
    assert should_rerecognize_region("ja", "ocr", 0.9)
    assert should_rerecognize_region("ja", "balanced", 0.3)
    assert not should_rerecognize_region("ja", "fast", 0.9)

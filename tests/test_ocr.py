import pytest

from lreader_engine.ocr import OcrEngine


def test_parse_spotting_output_builds_scaled_polygons() -> None:
    output = (
        "ここは…"
        "<|LOC_250|><|LOC_100|><|LOC_350|><|LOC_100|>"
        "<|LOC_350|><|LOC_500|><|LOC_250|><|LOC_500|>"
        "\nどこだ？"
        "<|LOC_400|><|LOC_200|><|LOC_500|><|LOC_200|>"
        "<|LOC_500|><|LOC_600|><|LOC_400|><|LOC_600|>"
        "<|im_end|>"
    )

    regions = OcrEngine.parse_spotting_output(output, width=720, height=1000)

    assert [region.text for region in regions] == ["ここは…", "どこだ？"]
    assert regions[0].polygon[0].x == pytest.approx(180)
    assert regions[0].polygon[0].y == pytest.approx(100)
    assert regions[1].polygon[2].x == pytest.approx(360)
    assert regions[1].polygon[2].y == pytest.approx(600)


def test_parse_spotting_output_ignores_incomplete_coordinates() -> None:
    output = "未完成<|LOC_100|><|LOC_200|>"

    assert OcrEngine.parse_spotting_output(output, 720, 1000) == []

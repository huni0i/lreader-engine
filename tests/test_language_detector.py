import pytest

from lreader_engine.language_detector import detect_text_language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("오늘은 어디로 갈까?", "ko"),
        ("今日はどこへ行く？", "ja"),
        ("今天去哪里？", "zh"),
        ("Where should we go today?", "en"),
        ("1234?!", None),
    ],
)
def test_detect_text_language(text: str, expected: str | None) -> None:
    assert detect_text_language(text) == expected

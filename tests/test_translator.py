from lreader_engine.translator import (
    TranslationEngine,
    clean_translation,
    contains_source_text,
    parse_numbered_translations,
)


def test_clean_translation_removes_generated_prompt_continuation() -> None:
    result = clean_translation(
        "So, what was decided?\nKorean: 그럼, 어떻게 했는데?\nEnglish:"
    )

    assert result == "So, what was decided?"


def test_clean_translation_removes_leading_language_label() -> None:
    assert clean_translation("English: Why am I going?") == "Why am I going?"


def test_translation_model_can_be_selected_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("LREADER_TRANSLATION_MODEL", "tencent/Hy-MT2-7B")

    assert TranslationEngine().model_id == "tencent/Hy-MT2-7B"


def test_parse_numbered_translations_preserves_item_order() -> None:
    result = "[1] 여기는\n[0] 어디야?"

    assert parse_numbered_translations(result, 2) == ["어디야?", "여기는"]


def test_parse_numbered_translations_rejects_missing_items() -> None:
    assert parse_numbered_translations("[0] 어디야?", 2) is None


def test_parse_numbered_translations_accepts_common_model_formats() -> None:
    assert parse_numbered_translations("0. 어디야?\n1. 여기는", 2) == [
        "어디야?",
        "여기는",
    ]
    assert parse_numbered_translations('["어디야?", "여기는"]', 2) == [
        "어디야?",
        "여기는",
    ]
    assert parse_numbered_translations("어디야?\n여기는", 2) == [
        "어디야?",
        "여기는",
    ]


def test_contains_source_text_rejects_numeric_false_positive() -> None:
    assert contains_source_text("ここは", "ja")
    assert not contains_source_text("379", "ja")

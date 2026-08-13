from lreader_engine.translator import clean_translation


def test_clean_translation_removes_generated_prompt_continuation() -> None:
    result = clean_translation(
        "So, what was decided?\nKorean: 그럼, 어떻게 했는데?\nEnglish:"
    )

    assert result == "So, what was decided?"


def test_clean_translation_removes_leading_language_label() -> None:
    assert clean_translation("English: Why am I going?") == "Why am I going?"

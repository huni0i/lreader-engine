from collections import Counter

from lreader_engine.models import TargetLanguage


def detect_text_language(text: str) -> TargetLanguage | None:
    counts: Counter[str] = Counter()

    for character in text:
        codepoint = ord(character)
        if (
            0x1100 <= codepoint <= 0x11FF
            or 0x3130 <= codepoint <= 0x318F
            or 0xAC00 <= codepoint <= 0xD7A3
        ):
            counts["ko"] += 1
        elif 0x3040 <= codepoint <= 0x30FF:
            counts["ja"] += 1
        elif 0x3400 <= codepoint <= 0x9FFF:
            counts["zh"] += 1
        elif character.isascii() and character.isalpha():
            counts["en"] += 1

    if counts["ko"]:
        return "ko"
    if counts["ja"]:
        return "ja"
    if counts["zh"]:
        return "zh"
    if counts["en"]:
        return "en"
    return None

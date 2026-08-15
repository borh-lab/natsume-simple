from collections import Counter
from collections.abc import Iterable


def is_japanese(line: str, min_length: int = 200) -> bool:
    """Filter a single line to determine if it is likely Japanese text.

    Args:
        line: The line of text to filter.
        min_length: Minimum length of the line to keep (default: 200).

    Returns:
        True if Japanese characters make up at least 50% of the text.

    Examples:
        >>> is_japanese("これは日本語の文章です。", min_length=5)
        True
        >>> is_japanese("abc", min_length=5)
        False
        >>> is_japanese("This is English text", min_length=5)
        False
        >>> is_japanese("123.456.789", min_length=5)
        False
        >>> is_japanese("日本語とEnglishの混ざった文", min_length=5)  # Mixed but mostly Japanese
        True
        >>> is_japanese("This is mostly English with some 日本語", min_length=5)  # Mixed but mostly English
        False
        >>> is_japanese("テスト", min_length=2)  # Katakana
        True
        >>> is_japanese("ひらがな", min_length=2)  # Hiragana
        True
        >>> is_japanese("漢字", min_length=2)  # Kanji
        True
        >>> is_japanese("！？＆", min_length=2)  # Japanese punctuation
        True
        >>> is_japanese("Ｈｅｌｌｏ", min_length=2)  # Fullwidth romaji
        True
    """
    line = line.strip()

    if not line:
        return False

    def is_japanese_char(c: str) -> bool:
        code = ord(c)
        return (
            0x3040 <= code <= 0x309F  # Hiragana
            or 0x30A0 <= code <= 0x30FF  # Katakana
            or 0x4E00 <= code <= 0x9FFF  # Kanji
            or 0xFF00 <= code <= 0xFF5E  # Fullwidth ASCII variants
            or 0x3000 <= code <= 0x303F  # Japanese punctuation and symbols
            or 0x31F0 <= code <= 0x31FF  # Additional CJK symbols and punctuation
            or 0x3400 <= code <= 0x4DBF  # Additional Kanji
        )

    if len(line) < min_length:
        # For short strings, require 100% Japanese characters
        return all(is_japanese_char(c) for c in line)

    # For longer strings, require at least 50% Japanese characters
    japanese_char_count = sum(1 for c in line if is_japanese_char(c))
    return (japanese_char_count / len(line)) >= 0.5


def split_japanese_sentences(
    text_units: tuple[str, ...],
    *,
    splitter: object,
    observations: Counter[str] | None = None,
) -> Iterable[str]:
    """Split paragraphs and retain the public Japanese-content policy."""
    paragraphs = [
        paragraph.strip()
        for text in text_units
        for paragraph in text.splitlines()
        if paragraph.strip()
    ]
    for group in splitter.split(paragraphs):  # type: ignore[attr-defined]
        for sentence in group:
            candidate = sentence.strip()
            if not candidate:
                continue
            if observations is not None:
                observations["candidate"] += 1
            if is_japanese(candidate, min_length=5):
                if observations is not None:
                    observations["retained"] += 1
                yield candidate
            elif observations is not None:
                observations["dropped"] += 1

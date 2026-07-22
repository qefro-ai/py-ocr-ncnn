"""Convert visual-order Arabic OCR output to logical Unicode order."""

from __future__ import annotations

import re


def _arabic_char_count(text: str) -> int:
    return sum(
        1
        for ch in text
        if (
            "\u0600" <= ch <= "\u06FF"
            or "\u0750" <= ch <= "\u077F"
            or "\uFB50" <= ch <= "\uFDFF"
            or "\uFE70" <= ch <= "\uFEFF"
        )
    )


def _is_arabic_token(token: str) -> bool:
    return bool(token) and all(
        (
            "\u0600" <= ch <= "\u06FF"
            or "\u0750" <= ch <= "\u077F"
            or "\uFB50" <= ch <= "\uFDFF"
            or "\uFE70" <= ch <= "\uFEFF"
        )
        for ch in token
    )


def _looks_like_visual_order_arabic(text: str) -> bool:
    """True when Arabic looks glyph/word-reversed (OCR LTR box order)."""
    ar = _arabic_char_count(text)
    if ar < 20:
        return False
    tokens = re.findall(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+", text)
    if len(tokens) < 3:
        return False
    logical_al = sum(1 for token in tokens if token.startswith("ال") or token.startswith("وال"))
    visual_al = sum(1 for token in tokens if token.endswith("لا") or token.endswith("لاو"))
    if visual_al >= 2 and visual_al > logical_al:
        return True
    reverses = {"ةكرش", "رونلا", "تاينقتلل", "تسسألت", "تسسأت", "ضايرلا", "يفظوم"}
    return sum(1 for token in tokens if token in reverses) >= 2


def _fix_arabic_glyphs_in_phrase(text: str) -> str:
    parts = re.findall(
        r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+|[^\s]",
        text,
    )
    if not parts:
        return text

    arabic_parts = [part for part in parts if _is_arabic_token(part)]
    if len(arabic_parts) >= 2:
        fixed = [part[::-1] if _is_arabic_token(part) else part for part in parts]
        fixed = list(reversed(fixed))
        rebuilt: list[str] = []
        for index, token in enumerate(fixed):
            if index > 0:
                prev = fixed[index - 1]
                if (
                    (_is_arabic_token(token) or token.isalnum())
                    and (_is_arabic_token(prev) or prev[-1].isalnum())
                    and prev not in ".,:;!?،؛؟-–—"
                    and token not in ".,:;!?،؛؟"
                ):
                    rebuilt.append(" ")
            rebuilt.append(token)
        return "".join(rebuilt)

    out: list[str] = []
    for index, part in enumerate(parts):
        token = part[::-1] if _is_arabic_token(part) else part
        if index > 0:
            prev = out[-1] if out else ""
            if (
                prev
                and token
                and (_is_arabic_token(prev) or prev[-1].isalnum())
                and (_is_arabic_token(token) or token[0].isalnum())
                and prev[-1] not in ".,:;!?،؛؟"
                and token[0] not in ".,:;!?،؛؟"
            ):
                out.append(" ")
        out.append(token)
    return "".join(out)


def _arabic_visual_segment_to_logical(segment: str) -> str:
    tokens = re.findall(
        r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+|[^\s]",
        segment,
    )
    if not tokens:
        return segment
    fixed = [token[::-1] if _is_arabic_token(token) else token for token in tokens]
    fixed = list(reversed(fixed))
    out: list[str] = []
    for index, token in enumerate(fixed):
        if index > 0:
            prev = fixed[index - 1]
            if (
                (_is_arabic_token(token) or token[0].isalnum())
                and (_is_arabic_token(prev) or prev[-1].isalnum())
                and prev not in ".,:;!?،؛؟-–—"
                and token not in ".,:;!?،؛؟"
            ):
                out.append(" ")
        out.append(token)
    return "".join(out)


def arabic_visual_line_to_logical(line: str) -> str:
    parts = re.findall(
        r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"
        r"(?:[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s،؛؟.:\-–—]*)"
        r"|[A-Za-z0-9@._+\-:/]+"
        r"|[^\s]",
        line,
    )
    if not parts:
        return line
    has_latin_or_digit = any(re.fullmatch(r"[A-Za-z0-9@._+\-:/]+", part) for part in parts)
    if not has_latin_or_digit:
        return _arabic_visual_segment_to_logical(line)

    out_parts = [
        _arabic_visual_segment_to_logical(part) if _arabic_char_count(part) >= 2 else part
        for part in parts
    ]
    out: list[str] = []
    for index, token in enumerate(out_parts):
        if index > 0:
            prev = out_parts[index - 1]
            if (
                prev
                and token
                and not prev[-1].isspace()
                and not token[0].isspace()
                and prev[-1] not in ".,:;!?،؛؟"
                and token[0] not in ".,:;!?،؛؟"
            ):
                out.append(" ")
        out.append(token)
    return "".join(out)


def fix_arabic_visual_order(text: str) -> str:
    """Convert visual-order Arabic OCR output to logical Unicode order."""
    if not _looks_like_visual_order_arabic(text):
        return text

    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    short_ar_lines = [
        line for line in nonempty if _arabic_char_count(line) >= 2 and len(line.split()) <= 3
    ]
    if len(nonempty) >= 5 and len(short_ar_lines) >= max(4, int(0.6 * len(nonempty))):
        phrases: list[str] = []
        for line in reversed(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if _arabic_char_count(stripped) >= 2:
                phrases.append(_fix_arabic_glyphs_in_phrase(stripped))
            else:
                phrases.append(stripped)
        result = " ".join(phrases)
        result = re.sub(r"\s+([.،؛؟!?])", r"\1", result)
        return re.sub(r"\s{2,}", " ", result).strip()

    fixed_lines = [
        arabic_visual_line_to_logical(line) if _arabic_char_count(line) >= 2 else line
        for line in lines
    ]
    return "\n".join(fixed_lines)

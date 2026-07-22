from __future__ import annotations

import pytest

from ncnn_ocr import __version__, supported_langs
from ncnn_ocr.lang import V5_REC_BY_LANG, normalize_lang, resolve_rec_spec
from ncnn_ocr.types import LineResult, Point


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_supported_langs_includes_tamil() -> None:
    langs = supported_langs("v5")
    assert "ta" in langs
    assert "ar" in langs


def test_normalize_lang() -> None:
    assert normalize_lang(" TA ") == "ta"
    assert normalize_lang(None) == "en"


def test_resolve_rec_spec_tamil() -> None:
    spec = resolve_rec_spec("ta")
    assert spec.rec_stem == "ta_PP-OCRv5_mobile_rec"
    assert spec.keys_file == "ppocr_keys_ta.txt"


def test_arabic_langs_share_model() -> None:
    ar = resolve_rec_spec("ar")
    fa = resolve_rec_spec("fa")
    assert ar == fa


def test_line_result_from_native() -> None:
    item = {
        "text": "hello",
        "confidence": 0.95,
        "box": [(0, 0), (10, 0), (10, 5), (0, 5)],
        "box_score": 0.9,
        "rotated": False,
        "angle_score": 0.8,
    }
    line = LineResult.from_native(item)
    assert line.text == "hello"
    assert line.confidence == pytest.approx(0.95)
    assert line.box[0] == Point(0, 0)


@pytest.mark.parametrize("lang", sorted(V5_REC_BY_LANG))
def test_all_langs_resolve(lang: str) -> None:
    spec = resolve_rec_spec(lang)
    assert spec.rec_stem
    assert spec.keys_file.endswith(".txt")

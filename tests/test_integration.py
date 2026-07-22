"""Integration tests requiring the native extension."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ncnn_ocr import NATIVE_AVAILABLE, OCR

pytestmark = pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native extension not built")


def test_recognize_synthetic_hello() -> None:
    image = np.ones((64, 200, 3), dtype=np.uint8) * 255
    cv2.putText(image, "Hello", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    ocr = OCR(lang="en")
    lines = ocr.recognize(image)

    assert lines
    assert any("Hello" in line.text for line in lines)
    assert ocr.recognize_text(image) == "Hello"

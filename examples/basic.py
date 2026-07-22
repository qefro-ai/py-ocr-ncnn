#!/usr/bin/env python3
"""Minimal ncnn-ocr usage example."""

from __future__ import annotations

import sys
from pathlib import Path

from ncnn_ocr import OCR


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python basic.py IMAGE [LANG]")
        return 1

    image_path = Path(sys.argv[1])
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"

    ocr = OCR(lang=lang)
    text = ocr.recognize_text(image_path)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

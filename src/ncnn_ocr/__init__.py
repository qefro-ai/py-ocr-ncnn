"""ncnn-ocr: high-performance PaddleOCR on Tencent NCNN."""

from __future__ import annotations

from .arabic import fix_arabic_visual_order
from .engine import OCR, NATIVE_AVAILABLE, recognize, recognize_text
from .lang import ProfileName, supported_langs
from .models import default_model_dir, hf_repo
from .types import LineResult, Point

__all__ = [
    "LineResult",
    "NATIVE_AVAILABLE",
    "OCR",
    "Point",
    "ProfileName",
    "default_model_dir",
    "fix_arabic_visual_order",
    "hf_repo",
    "recognize",
    "recognize_text",
    "supported_langs",
]

__version__ = "0.1.0"

"""High-level OCR engine."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .arabic import fix_arabic_visual_order
from .lang import ProfileName, normalize_lang, supported_langs
from .models import default_model_dir, write_runtime_config
from .types import LineResult

try:
    from ncnn_ocr._ncnn_ocr import Engine as _NativeEngine

    NATIVE_AVAILABLE = True
except ImportError:  # pragma: no cover - extension not built yet
    _NativeEngine = None  # type: ignore[assignment,misc]
    NATIVE_AVAILABLE = False

ImageInput = str | Path | NDArray[np.uint8]
_BGRImage = NDArray[np.uint8]

_engines: dict[tuple[str, str, str], _NativeEngine] = {}
_lock = threading.Lock()


class OCR:
    """PP-OCR inference on NCNN (det + cls + rec).

    Example::

        from ncnn_ocr import OCR

        ocr = OCR(lang="ta")
        text = ocr.recognize_text("page.png")
        lines = ocr.recognize(image_array)
    """

    def __init__(
        self,
        lang: str = "en",
        *,
        profile: ProfileName = "v5",
        model_dir: str | Path | None = None,
        fix_arabic: bool = True,
        det_overrides: dict[str, Any] | None = None,
    ) -> None:
        if not NATIVE_AVAILABLE:
            raise ImportError(
                "ncnn_ocr native extension is not installed. "
                "Build with: pip install . (requires OpenCV, ncnn, cmake)."
            )

        self.lang = normalize_lang(lang)
        self.profile: ProfileName = profile
        self.model_dir = Path(model_dir).expanduser() if model_dir else default_model_dir()
        self.fix_arabic = fix_arabic
        self.det_overrides = det_overrides
        self._engine = _get_engine(
            lang=self.lang,
            profile=self.profile,
            model_dir=self.model_dir,
            det_overrides=self.det_overrides,
        )

    @staticmethod
    def supported_langs(profile: ProfileName = "v5") -> list[str]:
        return supported_langs(profile)

    @staticmethod
    def native_available() -> bool:
        return NATIVE_AVAILABLE

    def recognize(self, image: ImageInput) -> list[LineResult]:
        """Detect and recognize text; returns one entry per line."""
        bgr = _load_bgr(image)
        raw = self._engine.recognize(bgr)
        return [LineResult.from_native(item) for item in raw]

    def recognize_text(self, image: ImageInput, *, join: str = "\n") -> str:
        """Return recognized text joined by ``join``."""
        lines = self.recognize(image)
        text = join.join(line.text for line in lines if line.text.strip())
        if self.fix_arabic and self.lang in {"ar", "fa", "ur"}:
            text = fix_arabic_visual_order(text)
        return text

    def batch(self, images: Sequence[ImageInput]) -> list[list[LineResult]]:
        return [self.recognize(image) for image in images]

    def batch_text(
        self,
        images: Sequence[ImageInput],
        *,
        join: str = "\n",
    ) -> list[str]:
        return [self.recognize_text(image, join=join) for image in images]


def _get_engine(
    *,
    lang: str,
    profile: ProfileName,
    model_dir: Path,
    det_overrides: dict[str, Any] | None,
) -> _NativeEngine:
    key = (lang, profile, str(model_dir.resolve()))
    with _lock:
        cached = _engines.get(key)
        if cached is not None:
            return cached

        config_path = write_runtime_config(
            model_dir=model_dir,
            profile=profile,
            lang=lang,
            det_overrides=det_overrides,
        )
        engine = _NativeEngine(str(config_path))
        _engines[key] = engine
        return engine


def _load_bgr(image: ImageInput) -> _BGRImage:
    if isinstance(image, (str, Path)):
        path = str(image)
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return bgr

    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 3:
        return np.ascontiguousarray(array, dtype=np.uint8)
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    raise ValueError("image must be a path or HxW / HxWx3 / HxWx4 numpy array")


def recognize(
    image: ImageInput,
    lang: str = "en",
    *,
    profile: ProfileName = "v5",
    **kwargs: Any,
) -> list[LineResult]:
    """One-shot recognize using a cached engine."""
    return OCR(lang=lang, profile=profile, **kwargs).recognize(image)


def recognize_text(image: ImageInput, lang: str = "en", **kwargs: Any) -> str:
    """One-shot text extraction using a cached engine."""
    return OCR(lang=lang, **kwargs).recognize_text(image)

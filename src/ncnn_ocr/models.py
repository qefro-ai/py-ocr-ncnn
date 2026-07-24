"""Download NCNN models and write runtime JSON configs."""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from .lang import ProfileName, resolve_det_cls_stems, resolve_rec_spec

DEFAULT_HF_REPO = "developerabu/pp-OCRv5-6-ncnn"
ARCHIVE_URL = "https://github.com/Avafly/PaddleOCR-ncnn-CPP/releases/download/v0.3.0/archive.tar.gz"

# Shared v5 det/cls may only exist in the upstream archive (not always on HF).
ARCHIVE_SHARED_STEMS = {
    "PP_OCRv5_mobile_det",
    "PP_LCNet_x0_25_textline_ori",
}


def default_model_dir() -> Path:
    override = (os.getenv("NCNN_OCR_MODEL_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "ncnn-ocr" / "models"


def hf_repo() -> str:
    return (os.getenv("NCNN_OCR_HF_REPO") or DEFAULT_HF_REPO).strip()


def _ensure_hf_file(filename: str, model_dir: Path) -> Path:
    dest = model_dir / filename
    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=hf_repo(),
        filename=filename,
        local_dir=str(model_dir),
    )
    return Path(path)


def _download_archive_shared(stem: str, model_dir: Path) -> None:
    param = model_dir / f"{stem}.param"
    bin_path = model_dir / f"{stem}.bin"
    if param.is_file() and bin_path.is_file():
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ncnn-ocr-archive-") as tmp:
        archive_path = Path(tmp) / "archive.tar.gz"
        urlretrieve(ARCHIVE_URL, archive_path)  # noqa: S310
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                name = Path(member.name).name
                if name in {f"{stem}.param", f"{stem}.bin"}:
                    member.name = name
                    tar.extract(member, path=model_dir)


def ensure_model_stem(stem: str, model_dir: Path) -> Path:
    """Ensure `{stem}.param` and `{stem}.bin` exist; return stem path prefix."""
    model_dir.mkdir(parents=True, exist_ok=True)
    param = model_dir / f"{stem}.param"
    bin_path = model_dir / f"{stem}.bin"

    if param.is_file() and bin_path.is_file():
        return model_dir / stem

    missing_suffixes = [
        suffix
        for suffix in (".param", ".bin")
        if not (model_dir / f"{stem}{suffix}").is_file()
    ]
    if missing_suffixes:
        hf_failed = False
        for suffix in missing_suffixes:
            try:
                _ensure_hf_file(f"{stem}{suffix}", model_dir)
            except Exception:
                hf_failed = True
                break
        if hf_failed:
            if stem in ARCHIVE_SHARED_STEMS:
                _download_archive_shared(stem, model_dir)
            else:
                raise

    if not param.is_file() or not bin_path.is_file():
        raise FileNotFoundError(f"Missing NCNN model files for stem {stem!r} in {model_dir}")
    return model_dir / stem


def ensure_keys_file(keys_file: str, model_dir: Path) -> Path:
    return _ensure_hf_file(keys_file, model_dir)


def write_runtime_config(
    *,
    model_dir: Path,
    profile: ProfileName,
    lang: str,
    config_dir: Path | None = None,
    det_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write a JSON config understood by the native OCREngine."""
    rec_spec = resolve_rec_spec(lang, profile)
    det_stem, cls_stem = resolve_det_cls_stems(profile)

    det_path = ensure_model_stem(det_stem, model_dir)
    cls_path = ensure_model_stem(cls_stem, model_dir)
    rec_path = ensure_model_stem(rec_spec.rec_stem, model_dir)
    keys_path = ensure_keys_file(rec_spec.keys_file, model_dir)

    det_cfg: dict[str, Any] = {
        "infer_threads": -1,
        "model_path": str(det_path),
        "padding": 0,
        "max_side_len": 768,
        "box_thres": 0.45,
        "bitmap_thres": 0.2,
        "unclip_ratio": 1.4,
        "fp16": False,
    }
    if det_overrides:
        det_cfg.update(det_overrides)

    payload = {
        "save": False,
        "det": det_cfg,
        "cls": {
            "infer_threads": 1,
            "reco_threads": -1,
            "model_path": str(cls_path),
            "enable": True,
            "most_angle": False,
            "fp16": False,
        },
        "rec": {
            "infer_threads": 1,
            "reco_threads": -1,
            "model_path": str(rec_path),
            "keys_path": str(keys_path),
            "fp16": False,
        },
    }

    out_dir = config_dir or (model_dir / "configs")
    out_dir.mkdir(parents=True, exist_ok=True)
    lang_key = (lang or "en").strip().lower()
    out_path = out_dir / f"runtime_{profile}_{lang_key}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path

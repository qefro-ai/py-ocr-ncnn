"""Language and model profile resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProfileName = Literal["v5", "v6_tiny", "v6_small", "v6_medium"]


@dataclass(frozen=True, slots=True)
class RecSpec:
    """Recognition model stem (no .param/.bin) and keys filename on HuggingFace."""

    rec_stem: str
    keys_file: str


# Shared PP-OCRv5 mobile det/cls (Avafly archive naming).
V5_DET_STEM = "PP_OCRv5_mobile_det"
V5_CLS_STEM = "PP_LCNet_x0_25_textline_ori"

# lang code -> v5 rec model on developerabu/pp-OCRv5-6-ncnn
V5_REC_BY_LANG: dict[str, RecSpec] = {
    "ta": RecSpec("ta_PP-OCRv5_mobile_rec", "ppocr_keys_ta.txt"),
    "te": RecSpec("te_PP-OCRv5_mobile_rec", "ppocr_keys_te.txt"),
    "hi": RecSpec("devanagari_PP-OCRv5_mobile_rec", "ppocr_keys_devanagari.txt"),
    "mr": RecSpec("devanagari_PP-OCRv5_mobile_rec", "ppocr_keys_devanagari.txt"),
    "ne": RecSpec("devanagari_PP-OCRv5_mobile_rec", "ppocr_keys_devanagari.txt"),
    "ar": RecSpec("arabic_PP-OCRv5_mobile_rec", "ppocr_keys_arabic.txt"),
    "fa": RecSpec("arabic_PP-OCRv5_mobile_rec", "ppocr_keys_arabic.txt"),
    "ur": RecSpec("arabic_PP-OCRv5_mobile_rec", "ppocr_keys_arabic.txt"),
    "en": RecSpec("en_PP-OCRv5_mobile_rec", "ppocr_keys_en.txt"),
    "latin": RecSpec("latin_PP-OCRv5_mobile_rec", "ppocr_keys_latin.txt"),
    "de": RecSpec("latin_PP-OCRv5_mobile_rec", "ppocr_keys_latin.txt"),
    "fr": RecSpec("latin_PP-OCRv5_mobile_rec", "ppocr_keys_latin.txt"),
    "es": RecSpec("latin_PP-OCRv5_mobile_rec", "ppocr_keys_latin.txt"),
    "ko": RecSpec("korean_PP-OCRv5_mobile_rec", "ppocr_keys_korean.txt"),
    "th": RecSpec("th_PP-OCRv5_mobile_rec", "ppocr_keys_th.txt"),
    "el": RecSpec("el_PP-OCRv5_mobile_rec", "ppocr_keys_el.txt"),
    "ru": RecSpec("eslav_PP-OCRv5_mobile_rec", "ppocr_keys_eslav.txt"),
    "uk": RecSpec("eslav_PP-OCRv5_mobile_rec", "ppocr_keys_eslav.txt"),
    "bg": RecSpec("cyrillic_PP-OCRv5_mobile_rec", "ppocr_keys_cyrillic.txt"),
    "ch": RecSpec("PP_OCRv5_mobile_rec", "ppocr_keys_v5.txt"),
    "zh": RecSpec("PP_OCRv5_mobile_rec", "ppocr_keys_v5.txt"),
}

V6_PROFILES: dict[str, tuple[str, str, str]] = {
    # profile -> (det_stem, rec_stem, keys_file)
    "v6_tiny": ("PP_OCRv6_tiny_det", "PP_OCRv6_tiny_rec", "ppocr_keys_v6_tiny.txt"),
    "v6_small": ("PP_OCRv6_small_det", "PP_OCRv6_small_rec", "ppocr_keys_v6.txt"),
    "v6_medium": ("PP_OCRv6_medium_det", "PP_OCRv6_medium_rec", "ppocr_keys_v6.txt"),
}


def normalize_lang(lang: str | None) -> str:
    return (lang or "en").strip().lower()


def supported_langs(profile: ProfileName = "v5") -> list[str]:
    if profile == "v5":
        return sorted(V5_REC_BY_LANG)
    return []


def resolve_rec_spec(lang: str, profile: ProfileName = "v5") -> RecSpec:
    key = normalize_lang(lang)
    if profile != "v5":
        _, rec_stem, keys_file = V6_PROFILES[profile]
        return RecSpec(rec_stem, keys_file)
    spec = V5_REC_BY_LANG.get(key)
    if spec is None:
        supported = ", ".join(sorted(V5_REC_BY_LANG))
        raise ValueError(f"Unsupported language {lang!r} for profile {profile!r}. Supported: {supported}")
    return spec


def resolve_det_cls_stems(profile: ProfileName) -> tuple[str, str]:
    if profile == "v5":
        return V5_DET_STEM, V5_CLS_STEM
    det_stem, _, _ = V6_PROFILES[profile]
    return det_stem, V5_CLS_STEM

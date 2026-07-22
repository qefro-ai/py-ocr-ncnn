from ncnn_ocr.arabic import _looks_like_visual_order_arabic, fix_arabic_visual_order


def test_arabic_logical_passthrough() -> None:
    logical = "كل هذه الأفكار المغلوطة حول استنكار النشوة وتمجيد الألم نشأت بالفعل"
    assert not _looks_like_visual_order_arabic(logical)
    assert fix_arabic_visual_order(logical) == logical


def test_arabic_mnn_line_dump_to_logical() -> None:
    raw = "\n".join(
        [
            "راكفألا",
            "هذه",
            "لك",
            "نأ",
            "كل",
            "حضوأ",
            "نأ",
            "دب ال",
            "نكل",
            "تأشن",
            "ملألا",
            "ديجمتو",
            "ةوشنلا",
            "راكنتسا",
            "لوح",
            "ةطولغملا",
            "فشتكتل",
            "ليصافتلا",
            "كل",
            "ض",
            "رعأسو",
            "لعفلاب",
        ]
    )
    assert _looks_like_visual_order_arabic(raw)
    fixed = fix_arabic_visual_order(raw)
    assert "كل هذه الأفكار" in fixed or "كل هذه الافكار" in fixed
    assert "استنكار" in fixed
    assert "راكفألا" not in fixed

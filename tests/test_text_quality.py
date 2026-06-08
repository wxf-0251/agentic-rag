from agentic_rag.agents import normalize_report_text


def test_normalize_report_text_fixes_common_chinese_wording():
    assert normalize_report_text("差异显着，影响显着。") == "差异显著，影响显著。"


def test_normalize_report_text_removes_common_mojibake_marker():
    assert "鈥" not in normalize_report_text("收入鈥?利润")


def test_normalize_report_text_repairs_gbk_mojibake():
    assert normalize_report_text("鏄庨槼鏅鸿兘") == "明阳智能"


def test_normalize_report_text_fixes_obvious_bad_phrases_from_reports():
    text = "技术自主化的氧化碳，同时增强了供应链负担。"
    normalized = normalize_report_text(text)
    assert "氧化碳" not in normalized
    assert "供应链负担" not in normalized
    assert "技术自主化的推进" in normalized
    assert "供应链韧性" in normalized

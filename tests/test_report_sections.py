from agentic_rag.ui.report_sections import extract_completed_sections


def test_extract_completed_sections_keeps_all_writer_sections():
    sections = [
        {"title": "第一章", "content": "one"},
        {"title": "第二章", "content": "two"},
    ]

    assert extract_completed_sections({"completed_sections": sections}) == sections


def test_extract_completed_sections_handles_missing_value():
    assert extract_completed_sections({}) == []


def test_extract_completed_sections_sorts_by_section_index():
    sections = [
        {"section_index": 2, "title": "第三章"},
        {"section_index": 0, "title": "第一章"},
        {"section_index": 1, "title": "第二章"},
    ]

    result = extract_completed_sections({"completed_sections": sections})

    assert [section["title"] for section in result] == ["第一章", "第二章", "第三章"]


def test_extract_completed_sections_keeps_latest_duplicate_section():
    sections = [
        {"section_index": 0, "title": "第一章旧稿"},
        {"section_index": 1, "title": "第二章"},
        {"section_index": 0, "title": "第一章新稿"},
    ]

    result = extract_completed_sections({"completed_sections": sections})

    assert [section["title"] for section in result] == ["第一章新稿", "第二章"]

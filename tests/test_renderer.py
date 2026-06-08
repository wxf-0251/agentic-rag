import agentic_rag.ui.renderer as renderer
from agentic_rag.ui.renderer import (
    build_report_html,
    build_report_markdown,
    build_report_pdf,
    dedupe_citations,
    normalize_citation_markers,
    parse_chart_tags,
    render_chart,
    should_render_chart,
    validate_inline_citations,
)


def test_parse_chart_tags_extracts_bar_chart():
    markdown = """Some text.
<chart type="bar" title="Market Share">
  x: ["A","B","C"]
  y: [30,50,20]
</chart>
More text."""
    charts = parse_chart_tags(markdown)
    assert len(charts) == 1
    assert charts[0]["type"] == "bar"
    assert charts[0]["title"] == "Market Share"
    assert charts[0]["x"] == ["A", "B", "C"]
    assert charts[0]["y"] == [30, 50, 20]


def test_parse_chart_tags_multiple():
    markdown = """Text.
<chart type="bar" title="Chart 1">
  x: ["X"]
  y: [1]
</chart>
<chart type="line" title="Chart 2">
  x: ["A","B"]
  y: [10,20]
</chart>"""
    charts = parse_chart_tags(markdown)
    assert len(charts) == 2
    assert charts[0]["type"] == "bar"
    assert charts[1]["type"] == "line"


def test_parse_chart_tags_no_charts():
    markdown = "Just plain text without any chart tags."
    charts = parse_chart_tags(markdown)
    assert charts == []


def test_render_chart_returns_figure():
    chart_data = {"type": "bar", "title": "Test", "x": ["A", "B"], "y": [10, 20]}
    fig = render_chart(chart_data)
    assert fig is not None


def test_render_chart_supports_horizontal_bar_and_scatter():
    assert render_chart({"type": "horizontal_bar", "title": "H", "x": ["Long label"], "y": [10]}) is not None
    assert render_chart({"type": "scatter", "title": "S", "x": [1, 2], "y": [3, 4]}) is not None


def test_should_render_chart_rejects_placeholder_score_chart():
    chart_data = {"type": "bar", "title": "主观评分", "x": ["政策支持", "监管环境", "竞争格局"], "y": [5, 4, 3]}
    assert should_render_chart(chart_data) is False


def test_should_render_chart_accepts_real_numeric_chart():
    chart_data = {"type": "bar", "title": "收入对比", "x": ["公司A", "公司B"], "y": [120.5, 230.0]}
    assert should_render_chart(chart_data) is True


def test_validate_inline_citations_flags_missing_references():
    content = "Revenue grew quickly [1], but the supplier share claim uses a missing citation [3]."
    citations = [
        {"id": "c1", "title": "Source 1", "url": "https://example.com/1", "snippet": "Revenue"},
        {"id": "c2", "title": "Source 2", "url": "https://example.com/2", "snippet": "Margin"},
    ]

    result = validate_inline_citations(content, citations)

    assert result["valid"] is False
    assert result["missing_numbers"] == [3]


def test_build_report_markdown_includes_appendix_sources():
    sections = [
        {"title": "Overview", "content": "The market is growing [1]."},
        {"title": "Risks", "content": "Supply remains concentrated [2]."},
    ]
    citations = [
        {"title": "Source 1", "url": "https://example.com/1", "snippet": "Market"},
        {"title": "Source 2", "url": "https://example.com/2", "snippet": "Supply"},
    ]

    markdown = build_report_markdown("Report Title", sections, citations)

    assert markdown.startswith("# Report Title")
    assert "## Overview" in markdown
    assert "## References" in markdown
    assert "1. [Source 1](https://example.com/1)" in markdown


def test_citations_for_prompt_includes_quality_metadata():
    citations = [
        {
            "id": "c1",
            "title": "Official Filing",
            "url": "https://example.com/1",
            "snippet": "Revenue 50.",
            "published_date": "2026-05-28",
            "freshness_status": "fresh",
            "authority_score": 95,
        }
    ]

    prompt_citations = renderer.citations_for_prompt(citations)

    assert prompt_citations[0]["freshness_status"] == "fresh"
    assert prompt_citations[0]["authority_score"] == 95
    assert prompt_citations[0]["published_date"] == "2026-05-28"


def test_build_report_pdf_returns_pdf_bytes():
    sections = [{"title": "Overview", "content": "The market is growing [1]."}]
    citations = [{"title": "Source 1", "url": "https://example.com/1", "snippet": "Market"}]

    pdf = build_report_pdf("Report Title", sections, citations)

    assert pdf.startswith(b"%PDF")


def test_build_report_pdf_prefers_edge_renderer(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("fallback renderer should not be called")

    monkeypatch.setattr(renderer, "_build_report_pdf_with_edge", lambda *args: b"%PDF-edge")
    monkeypatch.setattr(renderer, "_build_report_pdf_with_reportlab", fail_if_called)
    monkeypatch.setattr(renderer, "_build_report_pdf_with_matplotlib", fail_if_called)

    pdf = renderer.build_report_pdf("Report Title", [{"title": "Overview", "content": "Text"}], [])

    assert pdf == b"%PDF-edge"


def test_build_report_html_embeds_chart_images():
    sections = [
        {
            "title": "Overview",
            "content": """The market grew [1].
<chart type="bar" title="Growth">
  x: ["2024","2025"]
  y: [10,20]
</chart>""",
        }
    ]
    citations = [{"title": "Source 1", "url": "https://example.com/1", "snippet": "Market"}]

    html = build_report_html("Report Title", sections, citations)

    assert "data:image/png;base64," in html


def test_build_report_html_converts_markdown_headings():
    sections = [
        {
            "title": "Overview",
            "content": "## Overview\n### Market Size\nRevenue reached 50 billion yuan [1].",
        }
    ]
    citations = [{"title": "Source 1", "url": "https://example.com/1", "snippet": "50 billion yuan"}]

    html = build_report_html("Report Title", sections, citations)

    assert "<h2>Overview</h2>" in html
    assert "<h3>Market Size</h3>" in html
    assert "## Overview" not in html
    assert "### Market Size" not in html


def test_build_report_pdf_with_chart_returns_pdf_bytes():
    sections = [
        {
            "title": "Overview",
            "content": """The market grew [1].
<chart type="line" title="Growth">
  x: ["2024","2025"]
  y: [10,20]
</chart>""",
        }
    ]
    citations = [{"title": "Source 1", "url": "https://example.com/1", "snippet": "Market"}]

    pdf = build_report_pdf("Report Title", sections, citations)

    assert pdf.startswith(b"%PDF")


def test_build_report_pdf_falls_back_without_weasyprint_or_reportlab(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("weasyprint") or name.startswith("reportlab"):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(renderer, "_build_report_pdf_with_edge", lambda *args: (_ for _ in ()).throw(RuntimeError("no browser")))

    sections = [
        {
            "title": "Overview",
            "content": """The market grew [1].
<chart type="bar" title="Growth">
  x: ["2024","2025"]
  y: [10,20]
</chart>""",
        }
    ]
    citations = [{"title": "Source 1", "url": "https://example.com/1", "snippet": "Market"}]

    pdf = build_report_pdf("Report Title", sections, citations)

    assert pdf.startswith(b"%PDF")


def test_dedupe_citations_removes_repeated_sources():
    citations = [
        {"id": "c1", "title": "Same", "url": "https://example.com/a", "snippet": "one"},
        {"id": "c2", "title": "Same again", "url": "https://example.com/a", "snippet": "duplicate"},
        {"id": "c3", "title": "Other", "url": "https://example.com/b", "snippet": "two"},
    ]

    deduped = dedupe_citations(citations)

    assert len(deduped) == 2
    assert deduped[0]["number"] == 1
    assert deduped[1]["number"] == 2


def test_normalize_citation_markers_converts_c_ids_to_numbers():
    citations = [
        {"id": "c1", "title": "A", "url": "https://a.com", "snippet": ""},
        {"id": "c2", "title": "B", "url": "https://b.com", "snippet": ""},
    ]

    content = normalize_citation_markers("观点来自 [c1] 和【c2】。", citations)

    assert "[c1]" not in content
    assert "【c2】" not in content
    assert "[1]" in content
    assert "[2]" in content


def test_normalize_citation_markers_maps_duplicate_source_ids_to_first_number():
    citations = [
        {"id": "c1", "title": "A", "url": "https://a.com", "snippet": ""},
        {"id": "c2", "title": "A duplicate", "url": "https://a.com", "snippet": ""},
        {"id": "c3", "title": "B", "url": "https://b.com", "snippet": ""},
    ]

    content = normalize_citation_markers("重复来源 [c2] 和新来源 [c3]。", citations)

    assert "[c2]" not in content
    assert "[1]" in content
    assert "[2]" in content

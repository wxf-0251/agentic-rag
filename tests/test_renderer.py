from agentic_rag.ui.renderer import parse_chart_tags, render_chart


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

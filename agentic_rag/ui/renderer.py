import re
import json
from typing import List
import matplotlib.pyplot as plt


def parse_chart_tags(markdown: str) -> List[dict]:
    pattern = r'<chart type="(.*?)" title="(.*?)">\s*x:\s*(\[.*?\])\s*y:\s*(\[.*?\])\s*</chart>'
    matches = re.findall(pattern, markdown, re.DOTALL)
    charts = []
    for chart_type, title, x_str, y_str in matches:
        try:
            x = json.loads(x_str)
            y = json.loads(y_str)
            charts.append({"type": chart_type, "title": title, "x": x, "y": y})
        except json.JSONDecodeError:
            continue
    return charts


def render_chart(chart_data: dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    chart_type = chart_data["type"]
    x = chart_data["x"]
    y = chart_data["y"]

    if chart_type == "bar":
        ax.bar(x, y, color="#3b82f6")
    elif chart_type == "line":
        ax.plot(x, y, marker="o", color="#22c55e")
    elif chart_type == "pie":
        ax.pie(y, labels=x, autopct="%1.1f%%")

    ax.set_title(chart_data["title"])
    plt.tight_layout()
    return fig


def clean_markdown(markdown: str) -> str:
    """Remove chart tags, leaving only markdown text."""
    return re.sub(r"<chart.*?</chart>", "", markdown, flags=re.DOTALL)


def build_citation_html(citations: list) -> str:
    """Build HTML for citation list."""
    html = "<ol>"
    for i, c in enumerate(citations, 1):
        html += f'<li id="cite-{i}"><a href="{c["url"]}" target="_blank">{c["title"]}</a> — {c["snippet"]}</li>'
    html += "</ol>"
    return html

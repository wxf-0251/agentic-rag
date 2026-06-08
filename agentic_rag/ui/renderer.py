import base64
import html
import io
import re
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


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


def _citation_key(citation: dict) -> str:
    url = (citation.get("url") or "").strip().lower()
    if url:
        return f"url:{url.rstrip('/')}"
    title = re.sub(r"\s+", " ", citation.get("title", "")).strip().lower()
    snippet = re.sub(r"\s+", " ", citation.get("snippet", "")).strip().lower()[:160]
    return f"text:{title}|{snippet}"


def dedupe_citations(citations: list) -> list:
    """Preserve citation order while removing repeated URLs/titles."""
    key_to_item = {}
    deduped = []
    for citation in citations:
        key = _citation_key(citation)
        raw_id = str(citation.get("id", "")).strip()
        if key in key_to_item:
            if raw_id:
                key_to_item[key].setdefault("aliases", []).append(raw_id)
            continue
        item = dict(citation)
        item["number"] = len(deduped) + 1
        item.setdefault("id", f"source-{item['number']}")
        item["aliases"] = [item["id"]] if item.get("id") else []
        if raw_id and raw_id not in item["aliases"]:
            item["aliases"].append(raw_id)
        key_to_item[key] = item
        deduped.append(item)
    return deduped


def normalize_citation_markers(content: str, citations: list) -> str:
    """Convert model-facing citation IDs like [c1] / 【c1】 into reader-facing [1]."""
    deduped = dedupe_citations(citations)
    id_to_number = {}
    for idx, citation in enumerate(deduped, 1):
        for alias in citation.get("aliases", []):
            raw_id = str(alias).strip()
            if raw_id:
                id_to_number[raw_id.lower()] = idx
        id_to_number.setdefault(f"c{idx}", idx)

    def repl(match: re.Match) -> str:
        marker = match.group(1).lower()
        number = id_to_number.get(marker)
        return f"[{number}]" if number else match.group(0)

    return re.sub(r"[\[【]([cC]\d+)[\]】]", repl, content)


def citations_for_prompt(citations: list) -> list:
    """Return a compact, numbered citation list for the Writer/Reviewer prompts."""
    prompt_citations = []
    for citation in dedupe_citations(citations):
        prompt_citations.append({
            "number": citation["number"],
            "ref": f"[{citation['number']}]",
            "title": citation.get("title", ""),
            "url": citation.get("url", ""),
            "snippet": citation.get("snippet", ""),
            "source_type": citation.get("source_type", "unknown"),
            "published_date": citation.get("published_date", ""),
            "freshness_status": citation.get("freshness_status", "unknown"),
            "authority_score": citation.get("authority_score", 0),
        })
    return prompt_citations


def should_render_chart(chart_data: dict) -> bool:
    y = chart_data.get("y", [])
    if not y or not all(isinstance(value, (int, float)) for value in y):
        return False

    # Filter out LLM-made rubric charts such as 1-5 score charts.
    if y and all(float(value).is_integer() and 1 <= value <= 5 for value in y):
        return False

    return True


def render_chart(chart_data: dict):
    chart_type = chart_data["type"]
    x = chart_data["x"]
    y = chart_data["y"]
    if chart_type == "bar" and any(len(str(label)) > 8 for label in x):
        chart_type = "horizontal_bar"

    fig, ax = plt.subplots(figsize=(8, 4.6 if chart_type == "horizontal_bar" else 4))

    if chart_type == "bar":
        ax.bar(x, y, color="#3b82f6")
    elif chart_type == "horizontal_bar":
        ax.barh(x, y, color="#2563eb")
        ax.invert_yaxis()
    elif chart_type == "line":
        ax.plot(x, y, marker="o", color="#22c55e")
        ax.grid(True, axis="y", alpha=0.25)
    elif chart_type == "area":
        ax.fill_between(x, y, color="#22c55e", alpha=0.25)
        ax.plot(x, y, marker="o", color="#15803d")
    elif chart_type == "scatter":
        ax.scatter(x, y, color="#f97316", s=72)
        ax.grid(True, alpha=0.25)
    elif chart_type == "pie":
        ax.pie(y, labels=x, autopct="%1.1f%%")

    ax.set_title(chart_data["title"])
    if chart_type not in ("pie", "horizontal_bar"):
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    return fig


def render_chart_png(chart_data: dict) -> bytes:
    """Render a chart tag to PNG bytes."""
    fig = render_chart(chart_data)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def clean_markdown(markdown: str) -> str:
    """Remove chart tags, leaving only markdown text."""
    return re.sub(r"<chart.*?</chart>", "", markdown, flags=re.DOTALL)


def validate_inline_citations(content: str, citations: list) -> dict:
    """Check that numeric inline citations like [1] map to provided sources."""
    normalized = normalize_citation_markers(content, citations)
    used_numbers = sorted({int(match) for match in re.findall(r"\[(\d+)\]", normalized)})
    available_numbers = set(range(1, len(dedupe_citations(citations)) + 1))
    missing_numbers = [number for number in used_numbers if number not in available_numbers]
    unused_numbers = [number for number in available_numbers if number not in used_numbers]
    return {
        "valid": not missing_numbers,
        "used_numbers": used_numbers,
        "missing_numbers": missing_numbers,
        "unused_numbers": unused_numbers,
    }


def build_report_markdown(title: str, sections: list, citations: list) -> str:
    """Build a complete markdown report with a source appendix."""
    citations = dedupe_citations(citations)
    parts = [f"# {title}".strip()]
    for section in sections:
        section_title = section.get("title", "Untitled")
        content = normalize_citation_markers(clean_markdown(section.get("content", "")), citations).strip()
        parts.append(f"## {section_title}\n\n{content}".strip())

    if citations:
        reference_lines = ["## References"]
        for idx, citation in enumerate(citations, 1):
            source_title = citation.get("title", "Untitled source")
            url = citation.get("url", "")
            snippet = citation.get("snippet", "")
            quality = _format_source_quality(citation)
            if url:
                reference_lines.append(f"{idx}. [{source_title}]({url}) - {quality} - {snippet}")
            else:
                reference_lines.append(f"{idx}. {source_title} - {quality} - {snippet}")
        parts.append("\n".join(reference_lines))

    return "\n\n".join(parts).strip() + "\n"


def build_report_html(title: str, sections: list, citations: list) -> str:
    """Build simple printable HTML for PDF export."""
    citations = dedupe_citations(citations)
    body_lines = []
    body_lines.append(f"<h1>{html.escape(title)}</h1>")
    for section in sections:
        section_title = section.get("title", "Untitled")
        body_lines.append(f"<h2>{html.escape(section_title)}</h2>")
        content = normalize_citation_markers(clean_markdown(section.get("content", "")), citations)
        for line in content.splitlines():
            if line.strip():
                heading_text = _markdown_heading_text(line)
                if heading_text and heading_text == section_title:
                    continue
                body_lines.append(_markdown_line_to_html(line))
        for chart_data in parse_chart_tags(section.get("content", "")):
            if not should_render_chart(chart_data):
                continue
            png = render_chart_png(chart_data)
            encoded = base64.b64encode(png).decode("ascii")
            alt = html.escape(chart_data.get("title", "Chart"))
            body_lines.append(f'<img alt="{alt}" src="data:image/png;base64,{encoded}" />')

    if citations:
        body_lines.append("<h2>References</h2>")
        reference_lines = []
        for idx, citation in enumerate(citations, 1):
            source_title = citation.get("title", "Untitled source")
            url = citation.get("url", "")
            snippet = citation.get("snippet", "")
            quality = _format_source_quality(citation)
            reference_lines.append(f"{idx}. {source_title} - {quality} - {url} - {snippet}")
        for line in reference_lines:
            body_lines.append(f"<p>{html.escape(line)}</p>")

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    @page {{ size: A4; margin: 22mm 18mm; }}
    * {{ box-sizing: border-box; }}
    body {{
      color: #111827;
      font-family: "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", Arial, sans-serif;
      font-size: 13.5px;
      line-height: 1.72;
      margin: 0;
      print-color-adjust: exact;
      -webkit-print-color-adjust: exact;
    }}
    h1 {{
      border-bottom: 1px solid #d1d5db;
      font-size: 25px;
      line-height: 1.28;
      margin: 0 0 22px;
      padding-bottom: 14px;
    }}
    h2, h3, h4 {{
      break-after: avoid;
      break-inside: avoid;
      color: #1f2937;
    }}
    h2 {{
      font-size: 20px;
      line-height: 1.35;
      margin: 26px 0 12px;
    }}
    h3 {{
      font-size: 16px;
      line-height: 1.4;
      margin: 18px 0 8px;
    }}
    h4 {{
      font-size: 14px;
      line-height: 1.4;
      margin: 14px 0 6px;
    }}
    p {{ margin: 0 0 10px; text-align: justify; }}
    img {{
      break-inside: avoid;
      display: block;
      max-height: 135mm;
      max-width: 100%;
      margin: 14px auto 22px;
    }}
  </style>
</head>
<body>
{''.join(body_lines)}
</body>
</html>"""


def _markdown_line_to_html(line: str) -> str:
    stripped = line.strip()
    heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
    if heading:
        level = min(len(heading.group(1)), 6)
        return f"<h{level}>{html.escape(heading.group(2).strip())}</h{level}>"
    return f"<p>{html.escape(stripped)}</p>"


def _markdown_heading_text(line: str) -> str | None:
    heading = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    return heading.group(2).strip() if heading else None


def _format_source_quality(citation: dict) -> str:
    parts = []
    if citation.get("source_type"):
        parts.append(f"type={citation.get('source_type')}")
    if citation.get("freshness_status"):
        parts.append(f"freshness={citation.get('freshness_status')}")
    if citation.get("published_date"):
        parts.append(f"date={citation.get('published_date')}")
    if citation.get("authority_score") not in (None, ""):
        parts.append(f"authority={citation.get('authority_score')}")
    return "; ".join(parts) if parts else "quality=unknown"


def build_report_pdf(title: str, sections: list, citations: list) -> bytes:
    """Render the report to PDF bytes, preferring browser-quality HTML printing."""
    try:
        return _build_report_pdf_with_edge(title, sections, citations)
    except Exception:
        pass

    try:
        from weasyprint import HTML

        return HTML(string=build_report_html(title, sections, citations)).write_pdf()
    except Exception:
        try:
            return _build_report_pdf_with_reportlab(title, sections, citations)
        except Exception:
            return _build_report_pdf_with_matplotlib(title, sections, citations)


def _find_edge_executable() -> str | None:
    configured = os.environ.get("EDGE_EXECUTABLE") or os.environ.get("MS_EDGE_EXECUTABLE")
    if configured and Path(configured).exists():
        return configured

    for command in ("msedge", "microsoft-edge", "microsoft-edge-stable"):
        found = shutil.which(command)
        if found:
            return found

    candidate_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidate_paths:
        if Path(candidate).exists():
            return candidate
    return None


def _build_report_pdf_with_edge(title: str, sections: list, citations: list) -> bytes:
    edge = _find_edge_executable()
    if not edge:
        raise RuntimeError("Microsoft Edge executable was not found.")

    with tempfile.TemporaryDirectory(prefix="agentic-rag-pdf-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        html_path = tmp_path / "report.html"
        pdf_path = tmp_path / "report.pdf"
        profile_path = tmp_path / "edge-profile"
        html_path.write_text(build_report_html(title, sections, citations), encoding="utf-8")

        command = [
            edge,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-first-run",
            "--disable-extensions",
            f"--user-data-dir={profile_path}",
            f"--print-to-pdf={pdf_path}",
            "--no-pdf-header-footer",
            html_path.as_uri(),
        ]
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"Edge PDF export failed: {stderr}")
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            raise RuntimeError("Edge PDF export did not create a PDF file.")
        return pdf_path.read_bytes()


def _build_report_pdf_with_reportlab(title: str, sections: list, citations: list) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import Image, SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    citations = dedupe_citations(citations)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(html.escape(title), styles["Title"]), Spacer(1, 12)]

    for section in sections:
        story.append(Paragraph(html.escape(section.get("title", "Untitled")), styles["Heading2"]))
        content = normalize_citation_markers(clean_markdown(section.get("content", "")), citations).replace("\n", "<br/>")
        story.append(Paragraph(html.escape(content), styles["BodyText"]))
        for chart_data in parse_chart_tags(section.get("content", "")):
            if not should_render_chart(chart_data):
                continue
            image_buffer = io.BytesIO(render_chart_png(chart_data))
            story.append(Spacer(1, 8))
            story.append(Image(image_buffer, width=440, height=250, kind="proportional"))
        story.append(Spacer(1, 12))

    if citations:
        story.append(Paragraph("References", styles["Heading2"]))
        for idx, citation in enumerate(citations, 1):
            source = f"{idx}. {citation.get('title', 'Untitled source')} - {citation.get('url', '')}"
            story.append(Paragraph(html.escape(source), styles["BodyText"]))

    doc.build(story)
    return buffer.getvalue()


def _build_report_pdf_with_matplotlib(title: str, sections: list, citations: list) -> bytes:
    from matplotlib.backends.backend_pdf import PdfPages
    import textwrap

    citations = dedupe_citations(citations)
    buffer = io.BytesIO()

    def add_text_page(pdf: PdfPages, page_title: str, lines: list[str]) -> None:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, page_title, fontsize=15, weight="bold", va="top")
        y = 0.89
        for line in lines:
            for wrapped in textwrap.wrap(line, width=86) or [""]:
                fig.text(0.08, y, wrapped, fontsize=9, va="top")
                y -= 0.026
                if y < 0.08:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    fig = plt.figure(figsize=(8.27, 11.69))
                    y = 0.94
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    with PdfPages(buffer) as pdf:
        for section in sections:
            content = normalize_citation_markers(clean_markdown(section.get("content", "")), citations)
            lines = [line for line in content.splitlines() if line.strip()]
            add_text_page(pdf, section.get("title", "Untitled"), lines)
            for chart_data in parse_chart_tags(section.get("content", "")):
                if not should_render_chart(chart_data):
                    continue
                fig = render_chart(chart_data)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        if citations:
            refs = [
                f"{idx}. {citation.get('title', 'Untitled source')} - {citation.get('url', '')}"
                for idx, citation in enumerate(citations, 1)
            ]
            add_text_page(pdf, "References", refs)

    return buffer.getvalue()


def build_citation_html(citations: list) -> str:
    """Build HTML for citation list."""
    html = "<ol>"
    for i, c in enumerate(citations, 1):
        title = c.get("title", "")
        url = c.get("url", "")
        snippet = c.get("snippet", "")
        html += f'<li id="cite-{i}"><a href="{url}" target="_blank">{title}</a> - {snippet}</li>'
    html += "</ol>"
    return html

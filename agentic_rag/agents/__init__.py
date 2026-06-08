import re
import json
import os
import tempfile


def normalize_report_text(text: str) -> str:
    """Fix common text artifacts before content is shown in the report."""
    mojibake_markers = (
        "\u93c4", "\u9428", "\u951b", "\u6d93", "\u934f",
        "\u7ba1", "\u6d63", "\u9a9e", "\u9225", "\ufffd",
    )
    if any(marker in text for marker in mojibake_markers):
        try:
            repaired = text.encode("gbk").decode("utf-8")
            old_score = sum(text.count(marker) for marker in mojibake_markers)
            new_score = sum(repaired.count(marker) for marker in mojibake_markers)
            if new_score < old_score:
                text = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    replacements = {
        "\u663e\u7740": "\u663e\u8457",
        "\u9225?": "",
        "\u9225": "",
        "\U0001f527": "",
        "\u2705": "",
        "技术自主化的氧化碳": "技术自主化的推进",
        "供应链负担": "供应链韧性",
        "供应链信任": "供应链韧性",
        "市场消耗": "市场份额",
        "盈利增长数万台规模": "出货增长至数万台规模",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _escape_string_bodies(text: str) -> str:
    """Walk the text as if parsing JSON; inside string values, escape literal
    control chars (newlines/tabs/CR) and bare double quotes that aren't
    immediately followed by a JSON-structural token. This rescues LLM output
    where long Markdown content fields weren't properly escaped.
    """
    out = []
    i = 0
    n = len(text)
    in_string = False
    while i < n:
        ch = text[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue
        # inside a string
        if ch == "\\" and i + 1 < n:
            out.append(ch)
            out.append(text[i + 1])
            i += 2
            continue
        if ch == "\n":
            out.append("\\n"); i += 1; continue
        if ch == "\r":
            out.append("\\r"); i += 1; continue
        if ch == "\t":
            out.append("\\t"); i += 1; continue
        if ch == '"':
            # Decide whether this " ends the string. Look ahead, skipping
            # whitespace, for a JSON-structural char: , } ] :
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j >= n or text[j] in ",}]:":
                out.append('"')
                in_string = False
                i += 1
                continue
            # Escape a bare quote inside the string body.
            out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _try_repair_json(text: str) -> dict:
    """Attempt to repair common LLM JSON issues before parsing."""
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*(\}|\])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try with strict=False (allows control chars in strings)
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    # Aggressive: re-escape control chars and bare quotes inside string bodies
    repaired = _escape_string_bodies(text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    return json.loads(repaired, strict=False)


def parse_json(raw: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks and extra text."""
    raw = raw.strip()
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip leading and trailing ``` fences
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    stripped = re.sub(r"\n?\s*```\s*$", "", stripped)
    # If it still looks like a fence, try harder: remove all leading/trailing fences
    stripped = re.sub(r"^[`\s]+", "", stripped)
    stripped = re.sub(r"[`\s]+$", "", stripped)
    # Try parsing the de-fenced content (strict and non-strict)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(stripped, strict=False)
    except json.JSONDecodeError:
        pass
    # Fallback: find the outermost balanced { } pair by position
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first != -1 and last > first:
        candidate = stripped[first:last + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            pass
        # Try repairing common issues
        try:
            return _try_repair_json(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
    # Last resort: try regex to find JSON object
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(m.group(0), strict=False)
        except json.JSONDecodeError:
            pass
        try:
            return _try_repair_json(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    # Debug: save failed response for inspection
    try:
        debug_path = os.path.join(tempfile.gettempdir(), "parse_json_failure.txt")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(f"=== RAW ({len(raw)} chars) ===\n{raw}\n\n=== STRIPPED ({len(stripped)} chars) ===\n{stripped}")
        print(f"[DEBUG] Saved failed parse_json input to {debug_path}")
    except Exception:
        pass
    raise ValueError(f"Failed to parse JSON from response ({len(raw)} chars): {raw[:500]}")

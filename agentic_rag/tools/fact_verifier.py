from __future__ import annotations

import re

from agentic_rag.ui.renderer import dedupe_citations, normalize_citation_markers


NUMBER_PATTERN = re.compile(r"(?<![\w\[])(\d+(?:\.\d+)?%?)(?![\w\]])")
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _strip_citation_markers(text: str) -> str:
    return CITATION_PATTERN.sub("", text)


def _split_claims(content: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|\n+", content) if part.strip()]


def extract_numeric_facts(content: str) -> list[dict]:
    facts = []
    for claim in _split_claims(content):
        claim_without_refs = _strip_citation_markers(claim)
        citations = [int(value) for value in CITATION_PATTERN.findall(claim)]
        for match in NUMBER_PATTERN.finditer(claim_without_refs):
            facts.append({
                "value": match.group(1),
                "claim": claim,
                "citations": citations,
            })
    return facts


def _citation_evidence(citation: dict) -> str:
    return " ".join(
        str(citation.get(key, ""))
        for key in ("title", "snippet", "url", "content", "findings")
    )


def verify_numeric_claims(content: str, citations: list[dict]) -> dict:
    citations = dedupe_citations(citations)
    normalized = normalize_citation_markers(content, citations)
    facts = extract_numeric_facts(normalized)
    issues = []

    for fact in facts:
        if not fact["citations"]:
            issues.append({
                "type": "numeric_claim_without_citation",
                "value": fact["value"],
                "claim": fact["claim"],
            })
            continue

        evidence = []
        for number in fact["citations"]:
            if 1 <= number <= len(citations):
                evidence.append(_citation_evidence(citations[number - 1]))
        if not any(fact["value"] in text for text in evidence):
            issues.append({
                "type": "unsupported_numeric_claim",
                "value": fact["value"],
                "claim": fact["claim"],
                "citations": fact["citations"],
            })

    return {
        "valid": not issues,
        "facts": facts,
        "issues": issues,
    }

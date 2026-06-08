from agentic_rag.tools.fact_verifier import extract_numeric_facts, verify_numeric_claims


def test_extract_numeric_facts_ignores_inline_citation_numbers():
    facts = extract_numeric_facts("Revenue reached 50 billion yuan in 2026 [1]. Margin was 12% [2].")

    values = [fact["value"] for fact in facts]

    assert "50" in values
    assert "2026" in values
    assert "12%" in values
    assert "1" not in values
    assert "2" not in values


def test_verify_numeric_claims_passes_when_numbers_exist_in_cited_source():
    content = "Revenue reached 50 billion yuan in 2026 [1]."
    citations = [
        {
            "title": "Company filing",
            "url": "https://example.com/filing",
            "snippet": "The company reported revenue of 50 billion yuan in 2026.",
        }
    ]

    result = verify_numeric_claims(content, citations)

    assert result["valid"] is True
    assert result["issues"] == []


def test_verify_numeric_claims_fails_when_numbers_do_not_match_cited_source():
    content = "Revenue reached 50 billion yuan in 2026 [1]."
    citations = [
        {
            "title": "Company filing",
            "url": "https://example.com/filing",
            "snippet": "The company reported revenue of 40 billion yuan in 2026.",
        }
    ]

    result = verify_numeric_claims(content, citations)

    assert result["valid"] is False
    assert result["issues"][0]["type"] == "unsupported_numeric_claim"
    assert result["issues"][0]["value"] == "50"

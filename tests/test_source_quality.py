from datetime import date

from agentic_rag.tools.source_quality import (
    extract_source_date,
    is_source_fresh,
    rank_sources,
    score_source_authority,
)


def test_extract_source_date_from_metadata_and_text():
    assert extract_source_date({"published_date": "2026-05-18"}) == date(2026, 5, 18)
    assert extract_source_date({"content": "Published on 2025-12-31 with market data."}) == date(2025, 12, 31)


def test_freshness_gate_marks_old_latest_news_stale():
    today = date(2026, 6, 2)

    fresh = is_source_fresh(date(2026, 5, 20), source_type="latest_news", today=today)
    stale = is_source_fresh(date(2024, 1, 1), source_type="latest_news", today=today)

    assert fresh == "fresh"
    assert stale == "stale"


def test_source_authority_scores_official_above_forum_sources():
    official = score_source_authority("https://www.sse.com.cn/disclosure/listedinfo/announcement/")
    forum = score_source_authority("https://zhuanlan.zhihu.com/p/123")

    assert official > forum


def test_rank_sources_prioritizes_fresh_authoritative_sources():
    today = date(2026, 6, 2)
    results = [
        {
            "title": "Opinion post",
            "url": "https://zhuanlan.zhihu.com/p/old",
            "content": "Published on 2024-01-01. Tech stock opinion.",
        },
        {
            "title": "Exchange announcement",
            "url": "https://www.sse.com.cn/disclosure/listedinfo/announcement/",
            "content": "Published on 2026-05-28. Official company disclosure.",
        },
    ]

    ranked = rank_sources(results, source_type="latest_news", today=today)

    assert ranked[0]["title"] == "Exchange announcement"
    assert ranked[0]["freshness_status"] == "fresh"
    assert ranked[1]["freshness_status"] == "stale"

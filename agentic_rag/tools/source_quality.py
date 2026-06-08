from __future__ import annotations

import re
from datetime import date, datetime
from urllib.parse import urlparse


OFFICIAL_DOMAINS = (
    "gov.cn",
    "sec.gov",
    "sse.com.cn",
    "szse.cn",
    "hkexnews.hk",
    "hkex.com.hk",
    "nasdaq.com",
    "nyse.com",
)

MARKET_DATA_DOMAINS = (
    "finance.yahoo.com",
    "bloomberg.com",
    "reuters.com",
    "ft.com",
    "wsj.com",
    "caixin.com",
    "cls.cn",
    "stcn.com",
    "cninfo.com.cn",
    "eastmoney.com",
)

LOW_AUTHORITY_DOMAINS = (
    "zhihu.com",
    "youtube.com",
    "bilibili.com",
    "reddit.com",
    "x.com",
    "twitter.com",
    "medium.com",
)

DATE_KEYS = ("published_date", "publish_date", "date", "datetime", "created_at", "updated_at")


def _domain(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _parse_date(value: object) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(20\d{2}|19\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", text)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_source_date(source: dict) -> date | None:
    for key in DATE_KEYS:
        parsed = _parse_date(source.get(key))
        if parsed:
            return parsed
    metadata = source.get("metadata") or {}
    for key in DATE_KEYS:
        parsed = _parse_date(metadata.get(key))
        if parsed:
            return parsed
    return _parse_date(" ".join(str(source.get(key, "")) for key in ("title", "content", "snippet")))


def is_source_fresh(source_date: date | None, source_type: str, today: date | None = None) -> str:
    if source_date is None:
        return "unknown"
    today = today or datetime.now().date()
    max_age_days = 365 if source_type in ("latest_news", "financial_data", "mixed") else 1095
    age_days = (today - source_date).days
    if age_days < 0:
        return "future"
    return "fresh" if age_days <= max_age_days else "stale"


def score_source_authority(url: str, title: str = "") -> int:
    domain = _domain(url)
    title_lower = (title or "").lower()
    if any(domain.endswith(item) for item in OFFICIAL_DOMAINS):
        return 95
    if any(token in domain for token in ("ir.", "investor.", "annualreports")):
        return 90
    if any(keyword in title_lower for keyword in ("annual report", "10-k", "20-f", "prospectus", "filing", "公告", "年报")):
        return 85
    if any(domain.endswith(item) for item in MARKET_DATA_DOMAINS):
        return 75
    if any(domain.endswith(item) for item in LOW_AUTHORITY_DOMAINS):
        return 35
    if domain:
        return 55
    return 60


def enrich_source_quality(source: dict, source_type: str, today: date | None = None) -> dict:
    item = dict(source)
    source_date = extract_source_date(item)
    item["published_date"] = source_date.isoformat() if source_date else item.get("published_date", "")
    item["freshness_status"] = is_source_fresh(source_date, source_type, today=today)
    item["authority_score"] = score_source_authority(item.get("url", ""), item.get("title", ""))
    return item


def rank_sources(sources: list[dict], source_type: str, today: date | None = None) -> list[dict]:
    enriched = [enrich_source_quality(source, source_type, today=today) for source in sources]
    freshness_weight = {"fresh": 30, "future": 15, "unknown": 0, "stale": -25}
    return sorted(
        enriched,
        key=lambda item: (
            item.get("authority_score", 0) + freshness_weight.get(item.get("freshness_status"), 0),
            item.get("published_date") or "",
        ),
        reverse=True,
    )

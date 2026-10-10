"""Google News RSS search, US-English edition, weekly windows (after:/before: operators).

Writes ``data/raw/google_news_rss.json`` with up to 100 items per (query, week):
title (with " - Publisher" suffix), publisher, pubDate, link.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from midterm.collect.http import get
from midterm.config import RAW

# Issue-neutral queries: they select election coverage without presupposing which issues matter.
QUERIES = {"midterms": "midterms", "voters_poll": "voters poll", "campaign": "midterm campaign",
           "senate_race": "Senate race", "house_race": "House race", "voters_say": "voters say"}
START, END = "2025-11-03", "2026-10-08"
_ITEM = re.compile(r"<item>(.*?)</item>", re.S)


def _tag(s: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", s, re.S)
    return m.group(1) if m else None


def collect() -> None:
    s = datetime.fromisoformat(START).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(END).replace(tzinfo=timezone.utc)
    items = []
    while s < end:
        e = min(s + timedelta(days=7), end)
        for qk, q in QUERIES.items():
            r = get("https://news.google.com/rss/search",
                    {"q": f"{q} after:{s:%Y-%m-%d} before:{e:%Y-%m-%d}", "hl": "en-US", "gl": "US", "ceid": "US:en"}, pause=1.0)
            for it in _ITEM.findall(r.text):
                items.append({"q": qk, "win": f"{s:%Y-%m-%d}", "title": _tag(it, "title"), "pubDate": _tag(it, "pubDate"),
                              "source": _tag(it, "source"), "link": _tag(it, "link")})
        s = e
    out = {"meta": {"source": "news.google.com/rss/search hl=en-US gl=US", "queries": QUERIES, "window_days": 7,
                    "retrieved": datetime.now(timezone.utc).isoformat()}, "items": items}
    (RAW / "google_news_rss.json").write_text(json.dumps(out))


if __name__ == "__main__":
    collect()

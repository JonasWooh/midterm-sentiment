"""Issue-targeted Google News RSS headlines for the key issues (dense text sample).

Why: the issue-neutral election corpus (``google_news.py``) yields only a few
documents per issue-week once documents are split across issues, so weekly
issue-sentiment means are dominated by sampling noise (median reliability
0.07). This collector draws a dense, issue-specific sample instead.

Design
------
* one query per key issue, from ``config/topic_queries.yaml`` (the same keyword
  sets used for GDELT tone, so that the text index and GDELT measure the same
  construct and can be cross-validated);
* US-English edition (hl=en-US, gl=US);
* consecutive 2-day windows ``after:d before:d+2`` (Google News returns at most
  100 items per request; a 7-day window is almost always capped at 100, a 2-day
  window gives up to ~350 headlines per issue-week);
* resumable: every finished (issue, window) is appended as one JSON line to
  ``data/raw/google_news_issue_rss.jsonl``; a rerun skips finished requests.

    python -m midterm.collect.google_news_issues [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--out PATH]

``--from/--to`` restrict the window *start* dates (inclusive / exclusive) so
that disjoint ranges can be collected in parallel from different hosts; the
JSON-lines outputs are then concatenated (records are self-describing).
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta

import yaml

from midterm.collect.http import get
from midterm.config import RAW, ROOT

ISSUE2QUERY = {"DEMOC": "democracy_elections", "IRAN": "iran_war", "IMMIG": "immigration", "COST": "cost_of_living",
               "TRUMP": "trump_approval", "SOCIAL": "social_culture", "FOREIGN": "foreign_policy",
               "AI": "ai_data_centers", "HEALTH": "healthcare", "TARIFF": "tariffs_trade", "ENERGY": "energy_gas_prices"}
START, END, STEP_DAYS = date(2025, 11, 3), date(2026, 10, 8), 2
OUT = RAW / "google_news_issue_rss.jsonl"
_ITEM = re.compile(r"<item>(.*?)</item>", re.S)


def _tag(s: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", s, re.S)
    return m.group(1) if m else None


def windows() -> list[tuple[date, date]]:
    out, s = [], START
    while s < END:
        e = min(s + timedelta(days=STEP_DAYS), END)
        out.append((s, e))
        s = e
    return out


def collect(first: date = START, last: date = END, out=OUT) -> None:
    queries = yaml.safe_load((ROOT / "config" / "topic_queries.yaml").read_text())["queries"]
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["issue"], r["win_start"]))
    todo = [(k, s, e) for s, e in windows() if first <= s < last for k in ISSUE2QUERY if (k, s.isoformat()) not in done]
    print(f"{len(done)} requests done, {len(todo)} to go")
    with out.open("a") as f:
        for i, (k, s, e) in enumerate(todo):
            q = queries[ISSUE2QUERY[k]]
            r = get("https://news.google.com/rss/search",
                    {"q": f"{q} after:{s.isoformat()} before:{e.isoformat()}", "hl": "en-US", "gl": "US", "ceid": "US:en"},
                    pause=1.0, tries=10)
            items = [{"title": _tag(it, "title"), "pubDate": _tag(it, "pubDate"), "source": _tag(it, "source"),
                      "link": _tag(it, "link")} for it in _ITEM.findall(r.text)]
            f.write(json.dumps({"issue": k, "query": q, "win_start": s.isoformat(), "win_end": e.isoformat(),
                                "n": len(items), "items": items}) + "\n")
            f.flush()
            if i % 50 == 0:
                print(i, k, s, len(items), flush=True)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="first", type=date.fromisoformat, default=START)
    ap.add_argument("--to", dest="last", type=date.fromisoformat, default=END)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    collect(a.first, a.last, a.out)

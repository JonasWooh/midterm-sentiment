"""Issue-targeted news corpus (dense sample for the issue-sentiment index).

Input: ``data/raw/google_news_issue_rss.jsonl`` (``midterm.collect.google_news_issues``).
Output: ``data/interim/issue_docs.parquet`` with one row per (issue, headline):

    doc_id, issue, date (ET calendar day), title, text, sub (publisher), window

Cleaning rules (deterministic, same conventions as ``preprocess.load_news``):
* the " - Publisher" suffix is stripped, HTML entities decoded;
* items published outside their 2-day query window (+/- 1 day tolerance for
  time-zone and indexing lag) are dropped;
* headlines with fewer than 4 tokens are dropped;
* within an issue, duplicate headlines (normalised text) are collapsed, keeping
  the earliest publication. The same headline may legitimately belong to two
  issues (e.g. tariffs and cost of living), so duplicates *across* issues are
  kept; downstream every issue is aggregated separately.
"""
from __future__ import annotations

import hashlib
import html
import json

import pandas as pd

from midterm.config import INTERIM, RAW
from midterm.nlp.preprocess import ET, _clean

SRC = RAW / "google_news_issue_rss.jsonl"


def norm_key(s: pd.Series) -> pd.Series:
    return s.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def build(start: str, end: str) -> pd.DataFrame:
    rows = []
    for line in SRC.read_text().splitlines():
        r = json.loads(line)
        for it in r["items"]:
            rows.append({"issue": r["issue"], "win_start": r["win_start"], "win_end": r["win_end"], **it})
    df = pd.DataFrame(rows).dropna(subset=["title", "pubDate"])
    n_raw = len(df)
    df["source"] = df["source"].fillna("").map(html.unescape)
    t = df["title"].map(html.unescape)
    suffix = " - " + df["source"]
    df["title"] = [x[: -len(sf)] if sf.strip(" -") and x.endswith(sf) else x for x, sf in zip(t, suffix)]
    df["title"] = df["title"].map(_clean)
    df["ts"] = pd.to_datetime(df["pubDate"], format="%a, %d %b %Y %H:%M:%S %Z", utc=True)
    ws = pd.to_datetime(df["win_start"]).dt.tz_localize("UTC")
    we = pd.to_datetime(df["win_end"]).dt.tz_localize("UTC")
    inside = (df["ts"] >= ws - pd.Timedelta(days=1)) & (df["ts"] < we + pd.Timedelta(days=1))
    df = df[inside].copy()
    n_inside = len(df)
    df["key"] = norm_key(df["title"])
    df = df[df["key"].str.split().str.len() >= 4]
    df = df.sort_values("ts").drop_duplicates(["issue", "key"], keep="first")
    df["date"] = df["ts"].dt.tz_convert(ET).dt.tz_localize(None).dt.normalize()
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    out = pd.DataFrame({
        "doc_id": "i_" + df["issue"] + "_" + df["key"].map(lambda k: hashlib.md5(k.encode()).hexdigest()[:12]),
        "issue": df["issue"], "date": df["date"], "title": df["title"], "text": df["title"],
        "sub": df["source"], "window": df["win_start"],
    }).sort_values(["issue", "date", "doc_id"]).reset_index(drop=True)
    assert out["doc_id"].is_unique
    out.attrs.update({"raw_items": n_raw, "outside_window": n_raw - n_inside, "kept": len(out)})
    out.to_parquet(INTERIM / "issue_docs.parquet", index=False)
    return out


if __name__ == "__main__":
    from midterm.config import settings
    s = settings()["study"]
    d = build(s["start"], s["end"])
    print(d.attrs)
    print(d.groupby("issue").size())

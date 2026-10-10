"""Build a unified document table from the raw news (Google News RSS) and social (Reddit) pulls.

Output: ``data/interim/docs.parquet`` with columns

    doc_id, source ('news'|'reddit'), date (ET calendar day), text, title,
    sub / domain, weight, lean

Cleaning rules (all deterministic):
* news: publisher suffix stripped, items outside their query week dropped,
  duplicate headlines collapsed (earliest kept), titles < 4 tokens dropped;
* reddit: removed/deleted bodies are blanked, posts whose title+body has < 4
  tokens are dropped, exact duplicate titles inside a subreddit-day dropped
  (cross-posting bots);
* Reddit sampling is not proportional across communities (r/politics: the 100
  most recent posts of one day per week, thinned to <= 40; the other communities:
  all posts, thinned to <= 12 per community-day; see
  ``midterm.collect.reddit_arctic``), therefore
  social aggregates are *stratified*: computed per community first and then
  averaged with equal weight across the left / centre / right strata
  (see ``midterm.nlp.sentiment``). ``weight`` is 1 for every document.
"""
from __future__ import annotations

import html
import json
import re

import numpy as np
import pandas as pd

from midterm.config import INTERIM, RAW

ET = "America/New_York"
_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+")
_MD = re.compile(r"[*_>#`~]|\[(.*?)\]\((.*?)\)")

LEAN = {  # self-description of the communities, used only for stratified summaries
    "politics": "left", "democrats": "left", "AskALiberal": "left", "PoliticalDiscussion": "center",
    "moderatepolitics": "center", "Conservative": "right", "Republican": "right", "AskConservatives": "right",
}


def _clean(s: str | None) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    if s.strip() in {"[removed]", "[deleted]"}:
        return ""
    s = _URL.sub(" ", s)
    s = _MD.sub(lambda m: m.group(1) or " ", s)
    return _WS.sub(" ", s).strip()


def load_news() -> pd.DataFrame:
    """Google News RSS items: strip the " - Publisher" suffix, keep items whose
    publication date falls inside the query window, collapse duplicate headlines
    (syndication / the same story returned by several queries) keeping the
    earliest publication."""
    raw = json.loads((RAW / "google_news_rss.json").read_text())
    df = pd.DataFrame(raw["items"]).dropna(subset=["title", "pubDate"])
    df["source"] = df["source"].fillna("").map(html.unescape)
    t = df["title"].map(html.unescape)
    suffix = " - " + df["source"]
    df["title"] = [x[: -len(sf)] if sf.strip(" -") and x.endswith(sf) else x for x, sf in zip(t, suffix)]
    df["title"] = df["title"].map(_clean)
    df["ts"] = pd.to_datetime(df["pubDate"], format="%a, %d %b %Y %H:%M:%S %Z", utc=True)
    win = pd.to_datetime(df["win"]).dt.tz_localize("UTC")
    inside = (df["ts"] >= win - pd.Timedelta(days=1)) & (df["ts"] < win + pd.Timedelta(days=8))
    n_raw = len(df)
    df = df[inside].sort_values("ts")
    df["key"] = df["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    df = df[df["key"].str.split().str.len() >= 4]
    n_before = len(df)
    df = df.drop_duplicates("key", keep="first")
    out = pd.DataFrame({
        "doc_id": "n_" + df["link_fnv"].astype(str) + "_" + df.groupby("link_fnv").cumcount().astype(str),
        "source": "news",
        "date": df["ts"].dt.tz_convert(ET).dt.tz_localize(None).dt.normalize(),
        "title": df["title"], "text": df["title"],
        "sub": df["source"], "query": df["q"], "weight": 1.0, "lean": "news",
    })
    out.attrs.update({"raw": n_raw, "outside_window": n_raw - int(inside.sum()), "dedup_dropped": n_before - len(df)})
    return out


def load_reddit() -> pd.DataFrame:
    raw = json.loads((RAW / "reddit_posts.json").read_text())
    df = pd.DataFrame(raw["posts"]).drop_duplicates("id")
    df["title"] = df["title"].map(_clean)
    df["body"] = df["text"].map(_clean)
    df["text"] = (df["title"] + ". " + df["body"]).str.strip(" .")
    df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["date"] = df["ts"].dt.tz_convert(ET).dt.tz_localize(None).dt.normalize()
    df = df[df["text"].str.split().str.len() >= 4]
    df["tkey"] = df["title"].str.lower()
    df = df.drop_duplicates(["sub", "date", "tkey"])
    return pd.DataFrame({
        "doc_id": "r_" + df["id"], "source": "reddit", "date": df["date"],
        "title": df["title"], "text": df["text"], "sub": df["sub"], "query": "",
        "weight": 1.0, "lean": df["sub"].map(LEAN).fillna("other"),
        "n_comments": df["nc"], "score": df["score"],
    })


def build(start: str, end: str) -> pd.DataFrame:
    news = load_news()
    red = load_reddit()
    docs = pd.concat([news, red], ignore_index=True)
    docs = docs[(docs["date"] >= start) & (docs["date"] <= end)].reset_index(drop=True)
    docs.to_parquet(INTERIM / "docs.parquet", index=False)
    return docs


if __name__ == "__main__":
    from midterm.config import settings
    s = settings()["study"]
    d = build(s["start"], s["end"])
    print(d.groupby("source").size(), d.groupby(["source", "lean"]).size())

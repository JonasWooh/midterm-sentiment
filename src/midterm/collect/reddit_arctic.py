"""Reddit submissions from the Arctic Shift archive (https://arctic-shift.photon-reddit.com).

Sampling design (identical to browser_collectors/reddit_arctic.js):
* r/politics (~350 posts/day): the 100 most recent posts before 00:00 UTC, one day per week;
* r/Conservative: all posts, 10-day windows;
* r/AskConservatives, r/AskALiberal: all posts, 14-day windows;
* r/democrats, r/Republican, r/moderatepolitics, r/PoliticalDiscussion: all posts, 28-day windows.

Export thinning (applied before writing, identical to the browser export): within
each subreddit x UTC-day keep the posts with the smallest FNV-1a-32 hash of their
id - at most 40 for r/politics days and 12 for every other subreddit-day - and
truncate self-text to 200 characters. Hash order is unrelated to content, so this
is a uniform random subsample; ``n_day`` keeps the pre-thinning daily count.
Writes ``data/raw/reddit_posts.json``.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from midterm.collect.http import get
from midterm.config import RAW

API = "https://arctic-shift.photon-reddit.com/api/posts/search"
SUBS = {"politics": (7, "last100"), "Conservative": (10, "all"), "AskConservatives": (14, "all"),
        "AskALiberal": (14, "all"), "democrats": (28, "all"), "Republican": (28, "all"),
        "moderatepolitics": (28, "all"), "PoliticalDiscussion": (28, "all")}
CAP = {"politics": 40}
CAP_DEFAULT = 12
TEXT_CHARS = 200


def _fnv(s: str) -> int:
    """FNV-1a-32 over UTF-16 code units (same as the browser's __fnv)."""
    from midterm.bridge.ingest import fnv1a32
    return fnv1a32(s)


def thin(posts: list[dict]) -> list[dict]:
    from collections import defaultdict
    groups = defaultdict(list)
    for p in posts:
        day = datetime.fromtimestamp(p["t"], timezone.utc).strftime("%Y-%m-%d")
        groups[(p["sub"], day)].append(p)
    out = []
    for (sub, day), arr in groups.items():
        arr.sort(key=lambda p: _fnv(p["id"]))
        for p in arr[: CAP.get(sub, CAP_DEFAULT)]:
            out.append({**p, "text": p["text"][:TEXT_CHARS], "n_day": len(arr)})
    return out


FIELDS = "id,created_utc,title,selftext,score,num_comments,subreddit,link_flair_text"
START, END = "2025-11-03", "2026-10-08"


def collect_posts() -> None:
    start = datetime.fromisoformat(START).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(END).replace(tzinfo=timezone.utc)
    posts, truncated = [], []
    for sub, (wd, mode) in SUBS.items():
        s = start
        while s < end:
            e = s + timedelta(days=1) if mode == "last100" else min(s + timedelta(days=wd), end)
            before = e
            while True:   # page backwards when a window hits the 1000-post cap
                r = get(API, {"subreddit": sub, "after": s.isoformat(), "before": before.isoformat(),
                              "limit": "auto" if mode == "all" else 100, "sort": "desc", "fields": FIELDS}, pause=1.0)
                data = r.json()["data"]
                for p in data:
                    posts.append({"id": p["id"], "t": p["created_utc"], "sub": p["subreddit"], "title": p["title"],
                                  "text": (p.get("selftext") or "")[:400], "score": p["score"], "nc": p["num_comments"],
                                  "flair": p.get("link_flair_text"), "win": int(s.timestamp()), "n_win": len(data),
                                  "mode": mode})
                if mode != "all" or len(data) < 1000:
                    break
                truncated.append([sub, s.isoformat()])
                before = datetime.fromtimestamp(min(p["created_utc"] for p in data), timezone.utc)
            s += timedelta(days=wd)
    posts = thin(posts)
    out = {"meta": {"source": API, "subs": SUBS, "cap_per_sub_day": {"default": CAP_DEFAULT, **CAP},
                    "selftext_max_chars": TEXT_CHARS, "truncated": truncated,
                    "retrieved": datetime.now(timezone.utc).isoformat()}, "posts": posts}
    (RAW / "reddit_posts.json").write_text(json.dumps(out))


if __name__ == "__main__":
    collect_posts()

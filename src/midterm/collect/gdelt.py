"""GDELT DOC 2.0 API collectors.

``tone`` -> data/raw/gdelt_topic_timelines.json (daily average tone for each
               topic query in config/topic_queries.yaml; topic *salience* is
               measured from the project's own corpora, so volume timelines are
               not required)

GDELT asks clients to stay below one request per 5 seconds.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import yaml

from midterm.collect.http import get
from midterm.config import ROOT, RAW

API = "https://api.gdeltproject.org/api/v2/doc/doc"
LIMIT_MSG = "Please limit requests"
START, END = "2025-11-03", "2026-10-08"
fmt = lambda d: d.strftime("%Y%m%d%H%M%S")


def collect_tone() -> None:
    """Resumable: each query's timeline is cached in data/raw/gdelt_tone_cache/<key>.json
    as soon as it arrives; the combined file is written once every query is present."""
    queries = yaml.safe_load((ROOT / "config" / "topic_queries.yaml").read_text())["queries"]
    cache = RAW / "gdelt_tone_cache"
    cache.mkdir(exist_ok=True)
    s = datetime.fromisoformat(START).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(END).replace(tzinfo=timezone.utc)
    for key, q in queries.items():
        if (cache / f"{key}.json").exists():
            continue
        full = f"{q} sourcecountry:US sourcelang:english"
        r = get(API, {"query": full, "mode": "timelinetone", "format": "json", "startdatetime": fmt(s),
                      "enddatetime": fmt(e)}, pause=6, tries=40, rate_limited_text=LIMIT_MSG)
        (cache / f"{key}.json").write_text(json.dumps({"query": full, "timelinetone": r.json()["timeline"][0]["data"],
                                                       "retrieved": datetime.now(timezone.utc).isoformat()}))
        print("got", key, flush=True)
    out = {"meta": {"source": f"{API} mode=timelinetone", "start": START, "end": END},
           "series": {k: json.loads((cache / f"{k}.json").read_text()) for k in queries}}
    (RAW / "gdelt_topic_timelines.json").write_text(json.dumps(out))


if __name__ == "__main__":
    collect_tone()

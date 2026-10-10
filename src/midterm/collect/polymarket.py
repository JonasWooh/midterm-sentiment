"""Polymarket: event metadata (Gamma API) and hourly price history (CLOB API).

Writes ``data/raw/polymarket_prices_hourly.json``::

    {"meta": {...}, "series": {name: {"token": <clobTokenId>, "history": [[unix_ts, price], ...]}}}
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from midterm.collect.http import get
from midterm.config import RAW

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
# event id -> {groupItemTitle: series name}
EVENTS = {
    32228: {"Democrats Sweep": "dem_sweep", "D Senate, R House": "dsen_rhouse", "R Senate, D House": "rsen_dhouse",
            "Republicans Sweep": "rep_sweep", "Other": "bop_other"},
    32225: {"Democratic Party": "dem_house"},
    32224: {"Democratic Party": "dem_senate"},
    103161: {"": "blue_wave"},
}
START = "2025-07-11T00:00:00Z"


def yes_tokens() -> dict[str, str]:
    out = {}
    for eid, names in EVENTS.items():
        ev = get(f"{GAMMA}/events/{eid}").json()
        for m in ev["markets"]:
            key = m.get("groupItemTitle", "")
            if key in names:
                out[names[key]] = json.loads(m["clobTokenIds"])[0]   # first token = "Yes"
    return out


def history(token: str, start: int, end: int, fidelity_min: int = 60, chunk_days: int = 14) -> list[list]:
    h = []
    for s in range(start, end, chunk_days * 86400):
        e = min(s + chunk_days * 86400, end)
        r = get(f"{CLOB}/prices-history", {"market": token, "startTs": s, "endTs": e, "fidelity": fidelity_min}, pause=0.2)
        h += [[p["t"], p["p"]] for p in r.json().get("history", [])]
    return h


def main() -> None:
    start = int(datetime.fromisoformat(START.replace("Z", "+00:00")).timestamp())
    end = int(time.time())
    out = {"meta": {"source": "clob.polymarket.com/prices-history", "fidelity_min": 60,
                    "retrieved": datetime.now(timezone.utc).isoformat(), "format": "history=[[unix_ts, price]]"},
           "series": {}}
    for name, tok in yes_tokens().items():
        out["series"][name] = {"token": tok, "history": history(tok, start, end)}
    (RAW / "polymarket_prices_hourly.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()

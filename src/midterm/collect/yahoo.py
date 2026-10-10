"""Daily prices from the Yahoo Finance chart endpoint.

Writes ``data/raw/yahoo_daily.json`` with, per symbol, timestamps, close,
dividend/split-adjusted close and volume.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from midterm.collect.http import get
from midterm.config import RAW, settings

EXTRA = ["SPY", "RSP", "IWM", "XLV", "XLE", "XLU", "XLF", "ITA", "TAN", "CL=F", "^VIX", "^TNX", "DX-Y.NYB", "GLD"]


def main(start: str = "2024-01-01") -> None:
    b = settings()["basket"]
    syms = list(b["dem_sweep_beneficiaries"]) + list(b["rep_hold_beneficiaries"]) + EXTRA
    p1 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    p2 = int(time.time())
    out = {"meta": {"source": "query1.finance.yahoo.com/v8/finance/chart", "interval": "1d", "period1": start,
                    "retrieved": datetime.now(timezone.utc).isoformat()}, "data": {}}
    for s in syms:
        x = get(f"https://query1.finance.yahoo.com/v8/finance/chart/{s}",
                {"period1": p1, "period2": p2, "interval": "1d", "events": "div,splits"}, pause=0.3).json()["chart"]["result"][0]
        q = x["indicators"]["quote"][0]
        out["data"][s] = {"tz": x["meta"].get("exchangeTimezoneName"), "currency": x["meta"].get("currency"),
                          "t": x["timestamp"], "close": q["close"],
                          "adjclose": (x["indicators"].get("adjclose") or [{}])[0].get("adjclose"), "volume": q["volume"]}
    (RAW / "yahoo_daily.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()

"""Kalshi daily candlesticks (public trade-api v2, no auth needed for market data).

Writes ``data/raw/kalshi_candles.json``::

    {"meta": {...}, "series": {name: [{"ts", "close", "mean", "bid", "ask", "vol", "oi"}, ...]}}
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from midterm.collect.http import get
from midterm.config import RAW

API = "https://api.elections.kalshi.com/trade-api/v2"
MARKETS = {  # name: (series, market ticker, start date)
    "DD": ("KXBALANCEPOWERCOMBO", "KXBALANCEPOWERCOMBO-27FEB-DD", "2025-12-11"),
    "RR": ("KXBALANCEPOWERCOMBO", "KXBALANCEPOWERCOMBO-27FEB-RR", "2025-12-11"),
    "DR": ("KXBALANCEPOWERCOMBO", "KXBALANCEPOWERCOMBO-27FEB-DR", "2025-12-11"),
    "RD": ("KXBALANCEPOWERCOMBO", "KXBALANCEPOWERCOMBO-27FEB-RD", "2025-12-11"),
    "HOUSE_D": ("CONTROLH", "CONTROLH-2026-D", "2025-01-01"),
    "SENATE_D": ("CONTROLS", "CONTROLS-2026-D", "2025-01-01"),
}


def main() -> None:
    end = int(time.time())
    out = {"meta": {"source": f"{API} candlesticks period_interval=1440",
                    "retrieved": datetime.now(timezone.utc).isoformat(),
                    "tickers": {k: v[1] for k, v in MARKETS.items()}}, "series": {}}
    for name, (series, ticker, start) in MARKETS.items():
        st = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
        r = get(f"{API}/series/{series}/markets/{ticker}/candlesticks",
                {"start_ts": st, "end_ts": end, "period_interval": 1440}).json()
        out["series"][name] = [{
            "ts": c["end_period_ts"],
            "close": c.get("price", {}).get("close_dollars"), "mean": c.get("price", {}).get("mean_dollars"),
            "bid": c.get("yes_bid", {}).get("close_dollars"), "ask": c.get("yes_ask", {}).get("close_dollars"),
            "vol": c.get("volume_fp"), "oi": c.get("open_interest_fp")} for c in r["candlesticks"]]
    (RAW / "kalshi_candles.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()

"""Prediction-market probability series (Polymarket + Kalshi).

Outputs ``data/processed/prediction_markets_daily.csv`` with one row per
calendar day (America/New_York) holding, for every contract:

* ``<name>``      last traded / mid price at or before 16:00 ET (equity close),
* ``<name>_eod``  last price of the ET calendar day (used for daily text joins).

For the Polymarket "balance of power" event the four mutually exclusive
outcomes are also renormalised to sum to one (``pm_sweep_norm``) so that the
Democratic-sweep probability is not contaminated by the overround.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from midterm.config import PROCESSED, RAW, settings

ET = "America/New_York"


def _polymarket_hourly() -> pd.DataFrame:
    raw = json.loads((RAW / "polymarket_prices_hourly.json").read_text())
    frames = []
    for name, s in raw["series"].items():
        h = pd.DataFrame(s["history"], columns=["ts", "p"])
        if h.empty:
            continue
        h["ts"] = pd.to_datetime(h["ts"], unit="s", utc=True).dt.tz_convert(ET)
        h = h.drop_duplicates("ts").set_index("ts")["p"].sort_index().rename(name)
        frames.append(h)
    return pd.concat(frames, axis=1, sort=True)


def _sample_daily(hourly: pd.DataFrame, close_hour: int) -> pd.DataFrame:
    """Last observation at or before ``close_hour`` ET, and end-of-day value."""
    # Forward-fill on the native irregular grid, then evaluate at cut-off times.
    days = pd.date_range(hourly.index.min().normalize(), hourly.index.max().normalize(), freq="D", tz=ET)
    cut = days + pd.Timedelta(hours=close_hour)
    eod = days + pd.Timedelta(hours=23, minutes=59, seconds=59)
    ff = hourly.ffill()
    at_close = ff.reindex(ff.index.union(cut)).ffill().loc[cut]
    at_eod = ff.reindex(ff.index.union(eod)).ffill().loc[eod]
    at_close.index = days.tz_localize(None)
    at_eod.index = days.tz_localize(None)
    at_eod.columns = [f"{c}_eod" for c in at_eod.columns]
    return pd.concat([at_close, at_eod], axis=1)


def _kalshi_daily() -> pd.DataFrame:
    raw = json.loads((RAW / "kalshi_candles.json").read_text())
    out = {}
    for name, rows in raw["series"].items():
        df = pd.DataFrame(rows)
        # Candle end is ~00:00 ET of the next day (23:00/01:00 around DST
        # switches) -> snap to the nearest ET midnight and attribute the
        # candle to the day that just ended.
        end = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(ET).dt.tz_localize(None)
        day = (end + pd.Timedelta(hours=2)).dt.normalize() - pd.Timedelta(days=1)
        assert not day.duplicated().any(), name
        num = lambda c: pd.to_numeric(df[c], errors="coerce")
        mid = (num("bid") + num("ask")) / 2
        # Prefer the bid/ask mid (defined every day); fall back to last trade.
        price = mid.where(num("bid").gt(0) & num("ask").lt(1), num("close"))
        out[f"ks_{name}"] = pd.Series(price.values, index=day.values)
    df = pd.DataFrame(out).sort_index()
    df.index.name = "date"
    return df


def build() -> pd.DataFrame:
    cfg = settings()
    hourly = _polymarket_hourly()
    pm = _sample_daily(hourly, cfg["study"]["market_close_hour_et"])
    pm.columns = [f"pm_{c}" for c in pm.columns]
    legs = ["pm_dem_sweep", "pm_rep_sweep", "pm_rsen_dhouse", "pm_dsen_rhouse"]
    pm["pm_bop_sum"] = pm[legs].sum(axis=1, min_count=4)
    pm["pm_sweep_norm"] = pm["pm_dem_sweep"] / pm["pm_bop_sum"]
    ks = _kalshi_daily()
    ks_legs = ["ks_DD", "ks_RR", "ks_DR", "ks_RD"]
    ks["ks_sweep_norm"] = ks["ks_DD"] / ks[ks_legs].sum(axis=1, min_count=4)
    df = pm.join(ks, how="outer").sort_index()
    df.index.name = "date"
    # Consistency check: implied sweep from marginals under independence is a
    # *lower* reference, the joint contract is the object of interest.
    df["pm_sweep_if_indep"] = df["pm_dem_house"] * df["pm_dem_senate"]
    df["logit_sweep"] = np.log(df["pm_sweep_norm"] / (1 - df["pm_sweep_norm"]))
    df.to_csv(PROCESSED / "prediction_markets_daily.csv")
    return df


if __name__ == "__main__":
    d = build()
    print(d.loc["2025-11-01":, ["pm_dem_sweep", "pm_sweep_norm", "ks_DD", "ks_sweep_norm", "pm_dem_house", "pm_dem_senate"]].describe().round(3))

"""Mid-term equity basket: market-adjusted (abnormal) returns.

Method
------
1. Daily simple returns from dividend/split-adjusted closes (Yahoo ``adjclose``).
2. Market model per stock, estimated strictly *before* the study window
   (``basket.estimation_window`` in settings.yaml)::

       r_i,t = a_i + b_i * r_m,t + e_i,t      (OLS, m = SPY)

3. Abnormal return in the study window (primary, "market-adjusted")::

       AR_i,t = r_i,t - b_i * r_m,t

   The pre-period intercept is deliberately NOT carried forward: over an
   11-month window an estimated daily alpha of 10 bp compounds to ~25 % and
   would mechanically impose a trend on every cumulative series. The full
   market-model version r - (a + b r_m) is written as a robustness series
   (suffix ``_mm``).
4. Legs are equal-weighted, rebalanced daily:
       AR_D,t  = mean_i in D+  AR_i,t     (Democratic-sweep beneficiaries)
       AR_R,t  = mean_i in R+  AR_i,t     (Republican-hold beneficiaries)
       SPREAD_t = AR_D,t - AR_R,t         (the "mid-term basket")
   Cumulative abnormal returns (CAR) are simple sums of daily ARs.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from midterm.config import PROCESSED, RAW, TAB, settings


def load_prices() -> pd.DataFrame:
    raw = json.loads((RAW / "yahoo_daily.json").read_text())["data"]
    cols = {}
    for sym, d in raw.items():
        idx = (pd.to_datetime(d["t"], unit="s", utc=True)
               .tz_convert(d.get("tz") or "America/New_York").tz_localize(None).normalize())
        px = d["adjclose"] if d.get("adjclose") else d["close"]
        s = pd.Series(px, index=idx, dtype="float64")
        cols[sym] = s[~s.index.duplicated(keep="last")]
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df


def trading_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Returns on NYSE trading days (days on which SPY traded)."""
    px = prices[prices["SPY"].notna()]
    ret = px.pct_change(fill_method=None)
    return ret.iloc[1:]


def market_model(ret: pd.DataFrame, names: list[str], market: str, window: tuple[str, str]) -> pd.DataFrame:
    est = ret.loc[window[0]:window[1]]
    rows = []
    for n in names:
        d = est[[n, market]].dropna()
        X = np.column_stack([np.ones(len(d)), d[market].values])
        coef, *_ = np.linalg.lstsq(X, d[n].values, rcond=None)
        resid = d[n].values - X @ coef
        rows.append({"ticker": n, "alpha": coef[0], "beta": coef[1], "resid_sd": resid.std(ddof=2),
                     "r2": 1 - resid.var() / d[n].var(ddof=0), "n_obs": len(d)})
    return pd.DataFrame(rows).set_index("ticker")


def build() -> dict[str, pd.DataFrame]:
    cfg = settings()
    b = cfg["basket"]
    d_leg = list(b["dem_sweep_beneficiaries"])
    r_leg = list(b["rep_hold_beneficiaries"])
    mkt = b["market"]
    prices = load_prices()
    ret = trading_returns(prices)
    mm = market_model(ret, d_leg + r_leg, mkt, tuple(b["estimation_window"]))
    meta = pd.concat([
        pd.DataFrame(b["dem_sweep_beneficiaries"]).T.assign(leg="D+"),
        pd.DataFrame(b["rep_hold_beneficiaries"]).T.assign(leg="R+")])
    mm = mm.join(meta)
    mm.to_csv(TAB / "market_model_betas.csv")

    start, end = cfg["study"]["start"], cfg["study"]["end"]
    r = ret.loc[start:end]
    ar = pd.DataFrame({n: r[n] - mm.at[n, "beta"] * r[mkt] for n in d_leg + r_leg})
    ar_mm = pd.DataFrame({n: r[n] - (mm.at[n, "alpha"] + mm.at[n, "beta"] * r[mkt]) for n in d_leg + r_leg})
    out = pd.DataFrame({
        "mkt_ret": r[mkt],
        "ar_D": ar[d_leg].mean(axis=1),
        "ar_R": ar[r_leg].mean(axis=1),
        "raw_D": r[d_leg].mean(axis=1),
        "raw_R": r[r_leg].mean(axis=1),
        "armm_D": ar_mm[d_leg].mean(axis=1),
        "armm_R": ar_mm[r_leg].mean(axis=1),
    })
    out["spread"] = out["ar_D"] - out["ar_R"]
    out["spread_mm"] = out["armm_D"] - out["armm_R"]
    out["spread_raw"] = out["raw_D"] - out["raw_R"]
    for c in ["ar_D", "ar_R", "spread", "spread_mm", "spread_raw"]:
        out[f"car_{c}"] = out[c].cumsum()
    # Controls (daily changes)
    ctl = pd.DataFrame(index=out.index)
    ctl["oil_ret"] = prices["CL=F"].reindex(prices.index).ffill().pct_change(fill_method=None).reindex(out.index)
    ctl["d_tnx"] = prices["^TNX"].ffill().diff().reindex(out.index)
    ctl["d_vix"] = prices["^VIX"].ffill().diff().reindex(out.index)
    out = out.join(ctl)
    # Group-level ARs for attribution
    grp = mm["group"]
    for g in grp.unique():
        out[f"ar_grp_{g}"] = ar[grp[grp == g].index].mean(axis=1)
    out.index.name = "date"
    out.to_csv(PROCESSED / "basket_daily.csv")
    ar.to_csv(PROCESSED / "abnormal_returns_by_stock.csv")
    return {"basket": out, "ar": ar, "betas": mm}


if __name__ == "__main__":
    res = build()
    bk = res["basket"]
    print(res["betas"].round(3))
    print(bk[["car_ar_D", "car_ar_R", "car_spread", "car_spread_raw"]].iloc[[0, -1]].round(4))

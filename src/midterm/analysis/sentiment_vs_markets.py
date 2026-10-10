"""Correlation of key-topic sentiment with the probability of a Democratic sweep.

Probability measure: Polymarket "Balance of Power: D Senate, D House",
renormalised over the four mutually exclusive outcomes, sampled at 16:00 ET
(``pm_sweep_norm``); analysed on the log-odds scale, which is the natural
scale for changes in a probability that moves between 0.3 and 0.65. Kalshi
"KXBALANCEPOWERCOMBO-27FEB-DD" is used as an independent replication.

For every sentiment series X (key topics and the composite index) and the
log-odds Y the module reports

* levels: Pearson / Spearman with circular block-bootstrap CI, and ADF unit-
  root p-values (a level correlation between two persistent series is not
  evidence of a relationship by itself);
* changes: daily and weekly (W-FRI) first differences, Pearson with HAC t-test
  and bootstrap CI — the primary statistic;
* lead-lag: cross-correlation of changes for +/- 14 days / +/- 4 weeks;
* Granger causality in a bivariate VAR of changes (AIC lag, max 7 days / 4 weeks);
* multiple testing: Benjamini-Hochberg q-values within each (market, frequency) family.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from midterm.analysis.stats import adf_p, block_bootstrap_corr, ccf, granger, hac_corr_test, to_weekly
from midterm.config import PROCESSED, TAB


def _bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    v = p.to_numpy(dtype=float)
    ok = ~np.isnan(v)
    q = np.full_like(v, np.nan)
    pv = v[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return pd.Series(q, index=p.index)


def _pairs(sent: pd.DataFrame, y: pd.Series, freq: str) -> pd.DataFrame:
    rows = []
    for col in sent.columns:
        x = sent[col]
        d = pd.concat([x, y], axis=1).dropna()
        lv_r, lv_lo, lv_hi = block_bootstrap_corr(d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy(),
                                                  block=max(5, int(len(d) ** 0.5)))
        ch = hac_corr_test(x.diff(), y.diff())
        g = granger(x.diff(), y.diff(), maxlags=7 if freq == "D" else 4)
        rows.append({"series": col, "freq": freq, "n": len(d),
                     "level_r": lv_r, "level_ci_lo": lv_lo, "level_ci_hi": lv_hi,
                     "level_spearman": d.corr(method="spearman").iloc[0, 1],
                     "adf_p_x": adf_p(x), "adf_p_y": adf_p(y),
                     "diff_r": ch["r"], "diff_spearman": ch["spearman"], "diff_ci_lo": ch["ci_lo"],
                     "diff_ci_hi": ch["ci_hi"], "diff_t_hac": ch["t_hac"], "diff_p_hac": ch["p_hac"],
                     "granger_lags": g["lags"], "sent_to_mkt_p": g["x_to_y_p"], "mkt_to_sent_p": g["y_to_x_p"]})
    return pd.DataFrame(rows)


def weekly_to_daily_panel(weekly: pd.DataFrame, end: str) -> pd.DataFrame:
    """Place weekly (W-FRI) values on their week-end date (the last study day for a
    partial final week) so that ``to_weekly(..., 'last')`` recovers them exactly."""
    w = weekly.copy()
    w.index = [min(d, pd.Timestamp(end)) for d in w.index]
    return w


def run(sent_daily: pd.DataFrame, pm: pd.DataFrame, start: str, end: str, tag: str,
        daily_cols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """daily_cols: series observed daily (tested at daily and weekly frequency);
    all other columns are weekly series placed on week-end dates (weekly tests only)."""
    pm = pm.loc[start:end]
    y_d = pm["logit_sweep"].rename("logit_sweep")
    ks = pm["ks_sweep_norm"]
    y_ks = np.log(ks / (1 - ks)).rename("logit_sweep_kalshi")
    s = sent_daily.loc[start:end]
    dcols = daily_cols if daily_cols is not None else list(s.columns)
    daily = _pairs(s[dcols], y_d, "D")
    weekly = _pairs(to_weekly(s), to_weekly(y_d.to_frame())["logit_sweep"], "W")
    weekly_ks = _pairs(to_weekly(s), to_weekly(y_ks.to_frame())["logit_sweep_kalshi"], "W").assign(market="kalshi")
    daily_ks = pd.concat([_pairs(s[dcols], y_ks, "D").assign(market="kalshi"), weekly_ks])
    res = pd.concat([daily.assign(market="polymarket"), weekly.assign(market="polymarket"), daily_ks])
    # Benjamini-Hochberg false-discovery-rate control within each (market, frequency) family
    for col in ("diff_p_hac", "sent_to_mkt_p", "mkt_to_sent_p"):
        res[col.replace("_p", "_q") if col != "diff_p_hac" else "diff_q_bh"] = (
            res.groupby(["market", "freq"])[col].transform(_bh))
    res.to_csv(TAB / f"sentiment_vs_sweep_{tag}.csv", index=False)
    lags = []
    for col in s.columns:
        if col in dcols:
            lags.append(ccf(s[col].diff(), y_d.diff(), 14).assign(series=col, freq="D"))
        lags.append(ccf(to_weekly(s)[col].diff(), to_weekly(y_d.to_frame())["logit_sweep"].diff(), 4).assign(series=col, freq="W"))
    lags = pd.concat(lags)
    lags.to_csv(TAB / f"sentiment_vs_sweep_ccf_{tag}.csv", index=False)
    # rolling 60-day correlation of daily changes (stability)
    roll = pd.DataFrame({c: s[c].diff().rolling(60, min_periods=40).corr(y_d.diff()) for c in dcols})
    roll.to_csv(PROCESSED / f"rolling_corr_sentiment_sweep_{tag}.csv")
    return {"summary": res, "ccf": lags, "rolling": roll}


IRAN_WAR_START = "2026-02-28"   # US-Israeli strikes on Iran begin (see report, section 2)


def controlled_weekly(panel: pd.DataFrame, pm: pd.DataFrame, controls: pd.DataFrame, start: str, end: str,
                      series: list[str]) -> pd.DataFrame:
    """Weekly d logit(P_sweep) on d X, with and without macro controls (weekly WTI
    return, change in 10y yield, change in VIX), full sample and before/after the
    start of the Iran war. HAC standard errors."""
    from midterm.analysis.stats import hac_ols
    y = to_weekly(pm.loc[start:end, ["logit_sweep"]])["logit_sweep"].diff()
    X = to_weekly(panel.loc[start:end]).diff()
    C = to_weekly(controls, how="sum")
    rows = []
    for s in series:
        for per, (a, b) in {"full": (start, end), "pre_war": (start, IRAN_WAR_START),
                            "post_war": (IRAN_WAR_START, end)}.items():
            for spec, cols in {"no_controls": [], "macro_controls": list(C.columns)}.items():
                d = pd.concat([y.rename("y"), X[s].rename("x"), C[cols]], axis=1).loc[a:b].dropna()
                if len(d) < 12:
                    continue
                r = hac_ols(d["y"], d[["x"] + cols])
                sx, sy = d["x"].std(), d["y"].std()
                rows.append({"series": s, "period": per, "spec": spec, "n_weeks": int(r.nobs),
                             "beta": r.params["x"], "beta_std": r.params["x"] * sx / sy,
                             "t_hac": r.tvalues["x"], "p_hac": r.pvalues["x"], "r2": r.rsquared})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "sentiment_vs_sweep_weekly_controlled.csv", index=False)
    return out

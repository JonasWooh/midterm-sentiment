"""Weekly issue-sentiment index from the dense issue-targeted corpus (v2).

Inputs: ``issue_docs`` (``midterm.nlp.issue_corpus``) scored with
``midterm.nlp.sentiment_v2`` -> columns ``tone`` (predicted -2..+2 scale) and
``p_rel`` (P(relevant to the issue as a US public matter)).

Index
-----
For issue k and week w (W-FRI, aligned with the equity week):

    m_kw = sum_d p_rel_d * tone_d / sum_d p_rel_d

The same empirical-Bayes shrinkage as in ``midterm.nlp.sentiment.shrink`` is
applied (weights n_kw = sum_d p_rel_d; prior = trailing 8-week mean); with
~100-200 relevant headlines per issue-week the shrinkage is light.

Reliability
-----------
* model-based: n / (n + kappa) at the mean weekly weight (shrinkage diagnostics);
* split-half: documents of every issue-week are randomly split in two halves
  (seeded); the two half-indices are correlated across weeks, in levels and in
  week-on-week changes, and stepped up with the Spearman-Brown formula
  r_full = 2 r / (1 + r). This is assumption-light: it measures how much of the
  weekly movement would replicate in an independent sample of headlines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from midterm.config import PROCESSED, TAB
from midterm.nlp.sentiment import shrink

SEED = 20261103


def _weeks(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.offsets.Week(weekday=4), freq="W-FRI")


def _moments(d: pd.DataFrame, weeks: pd.DatetimeIndex, issues: list[str]):
    wk = d["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    g = d.assign(wk=wk, wt=d["p_rel"] * d["tone"])
    num = g.pivot_table(index="wk", columns="issue", values="wt", aggfunc="sum").reindex(index=weeks, columns=issues).fillna(0.0)
    n = g.pivot_table(index="wk", columns="issue", values="p_rel", aggfunc="sum").reindex(index=weeks, columns=issues).fillna(0.0)
    sig = {}
    for k in issues:
        x = d[d["issue"] == k]
        mu = np.average(x["tone"], weights=x["p_rel"])
        sig[k] = np.average((x["tone"] - mu) ** 2, weights=x["p_rel"])
    return num, n, pd.Series(sig)


def build(docs: pd.DataFrame, start: str, end: str, issues: list[str]) -> dict:
    weeks = _weeks(start, end)
    d = docs[(docs["date"] >= start) & (docs["date"] <= end)]
    num, n, s2 = _moments(d, weeks, issues)
    shrunk, diag = shrink(num, n, s2)
    raw = num / n.replace(0, np.nan)
    shrunk.index.name = raw.index.name = "week_end"
    shrunk.to_csv(PROCESSED / "issue_sentiment_v2_weekly.csv")
    raw.to_csv(PROCESSED / "issue_sentiment_v2_weekly_raw.csv")
    n.to_csv(PROCESSED / "issue_sentiment_v2_weekly_weight.csv")
    diag.to_csv(TAB / "issue_sentiment_v2_shrinkage.csv")
    sh = split_half(d, weeks, issues)
    sh.to_csv(TAB / "issue_sentiment_v2_split_half.csv")
    return {"index": shrunk, "raw": raw, "weight": n, "diag": diag, "split_half": sh}


def split_half(d: pd.DataFrame, weeks: pd.DatetimeIndex, issues: list[str], reps: int = 20) -> pd.DataFrame:
    """Split-half reliability of the weekly raw index (levels and weekly changes),
    averaged over ``reps`` random splits, Spearman-Brown corrected."""
    rng = np.random.default_rng(SEED)
    rows = []
    for r in range(reps):
        half = rng.integers(0, 2, len(d))
        out = []
        for h in (0, 1):
            num, n, _ = _moments(d[half == h], weeks, issues)
            out.append(num / n.replace(0, np.nan))
        for k in issues:
            a, b = out[0][k], out[1][k]
            lv = pd.concat([a, b], axis=1).dropna()
            ch = pd.concat([a.diff(), b.diff()], axis=1).dropna()
            rows.append({"rep": r, "issue": k, "r_level": lv.corr().iloc[0, 1], "r_change": ch.corr().iloc[0, 1]})
    s = pd.DataFrame(rows).groupby("issue")[["r_level", "r_change"]].mean()
    sb = lambda x: 2 * x / (1 + x)
    s["reliability_level"] = sb(s["r_level"])
    s["reliability_change"] = sb(s["r_change"])
    return s.reindex(issues)


def convergent_validity(index: pd.DataFrame, gdelt_daily: pd.DataFrame, code2gdelt: dict[str, str]) -> pd.DataFrame:
    """Correlation of the weekly text index with the weekly mean of GDELT full-text
    tone for the same issue query (levels and week-on-week changes)."""
    g = gdelt_daily.resample("W-FRI").mean()
    rows = []
    for k in index.columns:
        col = code2gdelt.get(k)
        if col not in g.columns:
            continue
        j = pd.concat([index[k], g[col]], axis=1).dropna()
        rows.append({"issue": k, "gdelt_query": col, "n_weeks": len(j), "r_level": j.corr().iloc[0, 1],
                     "r_change": j.diff().corr().iloc[0, 1]})
    out = pd.DataFrame(rows).set_index("issue")
    out.to_csv(TAB / "issue_sentiment_v2_vs_gdelt.csv")
    return out

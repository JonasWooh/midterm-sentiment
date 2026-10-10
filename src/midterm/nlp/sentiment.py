"""Issue-level text sentiment over time (weekly).

Document sentiment
------------------
``docs["sent"]``: the zero-shot score P(pos) - P(neg) of the pretrained
transformer ``cardiffnlp/twitter-roberta-base-sentiment-latest``
(``midterm.nlp.sentiment_v2.zero_shot_score``), standardised over the corpus.
On the random hold-outs of this corpus it agrees with the annotations at
Spearman 0.67 (news) / 0.65 (Reddit), against 0.41 / 0.47 for the v1
lexicon + logistic ensemble (``midterm.nlp.sentiment_model``, kept as the
documented baseline). Units are corpus standard deviations; higher = more positive.

This module builds the *legacy-corpus* indices (issue-neutral election news +
stratified Reddit, issue membership from the classifier). Because few
documents per issue-week remain after splitting that corpus by issue, their
reliability is low; the primary text index is the dense issue-targeted one in
``midterm.nlp.issue_index``.

Issue membership
----------------
w_dk = P(issue k | d) from ``midterm.nlp.issue_classifier`` when it is at least
``MIN_PROB`` (set by the pipeline), else 0.

Why weekly
----------
Issue-specific document flow is thin: for most issue x stream cells the
expected issue weight per *day* is below one document, so daily means are
dominated by sampling noise. The text index is therefore built on calendar
weeks ending Friday (aligned with the equity week). The daily news-tone
measure (GDELT) carries the daily analysis.

Estimator (per stream s in {news, left, centre, right} and issue k)
--------------------------------------------------------------------
    m_kw  = sum_d w_dk c_d / sum_d w_dk ,      n_kw = sum_d w_dk
    sigma2_k = sum_d w_dk (c_d - cbar_k)^2 / sum_d w_dk      (document-level variance)
    tau2_k   = max( var_w(m_kw) - mean_w(sigma2_k / n_kw),  0.1 var_w(m_kw) )
    kappa_k  = sigma2_k / tau2_k
    shrunk_kw = (n_kw m_kw + kappa_k mu_k,w-1) / (n_kw + kappa_k)

mu_k,w-1 is the trailing 8-week weighted mean (past data only). This is the
normal-normal empirical-Bayes posterior mean; the floor keeps a minimum of
10% of the observed between-week variance as signal so that a noisy
method-of-moments estimate cannot force complete pooling. The social index is
the equal-weight mean of the three Reddit strata; the combined index is the
mean of news and social.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from midterm.config import PROCESSED

PRIOR_WEEKS = 8
TAU2_FLOOR_SHARE = 0.10


def _weekly_moments(dates: pd.Series, c: np.ndarray, w: np.ndarray, weeks: pd.DatetimeIndex):
    K = w.shape[1]
    wk = pd.to_datetime(dates.values).to_period("W-FRI").end_time.normalize()
    num = pd.DataFrame(w * c[:, None]).groupby(wk).sum().reindex(weeks, fill_value=0.0)
    n = pd.DataFrame(w).groupby(wk).sum().reindex(weeks, fill_value=0.0)
    cbar = (w * c[:, None]).sum(0) / np.maximum(w.sum(0), 1e-12)
    sigma2 = (w * (c[:, None] - cbar) ** 2).sum(0) / np.maximum(w.sum(0), 1e-12)
    num.columns = n.columns = range(K)
    return num, n, pd.Series(sigma2, index=range(K))


def shrink(num: pd.DataFrame, n: pd.DataFrame, sigma2: pd.Series, prior_weeks: int = PRIOR_WEEKS):
    m = num / n.replace(0, np.nan)
    ok = n >= 1
    var_m = m.where(ok).var()
    noise = (sigma2 / n.where(ok)).mean()
    tau2 = np.maximum(var_m - noise, TAU2_FLOOR_SHARE * var_m)
    kappa = sigma2 / tau2
    roll_num = num.rolling(prior_weeks, min_periods=1).sum().shift(1)
    roll_n = n.rolling(prior_weeks, min_periods=1).sum().shift(1)
    mu = roll_num / roll_n.replace(0, np.nan)
    overall = num.sum() / n.sum().replace(0, np.nan)
    mu = mu.fillna(overall)
    shrunk = (n * m.fillna(0) + kappa * mu) / (n + kappa)
    diag = pd.DataFrame({"sigma2_doc": sigma2, "var_weekly_mean": var_m, "noise_var": noise, "tau2": tau2,
                         "kappa": kappa, "mean_weekly_weight": n.mean(),
                         "reliability_at_mean_n": n.mean() / (n.mean() + kappa)})
    return shrunk, diag


def build(docs: pd.DataFrame, M: np.ndarray, names: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """M: (n_docs x n_issues) issue weights. Returns weekly (W-FRI) indices."""
    assert "sent" in docs, "documents need a sentiment column 'sent'"
    weeks = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.offsets.Week(weekday=4), freq="W-FRI")
    labels = dict(enumerate(names))
    series, diags = {}, []
    streams = {
        "news": docs["source"].eq("news"),
        "left": docs["source"].eq("reddit") & docs["lean"].eq("left"),
        "center": docs["source"].eq("reddit") & docs["lean"].eq("center"),
        "right": docs["source"].eq("reddit") & docs["lean"].eq("right"),
    }
    for name, mask in streams.items():
        msk = mask.to_numpy()
        num, n, s2 = _weekly_moments(docs.loc[msk, "date"], docs.loc[msk, "sent"].to_numpy(), M[msk], weeks)
        sh, dg = shrink(num, n, s2)
        diags.append(dg.rename(index=labels).assign(stream=name))
        series[name] = sh.rename(columns=labels)
    out = {}
    social = (series["left"] + series["center"] + series["right"]) / 3.0
    for tag, df in {"news": series["news"], "social": social, "combined": (series["news"] + social) / 2.0,
                    "left": series["left"], "center": series["center"], "right": series["right"]}.items():
        df.index.name = "week_end"
        df.to_csv(PROCESSED / f"issue_sentiment_text_weekly_{tag}.csv")
        out[tag] = df
    pd.concat(diags).to_csv(PROCESSED / "sentiment_shrinkage_diagnostics.csv")
    return out

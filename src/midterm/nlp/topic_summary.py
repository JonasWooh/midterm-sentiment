"""Ranking of voter issues and daily issue salience.

Inputs: per-document issue posteriors P (``midterm.nlp.issue_classifier``).

Salience of issue j in stream s = PCC share of issue-bearing document mass:
    share_js = sum_{d in s} P_dj / sum_{d in s} sum_{k != NONE} P_dk
Social salience is the equal-weight mean of the left / centre / right Reddit
strata; the headline ("combined") salience is the mean of news and social.
The text ranking is reported next to external survey evidence
(config/external_polls.yaml) and the final key-topic list is the union of the
top text-salience issues and the issues that rank top-4 in national polls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from midterm.config import PROCESSED, ROOT, TAB
from midterm.nlp.issue_classifier import ISSUE_NAMES

STREAMS = ("news", "left", "center", "right")


def _mask(docs: pd.DataFrame, s: str) -> np.ndarray:
    if s == "news":
        return docs["source"].eq("news").to_numpy()
    return (docs["source"].eq("reddit") & docs["lean"].eq(s)).to_numpy()


def rank_issues(docs: pd.DataFrame, P: np.ndarray, classes: list[str]) -> pd.DataFrame:
    issue_idx = [i for i, c in enumerate(classes) if c != "NONE"]
    names = [classes[i] for i in issue_idx]
    out = {}
    issue_mass_share = {}
    for s in STREAMS:
        m = _mask(docs, s)
        mass = P[m][:, issue_idx].sum(axis=0)
        out[s] = mass / mass.sum()
        issue_mass_share[s] = mass.sum() / m.sum()
    tab = pd.DataFrame(out, index=names)
    tab["social"] = tab[["left", "center", "right"]].mean(axis=1)
    tab["combined"] = (tab["news"] + tab["social"]) / 2
    # trend: combined share, last 8 weeks vs first 8 weeks
    d = docs["date"]
    for tag, msk in (("first8w", (d < d.min() + pd.Timedelta(weeks=8)).to_numpy()),
                     ("last8w", (d > d.max() - pd.Timedelta(weeks=8)).to_numpy())):
        v = []
        for s in ("news", "reddit"):
            m = msk & docs["source"].eq(s).to_numpy()
            mass = P[m][:, issue_idx].sum(axis=0)
            v.append(mass / mass.sum())
        tab[f"share_{tag}"] = np.mean(v, axis=0)
    tab["trend_pp"] = 100 * (tab["share_last8w"] - tab["share_first8w"])
    tab["label"] = [ISSUE_NAMES[c] for c in tab.index]
    tab = tab.sort_values("combined", ascending=False)
    tab["text_rank"] = np.arange(1, len(tab) + 1)
    polls = yaml.safe_load((ROOT / "config" / "external_polls.yaml").read_text())
    tab["emerson_mip_pct"] = pd.Series(polls["emerson_2026_09"]["most_important_issue_pct"]).reindex(tab.index)
    tab["emerson_importance_1to5"] = pd.Series(polls["emerson_2026_09"]["importance_mean_1to5"]).reindex(tab.index)
    tab.to_csv(TAB / "issue_ranking.csv")
    pd.Series(issue_mass_share, name="issue_share_of_all_docs").to_csv(TAB / "issue_vs_nonissue_share.csv")
    return tab


def select_key_issues(tab: pd.DataFrame, n_text: int = 8) -> list[str]:
    text_top = [c for c in tab.index if c != "OTHER_ISSUE"][:n_text]
    poll_top = list(tab["emerson_mip_pct"].dropna().sort_values(ascending=False).index[:4])
    poll_imp = list(tab["emerson_importance_1to5"].dropna().sort_values(ascending=False).index[:4])
    key = list(dict.fromkeys(text_top + poll_top + poll_imp))
    return key


def issue_share_daily(docs: pd.DataFrame, P: np.ndarray, classes: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    idx = pd.date_range(start, end, freq="D")
    issue_idx = [i for i, c in enumerate(classes) if c != "NONE"]
    out = {}
    for tag in ("news", "reddit"):
        m = docs["source"].eq(tag).to_numpy()
        df = pd.DataFrame(P[m][:, issue_idx], columns=[classes[i] for i in issue_idx])
        df["date"] = docs.loc[m, "date"].to_numpy()
        g = df.groupby("date").sum().reindex(idx, fill_value=0.0)
        n = pd.Series(docs.loc[m].groupby("date").size()).reindex(idx, fill_value=0)
        share = g.div(n.replace(0, np.nan), axis=0)
        share.index.name = "date"
        share.to_csv(PROCESSED / f"issue_salience_daily_{tag}.csv")
        out[tag] = share
    return out

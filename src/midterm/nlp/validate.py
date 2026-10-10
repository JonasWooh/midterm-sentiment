"""Validation of document sentiment (VADER) against independent annotations.

The random evaluation sets ``data/annotations/reference_labels_news.csv`` (264
headlines) and ``reference_labels_reddit.csv`` (304 posts) contain codebook-based
reference labels reviewed and confirmed by researchers. Sentiment is in
{-1, 0, +1} (valence of the text as written). This evaluation estimates
agreement with the reference judgements; it does not establish an objective
gold standard.

Reported per set: Spearman rho between VADER compound and the label, 3-class
accuracy / macro-F1 / Cohen's kappa at VADER's conventional +/-0.05 cut-offs,
and the majority-class baseline. Classifier (issue) validation lives in
``midterm.nlp.issue_classifier.train_and_evaluate`` (true hold-out).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from midterm.config import ROOT, TAB


def _vader_class(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.05, 1, np.where(x <= -0.05, -1, 0))


def run(docs_scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ("news", "reddit"):
        a = pd.read_csv(ROOT / "data" / "annotations" / f"reference_labels_{name}.csv")
        d = a.merge(docs_scored[["doc_id", "vader"]], on="doc_id", how="inner")
        y, v = d["sentiment_label"].to_numpy(), d["vader"].to_numpy()
        vc = _vader_class(v)
        sub = d["issue_label"].ne("NONE").to_numpy()
        rows.append({
            "set": name, "n": len(d),
            "spearman_vader_vs_label": stats.spearmanr(v, y)[0],
            "spearman_issue_docs_only": stats.spearmanr(v[sub], y[sub])[0],
            "acc_3class": accuracy_score(y, vc), "macro_f1_3class": f1_score(y, vc, average="macro"),
            "kappa": cohen_kappa_score(y, vc),
            "majority_baseline_acc": pd.Series(y).value_counts(normalize=True).max(),
            "label_share_neg": float((y == -1).mean()), "label_share_pos": float((y == 1).mean()),
            "vader_share_neg": float((vc == -1).mean()), "vader_share_pos": float((vc == 1).mean()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "validation_sentiment_vader.csv", index=False)
    return out

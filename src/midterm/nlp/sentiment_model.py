"""Supervised document-sentiment model trained on codebook-based reference labels.

The lexicon baseline (VADER) agrees only weakly with a careful reader on this
corpus (Spearman ~0.37, kappa ~0.2-0.26; see validation_sentiment_vader.csv):
political headlines are written in a register where words such as "wins",
"support", "leads" or "deal" carry little valence while conflict vocabulary is
implicit. We therefore train a supervised model on researcher-reviewed reference labels:

* labels: sentiment in {-1, 0, +1} for 1,220 training documents and the two
  random evaluation sets (news 264, Reddit 304), reviewed and confirmed by researchers;
* features: TF-IDF word 1-2-grams + character 3-5-grams;
* model: multinomial logistic regression, C by 5-fold CV (macro-F1);
* document score: s_d = P(+1 | d) - P(-1 | d)  in [-1, 1];
* reported document sentiment = equal-weight ensemble of the standardised
  supervised score and the standardised VADER compound,
      e_d = 0.5 z(s_d) + 0.5 z(vader_d),
  with z computed over the whole corpus. Equal weights are the no-information
  default (not tuned); the hold-out tables report weights 0, .25, .5, .75, 1
  for transparency — the ensemble beats either component on both sets.

Evaluation is on the random hold-out sets only (the final model used for
scoring is refitted on all labelled documents). VADER is evaluated on the
same documents for comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from midterm.config import ROOT, TAB
from midterm.nlp.issue_classifier import _features, _fit_transform, _transform

A = ROOT / "data" / "annotations"


def _vader_class(x):
    return np.where(x >= 0.05, 1, np.where(x <= -0.05, -1, 0))


def train_and_evaluate(seed: int = 20261103) -> dict:
    tr = pd.read_csv(A / "reference_labels_train.csv")
    ev = pd.concat([pd.read_csv(A / "reference_labels_news.csv").assign(set="news", text=lambda d: d["title"]),
                    pd.read_csv(A / "reference_labels_reddit.csv").assign(set="reddit")], ignore_index=True)
    word, char = _features()
    X = _fit_transform(word, char, tr["text"].astype(str))
    grid = GridSearchCV(LogisticRegression(max_iter=4000), {"C": [0.3, 1, 3, 10, 30]}, scoring="f1_macro",
                        cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=-1)
    grid.fit(X, tr["sentiment_label"])
    clf = grid.best_estimator_
    sia = SentimentIntensityAnalyzer()
    rows = []
    for name, g in ev.groupby("set"):
        y = g["sentiment_label"].to_numpy()
        Pe = clf.predict_proba(_transform(word, char, g["text"].astype(str)))
        cls = list(clf.classes_)
        score = Pe[:, cls.index(1)] - Pe[:, cls.index(-1)]
        pred = np.array(cls)[Pe.argmax(1)]
        v = g["text"].astype(str).map(lambda t: sia.polarity_scores(t)["compound"]).to_numpy()
        vc = _vader_class(v)
        zs = (score - score.mean()) / score.std()
        zv = (v - v.mean()) / v.std()
        for wgt in (0.25, 0.5, 0.75):
            e = wgt * zs + (1 - wgt) * zv
            rows.append({"eval_set": name, "model": f"ensemble_w{wgt}", "n": len(g), "C": np.nan,
                         "spearman": stats.spearmanr(e, y)[0], "mean_score": float(e.mean()), "mean_label": float(y.mean())})
        rows += [
            {"eval_set": name, "model": "supervised_LR", "n": len(g), "C": grid.best_params_["C"],
             "spearman": stats.spearmanr(score, y)[0], "acc_3class": accuracy_score(y, pred),
             "macro_f1": f1_score(y, pred, average="macro"), "kappa": cohen_kappa_score(y, pred),
             "share_neg_pred": float((pred == -1).mean()), "share_neg_true": float((y == -1).mean()),
             "mean_score": float(score.mean()), "mean_label": float(y.mean())},
            {"eval_set": name, "model": "VADER", "n": len(g), "C": np.nan,
             "spearman": stats.spearmanr(v, y)[0], "acc_3class": accuracy_score(y, vc),
             "macro_f1": f1_score(y, vc, average="macro"), "kappa": cohen_kappa_score(y, vc),
             "share_neg_pred": float((vc == -1).mean()), "share_neg_true": float((y == -1).mean()),
             "mean_score": float(v.mean()), "mean_label": float(y.mean())},
        ]
    evaluation = pd.DataFrame(rows)
    evaluation.to_csv(TAB / "sentiment_model_evaluation.csv", index=False)
    allv = pd.concat([tr[["text", "sentiment_label"]], ev[["text", "sentiment_label"]]], ignore_index=True)
    w2, c2 = _features()
    final = LogisticRegression(max_iter=4000, C=grid.best_params_["C"]).fit(
        _fit_transform(w2, c2, allv["text"].astype(str)), allv["sentiment_label"])
    return {"evaluation": evaluation, "model": final, "word": w2, "char": c2}


def score(docs: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    """Columns: sent_lr, vader, sent (equal-weight ensemble of z-scores)."""
    clf = fitted["model"]
    P = clf.predict_proba(_transform(fitted["word"], fitted["char"], docs["text"].astype(str)))
    cls = list(clf.classes_)
    lr = P[:, cls.index(1)] - P[:, cls.index(-1)]
    sia = SentimentIntensityAnalyzer()
    v = docs["text"].astype(str).map(lambda t: sia.polarity_scores(t)["compound"]).to_numpy()
    z = lambda a: (a - a.mean()) / a.std()
    return pd.DataFrame({"sent_lr": lr, "vader": v, "sent": 0.5 * z(lr) + 0.5 * z(v)}, index=docs.index)

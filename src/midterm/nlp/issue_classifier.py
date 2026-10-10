"""Codebook-based topic discovery and supervised scaling of voter issues.

Why this module exists
----------------------
Unsupervised NMF on the corpus (``midterm.nlp.topics``) shows that election
discourse is dominated by *process* content (races, primaries, candidates,
results); issue content is spread thinly over many short texts and NMF does
not isolate several important issues (health care, cost of living, energy
prices) as separate topics, even after removing process vocabulary (stage-2
mean NPMI < 0). The workflow has three stages: (i) issue categories are
*discovered* by reading stage-1 NMF topics and open-coding random documents;
(ii) researchers review and confirm the codebook-based reference labels;
(iii) a transparent supervised model scales the labels to the full corpus.

Annotation sets (``data/annotations``)
* ``reference_labels_train.csv`` (1,220 docs): 500 random + 720 keyword-retrieved
  documents (retrieval only enriches rare issues; every label comes from
  reading the text, and 1/3 of retrieved documents were labelled NONE or a
  different issue);
* ``reference_labels_news.csv`` (264) and ``reference_labels_reddit.csv`` (304): purely
  random, stratified evaluation sets, never used for training.

Model
* features: TF-IDF of word 1-2-grams and character 3-5-grams (sublinear tf);
* classifier: multinomial logistic regression (L2, C chosen by 5-fold CV on
  the training set, scoring = macro-F1);
* evaluation on the random hold-out sets: accuracy, macro-F1, issue-vs-none
  accuracy and prevalence error;
* prevalence: probabilistic classify-and-count (PCC, mean posterior). The
  EM prior-shift correction of Saerens, Latinne & Decaestecker (2002) was
  also evaluated because the training sample over-represents rare issues by
  design; on the random hold-out sets it *increased* prevalence error
  (posteriors of a high-C logistic model are not calibrated enough for EM),
  so PCC is the reported estimator and EM priors are kept as a diagnostic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from midterm.config import INTERIM, ROOT, TAB

A = ROOT / "data" / "annotations"
MERGE = {"HOUSING": "COST"}            # 15 training examples -> folded into cost of living
NONISSUE = {"NONE"}
ISSUE_NAMES = {
    "COST": "Cost of living & economy", "ENERGY": "Gas & energy prices", "HEALTH": "Health care costs & coverage",
    "IMMIG": "Immigration & ICE enforcement", "IRAN": "Iran war & Middle East", "TARIFF": "Tariffs & trade",
    "DEMOC": "Democracy, voting rules & courts", "EPSTEIN": "Epstein files", "ABORT": "Abortion",
    "CRIME": "Crime, guns & public safety", "AI": "AI & data centers", "CLIMATE": "Climate & environment",
    "GOVT": "Taxes, spending & shutdown", "SOCIAL": "Culture war (trans, race, DEI)",
    "FOREIGN": "Other foreign policy (Ukraine, Venezuela, Greenland, China)",
    "TRUMP": "Trump approval & conduct", "OTHER_ISSUE": "Other policy issues",
}


def _load(name: str) -> pd.DataFrame:
    d = pd.read_csv(A / f"reference_labels_{name}.csv")
    d["y"] = d["issue_label"].replace(MERGE)
    return d


def _features():
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True, max_features=60000,
                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'\-]+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=80000)
    return word, char


def _fit_transform(word, char, texts):
    return sparse.hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()


def _transform(word, char, texts):
    return sparse.hstack([word.transform(texts), char.transform(texts)]).tocsr()


def em_prior_shift(P: np.ndarray, train_prior: np.ndarray, tol: float = 1e-6, max_iter: int = 1000):
    """Saerens et al. (2002): re-estimate class priors on unlabelled data and
    adjust posteriors accordingly. Returns (adjusted posteriors, new priors)."""
    prior = train_prior.copy()
    for _ in range(max_iter):
        w = P * (prior / train_prior)
        Q = w / w.sum(axis=1, keepdims=True)
        new = Q.mean(axis=0)
        if np.abs(new - prior).max() < tol:
            prior = new
            break
        prior = new
    return Q, prior


def train_and_evaluate(seed: int = 20261103) -> dict:
    tr = _load("train")
    ev = pd.concat([_load("news").assign(set="news", text=lambda d: d["title"]),
                    _load("reddit").assign(set="reddit")], ignore_index=True)
    word, char = _features()
    X = _fit_transform(word, char, tr["text"].astype(str))
    grid = GridSearchCV(LogisticRegression(max_iter=4000), {"C": [1, 3, 10, 30]}, scoring="f1_macro",
                        cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=-1)
    grid.fit(X, tr["y"])
    clf = grid.best_estimator_
    classes = clf.classes_
    train_prior = tr["y"].value_counts(normalize=True).reindex(classes).to_numpy()
    rows = []
    for name, g in ev.groupby("set"):
        Xe = _transform(word, char, g["text"].astype(str))
        P = clf.predict_proba(Xe)
        Q, prior = em_prior_shift(P, train_prior)
        pred_raw = classes[P.argmax(1)]
        pred = classes[Q.argmax(1)]
        y = g["y"].to_numpy()
        true_share = pd.Series(y).value_counts(normalize=True).reindex(classes, fill_value=0).to_numpy()
        is_issue_true = ~np.isin(y, list(NONISSUE))
        rows.append({
            "eval_set": name, "n": len(g), "C": grid.best_params_["C"], "cv_macro_f1_train": grid.best_score_,
            "acc_raw": accuracy_score(y, pred_raw), "acc_em": accuracy_score(y, pred),
            "macro_f1_em": f1_score(y, pred, average="macro"),
            "issue_vs_none_acc_raw": accuracy_score(is_issue_true, ~np.isin(pred_raw, list(NONISSUE))),
            "issue_vs_none_acc_em": accuracy_score(is_issue_true, ~np.isin(pred, list(NONISSUE))),
            "issue_share_true": is_issue_true.mean(),
            "issue_share_cc_raw": (~np.isin(pred_raw, list(NONISSUE))).mean(),
            "issue_share_pcc_raw": 1 - P[:, list(classes).index("NONE")].mean(),
            "issue_share_em": 1 - prior[list(classes).index("NONE")],
            "prevalence_mae_raw_pcc_pp": 100 * np.abs(P.mean(0) - true_share).mean(),
            "prevalence_mae_em_pp": 100 * np.abs(prior - true_share).mean(),
        })
    evaluation = pd.DataFrame(rows)
    evaluation.to_csv(TAB / "issue_classifier_evaluation.csv", index=False)
    # final model on all labelled data (train + evaluation sets)
    allv = pd.concat([tr[["text", "y"]], ev[["text", "y"]]], ignore_index=True)
    word_f, char_f = _features()
    Xa = _fit_transform(word_f, char_f, allv["text"].astype(str))
    final = LogisticRegression(max_iter=4000, C=grid.best_params_["C"]).fit(Xa, allv["y"])
    return {"evaluation": evaluation, "model": final, "word": word_f, "char": char_f,
            "train_prior": allv["y"].value_counts(normalize=True).reindex(final.classes_).to_numpy()}


def apply(docs: pd.DataFrame, fitted: dict) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Posterior issue probabilities for every document; per-stream prevalence
    by PCC (reported) and by EM prior-shift (diagnostic)."""
    clf, classes = fitted["model"], list(fitted["model"].classes_)
    X = _transform(fitted["word"], fitted["char"], docs["text"].astype(str))
    P = clf.predict_proba(X)
    pcc, em = {}, {}
    streams = {"news": docs["source"].eq("news")}
    for s in ("left", "center", "right"):
        streams[f"reddit_{s}"] = docs["source"].eq("reddit") & docs["lean"].eq(s)
    for name, m in streams.items():
        m = m.to_numpy()
        pcc[name] = P[m].mean(axis=0)
        em[name] = em_prior_shift(P[m], fitted["train_prior"])[1]
    prev = pd.DataFrame(pcc, index=classes)
    prev.to_csv(TAB / "issue_prevalence_pcc_by_stream.csv")
    pd.DataFrame(em, index=classes).to_csv(TAB / "issue_prevalence_em_by_stream_diagnostic.csv")
    np.save(INTERIM / "doc_issue_posterior.npy", P)
    pd.Series(classes).to_csv(INTERIM / "issue_classes.csv", index=False)
    return P, classes, prev

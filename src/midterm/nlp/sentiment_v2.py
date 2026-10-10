"""Issue-sentiment measurement, version 2 (dense issue corpus + transformer features).

Labels (``data/annotations/labels_issue_*.csv``; guidelines in
``data/annotations/GUIDELINES_issue_v2.md``): for each sampled headline,
``rel`` in {0, 1} (about the issue as a US public matter) and ``tone`` in
{-2, ..., +2}.

Features: for each text, the 3 logits of the pretrained sentiment head and the
768-d mean-pooled embedding of ``cardiffnlp/twitter-roberta-base-sentiment-latest``
(``midterm.nlp.transformer``), plus the issue indicator.

Tone model — candidates, selected by 5-fold cross-validated Spearman on the
*training* sample only:
  zero_shot   s = P(pos) - P(neg) from the pretrained head (no fitting);
  calibrated  ridge regression of the label on the 3 logits;
  embedding   ridge regression on standardised [embedding, logits, issue dummies],
              penalty chosen by inner CV.
Relevance model: L2 logistic regression on standardised [embedding, issue
dummies], penalty by inner CV (log-loss); output P(rel = 1 | d) is used as the
document weight in the issue index.

The selected specifications are refitted on the training sample and evaluated
once on the random hold-out sample (``evaluate``). For scoring the corpus they
are refitted on train + hold-out (``fit_final``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import (brier_score_loss, cohen_kappa_score, log_loss, mean_absolute_error,
                             roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from midterm.config import ROOT, TAB
from midterm.nlp import transformer as TR

A = ROOT / "data" / "annotations"
ISSUES = ["DEMOC", "IRAN", "IMMIG", "COST", "TRUMP", "SOCIAL", "FOREIGN", "AI", "HEALTH", "TARIFF", "ENERGY"]
ALPHAS = np.logspace(-1, 4, 16)
SEED = 20261103


def load(name: str) -> pd.DataFrame:
    s = pd.read_csv(A / f"sample_issue_{name}.csv")
    lab = pd.read_csv(A / f"labels_issue_{name}.csv")
    d = s.merge(lab, on="item", how="left", validate="one_to_one")
    assert d[["rel", "tone"]].notna().all().all(), f"unlabelled items in {name}"
    return d


def design(texts: pd.Series, issues: pd.Series) -> dict[str, np.ndarray]:
    logits, emb = TR.encode(texts.astype(str).tolist())
    dummies = (issues.to_numpy()[:, None] == np.array(ISSUES)[None, :]).astype(np.float32)
    p = softmax(logits, axis=1)
    return {"logits": logits, "emb": emb, "dummies": dummies, "zero_shot": p[:, 2] - p[:, 0]}


def _X(f: dict, spec: str) -> np.ndarray:
    if spec == "calibrated":
        return f["logits"]
    if spec == "embedding":
        return np.hstack([f["emb"], f["logits"], f["dummies"]])
    if spec == "relevance":
        return np.hstack([f["emb"], f["dummies"]])
    raise ValueError(spec)


def _tone_model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def _rel_model():
    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-4, 1, 11), cv=5, scoring="neg_log_loss",
                                                               max_iter=5000, random_state=SEED))


def predict_tone(spec: str, model, f: dict) -> np.ndarray:
    return f["zero_shot"] if spec == "zero_shot" else model.predict(_X(f, spec))


def select_tone(tr: pd.DataFrame, f: dict) -> pd.DataFrame:
    """5-fold CV on the training sample; returns mean/sd of fold Spearman per spec."""
    y = tr["tone"].to_numpy(float)
    rows = []
    for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(y)):
        fa = {key: v[a] for key, v in f.items()}
        fb = {key: v[b] for key, v in f.items()}
        for spec in ("zero_shot", "calibrated", "embedding"):
            m = None if spec == "zero_shot" else _tone_model().fit(_X(fa, spec), y[a])
            rows.append({"fold": k, "spec": spec, "spearman": stats.spearmanr(predict_tone(spec, m, fb), y[b])[0]})
    cv = pd.DataFrame(rows).groupby("spec")["spearman"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    return cv


def _tone_metrics(yhat: np.ndarray, y: np.ndarray, on_label_scale: bool) -> dict:
    out = {"n": len(y), "spearman": stats.spearmanr(yhat, y)[0], "pearson": stats.pearsonr(yhat, y)[0],
           "mean_pred": float(yhat.mean()), "mean_label": float(y.mean())}
    if on_label_scale:
        cls = np.clip(np.rint(yhat), -2, 2)
        c3 = lambda v: np.sign(np.where(np.abs(v) < 0.5, 0, v))
        out.update({"mae": mean_absolute_error(y, yhat), "kappa_5_quadratic": cohen_kappa_score(y, cls, weights="quadratic"),
                    "acc_3class": float((c3(yhat) == np.sign(y)).mean()),
                    "kappa_3class": cohen_kappa_score(np.sign(y), c3(yhat))})
    return out


def evaluate(tr: pd.DataFrame, ev: pd.DataFrame, ftr: dict, fev: dict, spec: str) -> dict:
    ytr, yev = tr["tone"].to_numpy(float), ev["tone"].to_numpy(float)
    rows = []
    for s in ("zero_shot", "calibrated", "embedding"):
        m = None if s == "zero_shot" else _tone_model().fit(_X(ftr, s), ytr)
        yhat = predict_tone(s, m, fev)
        rows.append({"spec": s, "selected": s == spec, "issue": "ALL", **_tone_metrics(yhat, yev, s != "zero_shot")})
        if s == spec:
            for k, g in ev.groupby("issue"):
                ix = g.index.to_numpy()
                rows.append({"spec": s, "selected": True, "issue": k, **_tone_metrics(yhat[ix], yev[ix], s != "zero_shot")})
    tone = pd.DataFrame(rows)
    rm = _rel_model().fit(_X(ftr, "relevance"), tr["rel"])
    p = rm.predict_proba(_X(fev, "relevance"))[:, 1]
    r = ev["rel"].to_numpy()
    rel = [{"issue": "ALL", "n": len(r), "auc": roc_auc_score(r, p), "brier": brier_score_loss(r, p), "log_loss": log_loss(r, p),
            "acc_at_0.5": float(((p >= .5) == r).mean()), "mean_pred": float(p.mean()), "mean_label": float(r.mean())}]
    for k, g in ev.groupby("issue"):
        ix = g.index.to_numpy()
        rel.append({"issue": k, "n": len(ix), "auc": roc_auc_score(r[ix], p[ix]) if 0 < r[ix].sum() < len(ix) else np.nan,
                    "brier": brier_score_loss(r[ix], p[ix]), "acc_at_0.5": float(((p[ix] >= .5) == r[ix]).mean()),
                    "mean_pred": float(p[ix].mean()), "mean_label": float(r[ix].mean())})
    rel = pd.DataFrame(rel)
    tone.to_csv(TAB / "sentiment_v2_tone_holdout.csv", index=False)
    rel.to_csv(TAB / "sentiment_v2_relevance_holdout.csv", index=False)
    return {"tone": tone, "relevance": rel}


def zero_shot_score(texts: pd.Series) -> np.ndarray:
    """P(pos) - P(neg) of the pretrained head; no fitting, no issue information."""
    logits, _ = TR.encode(texts.astype(str).tolist())
    p = softmax(logits, axis=1)
    return p[:, 2] - p[:, 0]


def legacy_corpus_eval() -> pd.DataFrame:
    """Zero-shot transformer score vs the v1 annotations of the election corpus
    (random hold-outs: news 264, Reddit 304; labels -1/0/+1)."""
    rows = []
    for name, col in (("news", "title"), ("reddit", "text")):
        d = pd.read_csv(A / f"reference_labels_{name}.csv")
        s = zero_shot_score(d[col].fillna(d.get("title", "")))
        rows.append({"eval_set": name, "model": "transformer_zero_shot", "n": len(d),
                     "spearman": stats.spearmanr(s, d["sentiment_label"])[0]})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "sentiment_zero_shot_legacy_holdout.csv", index=False)
    return out


def legacy_baseline(ev: pd.DataFrame) -> pd.DataFrame:
    """The v1 instrument (supervised LR + VADER ensemble, ``midterm.nlp.sentiment_model``)
    evaluated on the v2 hold-out, for a like-for-like comparison."""
    from midterm.nlp import sentiment_model as SMOD
    fitted = SMOD.train_and_evaluate()
    sc = SMOD.score(ev[["text"]], fitted)
    y = ev["tone"].to_numpy(float)
    rows = [{"spec": name, "n": len(y), "spearman": stats.spearmanr(sc[col], y)[0], "pearson": stats.pearsonr(sc[col], y)[0]}
            for name, col in (("v1_ensemble", "sent"), ("v1_supervised_LR", "sent_lr"), ("VADER", "vader"))]
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "sentiment_v2_legacy_baseline_holdout.csv", index=False)
    return out


def aggregate_validity(tr: pd.DataFrame, ev: pd.DataFrame, ftr: dict, fev: dict, spec: str,
                       dates: pd.Series, start: str) -> dict:
    """Do document errors cancel in time aggregates?

    Predictions without in-sample leakage: 5-fold out-of-fold predictions for the
    training sample and predictions of the train-fitted model for the hold-out.
    Labels and predictions (restricted to rel = 1 documents, as in the index) are
    averaged by 4-week block (pooled over issues) and by issue x half of the study
    window; the correlation of label means with prediction means measures the
    validity of the aggregated measure.
    """
    ytr = tr["tone"].to_numpy(float)
    oof = np.empty(len(tr))
    for a, b in KFold(5, shuffle=True, random_state=SEED).split(ytr):
        fa = {k: v[a] for k, v in ftr.items()}
        fb = {k: v[b] for k, v in ftr.items()}
        oof[b] = predict_tone(spec, None if spec == "zero_shot" else _tone_model().fit(_X(fa, spec), ytr[a]), fb)
    m = None if spec == "zero_shot" else _tone_model().fit(_X(ftr, spec), ytr)
    both = pd.concat([tr.assign(pred=oof), ev.assign(pred=predict_tone(spec, m, fev))], ignore_index=True)
    both["date"] = both["doc_id"].map(dates)
    both = both[both["rel"] == 1]
    both["block"] = ((both["date"] - pd.Timestamp(start)).dt.days // 28).astype(int)
    mid = pd.Timestamp(start) + (both["date"].max() - pd.Timestamp(start)) / 2
    both["half"] = np.where(both["date"] < mid, "H1", "H2")
    blk = both.groupby("block").agg(n=("tone", "size"), label=("tone", "mean"), pred=("pred", "mean"))
    ih = both.groupby(["issue", "half"]).agg(n=("tone", "size"), label=("tone", "mean"), pred=("pred", "mean"))
    chg = ih.unstack("half")
    d_label = chg[("label", "H2")] - chg[("label", "H1")]
    d_pred = chg[("pred", "H2")] - chg[("pred", "H1")]
    out = {"block_corr": float(blk[["label", "pred"]].corr().iloc[0, 1]), "n_blocks": len(blk),
           "issue_half_corr": float(ih[["label", "pred"]].corr().iloc[0, 1]), "n_issue_half": len(ih),
           "issue_change_corr": float(np.corrcoef(d_label, d_pred)[0, 1]),
           "issue_change_sign_agreement": float((np.sign(d_label) == np.sign(d_pred)).mean())}
    blk.to_csv(TAB / "sentiment_v2_aggregate_validity_blocks.csv")
    ih.to_csv(TAB / "sentiment_v2_aggregate_validity_issue_half.csv")
    pd.Series(out).to_csv(TAB / "sentiment_v2_aggregate_validity_summary.csv")
    return out


def retest(tr: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Test-retest agreement of the annotation (first pass vs blind second pass)."""
    first = pd.concat([tr, ev])[["doc_id", "rel", "tone"]]
    rt = load("retest")[["doc_id", "issue", "rel", "tone"]].merge(first, on="doc_id", suffixes=("_2", "_1"), validate="one_to_one")
    out = pd.DataFrame([{
        "n": len(rt), "rel_agreement": float((rt.rel_1 == rt.rel_2).mean()), "rel_kappa": cohen_kappa_score(rt.rel_1, rt.rel_2),
        "tone_exact": float((rt.tone_1 == rt.tone_2).mean()), "tone_within_1": float(((rt.tone_1 - rt.tone_2).abs() <= 1).mean()),
        "tone_kappa_quadratic": cohen_kappa_score(rt.tone_1, rt.tone_2, weights="quadratic"),
        "tone_spearman": stats.spearmanr(rt.tone_1, rt.tone_2)[0],
        "tone_kappa_3class": cohen_kappa_score(np.sign(rt.tone_1), np.sign(rt.tone_2)),
    }])
    out.to_csv(TAB / "annotation_retest_agreement.csv", index=False)
    return out


def fit_final(spec: str, labelled: pd.DataFrame, f: dict) -> dict:
    tone = None if spec == "zero_shot" else _tone_model().fit(_X(f, spec), labelled["tone"].to_numpy(float))
    rel = _rel_model().fit(_X(f, "relevance"), labelled["rel"])
    return {"spec": spec, "tone": tone, "rel": rel}


def score(fitted: dict, texts: pd.Series, issues: pd.Series) -> pd.DataFrame:
    f = design(texts, issues)
    return pd.DataFrame({"tone": predict_tone(fitted["spec"], fitted["tone"], f),
                         "p_rel": fitted["rel"].predict_proba(_X(f, "relevance"))[:, 1],
                         "zero_shot": f["zero_shot"]}, index=texts.index)


def human_validation(scored: pd.DataFrame, sheet=A / "human_validation_sheet.csv", B: int = 2000) -> pd.DataFrame | None:
    """Agreement of an independent human coder (110 hold-out items, filled in
    ``human_validation_sheet.csv`` under the same guidelines, without access to the
    annotation labels or model scores) with (a) the annotation labels and (b) the
    model. Returns None while the sheet is unfilled. 95% CIs by case bootstrap."""
    h = pd.read_csv(sheet)
    h = h[pd.to_numeric(h["rel"], errors="coerce").notna() & pd.to_numeric(h["tone"], errors="coerce").notna()]
    if h.empty:
        return None
    h = h.astype({"rel": int, "tone": int})
    lab = pd.read_csv(A / "labels_issue_eval.csv")
    ids = pd.read_csv(A / "sample_issue_eval.csv")[["item", "doc_id"]]
    d = (h[["item", "rel", "tone"]].merge(lab, on="item", suffixes=("_h", "_a"), validate="one_to_one")
         .merge(ids, on="item").merge(scored[["doc_id", "tone", "p_rel"]], on="doc_id", validate="one_to_one"))
    stats_fn = {
        "rel_kappa_human_vs_labels": lambda x: cohen_kappa_score(x.rel_h, x.rel_a),
        "tone_exact_human_vs_labels": lambda x: float((x.tone_h == x.tone_a).mean()),
        "tone_within1_human_vs_labels": lambda x: float(((x.tone_h - x.tone_a).abs() <= 1).mean()),
        "tone_kappa_quadratic_human_vs_labels": lambda x: cohen_kappa_score(x.tone_h, x.tone_a, weights="quadratic"),
        "tone_spearman_human_vs_labels": lambda x: stats.spearmanr(x.tone_h, x.tone_a)[0],
        "tone_spearman_human_vs_model": lambda x: stats.spearmanr(x.tone_h, x.tone)[0],
        "tone_spearman_labels_vs_model_same_items": lambda x: stats.spearmanr(x.tone_a, x.tone)[0],
        "rel_auc_human_vs_model": lambda x: roc_auc_score(x.rel_h, x.p_rel),
    }
    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in stats_fn}
    for _ in range(B):
        x = d.iloc[rng.integers(0, len(d), len(d))]
        for k, f in stats_fn.items():
            boot[k].append(f(x))
    out = pd.DataFrame([{"statistic": k, "n": len(d), "value": f(d), "ci_lo": np.nanpercentile(boot[k], 2.5),
                         "ci_hi": np.nanpercentile(boot[k], 97.5)} for k, f in stats_fn.items()])
    out = pd.concat([out, pd.DataFrame([
        {"statistic": "mean_tone_human", "n": len(d), "value": float(d.tone_h.mean())},
        {"statistic": "mean_tone_labels", "n": len(d), "value": float(d.tone_a.mean())},
        {"statistic": "rel_share_human", "n": len(d), "value": float(d.rel_h.mean())},
        {"statistic": "rel_share_labels", "n": len(d), "value": float(d.rel_a.mean())}])], ignore_index=True)
    out.to_csv(TAB / "human_validation_agreement.csv", index=False)
    return out

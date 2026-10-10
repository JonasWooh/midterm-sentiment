"""Topic modelling of the joint news + social corpus (two stages).

Stage 1 (``fit_select``): NMF on the full corpus. Election coverage is dominated
by horse-race vocabulary (races, primaries, candidates, results), so stage-1
topics are mostly *process* topics; they are classified issue/process in
``config/topic_labels_stage1.yaml``.

Stage 2 (``fit_issue_model``): vocabulary "deflation". For every term w the
share of its stage-1 loading mass that sits in process topics is
p_w = sum_{k in process} H_kw / sum_k H_kw. Terms with p_w >= ``PROCESS_CUT``
(0.8) are removed and NMF is refitted on the remaining vocabulary (documents
with no remaining terms drop out naturally). The resulting topics describe
what the discourse is *about* rather than which race it concerns; they are
mapped to issues in ``config/topic_labels.yaml``. K is again selected by NPMI.

Model: Non-negative Matrix Factorisation of a TF-IDF document-term matrix
(Frobenius loss, NNDSVDa initialisation). NMF is preferred to LDA for short
texts (headlines, post titles) because it does not rely on within-document
word-count statistics; LDA is fitted with the same K as a robustness check and
aligned to the NMF topics by Hungarian matching on top-word overlap.

Model selection: K is chosen from ``topics.n_topics_grid`` by maximising the
mean NPMI coherence of each topic's top-10 words, where NPMI is computed from
document co-occurrence over the *whole* corpus (Bouma 2009; Lau et al. 2014).
Topic diversity (share of unique words among all top-25 lists) and seed
stability (mean best-match Jaccard of top-10 lists across 3 random seeds) are
reported alongside.

The fitting corpus is balanced (equal numbers of news and Reddit documents,
drawn without replacement with a fixed seed) so neither source dictates the
topic space; all documents are then projected with ``transform``.
"""
from __future__ import annotations

import json
import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

from midterm.config import INTERIM, PROCESSED, TAB, settings

# Query terms appear by construction in the news sample and carry no topical
# information; generic discourse words are removed for the same reason.
DOMAIN_STOP = {
    "midterm", "midterms", "election", "elections", "voter", "voters", "poll", "polls", "polling", "survey",
    "surveys", "concerns", "concern", "issue", "issues", "says", "say", "said", "new", "year", "years", "week",
    "weeks", "day", "days", "just", "like", "people", "think", "know", "get", "got", "going", "want", "make",
    "really", "would", "could", "should", "also", "one", "two", "us", "u", "s", "don", "doesn", "didn", "isn",
    "im", "ive", "it's", "dont", "thing", "things", "way", "lot", "good", "time", "right", "need", "does",
    "did", "news", "report", "reports", "according", "amid", "vs", "video", "watch", "live", "update", "updates",
    "opinion", "analysis", "via", "gt", "amp", "nbsp", "000", "2025", "2026", "2024", "2027", "2028",
    "reddit", "post", "posts", "thread", "discussion", "question", "questions", "sub", "subreddit", "mod", "mods",
    "rule", "rules", "comment", "comments", "edit", "removed", "deleted", "ask", "asking", "conservatives",
    "liberals", "conservative", "liberal",
}
STOP = sorted(set(ENGLISH_STOP_WORDS) | DOMAIN_STOP)
TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z\-']{1,}[a-zA-Z]\b"


def _vectorizer(min_df: int, max_df: float) -> TfidfVectorizer:
    return TfidfVectorizer(stop_words=STOP, token_pattern=TOKEN, lowercase=True, ngram_range=(1, 2),
                           min_df=min_df, max_df=max_df, sublinear_tf=True, max_features=30000)


def npmi_coherence(top_idx: list[np.ndarray], Xbin: sparse.csc_matrix, eps: float = 1e-12) -> np.ndarray:
    """Mean pairwise NPMI of each topic's top words (document co-occurrence)."""
    n = Xbin.shape[0]
    df = np.asarray(Xbin.sum(axis=0)).ravel() / n
    out = []
    for idx in top_idx:
        sub = Xbin[:, idx]
        co = (sub.T @ sub).toarray() / n
        vals = []
        for a, b in combinations(range(len(idx)), 2):
            pab = co[a, b]
            if pab <= 0:
                vals.append(-1.0)
                continue
            pmi = np.log(pab / (df[idx[a]] * df[idx[b]] + eps))
            vals.append(pmi / (-np.log(pab) + eps))
        out.append(float(np.mean(vals)))
    return np.array(out)


def _top(H: np.ndarray, k: int) -> list[np.ndarray]:
    return [np.argsort(-row)[:k] for row in H]


def _jaccard_match(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    sa = [set(x) for x in a]
    sb = [set(x) for x in b]
    J = np.array([[len(x & y) / len(x | y) for y in sb] for x in sa])
    r, c = linear_sum_assignment(-J)
    return float(J[r, c].mean())


def fit_select(docs: pd.DataFrame) -> dict:
    cfg = settings()["topics"]
    rs = cfg["random_state"]
    rng = np.random.default_rng(rs)
    news_idx = docs.index[docs["source"] == "news"].to_numpy()
    red_idx = docs.index[docs["source"] == "reddit"].to_numpy()
    m = min(len(news_idx), len(red_idx))
    fit_idx = np.concatenate([rng.choice(news_idx, m, replace=False), rng.choice(red_idx, m, replace=False)])
    vec = _vectorizer(cfg["min_df"], cfg["max_df"])
    X_fit = vec.fit_transform(docs.loc[fit_idx, "text"])
    X_all = vec.transform(docs["text"])
    vocab = np.array(vec.get_feature_names_out())
    Xbin = (X_all > 0).astype(np.float64).tocsc()

    grid = []
    models = {}
    for K in cfg["n_topics_grid"]:
        tops = []
        for seed in (rs, rs + 1, rs + 2):
            nmf = NMF(n_components=K, init="nndsvda" if seed == rs else "random", random_state=seed,
                      max_iter=600, alpha_W=0.0, alpha_H=0.0)
            nmf.fit(X_fit)
            tops.append(_top(nmf.components_, 10))
            if seed == rs:
                models[K] = nmf
        coh = npmi_coherence(tops[0], Xbin)
        top25 = _top(models[K].components_, 25)
        diversity = len(set(np.concatenate(top25))) / (25 * K)
        stab = np.mean([_jaccard_match(tops[0], tops[1]), _jaccard_match(tops[0], tops[2])])
        grid.append({"K": K, "npmi_mean": coh.mean(), "npmi_min": coh.min(), "diversity": diversity,
                     "seed_stability": stab, "recon_err": models[K].reconstruction_err_})
    grid = pd.DataFrame(grid)
    grid.to_csv(TAB / "topic_model_selection.csv", index=False)
    K = int(grid.sort_values("npmi_mean", ascending=False).iloc[0]["K"])
    nmf = models[K]
    W = nmf.transform(X_all)
    H = nmf.components_

    # LDA robustness with same K on counts
    cv = CountVectorizer(vocabulary=vocab, token_pattern=TOKEN, ngram_range=(1, 2), stop_words=STOP)
    C_fit = cv.transform(docs.loc[fit_idx, "text"])
    lda = LatentDirichletAllocation(n_components=K, learning_method="batch", max_iter=50, random_state=rs)
    lda.fit(C_fit)
    lda_top = _top(lda.components_, 10)
    nmf_top = _top(H, 10)
    sa = [set(x) for x in nmf_top]
    sb = [set(x) for x in lda_top]
    Jm = np.array([[len(x & y) / len(x | y) for y in sb] for x in sa])
    r, c = linear_sum_assignment(-Jm)
    lda_align = pd.DataFrame({"nmf_topic": r, "lda_topic": c, "jaccard_top10": Jm[r, c],
                              "lda_top_words": [", ".join(vocab[lda_top[j]]) for j in c],
                              "lda_npmi": npmi_coherence([lda_top[j] for j in c], Xbin)})
    lda_align.to_csv(TAB / "topic_lda_alignment.csv", index=False)

    coh = npmi_coherence(nmf_top, Xbin)
    topics = pd.DataFrame({
        "topic": range(K),
        "top_words": [", ".join(vocab[i]) for i in _top(H, 15)],
        "npmi": coh,
    })
    return {"K": K, "W": W, "H": H, "vocab": vocab, "topics": topics, "grid": grid, "fit_n": len(fit_idx)}


PROCESS_CUT = 0.8


def fit_issue_model(docs: pd.DataFrame, H1: np.ndarray, vocab1: np.ndarray, stage1_kind: dict[int, str],
                    grid: list[int]) -> dict:
    cfg = settings()["topics"]
    rs = cfg["random_state"]
    proc = np.array([stage1_kind[k] == "process" for k in range(H1.shape[0])])
    mass = H1.sum(axis=0)
    p_w = np.divide(H1[proc].sum(axis=0), mass, out=np.zeros_like(mass), where=mass > 0)
    keep = vocab1[p_w < PROCESS_CUT]
    removed = pd.DataFrame({"term": vocab1, "p_process": p_w, "loading_mass": mass}).sort_values("loading_mass", ascending=False)
    removed["removed"] = removed["p_process"] >= PROCESS_CUT
    removed.to_csv(TAB / "stage2_vocabulary_deflation.csv", index=False)
    vec = TfidfVectorizer(vocabulary=keep, token_pattern=TOKEN, lowercase=True, ngram_range=(1, 2),
                          stop_words=STOP, sublinear_tf=True)
    X_all = vec.fit_transform(docs["text"])
    has = np.asarray(X_all.sum(axis=1)).ravel() > 0
    rng = np.random.default_rng(rs)
    news_idx = np.where(has & docs["source"].eq("news").to_numpy())[0]
    red_idx = np.where(has & docs["source"].eq("reddit").to_numpy())[0]
    m = min(len(news_idx), len(red_idx))
    fit_idx = np.concatenate([rng.choice(news_idx, m, replace=False), rng.choice(red_idx, m, replace=False)])
    X_fit = X_all[fit_idx]
    vocab = np.array(vec.get_feature_names_out())
    Xbin = (X_all[has] > 0).astype(np.float64).tocsc()
    rows, models = [], {}
    for K in grid:
        tops = []
        for seed in (rs, rs + 1, rs + 2):
            nmf = NMF(n_components=K, init="nndsvda" if seed == rs else "random", random_state=seed, max_iter=600)
            nmf.fit(X_fit)
            tops.append(_top(nmf.components_, 10))
            if seed == rs:
                models[K] = nmf
        coh = npmi_coherence(tops[0], Xbin)
        top25 = _top(models[K].components_, 25)
        rows.append({"K": K, "npmi_mean": coh.mean(), "npmi_min": coh.min(),
                     "diversity": len(set(np.concatenate(top25))) / (25 * K),
                     "seed_stability": np.mean([_jaccard_match(tops[0], tops[1]), _jaccard_match(tops[0], tops[2])]),
                     "recon_err": models[K].reconstruction_err_})
    g = pd.DataFrame(rows)
    g.to_csv(TAB / "topic_model_selection_stage2.csv", index=False)
    K = int(g.sort_values("npmi_mean", ascending=False).iloc[0]["K"])
    nmf = models[K]
    W = nmf.transform(X_all)
    H = nmf.components_
    topics = pd.DataFrame({"topic": range(K), "top_words": [", ".join(vocab[i]) for i in _top(H, 15)],
                           "npmi": npmi_coherence(_top(H, 10), Xbin)})
    return {"K": K, "W": W, "H": H, "vocab": vocab, "topics": topics, "grid": g, "fit_n": len(fit_idx),
            "n_removed_terms": int((p_w >= PROCESS_CUT).sum()), "n_kept_terms": int(len(keep)),
            "docs_with_issue_vocab": int(has.sum())}


def representative_docs(docs: pd.DataFrame, W: np.ndarray, n: int = 8) -> dict[int, list[str]]:
    Wn = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    rep = {}
    for k in range(W.shape[1]):
        score = Wn[:, k] * (W[:, k] > np.quantile(W[:, k], 0.5))
        idx = np.argsort(-score)[: n * 3]
        seen, out = set(), []
        for i in idx:
            t = docs.at[i, "title"][:160]
            if t.lower() in seen:
                continue
            seen.add(t.lower())
            out.append(f"[{docs.at[i, 'source']}/{docs.at[i, 'sub']}] {t}")
            if len(out) == n:
                break
        rep[k] = out
    return rep


def run() -> dict:
    import yaml
    from midterm.config import ROOT
    docs = pd.read_parquet(INTERIM / "docs.parquet")
    res = fit_select(docs)
    np.save(INTERIM / "stage1_W.npy", res["W"])
    np.save(INTERIM / "stage1_H.npy", res["H"])
    rep = representative_docs(docs, res["W"])
    res["topics"]["examples"] = [" || ".join(rep[k]) for k in res["topics"]["topic"]]
    res["topics"].to_csv(TAB / "topics_raw.csv", index=False)
    lab1 = ROOT / "config" / "topic_labels_stage1.yaml"
    if not lab1.exists():
        return {"stage1": res}
    kinds = {int(k): v["kind"] for k, v in yaml.safe_load(lab1.read_text())["topics"].items()}
    assert len(kinds) == res["K"], "stage-1 labels do not match the fitted K"
    res2 = fit_issue_model(docs, res["H"], res["vocab"], kinds, settings()["topics"]["n_topics_grid_stage2"])
    np.save(INTERIM / "doc_topic_W.npy", res2["W"])
    np.save(INTERIM / "topic_word_H.npy", res2["H"])
    (INTERIM / "vocab.json").write_text(json.dumps(res2["vocab"].tolist()))
    rep2 = representative_docs(docs, res2["W"])
    res2["topics"]["examples"] = [" || ".join(rep2[k]) for k in res2["topics"]["topic"]]
    res2["topics"].to_csv(TAB / "topics_stage2_raw.csv", index=False)
    return {"stage1": res, "stage2": res2}


if __name__ == "__main__":
    r = run()
    print(r["grid"].round(3))
    pd.set_option("display.max_colwidth", 200)
    print(r["topics"][["topic", "npmi", "top_words"]])

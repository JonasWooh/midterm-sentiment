"""Draw the annotation samples (fixed seeds; run once after data/interim/docs.parquet exists).

* sample_news.csv        - evaluation set, news: 22 headlines per month (stratified random)
* sample_reddit_eval.csv - evaluation set, Reddit: 38 posts per community (stratified random)
* sample_train.csv       - training set for the issue classifier: 500 random documents
                           + up to 45 keyword-retrieved documents per candidate issue
                           (retrieval only enriches rare classes; every document is
                           labelled by reading it, and evaluation uses the random sets only)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
docs = pd.read_parquet(ROOT / "data/interim/docs.parquet")
A = ROOT / "data/annotations"

news = docs[docs.source == "news"]
if not (A / "sample_news.csv").exists():
    s = news.groupby(news.date.dt.to_period("M")).sample(n=22, random_state=20261103, replace=False)
    s.sample(frac=1, random_state=1)[["doc_id", "date", "title"]].to_csv(A / "sample_news.csv", index=False)
ev_news = pd.read_csv(A / "sample_news.csv")

red = docs[docs.source == "reddit"]
ev_red = red.groupby("sub").sample(n=38, random_state=11).sample(frac=1, random_state=2)
excl = set(ev_news.doc_id) | set(ev_red.doc_id)
pool = docs[~docs.doc_id.isin(excl)]
tr_rand = pool.sample(500, random_state=13)
SEEDS = {
    "COST": r"inflation|cost of living|afford|grocer|prices|economy|recession|jobs|wages|taxes?\b",
    "ENERGY": r"gas price|gasoline|oil price|electricity|energy cost|utility bill",
    "HOUSING": r"housing|rent\b|mortgage|home prices",
    "HEALTH": r"health|medicaid|medicare|obamacare|\baca\b|premium|insurance",
    "IMMIG": r"immigra|\bice\b|deport|border|migrant|asylum",
    "IRAN": r"\biran|israel|hormuz|gaza|middle east",
    "TARIFF": r"tariff|trade war",
    "DEMOC": r"redistrict|gerrymand|voting rights|mail-in|save act|democracy|insurrection act|authoritarian|third term|cancel the midterms|election integrity",
    "EPSTEIN": r"epstein",
    "ABORT": r"abortion|roe\b|reproductive",
    "CRIME": r"crime|national guard|gun|shooting|violence",
    "AI": r"\bai\b|artificial intelligence|data center",
    "CLIMATE": r"climate|environment|emissions",
    "GOVT": r"shutdown|budget|deficit|debt|doge|spending|social security|federal workers",
    "SOCIAL": r"transgender|\btrans\b|lgbt|dei\b|woke|racism",
    "FOREIGN": r"ukraine|russia|china|greenland|venezuela|nato",
}
low = pool.text.str.lower()
used = set(tr_rand.doc_id)
parts = []
for c, pat in SEEDS.items():
    m = pool[low.str.contains(pat, regex=True) & ~pool.doc_id.isin(used)]
    k = m.sample(min(45, len(m)), random_state=17)
    used |= set(k.doc_id)
    parts.append(k.assign(seed=c))
tr = pd.concat([tr_rand.assign(seed="random")] + parts).sample(frac=1, random_state=5)
ev_red[["doc_id", "date", "sub", "title", "text"]].to_csv(A / "sample_reddit_eval.csv", index=False)
tr[["doc_id", "source", "sub", "seed", "text"]].to_csv(A / "sample_train.csv", index=False)
print(len(ev_red), len(tr), tr.seed.value_counts().to_dict(), file=sys.stderr)

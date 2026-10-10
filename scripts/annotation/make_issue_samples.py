"""Draw the v2 annotation samples from the issue-targeted corpus.

    python scripts/annotation/make_issue_samples.py

* train  : per issue, 10 headlines from each 4-week block of the study window
           (time-stratified so that the tone model sees every phase, e.g. pre/post
           Iran war); blocks with fewer documents contribute all they have and the
           shortfall is not re-allocated (sample size is then < 120 for that issue);
* eval   : per issue, a simple random sample of 50 headlines not in train
           (representative of the corpus, so hold-out metrics are unbiased);
* retest : 25 per issue drawn at random from train + eval, shuffled, with fresh ids.

Files: data/annotations/sample_issue_{train,eval,retest}.csv (columns: item, doc_id, issue, text).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from midterm.config import INTERIM, ROOT, settings  # noqa: E402

A = ROOT / "data" / "annotations"
SEED = 20261103


def main() -> None:
    st = settings()["study"]
    d = pd.read_parquet(INTERIM / "issue_docs.parquet")
    start = pd.Timestamp(st["start"])
    d["block"] = ((d["date"] - start).dt.days // 28).astype(int)
    rng = np.random.default_rng(SEED)
    train = []
    for (k, b), g in d.groupby(["issue", "block"], sort=True):
        n = min(10, len(g))
        train.append(g.iloc[rng.choice(len(g), n, replace=False)])
    train = pd.concat(train)
    rest = d[~d["doc_id"].isin(train["doc_id"])]
    ev = pd.concat([g.iloc[rng.choice(len(g), min(50, len(g)), replace=False)] for _, g in rest.groupby("issue", sort=True)])
    both = pd.concat([train, ev])
    rt = pd.concat([g.iloc[rng.choice(len(g), 25, replace=False)] for _, g in both.groupby("issue", sort=True)])
    rt = rt.iloc[rng.permutation(len(rt))]
    cols = ["item", "doc_id", "issue", "text"]
    for name, df, pre in (("train", train, "T"), ("eval", ev, "E"), ("retest", rt, "R")):
        df = df.iloc[rng.permutation(len(df))] if name != "retest" else df
        df = df.assign(item=[f"{pre}{i:04d}" for i in range(1, len(df) + 1)])
        df[cols].to_csv(A / f"sample_issue_{name}.csv", index=False)
        print(name, len(df), df.groupby("issue").size().to_dict())


if __name__ == "__main__":
    main()

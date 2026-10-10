"""Independent human validation of the v2 annotations and model.

    python scripts/annotation/human_validation.py make    # writes the blank sheet
    python scripts/annotation/human_validation.py score   # after the sheet has been filled in

`make` draws 10 hold-out items per key issue (seeded) into
data/annotations/human_validation_sheet.csv with empty `rel` and `tone`
columns; a human coder fills them in following GUIDELINES_issue_v2.md without
looking at the reference labels or model scores.

`score` reports agreement of the human coder with the annotation labels (relevance
kappa; tone exact, within-1, quadratic-weighted kappa, Spearman) and with the model
(Spearman of the model tone score, AUC of the relevance probability), with bootstrap
CIs, into outputs/tables/human_validation_agreement.csv
(``midterm.nlp.sentiment_v2.human_validation``; also run by the pipeline).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from midterm.config import INTERIM, ROOT, TAB  # noqa: E402

A = ROOT / "data" / "annotations"
SHEET = A / "human_validation_sheet.csv"
SEED = 20261103


def make() -> None:
    if SHEET.exists() and pd.read_csv(SHEET)[["rel", "tone"]].notna().any().any():
        raise SystemExit(f"{SHEET} already contains annotations; refusing to overwrite")
    ev = pd.read_csv(A / "sample_issue_eval.csv")
    rng = np.random.default_rng(SEED)
    pick = pd.concat([g.iloc[rng.choice(len(g), 10, replace=False)] for _, g in ev.groupby("issue", sort=True)])
    pick = pick.iloc[rng.permutation(len(pick))]
    pick.assign(rel="", tone="")[["item", "issue", "text", "rel", "tone"]].to_csv(SHEET, index=False)
    print("wrote", SHEET, len(pick))


def score() -> None:
    from midterm.nlp.sentiment_v2 import human_validation
    out = human_validation(pd.read_parquet(INTERIM / "issue_docs_scored.parquet"))
    print("sheet not filled in yet" if out is None else out.round(3).to_string(index=False))


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]]()

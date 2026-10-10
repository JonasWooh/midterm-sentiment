"""Unit tests for the deterministic building blocks of the pipeline."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from midterm.analysis.stats import block_bootstrap_corr, ccf, hac_corr_test
from midterm.bridge.ingest import fnv1a32, reassemble
from midterm.equities.basket import market_model
from midterm.nlp.sentiment import shrink
from midterm.nlp.issue_classifier import em_prior_shift


def _js_chunks(payload: str, C: int):
    """Emulate bridge.js: slice by UTF-16 code units and pad to C units."""
    units = payload.encode("utf-16-le", "surrogatepass")
    n = len(units) // 2
    out = []
    for i in range(0, (n + C - 1) // C):
        part = units[2 * i * C: 2 * (i + 1) * C].decode("utf-16-le", "surrogatepass")
        part = part + " " * (C - len(part.encode("utf-16-le", "surrogatepass")) // 2)
        out.append(f"#CHUNK{i}#" + part)
    return out, n


def test_bridge_roundtrip_with_split_surrogate(tmp_path):
    obj = {"a": "x" * 9 + "\U0001F5F3" + "y" * 20, "b": [1, 2, 3], "c": 'quote " and \\ backslash'}
    payload = json.dumps(obj, ensure_ascii=False)
    chunks, n = _js_chunks(payload, C=10)          # emoji straddles the first boundary
    files = []
    for i, ch in enumerate(chunks):
        p = tmp_path / f"c{i}.txt"
        enc = json.dumps(ch)
        p.write_text(json.dumps([{"type": "text", "text": enc + "\n\n(captured at origin https://x)"}]))
        files.append(p)
    out = reassemble(files[::-1], n, fnv1a32(payload))
    assert json.loads(out) == obj


def test_bridge_detects_corruption(tmp_path):
    payload = json.dumps({"k": "abcdefghij" * 5})
    chunks, n = _js_chunks(payload, C=16)
    chunks[1] = chunks[1].replace("c", "C", 1)
    files = []
    for i, ch in enumerate(chunks):
        p = tmp_path / f"c{i}.txt"
        p.write_text(json.dumps([{"type": "text", "text": json.dumps(ch)}]))
        files.append(p)
    with pytest.raises(ValueError):
        reassemble(files, n, fnv1a32(payload))


def test_market_model_recovers_beta():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=400)
    m = rng.normal(0, 0.01, 400)
    r = 0.0002 + 1.3 * m + rng.normal(0, 0.005, 400)
    df = pd.DataFrame({"SPY": m, "X": r}, index=idx)
    mm = market_model(df, ["X"], "SPY", ("2024-01-01", "2025-12-31"))
    assert abs(mm.at["X", "beta"] - 1.3) < 0.08


def test_em_prior_shift_recovers_known_priors():
    """Posteriors from a calibrated model under training priors (0.5, 0.5) applied to a
    test set whose true priors are (0.8, 0.2): EM should recover ~0.8."""
    rng = np.random.default_rng(4)
    y = rng.random(20000) < 0.2
    x = rng.normal(np.where(y, 1.0, -1.0), 1.0)
    lik1, lik0 = np.exp(-(x - 1) ** 2 / 2), np.exp(-(x + 1) ** 2 / 2)
    p1 = lik1 / (lik1 + lik0)                              # posterior under equal priors
    P = np.column_stack([1 - p1, p1])
    _, prior = em_prior_shift(P, np.array([0.5, 0.5]))
    assert abs(prior[1] - 0.2) < 0.02


def test_shrinkage_pulls_sparse_weeks_to_prior():
    idx = pd.date_range("2026-01-02", periods=40, freq="W-FRI")
    rng = np.random.default_rng(1)
    n = pd.DataFrame({0: rng.integers(20, 40, 40).astype(float)}, index=idx)
    n.iloc[30, 0] = 0.5                                   # one sparse week
    m = pd.DataFrame({0: rng.normal(0.0, 0.3, 40)}, index=idx)
    m.iloc[30, 0] = 2.0                                   # extreme mean in the sparse week
    num = m * n
    sh, diag = shrink(num, n, pd.Series({0: 1.0}))
    assert abs(sh.iloc[30, 0]) < 0.5 * 2.0                # heavily shrunk
    dense = sh.drop(index=idx[30])[0] - m.drop(index=idx[30])[0]
    assert dense.abs().median() < 0.1                     # well-measured weeks barely move


def test_ccf_sign_convention():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(size=300))
    y = x.shift(2) + 0.1 * rng.normal(size=300)           # x leads y by 2
    c = ccf(x, y, 4).set_index("lag")["r"]
    assert c.idxmax() == 2


def test_bootstrap_and_hac_size_under_null():
    """Under independence both procedures should reject roughly 5% of the time."""
    rej_b = rej_h = 0
    R = 120
    for seed in range(R):
        rng = np.random.default_rng(seed)
        x, y = rng.normal(size=200), rng.normal(size=200)
        _, lo, hi = block_bootstrap_corr(x, y, B=300, seed=seed)
        rej_b += not (lo < 0 < hi)
        rej_h += hac_corr_test(pd.Series(x), pd.Series(y))["p_hac"] < 0.05
    assert rej_b / R < 0.12 and rej_h / R < 0.12


def _synthetic_issue_docs(signal_sd: float, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-11-03", "2026-03-27", freq="D")
    week_effect = {w: rng.normal(0, signal_sd) for w in days.to_period("W-FRI").unique()}
    rows = []
    for d in days:
        mu = week_effect[d.to_period("W-FRI")]
        for _ in range(25):                                   # ~175 docs per issue-week
            rows.append({"issue": "X", "date": d, "tone": mu + rng.normal(0, 0.8), "p_rel": rng.uniform(0.5, 1.0)})
    return pd.DataFrame(rows)


def test_split_half_reliability_detects_signal_and_noise():
    from midterm.nlp.issue_index import split_half
    weeks = pd.date_range("2025-11-07", "2026-03-27", freq="W-FRI")
    strong = split_half(_synthetic_issue_docs(0.3), weeks, ["X"], reps=5)
    noise = split_half(_synthetic_issue_docs(0.0), weeks, ["X"], reps=5)
    assert strong.loc["X", "reliability_level"] > 0.9
    assert abs(noise.loc["X", "reliability_level"]) < 0.35

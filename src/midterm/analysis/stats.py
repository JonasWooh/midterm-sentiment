"""Time-series association tools used throughout the study.

All inference on autocorrelated series uses either Newey-West (HAC) standard
errors or a circular moving-block bootstrap; nothing assumes i.i.d. errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


def nw_lags(n: int) -> int:
    """Newey-West (1994) automatic bandwidth rule of thumb."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def block_bootstrap_corr(x: np.ndarray, y: np.ndarray, block: int | None = None, B: int = 2000,
                         seed: int = 7, method: str = "pearson") -> tuple[float, float, float]:
    """Circular moving-block bootstrap 95% CI for a correlation."""
    n = len(x)
    block = block or max(2, int(round(n ** (1 / 3))))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    f = (lambda a, b: stats.spearmanr(a, b)[0]) if method == "spearman" else (lambda a, b: np.corrcoef(a, b)[0, 1])
    draws = np.empty(B)
    for i in range(B):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        draws[i] = f(x[idx], y[idx])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return f(x, y), lo, hi


def hac_corr_test(x: pd.Series, y: pd.Series) -> dict:
    """Correlation with a HAC t-test (slope of standardised y on standardised x)."""
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 10:
        return {"n": len(d), "r": np.nan, "t_hac": np.nan, "p_hac": np.nan}
    zx = (d.iloc[:, 0] - d.iloc[:, 0].mean()) / d.iloc[:, 0].std()
    zy = (d.iloc[:, 1] - d.iloc[:, 1].mean()) / d.iloc[:, 1].std()
    res = sm.OLS(zy.values, sm.add_constant(zx.values)).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(d))})
    r = float(np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1])
    rho_s = float(stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0])
    _, lo, hi = block_bootstrap_corr(d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy())
    return {"n": len(d), "r": r, "spearman": rho_s, "ci_lo": lo, "ci_hi": hi,
            "t_hac": float(res.tvalues[1]), "p_hac": float(res.pvalues[1])}


def ccf(x: pd.Series, y: pd.Series, max_lag: int) -> pd.DataFrame:
    """corr(x_{t-l}, y_t) for l in [-max_lag, max_lag]; l > 0 means x leads y."""
    rows = []
    for l in range(-max_lag, max_lag + 1):
        d = pd.concat([x.shift(l), y], axis=1).dropna()
        r = d.corr().iloc[0, 1] if len(d) > 5 else np.nan
        rows.append({"lag": l, "r": r, "n": len(d), "se_approx": 1 / np.sqrt(max(len(d), 1))})
    return pd.DataFrame(rows)


def granger(x: pd.Series, y: pd.Series, maxlags: int) -> dict:
    """Bivariate VAR on (x, y) with AIC lag choice; Wald tests both directions."""
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    model = VAR(d)
    sel = model.select_order(maxlags)
    p = max(1, int(sel.aic))
    res = model.fit(p)
    t_xy = res.test_causality("y", ["x"], kind="f")
    t_yx = res.test_causality("x", ["y"], kind="f")
    return {"lags": p, "n": res.nobs, "x_to_y_F": float(t_xy.test_statistic), "x_to_y_p": float(t_xy.pvalue),
            "y_to_x_F": float(t_yx.test_statistic), "y_to_x_p": float(t_yx.pvalue)}


def adf_p(s: pd.Series) -> float:
    s = s.dropna()
    return float(adfuller(s, autolag="AIC")[1]) if len(s) > 20 else np.nan


def hac_ols(y: pd.Series, X: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    d = pd.concat([y, X], axis=1).dropna()
    return sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1:])).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(d))})


def to_weekly(df: pd.DataFrame, how: str = "last") -> pd.DataFrame:
    """Calendar weeks ending Friday (aligns with the equity trading week)."""
    r = df.resample("W-FRI")
    return r.last() if how == "last" else r.sum(min_count=1)

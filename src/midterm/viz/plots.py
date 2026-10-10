"""Static report figures (matplotlib). One y-scale per panel; multiple measures
are shown as vertically stacked panels sharing the time axis, never twin axes.
Figures are drawn at the report's text width (W inches) so that they are placed
at 1:1 scale and the font sizes below are the sizes printed in the PDF."""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from midterm.config import FIG

# Reference categorical palette (validated order; see dataviz skill palette.md)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
W = 6.8  # report text width, inches (US Letter, 0.85in margins)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "legend.frameon": False, "font.family": "DejaVu Sans",
    "figure.constrained_layout.use": True,
})


def _wrap(labels, width=30):
    """Wrap long category labels so they do not squeeze the plotting area."""
    return [textwrap.fill(str(l), width) for l in labels]


def _fmt_dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


def _save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return FIG / name


def prediction_markets(pm: pd.DataFrame, start: str, end: str):
    d = pm.loc[start:end]
    fig, ax = plt.subplots(figsize=(W, 3.3))
    ax.plot(d.index, d["pm_sweep_norm"], color=C[0], label="Polymarket: Democratic sweep (normalised)")
    ax.plot(d.index, d["ks_sweep_norm"], color=C[1], lw=1.2, label="Kalshi: Democratic sweep (normalised)")
    ax.plot(d.index, d["pm_dem_house"], color=C[2], lw=1.0, ls="--", label="Polymarket: Democrats win House")
    ax.plot(d.index, d["pm_dem_senate"], color=C[6], lw=1.0, ls=":", label="Polymarket: Democrats win Senate")
    ax.set_ylim(0, 1)
    ax.set_ylabel("probability")
    ax.set_title("Prediction-market probability of a Democratic sweep, Nov 2025 – Oct 2026")
    _fmt_dates(ax)
    fig.legend(loc="outside lower center", fontsize=7.5, ncol=2)
    return _save(fig, "fig01_prediction_markets.png")


def topic_prevalence(prev: pd.DataFrame):
    """prev: index = topic label, columns news / social share (0-1), sorted."""
    p = prev.sort_values("combined")
    fig, ax = plt.subplots(figsize=(W, 0.3 * len(p) + 1))
    y = np.arange(len(p))
    h = 0.38
    ax.barh(y + h / 2, p["news"], height=h, color=C[0], label="News (Google News RSS)")
    ax.barh(y - h / 2, p["social"], height=h, color=C[1], label="Social (Reddit, stratified)")
    ax.set_yticks(y, _wrap(p.index))
    ax.set_xlabel("share of topical document mass")
    fig.suptitle("Topic prevalence in the midterm discourse", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    return _save(fig, "fig02_topic_prevalence.png")


def topic_share_over_time(share: pd.DataFrame, topics: list[str], name: str, title: str):
    w = share[topics].resample("W-FRI").mean()
    fig, ax = plt.subplots(figsize=(W, 3.3))
    for i, t in enumerate(topics):
        ax.plot(w.index, w[t], color=C[i % len(C)], label=t)
    ax.set_ylabel("weekly share of documents")
    ax.set_title(title)
    _fmt_dates(ax)
    fig.legend(fontsize=7.5, ncol=3, loc="outside lower center")
    return _save(fig, name)


def sentiment_panels(sent: pd.DataFrame, topics: list[str], pm: pd.DataFrame, name: str, title: str):
    n = len(topics) + 1
    fig, axes = plt.subplots(n, 1, figsize=(W, 1.0 * n + 0.6), sharex=True)
    fig.suptitle(title, fontsize=10, fontweight="bold")
    axes[0].plot(pm.index, pm["pm_sweep_norm"], color=C[0])
    axes[0].set_title("P(Democratic sweep)", loc="left", fontsize=8.5, fontweight="normal", color=INK)
    for ax, t, c in zip(axes[1:], topics, C[1:] + C[:1]):
        ax.plot(sent.index, sent[t], color=c)
        ax.set_title(t, loc="left", fontsize=8.5, fontweight="normal", color=INK)
        ax.axhline(sent[t].mean(), color=INK2, lw=0.6, ls=":")
    _fmt_dates(axes[-1])
    return _save(fig, name)


def index_vs_sweep(idx: pd.Series, pm: pd.Series, name: str, title: str, ylab: str):
    d = pd.concat([idx.rename("idx"), pm.rename("p")], axis=1).dropna()
    z = (d - d.mean()) / d.std()
    fig, ax = plt.subplots(figsize=(W, 3.2))
    ax.plot(z.index, z["p"], color=C[0], label="Democratic-sweep log-odds (z-score)")
    ax.plot(z.index, z["idx"], color=C[1], label=f"{ylab} (z-score)")
    ax.set_ylabel("standard deviations from mean")
    ax.set_title(title)
    _fmt_dates(ax)
    fig.legend(fontsize=8, loc="outside lower center", ncol=2)
    return _save(fig, name)


def ccf_plot(ccf: pd.DataFrame, series: list[str], name: str, title: str, unit: str):
    fig, ax = plt.subplots(figsize=(W, 3.0))
    for i, s in enumerate(series):
        d = ccf[ccf["series"] == s]
        ax.plot(d["lag"], d["r"], marker="o", ms=3.5, color=C[i % len(C)], label=s)
    band = 1.96 / np.sqrt(ccf["n"].median())
    ax.axhspan(-band, band, color=GRID, alpha=0.6, lw=0)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel(f"lag ({unit}); > 0 = sentiment leads the market")
    ax.set_ylabel("correlation of changes")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, name)


def basket_car(b: pd.DataFrame, pm: pd.DataFrame, car_x: pd.Series):
    fig, axes = plt.subplots(3, 1, figsize=(W, 6.0), sharex=True)
    axes[0].plot(pm.index, pm["pm_sweep_norm"], color=C[0])
    axes[0].set_ylabel("P(Dem. sweep)")
    axes[0].set_title("Mid-term basket: cumulative market-adjusted returns vs. sweep probability")
    axes[1].plot(b.index, 100 * b["car_ar_D"], color=C[2], label="D+ leg (sweep beneficiaries)")
    axes[1].plot(b.index, 100 * b["car_ar_R"], color=C[7], label="R+ leg (sweep losers)")
    axes[1].set_ylabel("CAR, %")
    axes[1].legend(fontsize=7.5, loc="upper left")
    axes[2].plot(b.index, 100 * b["car_spread"], color=C[6], label="Spread D+ − R+")
    axes[2].plot(car_x.index, 100 * car_x, color=C[3], label="Spread, oil/rates/VIX-orthogonalised")
    axes[2].axhline(0, color=INK2, lw=0.6)
    axes[2].set_ylabel("CAR, %")
    axes[2].legend(fontsize=7.5, loc="lower left")
    _fmt_dates(axes[-1])
    return _save(fig, "fig08_basket_car.png")


def stock_betas(sens: pd.DataFrame):
    d = sens[(sens["spec"] == "with_controls") & sens["target"].str.startswith("stock_")].copy()
    d["ticker"] = d["target"].str[6:]
    d = d.sort_values(["leg", "beta_dlogit"])
    fig, ax = plt.subplots(figsize=(W, 5.6))
    y = np.arange(len(d))
    cols = [C[2] if l == "D+" else C[7] for l in d["leg"]]
    se = d["beta_dlogit"] / d["t_hac"]
    ax.errorbar(100 * d["beta_dlogit"], y, xerr=196 * se.abs(), fmt="none", ecolor=GRID, elinewidth=3)
    ax.scatter(100 * d["beta_dlogit"], y, c=cols, s=28, zorder=3)
    ax.set_yticks(y, [f"{t} ({g})" for t, g in zip(d["ticker"], d["group"])], fontsize=7)
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_xlabel("daily abnormal return (%) per +1.0 change in sweep log-odds  (95% HAC CI)")
    fig.suptitle("Stock sensitivity to Democratic-sweep odds (green = D+ leg, red = R+ leg)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", visible=False)
    return _save(fig, "fig09_stock_odds_betas.png")


def reliability_comparison(rel: pd.DataFrame):
    """rel: index = issue label; columns: 'v1 (legacy corpus, news): model-based',
    'v2: model-based (levels)', 'v2: split-half (weekly changes)'; values 0-1."""
    r = rel.iloc[::-1]
    fig, ax = plt.subplots(figsize=(W, 0.36 * len(r) + 1.5))
    y = np.arange(len(r))
    k = r.shape[1]
    h = 0.8 / k
    for i, col in enumerate(r.columns):
        ax.barh(y + (k - 1 - 2 * i) * h / 2, r[col].clip(lower=0), height=h * 0.92, color=C[i], label=col)
    ax.set_yticks(y, _wrap(r.index))
    ax.set_xlim(0, 1)
    ax.set_xlabel("reliability (share of weekly variance that is signal)")
    fig.suptitle("Reliability of the weekly issue-sentiment index: v1 vs v2", fontsize=10, fontweight="bold")
    ax.grid(axis="y", visible=False)
    fig.legend(loc="outside lower center", ncol=1, fontsize=7.5)
    return _save(fig, "fig10_index_reliability.png")

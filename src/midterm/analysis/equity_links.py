"""Links between the mid-term basket, election odds and the sentiment index.

1. Election-odds sensitivity (the validity test for the basket)::

       AR_t = a + b * dlogit(P_sweep)_t + c' controls_t + e_t      (HAC)

   for the spread, each leg, each policy group and each stock. Controls:
   WTI front-month return (the 2026 Iran-war oil shock moves the energy and
   defence names directly), change in the 10y yield, change in VIX. P is
   sampled at 16:00 ET, i.e. synchronously with the equity close.
2. Sentiment link: correlation of the basket's cumulative abnormal return
   with the sentiment index (levels, with the same caveats as above) and of
   weekly basket AR with weekly sentiment changes; lead-lag CCF.
3. Event study: the N days with the largest absolute change in the sweep
   log-odds; mean spread AR on up-days vs down-days with a permutation test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from midterm.analysis.stats import block_bootstrap_corr, ccf, hac_corr_test, hac_ols, to_weekly
from midterm.config import TAB

CONTROLS = ["oil_ret", "d_tnx", "d_vix"]


def odds_sensitivity(basket: pd.DataFrame, ar: pd.DataFrame, betas: pd.DataFrame, pm: pd.DataFrame) -> pd.DataFrame:
    dlo = pm["logit_sweep"].diff().rename("dlogit")
    # trading-day alignment: change in log-odds between consecutive 16:00 ET marks of trading days
    lo_td = pm["logit_sweep"].reindex(basket.index)
    dlo_td = lo_td.diff().rename("dlogit")
    X = pd.concat([dlo_td, basket[CONTROLS]], axis=1)
    targets = {"spread": basket["spread"], "leg_D": basket["ar_D"], "leg_R": basket["ar_R"],
               "spread_raw": basket["spread_raw"], "spread_mm": basket["spread_mm"]}
    targets |= {c.replace("ar_grp_", "group_"): basket[c] for c in basket.columns if c.startswith("ar_grp_")}
    targets |= {f"stock_{t}": ar[t] for t in ar.columns}
    rows = []
    for name, y in targets.items():
        for spec, cols in {"with_controls": ["dlogit"] + CONTROLS, "no_controls": ["dlogit"]}.items():
            r = hac_ols(y.rename("y"), X[cols])
            rows.append({"target": name, "spec": spec, "beta_dlogit": r.params["dlogit"], "t_hac": r.tvalues["dlogit"],
                         "p_hac": r.pvalues["dlogit"], "r2": r.rsquared, "n": int(r.nobs),
                         "leg": betas.at[name[6:], "leg"] if name.startswith("stock_") else "",
                         "group": betas.at[name[6:], "group"] if name.startswith("stock_") else ""})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "basket_odds_sensitivity.csv", index=False)
    return out


def weekly_odds_link(basket: pd.DataFrame, pm: pd.DataFrame) -> pd.DataFrame:
    wk_ar = to_weekly(basket[["spread", "ar_D", "ar_R"] + CONTROLS], how="sum")
    wk_lo = to_weekly(pm[["logit_sweep"]])["logit_sweep"].diff().reindex(wk_ar.index)
    rows = []
    for c in ["spread", "ar_D", "ar_R"]:
        r = hac_ols(wk_ar[c], pd.concat([wk_lo.rename("dlogit"), wk_ar[CONTROLS]], axis=1))
        t = hac_corr_test(wk_lo, wk_ar[c])
        rows.append({"target": c, "beta_dlogit": r.params["dlogit"], "t_hac": r.tvalues["dlogit"],
                     "p_hac": r.pvalues["dlogit"], "r_simple": t["r"], "r_ci_lo": t["ci_lo"], "r_ci_hi": t["ci_hi"], "n": int(r.nobs)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "basket_odds_weekly.csv", index=False)
    return out


def sentiment_link(basket: pd.DataFrame, sent: pd.DataFrame, pm: pd.DataFrame, tag: str) -> dict[str, pd.DataFrame]:
    rows, lags = [], []
    car = basket["car_spread"]
    # oil-orthogonalised spread: residual of daily spread on controls, then cumulated
    r = hac_ols(basket["spread"].rename("y"), basket[CONTROLS])
    resid = (basket["spread"] - r.fittedvalues.reindex(basket.index).fillna(0) + r.params["const"])
    car_x = resid.cumsum().rename("car_spread_ex_controls")
    wk_spread = to_weekly(basket[["spread"]], how="sum")["spread"]
    wk_resid = to_weekly(resid.to_frame("r"), how="sum")["r"]
    for col in sent.columns:
        # daily series: all trading days; weekly series: their week-end dates only
        s_td = sent[col].reindex(basket.index)
        for lbl, target in {"car_spread": car, "car_spread_ex_controls": car_x}.items():
            d = pd.concat([s_td, target], axis=1).dropna()
            lv, lo, hi = block_bootstrap_corr(d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy(), block=max(5, int(len(d) ** 0.5)))
            rows.append({"series": col, "target": lbl, "stat": "level_corr", "r": lv, "ci_lo": lo, "ci_hi": hi, "n": len(d), "p_hac": np.nan})
        ws = to_weekly(sent[[col]])[col].diff()
        for lbl, target in {"weekly_spread_ar": wk_spread, "weekly_spread_ar_ex_controls": wk_resid}.items():
            t = hac_corr_test(ws.reindex(target.index), target)
            rows.append({"series": col, "target": lbl, "stat": "weekly_change_corr", "r": t["r"], "ci_lo": t["ci_lo"],
                         "ci_hi": t["ci_hi"], "n": t["n"], "p_hac": t["p_hac"]})
        lags.append(ccf(ws.reindex(wk_spread.index), wk_spread, 4).assign(series=col))
    out = pd.DataFrame(rows)
    out.to_csv(TAB / f"sentiment_vs_basket_{tag}.csv", index=False)
    lags = pd.concat(lags)
    lags.to_csv(TAB / f"sentiment_vs_basket_ccf_{tag}.csv", index=False)
    return {"summary": out, "ccf": lags, "car_ex_controls": car_x}


def event_study(basket: pd.DataFrame, pm: pd.DataFrame, n_events: int = 15, seed: int = 11) -> pd.DataFrame:
    lo = pm["logit_sweep"].reindex(basket.index).diff()
    d = pd.concat([lo.rename("dlogit"), basket["spread"], basket["ar_D"], basket["ar_R"]], axis=1).dropna()
    ev = d.reindex(d["dlogit"].abs().sort_values(ascending=False).index[:n_events]).sort_index()
    sign = np.sign(ev["dlogit"])
    stat = float((sign * ev["spread"]).mean())  # signed AR: positive if basket moves with odds
    rng = np.random.default_rng(seed)
    null = np.array([(sign.values * rng.choice(d["spread"].values, n_events, replace=False)).mean() for _ in range(20000)])
    p = float((np.abs(null) >= abs(stat)).mean())
    ev = ev.assign(signed_spread=sign * ev["spread"])
    ev.attrs.update({"mean_signed_spread": stat, "perm_p": p})
    ev.to_csv(TAB / "event_study_largest_odds_moves.csv")
    pd.DataFrame([{"n_events": n_events, "mean_signed_spread_ar": stat, "perm_p_two_sided": p,
                   "mean_abs_dlogit": ev["dlogit"].abs().mean()}]).to_csv(TAB / "event_study_summary.csv", index=False)
    return ev


def odds_sensitivity_by_period(basket: pd.DataFrame, pm: pd.DataFrame, war_start: str = "2026-02-28") -> pd.DataFrame:
    lo = pm["logit_sweep"].reindex(basket.index).diff().rename("dlogit")
    X = pd.concat([lo, basket[CONTROLS]], axis=1)
    rows = []
    for per, sl in {"pre_war": slice(None, war_start), "post_war": slice(war_start, None), "full": slice(None, None)}.items():
        for tgt in ("spread", "ar_D", "ar_R"):
            r = hac_ols(basket[tgt].loc[sl].rename("y"), X.loc[sl])
            rows.append({"period": per, "target": tgt, "beta_dlogit": r.params["dlogit"], "t_hac": r.tvalues["dlogit"],
                         "p_hac": r.pvalues["dlogit"], "beta_oil": r.params["oil_ret"], "t_oil": r.tvalues["oil_ret"],
                         "n": int(r.nobs)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "basket_odds_sensitivity_by_period.csv", index=False)
    return out

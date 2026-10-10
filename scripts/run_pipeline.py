"""End-to-end analysis pipeline (raw files in data/raw -> tables, figures, results.json).

    python scripts/run_pipeline.py

Stages
  1 markets   prediction-market daily series (Polymarket, Kalshi)
  2 equities  mid-term basket market-adjusted returns
  3 docs      unified news + Reddit document table
  4 topics    unsupervised NMF topic discovery (stage 1; stage-2 deflated model as diagnostic)
  5 issues    Reference-label-trained issue classifier -> issue posteriors, salience ranking, key issues
  6 sentiment v2 issue-sentiment index from the dense issue-targeted news corpus (transformer
              tone + relevance models trained on v2 annotations; validation), legacy-corpus
              indices (news + stratified Reddit), GDELT issue tone, composites
  7 analysis  sentiment <-> P(Democratic sweep); basket <-> odds and sentiment
  8 figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from midterm.analysis import equity_links as EL  # noqa: E402
from midterm.analysis import sentiment_vs_markets as SVM  # noqa: E402
from midterm.config import INTERIM, PROCESSED, ROOT, settings  # noqa: E402
from midterm.equities import basket as BK  # noqa: E402
from midterm.markets import prediction as PM  # noqa: E402
from midterm.nlp import issue_classifier as IC  # noqa: E402
from midterm.nlp import issue_corpus as ICORP  # noqa: E402
from midterm.nlp import issue_index as II  # noqa: E402
from midterm.nlp import news_tone as NT  # noqa: E402
from midterm.nlp import preprocess as PP  # noqa: E402
from midterm.nlp import sentiment as SE  # noqa: E402
from midterm.nlp import sentiment_model as SMOD  # noqa: E402
from midterm.nlp import sentiment_v2 as S2  # noqa: E402
from midterm.collect.google_news_issues import ISSUE2QUERY  # noqa: E402
from midterm.nlp import topic_summary as TS  # noqa: E402
from midterm.nlp import topics as TP  # noqa: E402
from midterm.nlp import validate as VA  # noqa: E402
from midterm.viz import plots as PL  # noqa: E402

MIN_PROB = 0.30          # a document counts towards an issue's sentiment if P(issue) >= 0.30
CODE2GDELT = {"COST": "cost_of_living", "ENERGY": "energy_gas_prices", "IRAN": "iran_war", "TARIFF": "tariffs_trade",
              "IMMIG": "immigration", "HEALTH": "healthcare", "DEMOC": "democracy_elections", "AI": "ai_data_centers",
              "EPSTEIN": "epstein", "TRUMP": "trump_approval", "ABORT": "abortion", "CRIME": "crime_safety",
              "SOCIAL": "social_culture", "FOREIGN": "foreign_policy"}


def recs(df: pd.DataFrame, nd: int = 4) -> list[dict]:
    return json.loads(df.round(nd).to_json(orient="records"))


def main() -> None:
    cfg = settings()
    start, end = cfg["study"]["start"], cfg["study"]["end"]
    R: dict = {"study_window": [start, end]}

    # 1-2 markets and equities ---------------------------------------------------------------------
    pmd = PM.build()
    bk = BK.build()
    pmw = pmd.loc[start:end]
    R["markets"] = {
        "pm_sweep_start": float(pmw["pm_sweep_norm"].iloc[0]), "pm_sweep_end": float(pmw["pm_sweep_norm"].iloc[-1]),
        "pm_sweep_min": float(pmw["pm_sweep_norm"].min()), "pm_sweep_max": float(pmw["pm_sweep_norm"].max()),
        "ks_sweep_end": float(pmw["ks_sweep_norm"].dropna().iloc[-1]),
        "corr_pm_ks_level": float(pmw[["pm_sweep_norm", "ks_sweep_norm"]].corr().iloc[0, 1]),
        "corr_pm_ks_weekly_change": float(pmw[["pm_sweep_norm", "ks_sweep_norm"]].resample("W-FRI").last().diff().corr().iloc[0, 1]),
        "pm_house_end": float(pmw["pm_dem_house"].iloc[-1]), "pm_senate_end": float(pmw["pm_dem_senate"].iloc[-1]),
        "pm_bop_sum_mean": float(pmw["pm_bop_sum"].mean()),
    }

    # 3-4 corpus and unsupervised topics -----------------------------------------------------------
    docs = PP.build(start, end)
    R["corpus"] = {"n_docs": int(len(docs)), "by_source": docs["source"].value_counts().to_dict(),
                   "reddit_by_sub": docs.loc[docs.source == "reddit", "sub"].value_counts().to_dict()}
    tp = TP.run()
    R["nmf_stage1"] = {"K": tp["stage1"]["K"], "grid": recs(tp["stage1"]["grid"])}
    if "stage2" in tp:
        s2 = tp["stage2"]
        R["nmf_stage2"] = {k: s2[k] for k in ("K", "n_removed_terms", "n_kept_terms", "docs_with_issue_vocab")}
        R["nmf_stage2"]["grid"] = recs(s2["grid"])

    # 5 issues ----------------------------------------------------------------------------------------
    fitted = IC.train_and_evaluate()
    R["issue_classifier_eval"] = recs(fitted["evaluation"])
    P, classes, prev = IC.apply(docs, fitted)
    ranking = TS.rank_issues(docs, P, classes)
    key = TS.select_key_issues(ranking)
    R["issue_ranking"] = json.loads(ranking.round(4).to_json(orient="index"))
    R["key_issues"] = key
    salience = TS.issue_share_daily(docs, P, classes, start, end)

    # 6 sentiment -------------------------------------------------------------------------------------
    kidx = [classes.index(k) for k in key]
    M = P[:, kidx].copy()
    M[M < MIN_PROB] = 0.0
    # 6a legacy corpus (issue-neutral news + stratified Reddit): v1 instrument kept as baseline,
    #    document score = zero-shot transformer (better on this corpus' own hold-outs)
    sfit = SMOD.train_and_evaluate()
    R["sentiment_model_eval_v1"] = recs(sfit["evaluation"])
    R["sentiment_zero_shot_legacy_eval"] = recs(S2.legacy_corpus_eval())
    docs = docs.join(SMOD.score(docs, sfit)).rename(columns={"sent": "sent_v1"})
    zs = S2.zero_shot_score(docs["text"])
    docs["sent"] = (zs - zs.mean()) / zs.std()
    docs.to_parquet(INTERIM / "docs_scored.parquet", index=False)
    R["vader_validation"] = recs(VA.run(docs))
    sent = SE.build(docs, M, key, start, end)          # weekly (W-FRI) legacy-corpus indices
    R["text_index_reliability_legacy"] = json.loads(pd.read_csv(PROCESSED / "sentiment_shrinkage_diagnostics.csv")
                                                    .round(4).to_json(orient="records"))

    # 6b dense issue-targeted corpus (primary text index)
    assert set(key) <= set(ISSUE2QUERY), f"key issues without a targeted corpus: {set(key) - set(ISSUE2QUERY)}"
    idocs = ICORP.build(start, end)
    corpus_attrs = dict(idocs.attrs)
    tr, ev = S2.load("train"), S2.load("eval")
    ftr, fev = S2.design(tr["text"], tr["issue"]), S2.design(ev["text"], ev["issue"])
    cv = S2.select_tone(tr, ftr)
    spec = str(cv.index[0])
    hold = S2.evaluate(tr, ev, ftr, fev, spec)
    agg = S2.aggregate_validity(tr, ev, ftr, fev, spec, idocs.set_index("doc_id")["date"], start)
    lab = pd.concat([tr, ev], ignore_index=True)
    final = S2.fit_final(spec, lab, {k: np.concatenate([ftr[k], fev[k]]) for k in ftr})
    idocs = idocs.join(S2.score(final, idocs["text"], idocs["issue"]))
    idocs.to_parquet(INTERIM / "issue_docs_scored.parquet", index=False)
    hv = S2.human_validation(idocs)
    v2 = II.build(idocs, start, end, key)
    tone = NT.gdelt_tone_daily(start, end)
    tone_raw = pd.read_csv(PROCESSED / "gdelt_tone_daily_raw.csv", index_col=0, parse_dates=True)
    conv = II.convergent_validity(v2["index"], tone_raw, CODE2GDELT)
    R["sentiment_v2"] = {
        "corpus": {**corpus_attrs, "by_issue": idocs["issue"].value_counts().to_dict(),
                   "median_docs_per_issue_week": float(idocs.assign(w=idocs["date"].dt.to_period("W-FRI"))
                                                       .groupby(["issue", "w"]).size().median()),
                   "mean_p_rel": float(idocs["p_rel"].mean())},
        "labels": {"train": len(tr), "eval": len(ev), "rel_share_train": float(tr["rel"].mean()),
                   "tone_dist_train": tr["tone"].value_counts().sort_index().to_dict(),
                   "rel_share_eval": float(ev["rel"].mean()),
                   "tone_dist_eval": ev["tone"].value_counts().sort_index().to_dict()},
        "cv_selection": recs(cv.reset_index()), "selected_spec": spec,
        "holdout_tone": recs(hold["tone"]), "holdout_relevance": recs(hold["relevance"]),
        "legacy_baseline_on_v2_holdout": recs(S2.legacy_baseline(ev)),
        "aggregate_validity": agg, "retest": recs(S2.retest(tr, ev)),
        "human_validation": None if hv is None else recs(hv),
        "split_half": recs(v2["split_half"].reset_index()), "shrinkage": recs(v2["diag"].reset_index()),
        "vs_gdelt": recs(conv.reset_index()),
    }
    gkey = [k for k in key if CODE2GDELT.get(k) in tone.columns]
    tone_key = tone[[CODE2GDELT[k] for k in gkey]].set_axis(gkey, axis=1)
    w = ranking.loc[key, "combined"]
    idx_d = pd.DataFrame({"KISI_gdelt": NT.composite(tone_key, w.loc[gkey]),
                          "KISI_gdelt_eqw": NT.composite(tone_key, pd.Series(1.0, index=gkey))})
    v2i = v2["index"][key]
    idx_w = pd.DataFrame({
        "KISI_text": NT.composite(v2i, w),
        "KISI_text_eqw": NT.composite(v2i, pd.Series(1.0, index=key)),
        "KISI_text_legacy": NT.composite(sent["combined"][key], w),
        "KISI_text_legacy_news": NT.composite(sent["news"][key], w),
        "KISI_text_legacy_social": NT.composite(sent["social"][key], w),
        "KISI_text_legacy_left": NT.composite(sent["left"][key], w),
        "KISI_text_legacy_right": NT.composite(sent["right"][key], w),
    })
    idx_d.to_csv(PROCESSED / "kisi_gdelt_daily.csv")
    idx_w.to_csv(PROCESSED / "kisi_text_weekly.csv")
    wk_g = idx_d.resample("W-FRI").last()
    for tag in ("KISI_text", "KISI_text_legacy"):
        j = pd.concat([idx_w[tag], wk_g["KISI_gdelt"]], axis=1)
        R[f"corr_{tag}_vs_gdelt_weekly_level"] = float(j.corr().iloc[0, 1])
        R[f"corr_{tag}_vs_gdelt_weekly_change"] = float(j.diff().corr().iloc[0, 1])

    # 7 analysis --------------------------------------------------------------------------------------
    sal7 = ((salience["news"].rolling(7, min_periods=1).mean() + salience["reddit"].rolling(7, min_periods=1).mean()) / 2)[key]
    weekly_part = SVM.weekly_to_daily_panel(pd.concat([idx_w, v2i.add_prefix("text_")], axis=1), end)
    daily_part = pd.concat([idx_d, tone_key.add_prefix("gdelt_"), sal7.add_prefix("salience_")], axis=1)
    panel = daily_part.join(weekly_part, how="left")
    svm = SVM.run(panel, pmd, start, end, "main", daily_cols=list(daily_part.columns))
    R["sentiment_vs_sweep"] = recs(svm["summary"])
    ctl = bk["basket"][["oil_ret", "d_tnx", "d_vix"]]
    cw = SVM.controlled_weekly(panel, pmd, ctl, start, end,
                               ["KISI_gdelt", "KISI_gdelt_eqw", "KISI_text", "KISI_text_eqw", "KISI_text_legacy",
                                "KISI_text_legacy_news", "KISI_text_legacy_social"]
                               + [f"gdelt_{k}" for k in gkey]
                               + [f"text_{k}" for k in key])
    R["sentiment_vs_sweep_weekly_controlled"] = recs(cw)
    wo = pd.concat([ctl["oil_ret"].resample("W-FRI").sum(), pmd["logit_sweep"].resample("W-FRI").last().diff()],
                   axis=1, sort=True).loc[start:].dropna()
    R["corr_weekly_oil_return_vs_dlogit_sweep"] = float(wo.corr().iloc[0, 1])

    b = bk["basket"]
    sens = EL.odds_sensitivity(b, bk["ar"], bk["betas"], pmd)
    wk = EL.weekly_odds_link(b, pmd)
    sl = EL.sentiment_link(b, panel[["KISI_gdelt", "KISI_text"]], pmd, "main")
    ev = EL.event_study(b, pmd)
    R["basket_by_period"] = recs(EL.odds_sensitivity_by_period(b, pmd))
    R["basket"] = {
        "car_end": b[["car_ar_D", "car_ar_R", "car_spread", "car_spread_raw", "car_spread_mm"]].iloc[-1].round(4).to_dict(),
        "car_spread_ex_controls_end": float(sl["car_ex_controls"].iloc[-1]),
        "odds_sensitivity": recs(sens[~sens.target.str.startswith("stock_")]),
        "stock_sensitivity": recs(sens[sens.target.str.startswith("stock_") & sens.spec.eq("with_controls")]),
        "weekly_odds": recs(wk),
        "sentiment_link": recs(sl["summary"]),
        "event_study": {"mean_signed_spread": ev.attrs["mean_signed_spread"], "perm_p": ev.attrs["perm_p"]},
        "betas": json.loads(bk["betas"].round(4).to_json(orient="index")),
    }

    # 8 figures ---------------------------------------------------------------------------------------
    PL.prediction_markets(pmd, start, end)
    PL.topic_prevalence(ranking.loc[[k for k in ranking.index if k != "OTHER_ISSUE"], ["news", "social", "combined"]]
                        .rename(index=lambda c: IC.ISSUE_NAMES[c]))
    top6 = key[:6]
    PL.topic_share_over_time(sal7[top6].rename(columns=IC.ISSUE_NAMES), [IC.ISSUE_NAMES[k] for k in top6],
                             "fig03_issue_salience.png", "Salience of key issues (7-day mean share of documents, news + Reddit)")
    PL.sentiment_panels(v2i[top6].rename(columns=IC.ISSUE_NAMES), [IC.ISSUE_NAMES[k] for k in top6], pmw,
                        "fig04_issue_sentiment_text.png",
                        "Weekly issue tone in issue-targeted news\n(transformer model, relevance-weighted) and P(Democratic sweep)")
    g6 = [k for k in key if k in gkey][:6]
    PL.sentiment_panels(tone_key[g6].rename(columns=IC.ISSUE_NAMES), [IC.ISSUE_NAMES[k] for k in g6], pmw,
                        "fig05_issue_tone_gdelt.png", "Issue tone in US news (GDELT full text) and P(Democratic sweep)")
    PL.index_vs_sweep(idx_d["KISI_gdelt"], pmw["logit_sweep"], "fig06_kisi_vs_sweep.png",
                      "Key-Issue Sentiment Index (GDELT tone) vs Democratic-sweep log-odds", "KISI-GDELT")
    PL.ccf_plot(svm["ccf"][svm["ccf"]["freq"] == "W"], ["KISI_text", "KISI_gdelt"], "fig07_ccf_kisi_sweep.png",
                "Lead-lag: weekly changes in KISI vs weekly change in sweep log-odds", "weeks")
    leg = pd.read_csv(PROCESSED / "sentiment_shrinkage_diagnostics.csv", index_col=0)
    leg = leg[leg["stream"] == "news"]["reliability_at_mean_n"]
    PL.reliability_comparison(pd.DataFrame({
        "v1 (legacy corpus, news stream): model-based": leg.reindex(key),
        "v2 (issue-targeted corpus): model-based, levels": v2["diag"]["reliability_at_mean_n"].reindex(key),
        "v2: split-half, weekly changes": v2["split_half"]["reliability_change"].reindex(key),
    }).rename(index=IC.ISSUE_NAMES))
    PL.basket_car(b, pmw, sl["car_ex_controls"])
    PL.stock_betas(sens)

    (ROOT / "outputs" / "results.json").write_text(json.dumps(R, indent=1, default=float))
    print("done; key issues:", key)


if __name__ == "__main__":
    main()

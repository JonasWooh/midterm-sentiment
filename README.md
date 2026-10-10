# midterm-sentiment

Voter-issue discovery, issue sentiment, prediction-market odds and a "mid-term"
equity basket for the **2026 US midterm elections** (study window 2025-11-03 →
2026-10-07, data retrieved 2026-10-08).

The research report is [`reports/midterm_sentiment_report.pdf`](reports/midterm_sentiment_report.pdf)
(source: [`reports/midterm_sentiment_report.md`](reports/midterm_sentiment_report.md)).

## What the pipeline does

| Stage | Module | Output |
|---|---|---|
| Prediction markets | `midterm.markets.prediction` | Polymarket "Balance of Power: D Senate, D House" (renormalised over the 4 party-control combinations, sampled 16:00 ET) and Kalshi `KXBALANCEPOWERCOMBO-27FEB-DD` → `data/processed/prediction_markets_daily.csv` |
| Equity basket | `midterm.equities.basket` | 12 Democratic-sweep beneficiaries (D+) vs 12 Republican-hold beneficiaries (R+); betas estimated pre-window; market-adjusted returns → `basket_daily.csv` |
| Corpus | `midterm.nlp.preprocess` | 18.6k Google-News headlines (6 issue-neutral queries × 49 weeks) + 23.4k Reddit posts from 8 communities, stratified left / centre / right |
| Unsupervised topics | `midterm.nlp.topics` | NMF, K by NPMI coherence (stage 1, K=35); vocabulary-deflated stage 2 as a diagnostic |
| Issue classifier | `midterm.nlp.issue_classifier` | Researcher-reviewed training labels (1,220) + random hold-out (568) → TF-IDF logistic model; issue posteriors for every document |
| Issue ranking | `midterm.nlp.topic_summary` | salience by stream, triangulated with national polls (`config/external_polls.yaml`) → key issues |
| Issue-targeted corpus | `midterm.collect.google_news_issues`, `midterm.nlp.issue_corpus` | 89k Google-News headlines, one keyword query per key issue × 170 two-day windows (dense sample for sentiment) |
| Sentiment (v2, primary) | `midterm.nlp.transformer`, `midterm.nlp.sentiment_v2`, `midterm.nlp.issue_index` | RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) features; tone (−2…+2) and relevance models trained on 1,430 v2 annotations, selected by CV, evaluated on a 550-item random hold-out; relevance-weighted weekly issue index with empirical-Bayes shrinkage, split-half reliability, convergent validity vs GDELT |
| Sentiment (baselines / companions) | `midterm.nlp.sentiment_model`, `midterm.nlp.sentiment`, `midterm.nlp.news_tone` | v1 lexicon + logistic instrument (baseline); legacy election-corpus indices (news + stratified Reddit); GDELT full-text news tone per issue; Key-Issue Sentiment Indices (KISI) |
| Analysis | `midterm.analysis.*` | levels / changes / lead-lag / Granger / BH-FDR; weekly regressions with macro controls and pre/post Iran-war split; basket sensitivity to odds, sentiment link, event study |
| Figures | `midterm.viz.plots` | `outputs/figures/*.png` |

## Reproduce

```bash
pip install -r requirements.txt
pip install -e .
# (optional) re-collect raw data — needs direct HTTPS egress
python -m midterm.collect.polymarket ; python -m midterm.collect.kalshi ; python -m midterm.collect.yahoo
python -m midterm.collect.google_news ; python -m midterm.collect.reddit_arctic ; python -m midterm.collect.gdelt
python -m midterm.collect.google_news_issues          # resumable; --from/--to split the window range across hosts
# analysis (uses data/raw as shipped; writes data/interim, data/processed, outputs/)
PYTHONPATH=src python scripts/run_pipeline.py
PYTHONPATH=src pytest -q tests
# report PDF from reports/midterm_sentiment_report.md (needs XeLaTeX and the Times New Roman font)
python scripts/build_report.py
```

`data/interim/` (document tables and model matrices) and `data/processed/` (derived series) are not
versioned; the pipeline rebuilds them from `data/raw` and the supplied annotations. The transformer encodings depend on the hardware and library versions, so a rerun on another
machine can move reported statistics in the third or fourth decimal place; `outputs/` holds the run the
report is based on.

`outputs/results.json` collects every number quoted in the report.

### How the raw data were obtained in this run

The first collection ran in a desktop browser because the analysis sandbox had no internet egress
(`browser_collectors/*.js`, same endpoints and parameters as `src/midterm/collect/*.py`). Results were
exported in fixed-size chunks with a length + FNV-1a checksum and reassembled and verified by
`src/midterm/bridge/ingest.py` (unit-tested). After direct network access was enabled, the GDELT tone series
were re-collected with `python -m midterm.collect.gdelt` and matched the browser-collected files exactly (all
13 series, every day); the issue-targeted corpus (`data/raw/google_news_issue_rss.jsonl`) and the two new
GDELT queries were collected directly with the Python collectors. The transformer weights are downloaded from
the Hugging Face hub on first use; encodings are cached in `data/interim/transformer_cache/` (≈400 MB, not
versioned; the first full run re-encodes the ≈130k unique texts, about 45 minutes on two CPU cores).

### Annotations

`data/annotations/` holds the v1 samples and researcher-reviewed reference labels
(`reference_labels_{train,news,reddit}.csv`, with `issue_label`, `sentiment_label`,
and `annotation_status` fields) for the issue classifier and 3-level sentiment, and the v2 samples,
written guidelines (`GUIDELINES_issue_v2.md`) and labels (`labels_issue_{train,eval,retest}.csv`). An
independent human coder labelled 110 hold-out items (`human_validation_sheet.csv`, produced by
`scripts/annotation/human_validation.py make`); agreement with the labels and the model is computed by the
pipeline and written to `outputs/tables/human_validation_agreement.csv`.

## Layout

```
config/            settings, topic labels (stage 1), GDELT issue queries, external poll evidence
browser_collectors/ JS collectors used for the reported run
src/midterm/       package: collect, bridge, markets, equities, nlp, analysis, viz
scripts/           run_pipeline.py, build_report.py (Markdown → PDF), annotation/ (samplers, human-validation kit)
data/raw           raw API payloads (as retrieved)
data/annotations   annotation guidelines, samples and labels (v1 and v2)
data/interim       document table, model matrices (generated, not versioned)
data/processed     daily series (generated, not versioned)
outputs/           tables, figures, results.json
reports/           report (.md source and .pdf; figures are read from outputs/figures)
tests/             unit tests
```

## Data sources and terms

Polymarket Gamma & CLOB APIs, Kalshi trade API v2 (public market data), Yahoo Finance chart
endpoint, Google News RSS, GDELT DOC 2.0 API, Arctic Shift Reddit archive, Hugging Face model hub. Requests were rate-limited
to the providers' published limits. Reddit content is kept only as titles and the first 200
characters of self-text for research purposes.

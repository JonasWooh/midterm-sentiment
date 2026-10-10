# Annotation guidelines — issue-targeted headlines (v2)

Applies to `sample_issue_train.csv`, `sample_issue_eval.csv` and `sample_issue_retest.csv`
(drawn by `scripts/annotation/make_issue_samples.py` from `data/interim/issue_docs.parquet`).
Each item is a news headline retrieved by the keyword query of one key issue. The annotator
sees the item id, the issue name and the headline only (no date, publisher, model score or
other label). Labels were fixed before any model was trained on them.

## Label 1 — `rel` (relevance, 0/1)

`rel = 1` if the headline is about the named issue **as a matter of US public concern**:
conditions Americans experience (prices, coverage, safety), US policy or politics on the
issue, or events abroad in which the US government is a direct party (e.g. US strikes on
Iran, US tariffs, US policy on Ukraine or Venezuela).

`rel = 0` otherwise, in particular:
* the issue keyword appears but the story is about something else (e.g. a sports item that
  mentions Ukraine, a shooting story returned for an Iran query);
* a purely foreign domestic matter (another country's immigration rules or elections);
* technical, scientific or corporate content without a public-policy or household angle
  (e.g. a medical-imaging review that uses "artificial intelligence"; a hospital award);
* approval ratings of non-US politicians for the TRUMP issue.

Borderline cases: corporate data-center siting, energy or utility news that affects US
prices, and market forecasts for US gasoline/oil prices are **relevant** (they are part of
how the issue reaches voters).

## Label 2 — `tone` (−2 … +2)

The tone the headline conveys about **the state of the issue for the US public**, judged from
a neutral, non-partisan standpoint (as in news-tone measurement: conflict, harm, failure and
deterioration are negative; relief, improvement, agreement and resolution are positive):

| value | meaning | examples of cues |
|---|---|---|
| −2 | clearly bad news / strongly negative framing | crisis, deaths, war escalation, sharp price spikes, collapse, scandal |
| −1 | somewhat negative | rising costs, disputes, lawsuits, criticism, setbacks, polls showing disapproval |
|  0 | neutral, factual, procedural or mixed | announcements, explainers, schedules, balanced polls |
| +1 | somewhat positive | easing prices, deals, approvals, modest improvement |
| +2 | clearly good news / strongly positive framing | major relief, resolution, ceasefire holds, large declines in prices |

Rules: judge the headline as written (no outside knowledge of what happened later); a
partisan win/loss is coded by its conflict content, not by which party benefits; tone is
coded for every item, including `rel = 0` items (used only to train the tone model).

## Retest

`sample_issue_retest.csv` re-presents a random subset of already-labelled items in a new
random order with new ids, after the main pass, without access to the first labels; the two
passes give a test–retest (intra-annotator) reliability for both labels.

Reference labels were reviewed and confirmed by researchers under this codebook.
Independent coding agreement was assessed on 110 randomly selected hold-out
headlines (10 per issue); see report §4.3 for the statistics and §8 for the
limits of issue-specific agreement estimates.

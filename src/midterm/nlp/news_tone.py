"""GDELT full-text news tone per key issue, and the composite sentiment indices.

GDELT's "Average Tone" is the mean document tone (positive minus negative
word share, in percentage points) over *all* US English articles matching the
issue query on a day — i.e. it measures the tone of the national information
environment around an issue, computed on full article text, independently of
our headline/Reddit samples. Days missing in GDELT's timeline (feed outages)
stay missing; smoothing uses an EWMA that ignores gaps.

Composite Key-Issue Sentiment Index (KISI)
-----------------------------------------
For a set of key issues with importance weights w_j (their combined
prevalence share among the key issues, from ``key_topics_ranking.csv``)::

    KISI_t = sum_j w_j * z_jt ,   z_jt = (s_jt - mean_j) / sd_j

where s_jt is the smoothed issue sentiment (GDELT tone, or the stratified
Reddit+news VADER index). Higher = more positive tone around what voters care
about. The z-scoring uses full-window moments; this changes only location and
scale, never the timing of movements, so it cannot create lead-lag effects.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from midterm.config import PROCESSED, RAW


def gdelt_tone_daily(start: str, end: str) -> pd.DataFrame:
    raw = json.loads((RAW / "gdelt_topic_timelines.json").read_text())["series"]
    idx = pd.date_range(start, end, freq="D")
    cols = {}
    for k, v in raw.items():
        t = pd.DataFrame(v["timelinetone"])
        s = pd.Series(t["value"].to_numpy(), index=pd.to_datetime(t["date"], format="%Y%m%dT%H%M%SZ"))
        cols[k] = s.reindex(idx)
    df = pd.DataFrame(cols)
    df.index.name = "date"
    df.to_csv(PROCESSED / "gdelt_tone_daily_raw.csv")
    sm = df.ewm(halflife=3.5, ignore_na=True, min_periods=1).mean().where(df.notna().cumsum() > 0)
    sm.to_csv(PROCESSED / "gdelt_tone_daily_smoothed.csv")
    return sm


def composite(sent: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(sent.columns).fillna(0.0)
    w = w / w.sum()
    z = (sent - sent.mean()) / sent.std()
    return (z * w).sum(axis=1, min_count=1)

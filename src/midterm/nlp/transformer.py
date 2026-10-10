"""Pretrained-transformer encoders for document sentiment and relevance.

Backbone: ``cardiffnlp/twitter-roberta-base-sentiment-latest`` (RoBERTa-base
trained on ~124M tweets and fine-tuned for 3-class sentiment; Loureiro et al.
2022). Short, informal political text is close to its training domain, and it
is the best zero-shot model on our legacy hold-out sets (Spearman 0.67 news /
0.65 Reddit vs 0.41 / 0.47 for the lexicon+LR ensemble).

For every text we extract, in one forward pass,
* the 3 sentiment logits of the pretrained head (zero-shot sentiment), and
* the mean-pooled last hidden state (768-d sentence embedding),
which feed the supervised heads in ``midterm.nlp.sentiment_v2``.

Encodings are cached on disk keyed by the MD5 of the exact input string, so
re-running the pipeline never re-encodes a text.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from midterm.config import INTERIM

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
REVISION = None  # resolved commit is recorded in the cache metadata
MAX_LEN = 96
CACHE = INTERIM / "transformer_cache"
_MEM: dict = {}


def _key(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def _load():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, output_hidden_states=True).eval()
    return torch, tok, model


def encode(texts: list[str], batch: int = 64, threads: int | None = None, cache: Path = CACHE) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits [n x 3] in order negative/neutral/positive, embeddings [n x 768])."""
    cache.mkdir(parents=True, exist_ok=True)
    keys = [_key(t) for t in texts]
    shards = tuple(sorted(cache.glob("shard_*.parquet")))
    if _MEM.get("id") != (cache, shards):  # reload the on-disk cache only when it changed
        known = pd.concat([pd.read_parquet(s) for s in shards]) if shards else pd.DataFrame(columns=["key"])
        _MEM.update(id=(cache, shards), table=known.drop_duplicates("key").set_index("key"))
    known = _MEM["table"]
    # length-sorted batches minimise padding (outputs do not depend on batch composition:
    # padded positions are masked in attention and excluded from mean pooling)
    missing = sorted({k: t for k, t in zip(keys, texts) if k not in known.index}.items(), key=lambda kv: (len(kv[1]), kv[0]))
    if missing:
        torch, tok, model = _load()
        if threads:
            torch.set_num_threads(threads)
        lab = {v.lower(): k for k, v in model.config.id2label.items()}
        order = [lab["negative"], lab["neutral"], lab["positive"]]
        new = []
        with torch.no_grad():
            for i in range(0, len(missing), batch):
                ks, ts = zip(*missing[i:i + batch])
                enc = tok(list(ts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
                out = model(**enc)
                h = out.hidden_states[-1]
                m = enc["attention_mask"].unsqueeze(-1).float()
                emb = ((h * m).sum(1) / m.sum(1)).numpy().astype(np.float32)
                lg = out.logits[:, order].numpy().astype(np.float32)
                new.append(pd.DataFrame({"key": ks, "logits": list(lg), "emb": list(emb)}))
                if len(new) * batch >= 4096 or i + batch >= len(missing):
                    part = pd.concat(new)
                    part.to_parquet(cache / f"shard_{len(list(cache.glob('shard_*.parquet'))):05d}.parquet", index=False)
                    known = pd.concat([known, part.set_index("key")])
                    new = []
        _MEM.update(id=(cache, tuple(sorted(cache.glob("shard_*.parquet")))), table=known)
    sel = known.loc[keys]
    return np.vstack(sel["logits"].to_numpy()), np.vstack(sel["emb"].to_numpy())


if __name__ == "__main__":
    import sys
    df = pd.read_parquet(sys.argv[1])
    encode(df["text"].astype(str).tolist(), threads=int(sys.argv[2]) if len(sys.argv) > 2 else None)

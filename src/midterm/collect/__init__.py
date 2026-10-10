"""Pure-Python data collectors.

These produce exactly the raw files that the browser collectors in
``browser_collectors/`` produced for the reported run (same endpoints, same
parameters, same JSON layout), so the pipeline can be re-run from scratch
wherever direct HTTPS egress is available::

    python -m midterm.collect.polymarket
    python -m midterm.collect.kalshi
    python -m midterm.collect.yahoo
    python -m midterm.collect.google_news
    python -m midterm.collect.gdelt
    python -m midterm.collect.reddit_arctic
"""

"""Small HTTP helper with polite retry/back-off."""
from __future__ import annotations

import time

import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/128.0 Safari/537.36 (midterm-sentiment academic research)"}


def get(url: str, params: dict | None = None, *, tries: int = 8, pause: float = 1.0, timeout: float = 60.0,
        rate_limited_text: str | None = None) -> requests.Response:
    last = None
    for a in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200 and not (rate_limited_text and rate_limited_text in r.text[:200]):
                time.sleep(pause)
                return r
            reset = r.headers.get("x-ratelimit-reset")
            wait = float(reset) + 1 if reset else min(60, 5 * (a + 1))
            last = f"{r.status_code}: {r.text[:120]}"
        except requests.RequestException as e:  # network error -> back off
            wait, last = min(60, 5 * (a + 1)), str(e)
        time.sleep(wait)
    raise RuntimeError(f"GET failed after {tries} tries: {url} ({last})")

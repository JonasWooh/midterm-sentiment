"""Reassemble payloads exported from the browser bridge into data/raw.

Context
-------
In the execution environment used for this study, outbound HTTP from Python was
blocked by an egress allow-list, while a desktop browser could reach the public
APIs. The collectors in ``browser_collectors/*.js`` therefore ran in the browser,
serialised their result with ``JSON.stringify`` into ``window.__buf``, and
exported it in fixed-size chunks of the form::

    '#CHUNK<i>#' + __buf.slice(i*C, (i+1)*C).padEnd(C, ' ')

Each chunk came back as a tool-result file containing a JSON array of
``{"type": "text", "text": <json-encoded string> + optional trailer}``.
This module decodes those files, strips padding, concatenates chunks in index
order and verifies the declared total length and a 32-bit FNV-1a checksum
computed in the browser over the same string (``__buf``).

When network access is available the pure-Python collectors in
``midterm.collect`` produce the same raw files directly and this module is not
needed.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_CHUNK_RE = re.compile(r"^#CHUNK(\d+)#")


def _units(s: str) -> bytes:
    """UTF-16LE code units (JS string semantics; lone surrogates preserved)."""
    return s.encode("utf-16-le", "surrogatepass")


def fnv1a32_units(b: bytes) -> int:
    """32-bit FNV-1a over UTF-16 code units (matches ``__fnv`` in bridge.js)."""
    h = 0x811C9DC5
    for i in range(0, len(b), 2):
        h ^= b[i] | (b[i + 1] << 8)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def fnv1a32(s: str) -> int:
    return fnv1a32_units(_units(s))


def _decode_tool_file(path: Path) -> str:
    items = json.loads(path.read_text())
    text = items[0]["text"]
    # The bridge may append a trailer such as "\n\n(captured at origin ...)".
    end = text.rfind('"')
    if not text.startswith('"') or end <= 0:
        raise ValueError(f"{path}: payload is not a JSON string")
    return json.loads(text[: end + 1])


def reassemble(paths: list[Path], total_len: int, checksum: int | None) -> str:
    # JS slices strings by UTF-16 code unit, so a surrogate pair can straddle two
    # chunks; all length/padding arithmetic is therefore done on UTF-16 units.
    chunks: dict[int, bytes] = {}
    for p in paths:
        s = _decode_tool_file(p)
        m = _CHUNK_RE.match(s)
        if not m:
            raise ValueError(f"{p}: missing chunk header")
        idx = int(m.group(1))
        if idx in chunks:
            raise ValueError(f"duplicate chunk {idx}")
        chunks[idx] = _units(s[m.end():])
    if sorted(chunks) != list(range(len(chunks))):
        raise ValueError(f"non-contiguous chunk indices {sorted(chunks)}")
    joined = b"".join(chunks[i] for i in range(len(chunks)))
    if len(joined) // 2 < total_len:
        raise ValueError(f"payload truncated: {len(joined) // 2} < {total_len}")
    body, pad = joined[: 2 * total_len], joined[2 * total_len:]
    if pad.decode("utf-16-le", "surrogatepass").strip():
        raise ValueError("non-padding content after declared length")
    if checksum is not None and fnv1a32_units(body) != checksum:
        raise ValueError("checksum mismatch")
    payload = body.decode("utf-16-le", "surrogatepass")
    payload = payload.encode("utf-16", "surrogatepass").decode("utf-16")  # re-pair surrogates
    json.loads(payload)  # must be valid JSON
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output file under data/raw")
    ap.add_argument("--len", type=int, required=True, dest="total_len")
    ap.add_argument("--fnv", type=int, default=None)
    ap.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args()
    payload = reassemble(a.files, a.total_len, a.fnv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload)
    print(f"wrote {out} ({len(payload):,} chars, {len(a.files)} chunk(s))")


if __name__ == "__main__":
    main()

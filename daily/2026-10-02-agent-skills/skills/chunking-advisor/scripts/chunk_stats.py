"""Tier-3 skill script: measure chunk-size distribution of a corpus.

Run by the agent (not loaded into its context) to ground chunking advice in
data. Usage:
    python scripts/chunk_stats.py --chars 512 --overlap 75 file1.txt file2.txt
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path


def recursive_split(text: str, size: int, overlap: int) -> list[str]:
    """Minimal recursive character splitter (\\n\\n -> \\n -> ' ' -> '')."""
    separators = ["\n\n", "\n", " ", ""]
    chunks: list[str] = []

    def _split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= size or not seps:
            return [t]
        sep, rest = seps[0], seps[1:]
        parts = t.split(sep) if sep else list(t)
        out: list[str] = []
        buf = ""
        for p in parts:
            piece = p if not sep else (sep + p if buf else p)
            if len(buf) + len(piece) > size and buf:
                out.append(buf)
                buf = buf[-overlap:] if overlap < len(buf) else ""
            buf += piece
        if buf:
            out.append(buf)
        result: list[str] = []
        for c in out:
            result.extend(_split(c, rest) if len(c) > size else [c])
        return result

    for c in _split(text, separators):
        chunks.append(c)
    return chunks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chunk-size stats for a text corpus")
    ap.add_argument("files", nargs="+", help="text files to analyse")
    ap.add_argument("--chars", type=int, default=512, help="target chunk size in chars")
    ap.add_argument("--overlap", type=int, default=75, help="overlap in chars")
    args = ap.parse_args(argv)

    sizes: list[int] = []
    n_files = 0
    for f in args.files:
        text = Path(f).read_text(encoding="utf-8")
        n_files += 1
        sizes.extend(len(c) for c in recursive_split(text, args.chars, args.overlap))

    if not sizes:
        print("no chunks produced", file=sys.stderr)
        return 1
    print(f"files: {n_files}  chunks: {len(sizes)}")
    print(f"mean: {statistics.mean(sizes):.1f} chars")
    print(f"median: {statistics.median(sizes):.1f} chars")
    print(f"min/max: {min(sizes)}/{max(sizes)} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

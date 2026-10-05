"""Document chunking strategies under test.

Each strategy turns a markdown document into a list of Chunk objects.
The benchmark compares them on retrieval quality, so every strategy must be
deterministic and must preserve the source text verbatim (no rewriting).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: str
    strategy: str
    start: int  # char offset of the chunk inside the source document
    end: int


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with a dependency-free regex splitter."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    # Merge fragments that are clearly not sentence ends (e.g. "e.g. Smith").
    merged: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged and len(merged[-1].split()) <= 3 and not merged[-1].endswith((".", "!", "?")):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return merged


def _make_chunks(doc_id: str, strategy: str, pieces: list[tuple[str, int, int]]) -> list[Chunk]:
    chunks = []
    for i, (text, start, end) in enumerate(pieces):
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                text=text,
                doc_id=doc_id,
                chunk_id=f"{doc_id}#{strategy}#{i}",
                strategy=strategy,
                start=start,
                end=end,
            )
        )
    return chunks


def fixed_size_chunks(
    doc_id: str, text: str, size: int = 1000, overlap: int = 100, strategy: str = "fixed"
) -> list[Chunk]:
    """Naive character-window chunking with a sliding overlap."""
    if size <= 0:
        raise ValueError("size must be positive")
    overlap = max(0, min(overlap, size - 1))
    pieces: list[tuple[str, int, int]] = []
    step = size - overlap
    for start in range(0, len(text), step):
        end = min(start + size, len(text))
        pieces.append((text[start:end], start, end))
        if end == len(text):
            break
    return _make_chunks(doc_id, strategy, pieces)


def sentence_chunks(
    doc_id: str,
    text: str,
    max_chars: int = 1000,
    overlap_sentences: int = 1,
    strategy: str = "sentence",
) -> list[Chunk]:
    """Greedily pack whole sentences into chunks up to max_chars.

    Sentences are never cut mid-way; the last `overlap_sentences` sentences of
    a chunk are repeated at the start of the next one for context continuity.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    sentences = split_sentences(text)
    if not sentences:
        return []
    pieces: list[tuple[str, int, int]] = []
    cursor = 0  # char offset into the original text for `start`
    i = 0
    # Map each sentence back to a char offset so Chunk.start stays truthful.
    offsets: list[int] = []
    search_from = 0
    for sent in sentences:
        idx = text.find(sent, search_from)
        if idx == -1:
            idx = search_from
        offsets.append(idx)
        search_from = idx + len(sent)

    while i < len(sentences):
        current: list[str] = []
        length = 0
        start = offsets[i]
        j = i
        while j < len(sentences):
            sent = sentences[j]
            extra = len(sent) + (1 if current else 0)
            if current and length + extra > max_chars:
                break
            current.append(sent)
            length += extra
            j += 1
        if not current:  # single sentence longer than max_chars: keep it whole
            current = [sentences[j]]
            j += 1
        end = offsets[j - 1] + len(sentences[j - 1])
        pieces.append((" ".join(current), start, end))
        # Overlap by repeating trailing sentences, but always make progress.
        i = max(j - overlap_sentences, i + 1) if j < len(sentences) else j
    return _make_chunks(doc_id, strategy, pieces)


def heading_chunks(
    doc_id: str, text: str, max_chars: int = 1000, strategy: str = "heading"
) -> list[Chunk]:
    """Structure-aware chunking: split on markdown headings, then sentence-pack
    any section that still exceeds max_chars. The heading line is prepended to
    each of its chunks so retrieved chunks stay self-describing."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    matches = list(_HEADING_RE.finditer(text))
    sections: list[tuple[str, str, int]] = []  # (heading, body, body_start_offset)
    if not matches:
        sections.append(("", text, 0))
    else:
        if matches[0].start() > 0:  # preamble before the first heading
            sections.append(("", text[: matches[0].start()], 0))
        for n, m in enumerate(matches):
            body_start = m.end()
            body_end = matches[n + 1].start() if n + 1 < len(matches) else len(text)
            sections.append((m.group(0).strip(), text[body_start:body_end], body_start))

    pieces: list[tuple[str, int, int]] = []
    for heading, body, body_start in sections:
        body = body.strip()
        if not body:
            continue
        if len(body) + len(heading) + 1 <= max_chars:
            chunk_text = f"{heading}\n{body}" if heading else body
            pieces.append((chunk_text, body_start, body_start + len(body)))
        else:
            # Oversized section: fall back to sentence packing, heading prepended.
            for sub in sentence_chunks(doc_id, body, max_chars=max_chars - len(heading) - 1,
                                      overlap_sentences=1, strategy=strategy):
                chunk_text = f"{heading}\n{sub.text}" if heading else sub.text
                pieces.append((chunk_text, body_start + sub.start, body_start + sub.end))
    return _make_chunks(doc_id, strategy, pieces)

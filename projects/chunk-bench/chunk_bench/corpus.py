"""Corpus loading: reads the bundled markdown docs from data/corpus/."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    corpus_dir = Path(corpus_dir)
    docs: list[Document] = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        first_line = text.splitlines()[0].strip()
        title = first_line.lstrip("# ").strip() if first_line.startswith("#") else path.stem
        docs.append(Document(doc_id=path.stem, title=title, text=text))
    if not docs:
        raise FileNotFoundError(f"No markdown documents found in {corpus_dir}")
    return docs

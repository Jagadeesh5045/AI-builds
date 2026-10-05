"""Gold QA set loading and validation helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .metrics import normalize


@dataclass
class QAPair:
    question: str
    answer_span: str
    doc_id: str


def load_qa_pairs(qa_path: str | Path) -> list[QAPair]:
    data = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    pairs = [QAPair(question=item["question"], answer_span=item["answer_span"],
                    doc_id=item["doc_id"]) for item in data]
    if not pairs:
        raise ValueError(f"No QA pairs found in {qa_path}")
    return pairs


def validate_qa_against_corpus(pairs: list[QAPair], docs: dict[str, str]) -> list[str]:
    """Return a list of problems; empty means every answer span is genuinely
    present (normalised) in its stated document. Used by the test suite."""
    problems = []
    for pair in pairs:
        doc_text = docs.get(pair.doc_id)
        if doc_text is None:
            problems.append(f"unknown doc_id {pair.doc_id!r} for question {pair.question!r}")
            continue
        if normalize(pair.answer_span) not in normalize(doc_text):
            problems.append(
                f"answer span not found in doc {pair.doc_id!r}: {pair.answer_span!r}"
            )
    return problems

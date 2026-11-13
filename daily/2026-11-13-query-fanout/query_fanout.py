"""
Query fan-out planner: decompose a complex question, retrieve per part, report coverage.

Trending context (13 Nov 2026): Google released Gemini 4 Hydrogen, a fast
agentic model, and rolled it into Search / AI Mode, where Google highlights
better handling of complicated multi-part questions and enhanced reasoning
for *query fan-out*: instead of searching once with the whole tangled
question, the system splits it into sub-queries, searches for each one, and
synthesises an answer that covers every part.

This demo rebuilds that mechanism in pure stdlib Python over a small local
corpus, so every step is inspectable:

  1. decompose(question)     -> rule-based split into sub-questions
  2. TfidfIndex.search(q)    -> cosine retrieval over the corpus
  3. fanout(question)        -> one search per sub-question, merged evidence
  4. coverage report         -> each sub-question marked covered / weak /
                                missing from its best score, never guessed
  5. single-shot baseline    -> one search with the whole question, for an
                                honest comparison of facet coverage

The point for data / RAG work: a whole-question embedding or TF-IDF query
dilutes each facet, so a dashboard question that asks about cost AND latency
AND rollback tends to retrieve whichever facet dominates the wording. Fan-out
gives every facet its own retrieval budget, and the coverage report makes an
unanswerable part visible instead of letting the synthesis step hallucinate
over the gap.

No model, no network, no dependencies. Run:  python3 query_fanout.py
Tests:                                       python3 -m pytest test_query_fanout.py
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Corpus: one document per facet of a warehouse-migration decision.
# Wording is deliberately facet-specific so retrieval has a real signal.
# --------------------------------------------------------------------------

CORPUS: list[dict] = [
    {
        "id": "cost",
        "title": "Monthly warehouse cost",
        "text": (
            "The migrated warehouse costs 1840 pounds per month in compute "
            "credits at the current tier. Storage adds 96 pounds per month. "
            "Moving up one pricing tier raises the monthly cost to 2600 "
            "pounds but doubles the compute credits included."
        ),
    },
    {
        "id": "latency",
        "title": "Dashboard query latency",
        "text": (
            "Dashboard queries are fast after the migration, with a p95 "
            "latency of 1.8 seconds, down from 6.4 seconds. The morning "
            "refresh finishes in 11 minutes and cached dashboards load "
            "fast, in under a second. Cold queries on the largest table "
            "take 4 seconds."
        ),
    },
    {
        "id": "data_quality",
        "title": "Data quality checks before load",
        "text": (
            "Seventeen data quality checks run before every load: null "
            "checks, duplicate detection, referential integrity, row count "
            "reconciliation and freshness checks. Rows that fail a check are "
            "quarantined, 503 rows were quarantined in the last run, and the "
            "load is blocked if a critical check fails."
        ),
    },
    {
        "id": "rollback",
        "title": "Rollback plan if cutover fails",
        "text": (
            "If the cutover fails, rollback restores the previous warehouse "
            "snapshot within 45 minutes. The legacy system stays read-only "
            "for seven days after cutover so traffic can switch back. A "
            "rollback rehearsal is run the week before the migration."
        ),
    },
    {
        "id": "security",
        "title": "Access control and row level security",
        "text": (
            "Access is controlled with role based permissions and row level "
            "security, so analysts only see rows for their own region. Data "
            "is encrypted at rest and in transit, and every access to "
            "restricted tables is written to an audit log."
        ),
    },
    {
        "id": "timeline",
        "title": "Migration timeline and cutover weekend",
        "text": (
            "The migration runs in three phases over six weeks. Phase one "
            "copies historical data, phase two dual writes new data, and the "
            "cutover happens on the final weekend. Training sessions for "
            "analysts are scheduled in week five, before the cutover."
        ),
    },
]

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "how", "what", "which", "who", "when", "why", "can", "could", "will",
    "would", "we", "i", "it", "in", "on", "of", "to", "for", "and", "or",
    "if", "per", "each", "our", "their", "they", "them", "this", "that",
    "with", "from", "into", "after", "before", "much", "many",
}

_INTERROGATIVE = (
    r"(?:how|what|which|who|whom|when|why|can|could|do|does|did|is|are|"
    r"was|were|will|would)"
)


def tokenise(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


# --------------------------------------------------------------------------
# Step 1: decomposition
# --------------------------------------------------------------------------

def decompose(question: str) -> list[str]:
    """
    Split a complex question into sub-questions with transparent rules:

    * semicolons always separate parts
    * ", and <interrogative>" and " and <interrogative>" separate parts
    * ", <interrogative>" separates parts in a comma list
    * a leading "also" on a part is dropped

    A simple question returns a single-element list: no fan-out is spent
    where there is nothing to fan out.
    """
    q = question.strip().rstrip("?").strip()
    if not q:
        return []
    parts: list[str] = []
    for chunk in q.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        chunk = re.split(
            r",?\s+and\s+(?=" + _INTERROGATIVE + r"\b)", chunk,
            flags=re.IGNORECASE,
        )
        split_parts: list[str] = []
        for piece in chunk:
            split_parts.extend(
                re.split(
                    r",\s+(?=" + _INTERROGATIVE + r"\b)", piece,
                    flags=re.IGNORECASE,
                )
            )
        for piece in split_parts:
            piece = re.sub(r"^also\s+", "", piece.strip(), flags=re.IGNORECASE)
            piece = piece.strip(" ,")
            if piece:
                parts.append(piece)
    return parts


# --------------------------------------------------------------------------
# Step 2: TF-IDF retrieval (stdlib only)
# --------------------------------------------------------------------------

class TfidfIndex:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.doc_tokens = [tokenise(d["title"] + " " + d["text"]) for d in docs]
        n = len(docs)
        df: dict[str, int] = {}
        for toks in self.doc_tokens:
            for tok in set(toks):
                df[tok] = df.get(tok, 0) + 1
        self.idf = {
            tok: math.log((1 + n) / (1 + count)) + 1.0
            for tok, count in df.items()
        }
        self.doc_vectors = [self._vectorise(toks) for toks in self.doc_tokens]

    def _vectorise(self, toks: list[str]) -> dict[str, float]:
        vec: dict[str, float] = {}
        for tok in toks:
            vec[tok] = vec.get(tok, 0.0) + self.idf.get(tok, 1.0)
        return vec

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        """Return [(doc_id, cosine_score), ...] best first."""
        qv = self._vectorise(tokenise(query))
        scored = [
            (self.docs[i]["id"], self._cosine(qv, dv))
            for i, dv in enumerate(self.doc_vectors)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


# --------------------------------------------------------------------------
# Step 3 + 4: fan-out with a coverage report
# --------------------------------------------------------------------------

COVERED_THRESHOLD = 0.20   # best score at or above this: facet is covered
WEAK_THRESHOLD = 0.08      # below covered but above this: weak evidence only


@dataclass
class SubResult:
    sub_question: str
    best_doc: str | None
    best_score: float
    status: str  # "covered" | "weak" | "missing"


@dataclass
class FanoutResult:
    question: str
    sub_results: list[SubResult] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        if not self.sub_results:
            return 0.0
        covered = sum(1 for s in self.sub_results if s.status == "covered")
        return covered / len(self.sub_results)

    @property
    def docs_used(self) -> set[str]:
        return {s.best_doc for s in self.sub_results if s.best_doc}


def fanout(question: str, index: TfidfIndex) -> FanoutResult:
    result = FanoutResult(question=question)
    for sub in decompose(question):
        hits = index.search(sub, k=1)
        doc_id, score = hits[0] if hits else (None, 0.0)
        if score >= COVERED_THRESHOLD:
            status = "covered"
        elif score >= WEAK_THRESHOLD:
            status = "weak"
        else:
            status = "missing"
        result.sub_results.append(
            SubResult(
                sub_question=sub,
                best_doc=doc_id if status != "missing" else None,
                best_score=round(score, 4),
                status=status,
            )
        )
    return result


def single_shot(question: str, index: TfidfIndex, k: int = 3) -> list[str]:
    """Baseline: one search with the whole complex question."""
    return [doc_id for doc_id, score in index.search(question, k=k) if score >= WEAK_THRESHOLD]


def facet_coverage(doc_ids: set[str] | list[str], wanted: set[str]) -> float:
    if not wanted:
        return 0.0
    return len(set(doc_ids) & wanted) / len(wanted)


def synthesise(result: FanoutResult, index: TfidfIndex) -> str:
    """
    Assemble an answer skeleton: one cited line per sub-question.
    Missing parts are stated as missing, never filled in by guessing.
    """
    by_id = {d["id"]: d for d in index.docs}
    lines = []
    for s in result.sub_results:
        if s.status == "missing" or s.best_doc is None:
            lines.append(f"- UNANSWERED: {s.sub_question} (no evidence in corpus)")
        else:
            doc = by_id[s.best_doc]
            lines.append(
                f"- {s.sub_question} -> [{doc['id']}] {doc['title']} "
                f"(score {s.best_score:.2f}, {s.status})"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

DEMO_QUESTIONS = [
    "What does the warehouse migration cost each month, how fast will "
    "dashboards load, and how do we roll back if the cutover fails?",
    "Which data quality checks run before load and who can access "
    "restricted rows?",
    "What is the monthly cost of the warehouse?",  # simple: no fan-out needed
    "What does the warehouse cost each month and what training is provided "
    "for executives on presentation skills?",  # second facet is not in corpus
]


def main() -> None:
    index = TfidfIndex(CORPUS)
    for question in DEMO_QUESTIONS:
        result = fanout(question, index)
        baseline_docs = single_shot(question, index, k=3)
        print("=" * 72)
        print(f"Q: {question}")
        print(f"Sub-questions: {len(result.sub_results)} | "
              f"fan-out coverage: {result.coverage:.0%} | "
              f"baseline docs: {baseline_docs}")
        print(synthesise(result, index))
    print("=" * 72)


if __name__ == "__main__":
    main()

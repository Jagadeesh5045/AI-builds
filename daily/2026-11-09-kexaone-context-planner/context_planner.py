"""
K-EXAONE 2.5-style context planner.

Trending context (9 Nov 2026): LG AI Research released K-EXAONE 2.5 with a
free API on the EXAONE API Platform, announced at NVIDIA AI Day Seoul.
Reported specs (LG / platform docs, vendor-reported):
  - 524,288-token combined limit: input + max_tokens must not exceed it,
    otherwise the API returns HTTP 400
  - reasoning is always on: reasoning_effort is low / high / max
    (default max) and cannot be switched off; reasoning tokens count
    toward max_tokens
  - OpenAI-compatible POST /v1/chat/completions, model k-exaone_v2.5
  - text only, Korean and English

This demo is NOT the model. It is the client-side planning problem the
spec creates: before you send anything, you must budget three things
inside one hard 524,288 ceiling
  1. the prompt / retrieved context you pack in,
  2. the reasoning the model will always spend,
  3. the visible answer you want back.
If you forget (2) - the classic mistake with optional-thinking models -
the request either 400s or the answer is silently starved.

Stdlib only. Token counts here use a deterministic estimator, not LG's
tokeniser; the arithmetic and invariants are the point.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum

CONTEXT_LIMIT = 524_288          # exact combined ceiling from the docs
MODEL_ID = "k-exaone_v2.5"

_WORD = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']")


def estimate_tokens(text: str) -> int:
    """Deterministic rough token estimate: word / punctuation pieces.

    Good enough for planning arithmetic and fully reproducible in tests.
    Empty text is 0 tokens.
    """
    if not text:
        return 0
    return len(_WORD.findall(text))


def _terms(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text) if t.isalnum()]


class Effort(str, Enum):
    LOW = "low"
    HIGH = "high"
    MAX = "max"
    # Deliberately no OFF member: K-EXAONE 2.5 reasoning cannot be
    # turned off. parse_effort() rejects "off" explicitly (see below).


# Illustrative reasoning reserves for this demo only. LG publishes the
# effort names and the default (max), not token counts per tier, so
# these numbers are the planner's own policy, labelled as such.
REASONING_RESERVE = {
    Effort.LOW: 2_048,
    Effort.HIGH: 16_384,
    Effort.MAX: 65_536,
}


def parse_effort(value: str | Effort | None) -> Effort:
    """Parse an effort value. None means the documented default: max."""
    if value is None:
        return Effort.MAX
    if isinstance(value, Effort):
        return value
    v = str(value).strip().lower()
    if v in ("off", "none", "false", "0", "disabled"):
        raise ValueError(
            "reasoning cannot be turned off on K-EXAONE 2.5: "
            "use low, high or max"
        )
    try:
        return Effort(v)
    except ValueError:
        raise ValueError(f"unknown reasoning_effort: {value!r}") from None


class ContextOverflowError(ValueError):
    """Mirror of the API's 400: input + max_tokens exceeds the limit."""

    def __init__(self, input_tokens: int, max_tokens: int, limit: int):
        self.input_tokens = input_tokens
        self.max_tokens = max_tokens
        self.limit = limit
        super().__init__(
            f"input ({input_tokens}) + max_tokens ({max_tokens}) = "
            f"{input_tokens + max_tokens} exceeds limit {limit} (HTTP 400)"
        )


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str = ""

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass
class Plan:
    effort: Effort
    reasoning_reserve: int
    answer_reserve: int
    input_tokens: int          # query + packed chunks + overhead
    max_tokens: int            # reasoning_reserve + answer_reserve
    kept: list[Chunk] = field(default_factory=list)
    dropped: list[Chunk] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.input_tokens + self.max_tokens

    @property
    def headroom(self) -> int:
        return CONTEXT_LIMIT - self.total

    def validate(self, limit: int = CONTEXT_LIMIT) -> bool:
        if self.total > limit:
            raise ContextOverflowError(self.input_tokens, self.max_tokens, limit)
        if self.reasoning_reserve <= 0:
            raise ValueError("always-on invariant broken: no reasoning reserve")
        return True


STOPWORDS = {
    "a", "an", "the", "in", "on", "of", "to", "and", "or", "is", "was",
    "did", "does", "had", "has", "which", "why", "what", "when", "for",
    "by", "at", "it", "its", "this", "that", "with", "from",
}


def relevance(query: str, chunk: Chunk) -> float:
    """TF overlap score in [0, 1]: share of query terms covered, with a
    small bonus for repeated coverage. Deterministic, no embeddings.
    Stopwords are excluded so 'the / in / which' cannot fake relevance."""
    q = {t for t in _terms(query) if t not in STOPWORDS}
    if not q:
        return 0.0
    c_terms = [t for t in _terms(chunk.text) if t not in STOPWORDS]
    if not c_terms:
        return 0.0
    covered = q.intersection(c_terms)
    if not covered:
        return 0.0
    coverage = len(covered) / len(q)
    freq = sum(c_terms.count(t) for t in covered)
    return min(1.0, coverage + 0.05 * math.log1p(freq))


OVERHEAD_TOKENS = 32  # system / formatting allowance in this demo


def plan_request(
    query: str,
    chunks: list[Chunk],
    *,
    effort: str | Effort | None = None,
    answer_tokens: int = 1_024,
    limit: int = CONTEXT_LIMIT,
) -> Plan:
    """Build a request plan that always fits the combined ceiling.

    Strategy: reserve reasoning first (it is not optional), reserve the
    answer, then greedily pack the most relevant chunks into what is
    left. Chunks that do not fit are dropped in relevance order, never
    silently truncated mid-chunk.
    """
    if answer_tokens <= 0:
        raise ValueError("answer_tokens must be positive")
    eff = parse_effort(effort)
    reasoning = REASONING_RESERVE[eff]
    max_tokens = reasoning + answer_tokens
    query_tokens = estimate_tokens(query)
    fixed_input = query_tokens + OVERHEAD_TOKENS
    if fixed_input + max_tokens > limit:
        raise ContextOverflowError(fixed_input, max_tokens, limit)

    budget_for_chunks = limit - max_tokens - fixed_input
    scored = sorted(
        chunks,
        key=lambda c: (-relevance(query, c), c.chunk_id),
    )
    scores = {c.chunk_id: round(relevance(query, c), 4) for c in chunks}
    kept: list[Chunk] = []
    dropped: list[Chunk] = []
    used = 0
    for c in scored:
        # Zero-relevance chunks are dropped as irrelevant, not packed
        # just because space remains: they dilute the context and cost
        # reasoning time for no evidence.
        if scores[c.chunk_id] <= 0.0:
            dropped.append(c)
        elif used + c.tokens <= budget_for_chunks:
            kept.append(c)
            used += c.tokens
        else:
            dropped.append(c)
    plan = Plan(
        effort=eff,
        reasoning_reserve=reasoning,
        answer_reserve=answer_tokens,
        input_tokens=fixed_input + used,
        max_tokens=max_tokens,
        kept=kept,
        dropped=dropped,
        scores=scores,
    )
    plan.validate(limit)
    return plan


def build_payload(plan: Plan, query: str) -> dict:
    """OpenAI-compatible payload shape for POST /v1/chat/completions."""
    context = "\n\n".join(
        f"[{c.chunk_id}] {c.text}" for c in plan.kept
    )
    return {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "Answer using only the context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        "max_tokens": plan.max_tokens,
        "reasoning_effort": plan.effort.value,
    }


def demo_corpus() -> list[Chunk]:
    """Synthetic analytics corpus: regional sales notes, mixed relevance."""
    chunks: list[Chunk] = []
    regions = ["North", "South", "East", "West", "Central"]
    for i, region in enumerate(regions):
        chunks.append(Chunk(
            f"sales-{region.lower()}",
            f"{region} region revenue grew in Q3. {region} revenue was driven "
            f"by renewals and two large orders. Returns stayed low in {region}.",
            source="q3-report",
        ))
    chunks.append(Chunk(
        "returns-deep-dive",
        "Returns rose in the South in Q3 because a packaging defect damaged "
        "fragile items. Returns cost and refund handling are detailed here.",
        source="ops-note",
    ))
    chunks.append(Chunk(
        "hr-note",
        "The office picnic is planned for Friday. Please bring a dish and "
        "sign up for the football match after lunch.",
        source="hr",
    ))
    chunks.append(Chunk(
        "weather-note",
        "Rain is expected over the weekend with cooler temperatures and "
        "strong winds along the coast on Sunday.",
        source="news",
    ))
    for i in range(12):
        chunks.append(Chunk(
            f"filler-{i:02d}",
            " ".join(["general background information about company history"] * 20),
            source="archive",
        ))
    return chunks


def main() -> None:
    query = "Which region had revenue growth in Q3 and why did returns rise in the South?"
    corpus = demo_corpus()
    print(f"K-EXAONE 2.5 context planner demo")
    print(f"Combined limit: {CONTEXT_LIMIT:,} tokens (input + max_tokens)")
    print(f"Corpus: {len(corpus)} chunks, {sum(c.tokens for c in corpus):,} est. tokens")
    print(f"Query: {query}\n")
    for effort in (Effort.LOW, Effort.HIGH, Effort.MAX):
        plan = plan_request(query, corpus, effort=effort, answer_tokens=1024)
        top = plan.kept[0].chunk_id if plan.kept else "-"
        print(
            f"effort={effort.value:4s} reasoning_reserve={plan.reasoning_reserve:6,} "
            f"max_tokens={plan.max_tokens:6,} input={plan.input_tokens:4,} "
            f"total={plan.total:6,} kept={len(plan.kept)} dropped={len(plan.dropped)} "
            f"top_chunk={top}"
        )
    # Small-limit run: same planner, tighter ceiling, shows dropping.
    # Note the floor: even at low effort the always-on reserve is
    # 2,048 tokens, so a limit below ~2,400 cannot be planned at all.
    small = plan_request(query, corpus, effort=Effort.LOW,
                         answer_tokens=256, limit=3000)
    print(f"\nTight limit 3,000, effort=low: kept {[c.chunk_id for c in small.kept]}")
    print(f"Dropped {len(small.dropped)} chunks, total {small.total} <= 3,000: "
          f"{small.total <= 3000}")
    payload = build_payload(small, query)
    print(f"Payload model={payload['model']} reasoning_effort="
          f"{payload['reasoning_effort']} max_tokens={payload['max_tokens']}")
    try:
        parse_effort("off")
    except ValueError as exc:
        print(f"effort=off correctly rejected: {exc}")


if __name__ == "__main__":
    main()

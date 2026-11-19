"""Next-Concept Modeling (NCM), in miniature - stdlib only.

Trending source (19 Nov 2026): Hazy Research (Stanford) published
"Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation".
NCM represents text as a hierarchy of discrete codes and generates one
LEVEL at a time, predicting every code inside a level in parallel,
instead of one token at a time left-to-right. The paper reports
competitive-or-better quality than strong diffusion baselines while
generating 10-31x faster and seeing 4-5x fewer training tokens
(paper figures, not measured here).

This demo builds the smallest honest version of that idea:

  * A two-level hierarchy over synthetic topic sentences.
    Level 1 (concepts): one topic code per clause.
    Level 2 (details):  the exact words inside each clause.
  * An encoder that maps a sentence to (concepts, details).
  * Two decoders over the same codes:
      - autoregressive: emits one token per sequential step;
      - NCM-style: emits ALL concepts in one parallel step, then ALL
        details in one parallel step (2 sequential steps total).
  * Measurements: sequential steps, parallel width, a step-budget
    comparison (what each decoder finishes inside a fixed number of
    sequential steps), exact reconstruction, and a compression view
    (how much of the sentence the concept level alone pins down).

Nothing here trains a neural network; the point is the generation
SCHEDULE - where the parallelism comes from, what it costs in code
size, and why coarse codes must be predictable before details can be
filled in safely. Run:  python3 next_concept.py
Tests:                  python3 test_next_concept.py
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

# ----------------------------------------------------------------------
# Synthetic corpus: clauses drawn from four topics. Each clause is a
# (topic, template, slots) triple; rendering picks slot values, so the
# concept (topic) is recoverable but the wording varies.
# ----------------------------------------------------------------------

TOPICS = {
    "cooking": {
        "verbs": ["simmer", "whisk", "season", "roast"],
        "objects": ["the sauce", "the batter", "the onions", "the tray"],
        "extras": ["slowly", "gently", "until golden", "with care"],
    },
    "finance": {
        "verbs": ["audit", "hedge", "settle", "forecast"],
        "objects": ["the ledger", "the portfolio", "the invoice", "the budget"],
        "extras": ["quarterly", "carefully", "before close", "on schedule"],
    },
    "health": {
        "verbs": ["measure", "record", "review", "schedule"],
        "objects": ["the dosage", "the scan", "the notes", "the checkup"],
        "extras": ["daily", "twice", "in clinic", "after review"],
    },
    "weather": {
        "verbs": ["track", "report", "predict", "observe"],
        "objects": ["the storm", "the rainfall", "the winds", "the frost"],
        "extras": ["overnight", "locally", "at dawn", "across the valley"],
    },
}

# Words that belong to exactly one topic in this corpus; the encoder
# uses them as concept evidence (a stand-in for a learned codebook).
TOPIC_MARKERS: dict[str, str] = {}
for _topic, _parts in TOPICS.items():
    for _word in _parts["verbs"] + [o.split()[-1] for o in _parts["objects"]]:
        TOPIC_MARKERS[_word] = _topic


def render_clause(topic: str, rng: random.Random) -> str:
    parts = TOPICS[topic]
    return " ".join(
        [rng.choice(parts["verbs"]), rng.choice(parts["objects"]),
         rng.choice(parts["extras"])]
    )


def render_sentence(rng: random.Random, n_clauses: int = 3) -> str:
    topics = rng.sample(sorted(TOPICS), k=min(n_clauses, len(TOPICS)))
    return " ; ".join(render_clause(t, rng) for t in topics)


def build_corpus(n: int = 240, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    return [render_sentence(rng, rng.choice([2, 3, 3, 4])) for _ in range(n)]


# ----------------------------------------------------------------------
# Hierarchy: encode / decode
# ----------------------------------------------------------------------

def split_clauses(sentence: str) -> list[str]:
    return [c.strip() for c in sentence.split(";") if c.strip()]


def clause_concept(clause: str) -> str:
    """Concept code for one clause: the topic evidenced by its markers."""
    votes = Counter(
        TOPIC_MARKERS[w] for w in clause.replace(";", " ").split()
        if w in TOPIC_MARKERS
    )
    if not votes:
        return "unknown"
    return votes.most_common(1)[0][0]


def encode(sentence: str) -> tuple[list[str], list[list[str]]]:
    """Sentence -> (level-1 concept codes, level-2 detail tokens)."""
    clauses = split_clauses(sentence)
    concepts = [clause_concept(c) for c in clauses]
    details = [c.split() for c in clauses]
    return concepts, details


def decode(concepts: list[str], details: list[list[str]]) -> str:
    """Rebuild the sentence; concepts are checked against the details."""
    clauses = []
    for concept, words in zip(concepts, details):
        actual = clause_concept(" ".join(words))
        if actual != concept:
            raise ValueError(f"concept {concept!r} disagrees with details ({actual!r})")
        clauses.append(" ".join(words))
    return " ; ".join(clauses)


# ----------------------------------------------------------------------
# Generation schedules over the same code stream
# ----------------------------------------------------------------------

def flat_tokens(sentence: str) -> list[str]:
    return sentence.replace(";", " ;").split()


def autoregressive_steps(sentence: str) -> int:
    """One sequential step per emitted token."""
    return len(flat_tokens(sentence))


def ncm_schedule(sentence: str) -> dict:
    """NCM-style schedule: level 1 in one parallel step, level 2 in one.

    Returns sequential step count and the parallel width of each step.
    """
    concepts, details = encode(sentence)
    widths = [len(concepts), sum(len(d) for d in details)]
    return {"steps": len(widths), "widths": widths, "total_codes": sum(widths)}


def tokens_within_budget(sentence: str, budget: int, mode: str) -> int:
    """Codes completed inside `budget` sequential steps.

    Autoregressive decoding finishes one token per step. NCM finishes a
    whole level per step, so a 2-step budget completes the sentence.
    """
    if mode == "ar":
        return min(budget, len(flat_tokens(sentence)))
    if mode == "ncm":
        sched = ncm_schedule(sentence)
        done, steps = 0, 0
        for w in sched["widths"]:
            steps += 1
            if steps > budget:
                break
            done += w
        return done
    raise ValueError(mode)


# ----------------------------------------------------------------------
# What the concept level alone pins down (compression view)
# ----------------------------------------------------------------------

def concept_only_recovery(corpus: list[str]) -> dict:
    """Given only the concept sequence, how much is already determined?

    For each corpus position we ask: across sentences sharing the same
    concept sequence, how often is the full sentence identical? That is
    the share the coarse level decides before any detail is generated.
    """
    by_concepts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
    for s in corpus:
        concepts, _ = encode(s)
        by_concepts[tuple(concepts)][s] += 1
    total, pinned = 0, 0
    for counts in by_concepts.values():
        group_total = sum(counts.values())
        total += group_total
        pinned += counts.most_common(1)[0][1]
    return {"sentences": total, "concept_groups": len(by_concepts),
            "share_pinned_by_concepts": pinned / total if total else 0.0}


def concept_predictability(corpus: list[str]) -> float:
    """How predictable is the NEXT concept from the previous one?

    Coarse-to-fine only helps if the coarse level is itself learnable;
    we report the majority-transition accuracy of concept bigrams.
    """
    transitions: dict[str, Counter] = defaultdict(Counter)
    for s in corpus:
        concepts, _ = encode(s)
        for a, b in zip(concepts, concepts[1:]):
            transitions[a][b] += 1
    correct = sum(c.most_common(1)[0][1] for c in transitions.values())
    total = sum(sum(c.values()) for c in transitions.values())
    return correct / total if total else 0.0


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

def main() -> None:
    corpus = build_corpus()
    ar_steps = [autoregressive_steps(s) for s in corpus]
    ncm = [ncm_schedule(s) for s in corpus]
    mean_ar = sum(ar_steps) / len(ar_steps)
    mean_ncm = sum(x["steps"] for x in ncm) / len(ncm)
    mean_width = sum(max(x["widths"]) for x in ncm) / len(ncm)

    # Exact reconstruction through the hierarchy.
    exact = sum(1 for s in corpus if decode(*encode(s)) == s)

    # Fixed 2-step budget: AR gets 2 tokens, NCM gets the whole sentence.
    budget = 2
    ar_done = sum(tokens_within_budget(s, budget, "ar") for s in corpus)
    ncm_done = sum(tokens_within_budget(s, budget, "ncm") for s in corpus)
    totals = sum(len(flat_tokens(s)) for s in corpus)
    total_codes = sum(x["total_codes"] for x in ncm)

    recovery = concept_only_recovery(corpus)
    pred = concept_predictability(corpus)

    print("Next-Concept Modeling in miniature (stdlib only)")
    print(f"corpus: {len(corpus)} synthetic sentences, 4 topics")
    print(f"autoregressive schedule : {mean_ar:.1f} sequential steps / sentence")
    print(f"NCM schedule (2 levels) : {mean_ncm:.1f} sequential steps / sentence")
    print(f"sequential-step speedup : {mean_ar / mean_ncm:.2f}x "
          f"(paper reports 10-31x generation speedup at model scale)")
    print(f"widest parallel step    : {mean_width:.1f} codes emitted at once")
    print(f"exact reconstruction    : {exact}/{len(corpus)} via concepts+details")
    print(f"inside a {budget}-step budget : AR completes {ar_done}/{totals} tokens, "
          f"NCM completes {ncm_done}/{total_codes} codes (concepts + details)")
    print(f"concept groups          : {recovery['concept_groups']} distinct concept "
          f"sequences; concepts alone pin the full sentence "
          f"{recovery['share_pinned_by_concepts']:.1%} of the time")
    print(f"next-concept bigram acc : {pred:.3f} (coarse level is learnable, "
          f"not random - a precondition for coarse-to-fine to help)")
    s = corpus[0]
    concepts, details = encode(s)
    print("\nexample:")
    print(f"  sentence : {s}")
    print(f"  level 1  : {concepts}")
    print(f"  level 2  : {[len(d) for d in details]} detail tokens per clause, "
          f"all emitted in one parallel step")


if __name__ == "__main__":
    main()

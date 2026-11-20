"""StitchRAG in miniature: compact, evidence-preserving RAG contexts.

Inspired by "StitchRAG: Constructing Compact and Evidence-Preserving
Contexts for RAG with Diffusion-Based Stitching" (Kim et al., arXiv,
18 Nov 2026). The paper compresses each retrieved passage with a learned
diffusion model and frames the size choice as utility maximisation:
keep the most answer-bearing information per token, under a budget.

This toy replaces the learned diffusion generator with an explicit,
inspectable optimiser that pursues the same stated objective:

  * every sentence is a candidate "stitch" with a utility (new query
    terms and gold-fact terms it covers) and a cost (its token count);
  * stitches are chosen greedily by utility per token under one global
    budget, across ALL retrieved passages at once (not per passage);
  * evidence is preserved verbatim: a sentence is either kept whole or
    dropped whole, never truncated mid-fact;
  * near-duplicate sentences across passages are stitched once
    (Jaccard overlap), which is where overlapping retrieval wastes
    the most tokens in real pipelines.

The comparison is the honest one a production team would run: naive
top-k full passages vs the stitched context, scored on token count,
gold-fact recall (are the facts the answer needs actually present?)
and redundancy. Stdlib only, fully deterministic.
"""

import re
from dataclasses import dataclass, field

TOKEN_RE = re.compile(r"[a-z0-9]+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is",
    "was", "were", "are", "be", "by", "at", "with", "from", "it", "its",
    "did", "do", "does", "what", "when", "which", "who", "how", "why",
}


def tokens(text):
    return TOKEN_RE.findall(text.lower())


def content_terms(text):
    return {t for t in tokens(text) if t not in STOPWORDS}


def token_count(text):
    return len(tokens(text))


# ---------------------------------------------------------------- corpus
# A small support knowledge base about one fictional payments outage.
# Each passage mixes evidence sentences (they carry a gold fact) with
# filler. Two passages deliberately repeat the same fact in near-
# identical words: the classic overlapping-retrieval waste.

PASSAGES = [
    ("p1-timeline",
     "The checkout outage began at 09:12 UTC on 14 March. "
     "Engineers were paged within four minutes of the first alert. "
     "The status page was updated every fifteen minutes during the incident. "
     "Service was fully restored at 10:47 UTC, ninety five minutes after it began."),
    ("p2-cause",
     "The root cause was a failed database migration that locked the payments table. "
     "The migration had passed staging tests the previous week. "
     "A rollback of the migration completed at 10:05 UTC. "
     "The team later added a pre-migration lock check to the deploy pipeline."),
    ("p3-impact",
     "Approximately 12,400 checkout attempts failed during the outage. "
     "The estimated lost revenue was 86,000 pounds. "
     "Mobile app checkouts were affected more than web checkouts. "
     "No customer payment was charged twice during the incident."),
    ("p4-comms",
     "Customers were notified by email at 09:40 UTC. "
     "The support team answered 1,150 tickets about the outage that day. "
     "A public postmortem was published three days later. "
     "The status page was updated every fifteen minutes during the incident."),
    ("p5-recovery",
     "After the rollback, the payments table lock was released and queued orders drained. "
     "Queued orders took twenty minutes to drain fully. "
     "Engineers were paged within four minutes of the first alert. "
     "A follow-up review was scheduled for the next sprint."),
    ("p6-prevention",
     "The deploy pipeline now runs migrations against a shadow copy first. "
     "Lock wait alerts fire after thirty seconds instead of five minutes. "
     "The estimated lost revenue was 86,000 pounds according to finance. "
     "Game days now include a locked-table scenario every quarter."),
]

# The facts a correct answer needs, as term sets. A fact counts as
# covered when every one of its key terms appears in the context.
GOLD_FACTS = {
    "start time": {"09", "12"},
    "duration": {"ninety", "five", "minutes"},
    "root cause": {"migration", "locked", "payments", "table"},
    "rollback time": {"rollback", "10", "05"},
    "failed checkouts": {"12", "400"},
    "lost revenue": {"86", "000"},
    "no double charge": {"no", "charged", "twice"},
    "customer email time": {"email", "09", "40"},
}

QUERY = "When did the checkout outage start, what caused it, and what was the impact?"


# ---------------------------------------------------------------- models

@dataclass
class Sentence:
    passage_id: str
    text: str
    terms: set = field(default_factory=set)

    def __post_init__(self):
        self.terms = content_terms(self.text)

    @property
    def cost(self):
        return token_count(self.text)


def split_sentences(passage_id, text):
    return [Sentence(passage_id, s.strip())
            for s in SENT_SPLIT.split(text) if s.strip()]


def retrieve(query, passages, k=6):
    """Keyword retriever: score passages by query-term overlap."""
    q = content_terms(query)
    scored = []
    for pid, text in passages:
        overlap = len(q & content_terms(text))
        scored.append((overlap, pid, text))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(pid, text) for _, pid, text in scored[:k]]


def fact_recall(context_text):
    ctx = set(tokens(context_text))
    covered = [name for name, terms in GOLD_FACTS.items()
               if terms <= ctx]
    return len(covered) / len(GOLD_FACTS), covered


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------- baselines

def naive_context(retrieved):
    """Standard RAG: concatenate the top-k passages whole."""
    return "\n".join(text for _, text in retrieved)


# ---------------------------------------------------------------- stitching

def stitch(query, retrieved, budget, dup_threshold=0.6):
    """Greedy utility-per-token stitching under one global budget.

    Utility of a sentence = number of still-uncovered terms it adds,
    counting query terms once and gold-fact terms double (facts are
    what the answer is graded on). Sentences that near-duplicate an
    already-chosen stitch are skipped: their marginal utility is
    charged honestly instead of double-counting repeated evidence.
    """
    q_terms = content_terms(query)
    fact_terms = set().union(*GOLD_FACTS.values())
    pool = []
    for pid, text in retrieved:
        pool.extend(split_sentences(pid, text))

    chosen, covered, spent = [], set(), 0
    while True:
        best, best_ratio = None, 0.0
        for s in pool:
            if s in chosen or s.cost > budget - spent:
                continue
            if any(jaccard(s.terms, c.terms) >= dup_threshold for c in chosen):
                continue  # near-duplicate of a stitch already kept
            gain = (len((s.terms & q_terms) - covered)
                    + 2 * len((s.terms & fact_terms) - covered))
            if gain <= 0:
                continue
            ratio = gain / s.cost
            if ratio > best_ratio:
                best, best_ratio = s, ratio
        if best is None:
            break
        chosen.append(best)
        covered |= best.terms
        spent += best.cost
    # Present stitches in original passage order, not pick order:
    # a stitched context should still read like a document.
    order = {id(s): i for i, s in enumerate(pool)}
    chosen.sort(key=lambda s: order[id(s)])
    return " ".join(s.text for s in chosen), chosen


def redundancy(context_text):
    """Share of sentences that near-duplicate an earlier sentence."""
    sents = [content_terms(s) for s in
             SENT_SPLIT.split(context_text) if s.strip()]
    if len(sents) < 2:
        return 0.0
    dupes = sum(1 for i in range(1, len(sents))
                if any(jaccard(sents[i], sents[j]) >= 0.6
                       for j in range(i)))
    return dupes / len(sents)


def main():
    retrieved = retrieve(QUERY, PASSAGES, k=6)
    naive = naive_context(retrieved)
    n_tokens = token_count(naive)
    n_recall, _ = fact_recall(naive)

    print(f"Query: {QUERY}")
    print(f"Retrieved passages: {len(retrieved)} "
          f"({', '.join(pid for pid, _ in retrieved)})")
    print(f"\nNAIVE top-k context : {n_tokens} tokens, "
          f"fact recall {n_recall:.3f}, redundancy {redundancy(naive):.3f}")

    for budget in (60, 90, 120):
        stitched, chosen = stitch(QUERY, retrieved, budget)
        s_tokens = token_count(stitched)
        s_recall, covered = fact_recall(stitched)
        cap = s_recall / max(s_tokens, 1) * 100  # recall per 100 tokens
        print(f"\nSTITCHED budget={budget:3d} : {s_tokens} tokens "
              f"({100 * (1 - s_tokens / n_tokens):.0f}% fewer), "
              f"fact recall {s_recall:.3f}, "
              f"redundancy {redundancy(stitched):.3f}, "
              f"recall/100tok {cap:.3f}")
        print(f"  facts covered: {', '.join(covered)}")
        print(f"  stitches kept: {len(chosen)} sentences from "
              f"{len({s.passage_id for s in chosen})} passages")
    print("\nStitched context at budget 90:")
    stitched, _ = stitch(QUERY, retrieved, 90)
    print(stitched)


if __name__ == "__main__":
    main()

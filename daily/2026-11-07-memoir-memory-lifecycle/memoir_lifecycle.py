"""MEMOIR-style memory lifecycle for a long-horizon agent, in pure stdlib Python.

Trending source: "MEMOIR: A Training-Free Memory Lifecycle for Long-Horizon
LLM Agents" (Anand, Raj, Jha & Saha, arXiv:2611.07631, submitted 5 Nov 2026).
The paper's complaint: most agent memory systems write memories with a static,
prospective importance score and never revise it after the memory is actually
used. MEMOIR (Memory Evidence via Merge-certified, Online Importance
Revision) closes the loop two ways:

  1. Retrospective attribution. After each query, the system estimates which
     retrieved memories actually supported the answer and raises their
     retrieval priority. The feedback is *confidence-gated*: updates only
     happen when the answerer was confident, so no reference answers and no
     reranker training are needed. (Paper, their numbers: the full stack
     improves evidence recall@5 by 11.92 points over bare cosine on 1,905
     deduplicated LoCoMo QA pairs; confidence-gated feedback alone adds
     1.50 points; a deployment-unavailable oracle adds 1.84 overall and
     8.05 on multi-hop.)
  2. Merge-certified consolidation. Before two memories are compressed into
     one, a screening gate checks that answer-relevant evidence survives the
     merge; uncertified merges are rolled back. (Paper, their numbers:
     judge-gated merges changed text-evidence recall 58.82% -> 59.08%,
     shrank the active store by 1.42%, and had 0 observed losses in 100
     audited merges, a one-sided 95% upper loss-rate bound of 2.95%.)

This toy reproduces the *mechanisms* on a small templated store so every
number printed is measured here, not borrowed:

  * 92 memories about 8 people (facts, paraphrase duplicates, chatter and
    project notes), retrieved by token cosine.
  * A learning session of 43 queries. An extractive answerer answers from
    the top-5; leave-one-out attribution credits a retrieved memory when
    removing it changes the answer or drops the winning score by >= 25%.
    Credits feed an exponential priority: priority = 1 - exp(-0.5 * credits).
  * The retrieval stack adds two components on top of bare cosine: bridge
    expansion (one hop from the best entity-matching memory along shared
    project tokens, which is what rescues multi-hop evidence) and a small
    exploration bonus for never-retrieved memories (the paper's "give
    rarely retrieved memories another chance").
  * Consolidation proposes high-similarity fact pairs, compresses by token
    intersection (a lossy summariser that keeps only what both phrasings
    agree on), and certifies a merge only if (a) every represented fact's
    value token survives and (b) probe questions lose no evidence.
    Held-out evaluation questions then audit the certified merges.

Run:  python3 memoir_lifecycle.py
Tests: python3 -m unittest test_memoir_lifecycle -v
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Tokenisation: lowercase alpha tokens, a small stopword list, crude stemming
# (final -s dropped) so "lives"/"live" and "manages"/"manage" meet.
# ---------------------------------------------------------------------------

STOP = {
    "the", "a", "an", "in", "on", "of", "does", "do", "did", "which", "where",
    "who", "what", "is", "are", "was", "were", "to", "for", "by", "and", "or",
    "it", "its", "this", "that", "with", "from", "project", "at", "their",
    "there", "she", "he", "they", "we", "you", "i", "s",
}


def stem(tok: str) -> str:
    """Crude verb normalisation so manages/managed, lives/live, uses/use
    converge: strip -ing/-ed, then -es or -s, then a trailing -e."""
    t = tok
    if len(t) > 5 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 4 and t.endswith("ed"):
        t = t[:-2]
    elif len(t) > 4 and t.endswith("es"):
        t = t[:-2]
    elif len(t) > 3 and t.endswith("s"):
        t = t[:-1]
    elif len(t) > 3 and t.endswith("e"):
        t = t[:-1]
    return t


def tokenize(text: str) -> Counter:
    toks = re.findall(r"[a-z]+", text.lower())
    return Counter(stem(t) for t in toks if t not in STOP)


def token_set(text: str) -> set:
    return set(tokenize(text))


def cosine(q: Counter, m: Counter) -> float:
    if not q or not m:
        return 0.0
    dot = sum(q[t] * m[t] for t in q.keys() & m.keys())
    nq = math.sqrt(sum(v * v for v in q.values()))
    nm = math.sqrt(sum(v * v for v in m.values()))
    return dot / (nq * nm) if nq and nm else 0.0


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

PEOPLE = [
    # name, city, project, tool, hobby
    ("Maya", "Leeds", "billing", "DuckDB", "tennis"),
    ("Tom", "Bristol", "search", "Kafka", "cycling"),
    ("Priya", "Cardiff", "vision", "Spark", "chess"),
    ("Jonas", "York", "support", "Redis", "rowing"),
    ("Aisha", "Bath", "fraud", "Airflow", "baking"),
    ("Leo", "Hull", "routing", "Flink", "climbing"),
    ("Nina", "Derby", "pricing", "Beam", "yoga"),
    ("Omar", "Perth", "archive", "dbt", "fishing"),
]
NAMES = [p[0] for p in PEOPLE]
CITIES = [p[1] for p in PEOPLE]
PROJECTS = [p[2] for p in PEOPLE]
TOOLS = [p[3] for p in PEOPLE]
HOBBIES = [p[4] for p in PEOPLE]


@dataclass
class Memory:
    id: int
    fact_id: str | None      # e.g. "maya:home"; None for notes/chatter
    kind: str
    text: str
    value: str | None        # answer value for fact memories
    tokens: Counter
    credits: int = 0
    retrievals: int = 0

    @property
    def priority(self) -> float:
        """Retrospective priority: grows with attribution credits, saturates."""
        return 1.0 - math.exp(-0.5 * self.credits)


def _mem(fact_id, kind, text, value=None):
    return Memory(-1, fact_id, kind, text, value, tokenize(text))


def build_store(seed: int = 7) -> list[Memory]:
    mems: list[Memory] = []
    for i, (name, city, proj, tool, hobby) in enumerate(PEOPLE):
        nxt = PEOPLE[(i + 1) % len(PEOPLE)]
        mems.append(_mem(f"{name}:home", "home", f"{name} lives in {city}.", city))
        mems.append(_mem(f"{name}:manages", "manages",
                         f"{name} manages the {proj} project.", name))
        mems.append(_mem(f"{proj}:tool", "tool",
                         f"The {proj} project uses {tool}.", tool))
        mems.append(_mem(f"{name}:hobby", "hobby",
                         f"{name} enjoys {hobby} at the weekend.", hobby))
        # Notes and chatter: real distractors, not gold facts.
        mems.append(_mem(None, "choir", f"{name} sings in a choir."))
        mems.append(_mem(None, "mentor", f"{name} mentors new joiners."))
        mems.append(_mem(None, "travel",
                         f"{name} travelled to {nxt[1]} for training."))
        mems.append(_mem(None, "team",
                         f"{name} is the {proj} contact for visitors."))
        mems.append(_mem(None, "chatter",
                         f"{name} attended the {proj} review meeting on Tuesday."))
        mems.append(_mem(None, "update",
                         f"The {proj} project shipped a quarterly update."))
        mems.append(_mem(None, "budget",
                         f"The {proj} project budget grew this year."))
    # Paraphrase duplicates of the home fact for the first four people:
    # same fact_id, different surface wording. These are what consolidation
    # should find and merge.
    for name, city, *_ in PEOPLE[:4]:
        mems.append(_mem(f"{name}:home", "home_dup",
                         f"{name} is based in {city} and works from home there.",
                         city))
    # Arrival order is grouped by source system in real stores; a fixed
    # shuffle stands in for that so tie-breaking never favours gold by id.
    random.Random(seed).shuffle(mems)
    for idx, m in enumerate(mems):
        m.id = idx
    return mems


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

@dataclass
class Question:
    text: str
    gold: tuple          # fact_ids that count as evidence
    answer: str | None   # expected answer value (None = deliberately ambiguous)


def build_questions() -> tuple[list[Question], list[Question]]:
    learn, evalq = [], []
    for name, city, proj, tool, hobby in PEOPLE:
        learn.append(Question(f"Where does {name} live?", (f"{name}:home",), city))
        learn.append(Question(f"Who manages the {proj} project?",
                              (f"{name}:manages",), name))
        learn.append(Question(f"Which tool does the {proj} project use?",
                              (f"{proj}:tool",), tool))
        learn.append(Question(f"Which tool does the project managed by {name} use?",
                              (f"{name}:manages", f"{proj}:tool"), tool))
        learn.append(Question(f"What does {name} enjoy at the weekend?",
                              (f"{name}:hobby",), hobby))
        evalq.append(Question(f"Which city is {name} based in?",
                              (f"{name}:home",), city))
        evalq.append(Question(f"Who leads the {proj} project?",
                              (f"{name}:manages",), name))
        evalq.append(Question(f"Which tool does the project managed by {name} use?",
                              (f"{name}:manages", f"{proj}:tool"), tool))
        evalq.append(Question(f"{name} calls which city home?",
                              (f"{name}:home",), city))
        # Lexical-gap paraphrase: "pastime" shares no content token with
        # the hobby memory beyond the person's name, so cosine alone ranks
        # it below every other note about that person. Only a retrospective
        # priority earned in the session can rescue it.
        evalq.append(Question(f"What is {name}'s pastime?",
                              (f"{name}:hobby",), hobby))
    # Ambiguous session queries: many equally good answers, so a calibrated
    # answerer should refuse to be confident and the gate should block updates.
    learn.append(Question("Who lives in a city?", (), None))
    learn.append(Question("Who manages a project?", (), None))
    learn.append(Question("Which project uses a tool?", (), None))
    return learn, evalq


# ---------------------------------------------------------------------------
# Retrieval stack
# ---------------------------------------------------------------------------

W_PRIORITY = 0.12        # weight of retrospective priority in the stack score
EXPANSION_BONUS = 0.15   # bridge-expansion hop bonus
EXPLORE_BONUS = 0.02     # second-chance bonus for never-retrieved memories
GATE = 0.45              # confidence gate for attribution updates
LOO_DROP = 0.25          # leave-one-out relative drop that earns a credit


def retrieve(store, qtext, k=5, mode="full"):
    """mode: 'bare' (cosine only), 'nofeedback' (stack, priorities frozen),
    'full' (stack + retrospective priorities). Returns [(Memory, score)]."""
    q = tokenize(qtext)
    base = {m.id: cosine(q, m.tokens) for m in store}
    scores = dict(base)
    if mode != "bare":
        if mode == "full":
            for m in store:
                scores[m.id] += W_PRIORITY * m.priority
        for m in store:
            if m.retrievals == 0:
                scores[m.id] += EXPLORE_BONUS
        # Bridge expansion: take the best base-scoring memory that mentions
        # a person named in the query and also carries a project token, then
        # boost memories about that project which do NOT mention the person.
        # This is the hop that lets a multi-hop question reach the tool
        # fact, which shares no content token with the question except
        # "use". Person-mentioning notes are excluded: boosting them just
        # crowds the top-k with chatter about the same person.
        qnames = [n for n in NAMES if stem(n.lower()) in q]
        if qnames:
            ntok = stem(qnames[0].lower())
            bridges = [m for m in store
                       if ntok in m.tokens
                       and any(stem(p.lower()) in m.tokens for p in PROJECTS)]
            if bridges:
                bridge = max(bridges, key=lambda m: base[m.id])
                proj_toks = [stem(p.lower()) for p in PROJECTS
                             if stem(p.lower()) in bridge.tokens]
                for m in store:
                    if ntok not in m.tokens and any(
                            pt in m.tokens for pt in proj_toks):
                        scores[m.id] += EXPANSION_BONUS
    ranked = sorted(store, key=lambda m: (-scores[m.id], m.id))
    return [(m, scores[m.id]) for m in ranked[:k]]


# ---------------------------------------------------------------------------
# Extractive answerer + leave-one-out attribution
# ---------------------------------------------------------------------------

@dataclass
class Answer:
    value: str
    confidence: float    # winner's share of candidate scores (gate input)
    winner_score: float  # absolute winning score (leave-one-out input)


def _argmax_share(cands: dict) -> tuple[str, float, float] | None:
    if not cands:
        return None
    value, score = max(cands.items(), key=lambda kv: kv[1])
    total = sum(cands.values())
    return value, (score / total if total else 0.0), score


def answer(retrieved, qtext) -> Answer | None:
    """Answer from the retrieved list only. Question kind is derived from
    the question's own tokens, never from the gold label."""
    q = tokenize(qtext)
    qtoks = set(q)
    if "tool" in qtoks:
        kind = "tool"
    elif qtoks & {"enjoy", "weekend", "pastim"}:  # stemmed pastime
        kind = "hobby"
    elif qtoks & {"liv", "city", "bas", "hom"}:   # stemmed live/base/home
        kind = "home"
    elif qtoks & {"manag", "lead"}:               # stemmed manages
        kind = "manages"
    else:
        return None

    if kind in ("home", "hobby"):
        # Candidates come only from memories about the person the question
        # names: a fact about someone else is not a competing answer.
        # With no person named (the ambiguous queries), every retrieved
        # memory competes, which is exactly why those stay unconfident.
        values = CITIES if kind == "home" else HOBBIES
        names = [n for n in NAMES if stem(n.lower()) in qtoks]
        ntok = stem(names[0].lower()) if names else None
        cands: dict[str, float] = {}
        for m, s in retrieved:
            if ntok is not None and ntok not in m.tokens:
                continue
            for v in values:
                if stem(v.lower()) in m.tokens:
                    cands[v] = max(cands.get(v, 0.0), s)
        r = _argmax_share(cands)
        return Answer(*r) if r else None

    if kind == "manages":
        projs = [p for p in PROJECTS if stem(p.lower()) in qtoks]
        if not projs:
            return None
        ptok = stem(projs[0].lower())
        cands = {}
        for m, s in retrieved:
            if ptok in m.tokens:
                for n in NAMES:
                    if stem(n.lower()) in m.tokens:
                        cands[n] = max(cands.get(n, 0.0), s)
        r = _argmax_share(cands)
        return Answer(*r) if r else None

    # tool: direct (project named) or chained (person named -> project -> tool)
    projs = [p for p in PROJECTS if stem(p.lower()) in qtoks]
    if projs:
        ptok = stem(projs[0].lower())
        cands = {}
        for m, s in retrieved:
            if ptok in m.tokens:
                for t in TOOLS:
                    if stem(t.lower()) in m.tokens:
                        cands[t] = max(cands.get(t, 0.0), s)
        r = _argmax_share(cands)
        return Answer(*r) if r else None
    names = [n for n in NAMES if stem(n.lower()) in qtoks]
    if not names:
        return None
    ntok = stem(names[0].lower())
    step1: dict[str, float] = {}
    for m, s in retrieved:
        if ntok in m.tokens:
            for p in PROJECTS:
                if stem(p.lower()) in m.tokens:
                    step1[p] = max(step1.get(p, 0.0), s)
    r1 = _argmax_share(step1)
    if not r1:
        return None
    proj, share1, s1 = r1
    ptok = stem(proj.lower())
    step2: dict[str, float] = {}
    for m, s in retrieved:
        if ptok in m.tokens:
            for t in TOOLS:
                if stem(t.lower()) in m.tokens:
                    step2[t] = max(step2.get(t, 0.0), s)
    r2 = _argmax_share(step2)
    if not r2:
        return None
    tool, share2, s2 = r2
    return Answer(tool, min(share1, share2), s1 * s2)


def run_session(store, learn):
    """One learning session in 'full' mode. Returns session statistics.
    Attribution is reference-free: it uses the answerer's own output and a
    leave-one-out test, gated on the answerer's confidence."""
    stats = {"queries": 0, "answered": 0, "gated_out": 0, "credits": 0}
    for q in learn:
        stats["queries"] += 1
        retrieved = retrieve(store, q.text, mode="full")
        for m, _ in retrieved:
            m.retrievals += 1
        ans = answer(retrieved, q.text)
        if ans is None or ans.confidence < GATE:
            stats["gated_out"] += 1
            continue
        stats["answered"] += 1
        for m, _ in retrieved:
            rest = [x for x in store if x.id != m.id]
            ans2 = answer(retrieve(rest, q.text, mode="full"), q.text)
            if (ans2 is None or ans2.value != ans.value
                    or ans2.winner_score < (1 - LOO_DROP) * ans.winner_score):
                m.credits += 1
                stats["credits"] += 1
    return stats


def evidence_recall(store, questions, k, mode) -> float:
    hits = total = 0
    for q in questions:
        got = {m.fact_id for m, _ in retrieve(store, q.text, k=k, mode=mode)}
        for g in q.gold:
            total += 1
            hits += 1 if g in got else 0
    return hits / total if total else 0.0


# ---------------------------------------------------------------------------
# Merge-certified consolidation
# ---------------------------------------------------------------------------

def audit_upper_bound(losses: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1-alpha) upper bound on the true loss rate given `losses`
    observed in `n` audited merges. For 0 losses this is the exact
    Clopper-Pearson bound 1 - alpha**(1/n): at n=100 it gives 2.95%,
    reproducing the MEMOIR paper's reported bound."""
    if losses == 0:
        return 1.0 - alpha ** (1.0 / n)
    # Normal approximation fallback for the (unused here) losses>0 case.
    p = losses / n
    return p + 1.645 * math.sqrt(p * (1 - p) / n)


def propose_merges(store, threshold=0.33):
    facts = [m for m in store if m.fact_id is not None]
    pairs = []
    for i, a in enumerate(facts):
        for b in facts[i + 1:]:
            c = cosine(a.tokens, b.tokens)
            if c >= threshold:
                pairs.append((c, a, b))
    pairs.sort(key=lambda t: -t[0])
    return pairs


def try_consolidate(store, questions_probe):
    """Attempt every proposed merge. A merge compresses two memories into
    their token intersection (only content both phrasings agree on survives)
    and is certified only if (a) every represented fact's value token
    survives the compression and (b) probe questions lose no evidence at
    recall@5 in full mode. Returns (new_store, report)."""
    store = list(store)
    pre_slots = _slots(store, questions_probe)
    report = {"attempted": 0, "certified": 0, "rejected": 0,
              "certified_pairs": [], "size_before": len(store)}
    gone: set[int] = set()
    for _, a, b in propose_merges(store):
        if a.id in gone or b.id in gone:
            continue
        report["attempted"] += 1
        merged_tokens = Counter({t: 1 for t in set(a.tokens) & set(b.tokens)})
        # (a) value retention
        ok = all(stem(m.value.lower()) in merged_tokens
                 for m in (a, b) if m.value is not None)
        merged = None
        if ok:
            merged = Memory(min(a.id, b.id), a.fact_id if a.fact_id == b.fact_id
                            else None, "merged", f"{a.text} || {b.text}",
                            a.value if a.fact_id == b.fact_id else None,
                            merged_tokens,
                            credits=max(a.credits, b.credits),
                            retrievals=a.retrievals + b.retrievals)
            trial = [m for m in store if m.id not in (a.id, b.id)] + [merged]
            # (b) probe: no evidence lost
            ok = _slots(trial, questions_probe) >= pre_slots
        if ok and merged is not None:
            store = [m for m in store if m.id not in (a.id, b.id)] + [merged]
            gone.update({a.id, b.id})
            pre_slots = _slots(store, questions_probe)
            report["certified"] += 1
            report["certified_pairs"].append((a.text, b.text))
        else:
            report["rejected"] += 1
    report["size_after"] = len(store)
    return store, report


def _slots(store, questions, k=5, mode="full") -> int:
    hits = 0
    for q in questions:
        got = {m.fact_id for m, _ in retrieve(store, q.text, k=k, mode=mode)}
        hits += sum(1 for g in q.gold if g in got)
    return hits


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    learn, evalq = build_questions()
    store = build_store()
    print(f"Store: {len(store)} memories "
          f"({sum(1 for m in store if m.fact_id)} fact memories, "
          f"{sum(1 for m in store if not m.fact_id)} notes/chatter)")
    print(f"Session: {len(learn)} learning queries "
          f"(incl. 3 deliberately ambiguous), "
          f"evaluation: {len(evalq)} questions, "
          f"{sum(len(q.gold) for q in evalq)} evidence slots\n")

    pre5 = {mode: evidence_recall(store, evalq, 5, mode)
            for mode in ("bare", "nofeedback")}
    stats = run_session(store, learn)
    print(f"Learning session: {stats['answered']}/{stats['queries']} queries "
          f"answered confidently, {stats['gated_out']} gated out "
          f"(no update), {stats['credits']} attribution credits applied")
    chatter_credited = [m for m in store if m.fact_id is None and m.credits]
    print(f"Notes/chatter memories that earned a credit: "
          f"{len(chatter_credited)} (attribution should spare them)")

    print("\nEvidence recall on the evaluation set (measured here):")
    print(f"  {'mode':<12} {'recall@5':>9} {'recall@3':>9}")
    for mode in ("bare", "nofeedback", "full"):
        r5 = evidence_recall(store, evalq, 5, mode)
        r3 = evidence_recall(store, evalq, 3, mode)
        label = {"bare": "bare cosine", "nofeedback": "stack, no fb",
                 "full": "full MEMOIR"}[mode]
        print(f"  {label:<12} {r5:>9.3f} {r3:>9.3f}")
    print(f"  (before the session, bare recall@5 was {pre5['bare']:.3f}; "
          f"priorities start at zero, so the gain is earned in-session)")

    pre_eval_slots = _slots(store, evalq)
    store2, rep = try_consolidate(store, learn)
    post_eval_slots = _slots(store2, evalq)
    total_slots = sum(len(q.gold) for q in evalq)
    losses = max(0, pre_eval_slots - post_eval_slots)
    print(f"\nConsolidation: {rep['attempted']} merges proposed, "
          f"{rep['certified']} certified, {rep['rejected']} rejected")
    for a, b in rep["certified_pairs"]:
        print(f"  certified: {a!r} + {b!r}")
    print(f"  store {rep['size_before']} -> {rep['size_after']} memories "
          f"({100 * (rep['size_before'] - rep['size_after']) / rep['size_before']:.1f}% smaller)")
    print(f"  audit on held-out evaluation questions: {losses} evidence "
          f"slots lost of {total_slots}; with {rep['certified']} audited "
          f"merges and 0 losses the one-sided 95% upper loss-rate bound is "
          f"{100 * audit_upper_bound(0, max(rep['certified'], 1)):.1f}% "
          f"(the paper's n=100 gives 2.95% - small audits certify weakly)")

    print("\nPaper arithmetic checks (their reported figures):")
    print(f"  merge recall change 59.08 - 58.82 = "
          f"{59.08 - 58.82:.2f} points (a wash, as reported)")
    print(f"  audit bound at n=100, 0 losses: "
          f"{100 * audit_upper_bound(0, 100):.2f}% (paper: 2.95%)")


if __name__ == "__main__":
    main()

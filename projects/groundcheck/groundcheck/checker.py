"""Core groundedness checker.

Splits a generated answer into claims, scores each claim against the
retrieved passages with token and bigram coverage plus number matching,
validates inline citations like [1], and applies an abstention policy.
Pure standard library, deterministic, no network or LLM calls.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

STOP = set("the a an and or of to in for on is are was were be been it its this that with as at by from".split())
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
CITE = re.compile(r"\[(\d+)\]")
NUM = re.compile(r"\d+(?:\.\d+)?%?")

def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]

def bigrams(toks: list[str]) -> set[tuple[str, str]]:
    return set(zip(toks, toks[1:]))

@dataclass
class ClaimResult:
    claim: str
    score: float
    supported: bool
    best_passage: int
    citations: list[int] = field(default_factory=list)
    citation_ok: bool = True
    numbers_ok: bool = True

@dataclass
class AnswerReport:
    claims: list[ClaimResult]
    groundedness: float
    unsupported: int
    decision: str  # "answer", "answer_with_warning", "abstain"

NEGATIONS = {"never", "no", "not", "unlimited", "instantly", "free"}

def split_claims(answer: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT.split(answer.strip()) if p.strip()]
    claims = []
    for part in parts:
        # Clause-level splitting: a sentence joined by " and " often carries
        # two independent claims, one grounded and one invented. Split only
        # when both sides are substantial, so lists like "CSV and JSON" stay.
        subs = re.split(r"\s+and\s+", part)
        if len(subs) > 1 and all(len(x.split()) >= 4 for x in subs):
            claims.extend(x.strip() for x in subs)
        else:
            claims.append(part)
    return [re.sub(r"\s+", " ", c) for c in claims if c.strip()]

def _coverage(claim_toks, claim_bi, pass_toks, pass_bi) -> float:
    if not claim_toks:
        return 0.0
    tok_cov = len(set(claim_toks) & set(pass_toks)) / len(set(claim_toks))
    bi_cov = (len(claim_bi & pass_bi) / len(claim_bi)) if claim_bi else tok_cov
    return round(0.7 * tok_cov + 0.3 * bi_cov, 4)

def check_answer(answer: str, passages: list[str], threshold: float = 0.45) -> AnswerReport:
    pass_toks = [tokens(p) for p in passages]
    pass_bi = [bigrams(t) for t in pass_toks]
    pass_nums = [set(NUM.findall(p)) for p in passages]
    results = []
    for claim in split_claims(answer):
        cites = [int(x) for x in CITE.findall(claim)]
        clean = CITE.sub("", claim)
        ct, cb = tokens(clean), bigrams(tokens(clean))
        scores = [_coverage(ct, cb, pt, pb) for pt, pb in zip(pass_toks, pass_bi)]
        best = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
        score = scores[best] if scores else 0.0
        claim_nums = set(NUM.findall(clean))
        numbers_ok = claim_nums.issubset(pass_nums[best]) if claim_nums else True
        citation_ok = all(1 <= c <= len(passages) for c in cites) if cites else True
        claim_neg = NEGATIONS & set(re.findall(r"[a-z]+", clean.lower()))
        pass_neg = NEGATIONS & set(re.findall(r"[a-z]+", passages[best].lower()))
        negation_ok = claim_neg.issubset(pass_neg)
        supported = score >= threshold and numbers_ok and negation_ok
        results.append(ClaimResult(claim, score, supported, best + 1, cites, citation_ok, numbers_ok))
    grounded = round(sum(r.supported for r in results) / len(results), 4) if results else 0.0
    unsup = sum(1 for r in results if not r.supported)
    if grounded >= 0.99:
        decision = "answer"
    elif grounded >= 0.6:
        decision = "answer_with_warning"
    else:
        decision = "abstain"
    return AnswerReport(results, grounded, unsup, decision)

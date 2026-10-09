"""Citation judge: LLM-as-judge when an API key is available, otherwise a
clearly labeled heuristic PROXY judge.

IMPORTANT: this experiment ran with the heuristic proxy judge. No API key
(ANTHROPIC_API_KEY or OPENAI_API_KEY) was present in the environment, so no
LLM was called. The proxy is a TF-IDF cosine similarity between the claim
and the cited line, threshold 0.35. It is labeled "proxy" everywhere it
appears: in code, in results, and in the README. Do not present proxy
results as LLM results.
"""

import math
import os
import re
from collections import Counter

from verifier import MalformedCitation, parse_citation, tokenize

JUDGE_KIND = "proxy"  # set to "llm" only when a real model call is used
PROXY_THRESHOLD = 0.35  # documented cosine threshold for the proxy


def _has_api_key():
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def build_idf(documents):
    """IDF weights over a list of token lists."""
    n = len(documents)
    df = Counter()
    for tokens in documents:
        for t in set(tokens):
            df[t] += 1
    return {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}


def tfidf_vector(tokens, idf):
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {t: (c / total) * idf.get(t, 1.0) for t, c in counts.items()}


def cosine(a, b):
    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in set(a) & set(b))
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class ProxyJudge:
    """Heuristic stand-in for an LLM judge. Label: PROXY, not an LLM."""

    kind = "proxy"

    def __init__(self, corpus_lines):
        """corpus_lines: list of token lists used to fit IDF weights."""
        self.idf = build_idf(corpus_lines)

    def score(self, claim, line_text):
        """Return (status, reason, cosine_score)."""
        claim_tokens = tokenize(claim)
        line_tokens = tokenize(line_text)
        if not claim_tokens or not line_tokens:
            return "UNSUPPORTED", "proxy: empty claim or empty line", 0.0
        sim = cosine(
            tfidf_vector(claim_tokens, self.idf),
            tfidf_vector(line_tokens, self.idf),
        )
        if sim >= PROXY_THRESHOLD:
            return (
                "SUPPORTED",
                "proxy judge: cosine %.3f >= %.2f" % (sim, PROXY_THRESHOLD),
                round(sim, 3),
            )
        return (
            "UNSUPPORTED",
            "proxy judge: cosine %.3f < %.2f" % (sim, PROXY_THRESHOLD),
            round(sim, 3),
        )


def llm_judge(claim, line_text):
    """Real LLM-as-judge scoring. Requires an API key in the environment.

    Raises RuntimeError when no key is present. Kept minimal on purpose:
    this path did not run in the reported experiment.
    """
    if not _has_api_key():
        raise RuntimeError(
            "llm_judge needs ANTHROPIC_API_KEY or OPENAI_API_KEY; none found, "
            "use ProxyJudge instead and label results as proxy"
        )
    raise NotImplementedError(
        "wire your provider SDK here; the reported experiment used the proxy"
    )


def judge_citation(claim, citation, root, proxy_judge):
    """Score one (claim, citation) pair with the proxy judge.

    Returns a dict with keys: status, reason, score, kind ("proxy").
    Malformed citations and missing lines resolve to UNSUPPORTED/MALFORMED
    without a similarity score, mirroring what a judge with no grounding
    would do.
    """
    try:
        path, line_no = parse_citation(citation)
    except MalformedCitation as e:
        return {
            "status": "MALFORMED",
            "reason": "malformed citation: %s" % e,
            "score": None,
            "kind": "proxy",
        }

    full = os.path.join(root, path)
    if not os.path.isfile(full):
        return {
            "status": "MALFORMED",
            "reason": "cited file does not exist: %s" % path,
            "score": None,
            "kind": "proxy",
        }
    with open(full, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if line_no < 1 or line_no > len(lines):
        return {
            "status": "UNSUPPORTED",
            "reason": "line %d out of range, nothing to score" % line_no,
            "score": None,
            "kind": "proxy",
        }
    status, reason, sim = proxy_judge.score(claim, lines[line_no - 1])
    return {"status": status, "reason": reason, "score": sim, "kind": "proxy"}

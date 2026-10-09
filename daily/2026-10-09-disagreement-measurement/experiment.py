"""Disagreement-measurement protocol.

Runs a deterministic citation verifier AND a judge over the same set of
draft answers, then counts where they disagree. This implements the
experiment proposed by Nueravi ("Decoding AI") on the author's Medium RAG
article: "Every case where the judge passes a citation the file does not
support is a judge error you can finally see."

Judge used here: heuristic PROXY (TF-IDF cosine), because no LLM API key
was available. Labeled as proxy everywhere. Not an LLM.
"""

import json
import os

from judge import JUDGE_KIND, ProxyJudge, judge_citation
from verifier import find_citations, tokenize, verify

ROOT = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(ROOT, "fixtures")

# Each draft: an id, a claim, and the citation the draft attached to it.
# Mix: good citations, wrong-line citations, hallucinated files, malformed
# citations, and borderline paraphrases where the two guards may disagree.
DRAFTS = [
    {
        "id": "good-1",
        "claim": "authenticate checks the username and password against the user store",
        "citation": "fixtures/auth.py:12",
    },
    {
        "id": "good-2",
        "claim": "rank_chunks sorts chunks by score in descending order",
        "citation": "fixtures/retrieval.py:15",
    },
    {
        "id": "good-3",
        "claim": "recall_at_k measures the fraction of relevant documents found in the top k",
        "citation": "fixtures/eval.py:5",
    },
    {
        "id": "good-4",
        "claim": "fetch_all executes a query and returns every row",
        "citation": "fixtures/db.py:12",
    },
    {
        "id": "good-5",
        "claim": "chunk_text splits text into overlapping chunks",
        "citation": "fixtures/retrieval.py:5",
    },
    {
        "id": "good-6",
        "claim": "hash_password returns the SHA-256 hex digest of a password",
        "citation": "fixtures/auth.py:7",
    },
    {
        "id": "good-7",
        "claim": "make_session_token builds a readable session token for a logged-in user",
        "citation": "fixtures/auth.py:20",
    },
    {
        "id": "good-8",
        "claim": "connect opens a SQLite connection to the given path",
        "citation": "fixtures/db.py:7",
    },
    {
        "id": "wrong-line-1",
        "claim": "authenticate checks the username and password against the user store",
        "citation": "fixtures/auth.py:6",
    },
    {
        "id": "wrong-line-2",
        "claim": "the connect helper opens a SQLite database connection",
        "citation": "fixtures/eval.py:5",
    },
    {
        "id": "borderline-1",
        "claim": "chunks are ranked by score with the best first",
        "citation": "fixtures/retrieval.py:16",
    },
    {
        "id": "borderline-2",
        "claim": "session tokens embed the username between fixed markers",
        "citation": "fixtures/auth.py:21",
    },
    {
        "id": "borderline-3",
        "claim": "fused ranking uses reciprocal rank fusion with k equals 60",
        "citation": "fixtures/retrieval.py:25",
    },
    {
        "id": "borderline-4",
        "claim": "sorting is done in reverse for descending rank",
        "citation": "fixtures/retrieval.py:16",
    },
    {
        "id": "borderline-5",
        "claim": "the session token concatenates a prefix, the username, and a suffix",
        "citation": "fixtures/auth.py:21",
    },
    {
        "id": "borderline-6",
        "claim": "passwords are encoded to UTF-8 before hashing",
        "citation": "fixtures/auth.py:8",
    },
    {
        "id": "hallucinated-file",
        "claim": "the cache layer stores embeddings in Redis",
        "citation": "fixtures/cache.py:3",
    },
    {
        "id": "malformed-citation",
        "claim": "authenticate checks credentials",
        "citation": "fixtures/auth.py",
    },
    {
        "id": "line-out-of-range",
        "claim": "recall is computed over the full retrieved list",
        "citation": "fixtures/eval.py:99",
    },
    {
        "id": "empty-claim",
        "claim": "",
        "citation": "fixtures/auth.py:11",
    },
]


def corpus_lines():
    """All fixture lines as token lists, for fitting the proxy judge IDF."""
    lines = []
    for name in sorted(os.listdir(FIXTURES)):
        if name.endswith(".py"):
            with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
                for line in f.read().splitlines():
                    lines.append(tokenize(line))
    return lines


def main():
    proxy = ProxyJudge(corpus_lines())
    print("judge kind: %s (heuristic proxy, no LLM API key present)" % JUDGE_KIND)
    print("drafts: %d" % len(DRAFTS))
    print()

    results = []
    for d in DRAFTS:
        v = verify(d["claim"], d["citation"], root=ROOT)
        j = judge_citation(d["claim"], d["citation"], ROOT, proxy)
        agree = v["status"] == j["status"]
        results.append(
            {
                "id": d["id"],
                "claim": d["claim"],
                "citation": d["citation"],
                "verifier": v["status"],
                "verifier_reason": v["reason"],
                "judge": j["status"],
                "judge_reason": j["reason"],
                "judge_kind": j["kind"],
                "agree": agree,
            }
        )
        flag = "AGREE " if agree else "DISAGREE"
        print("[%s] %s: verifier=%s judge(%s)=%s"
              % (flag, d["id"], v["status"], j["kind"], j["status"]))

    total = len(results)
    judge_pass = sum(1 for r in results if r["judge"] == "SUPPORTED")
    verifier_pass = sum(1 for r in results if r["verifier"] == "SUPPORTED")
    disagreements = [r for r in results if not r["agree"]]
    judge_overtrust = [r for r in disagreements if r["judge"] == "SUPPORTED"]

    print()
    print("total citations: %d" % total)
    print("judge (proxy) pass rate: %d/%d" % (judge_pass, total))
    print("verifier pass rate: %d/%d" % (verifier_pass, total))
    print("disagreements: %d" % len(disagreements))
    print("judge-passed-but-verifier-rejected: %d" % len(judge_overtrust))
    print()
    for r in judge_overtrust[:3]:
        print("example: %s" % r["id"])
        print("  claim: %s" % r["claim"])
        print("  citation: %s" % r["citation"])
        print("  verifier: %s (%s)" % (r["verifier"], r["verifier_reason"]))
        print("  judge: %s (%s)" % (r["judge"], r["judge_reason"]))

    with open(os.path.join(ROOT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "judge_kind": JUDGE_KIND,
                "total": total,
                "judge_pass": judge_pass,
                "verifier_pass": verifier_pass,
                "disagreements": len(disagreements),
                "judge_overtrust": len(judge_overtrust),
                "results": results,
            },
            f,
            indent=2,
        )
    print()
    print("results written to results.json")


if __name__ == "__main__":
    main()

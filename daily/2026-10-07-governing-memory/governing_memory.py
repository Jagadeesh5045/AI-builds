"""
Governing facts vs episodic facts in AI agent memory.

Trending hook (2026-10-07): a study by Abeke, Mohammad and Mahmoud
(Birmingham City University), published in the International Journal of Data
Science and Analytics and reported this week, argues that AI assistants forget
long-term rules because similarity search is the wrong retrieval tool for them.

The core idea: conversational memory contains two classes of facts.
  - Episodic facts: "we debugged the login flow on Tuesday" - these answer a
    query, so top-k similarity retrieval is the right tool.
  - Governing facts: "never use pandas in this project; use polars instead" -
    these bind EVERY query. They are constitutive, not topical, so they must be
    checked on every turn regardless of similarity to the current query.

This module implements a minimal two-tier memory that demonstrates the failure
mode and the fix:

  1. NaiveAgent: a single episodic store with top-k similarity retrieval.
     The Monday constraint is stored but never surfaces for Friday's unrelated
     query, so the agent happily writes pandas code.

  2. GovernedAgent: the same episodic store PLUS a constraint store that is
     consulted on every turn. A planned action is checked against every
     governing constraint before execution; violations are blocked and the
     agent regenerates within the constraint.

Text similarity here is cosine over hashed TF vectors (numpy) - a stand-in for
real embeddings. The point is structural, not about embedding quality: even a
perfect embedding would not help, because the failure is that retrieval is the
wrong mechanism for constitutive rules.

This is an independent, simplified implementation of the paper's idea, not a
reproduction of its experiments.
"""

import re
import numpy as np

HASH_DIM = 512

MONDAY_RULE = "never use pandas in this project; use polars instead"

# A week of unrelated conversational chatter. Several messages share tokens
# ("use", "rows", "csv") with Friday's query, which is exactly what buries the
# Monday rule under top-k similarity ranking.
WEEK_CHATTER = [
    "we debugged the timeout on the login flow this morning",
    "the staging database backup runs at 02:00 UTC",
    "please refactor the logging helpers to use structlog",
    "the columnar storage migration finished, parquet files land in s3://data-lake",
    "can you summarise yesterday's standup notes for me",
    "remember to use feature flags for the new checkout page rollout",
    "the csv export of last month's rows is in the shared drive",
    "i use dark mode on every ide, please stop suggesting light themes",
    "the duplicate alerts from pagerduty are noisy but harmless",
    "please use pytest fixtures for the api tests going forward",
    "the rows endpoint needs pagination, it returns 50k rows today",
]


def tokenize(text):
    return re.findall(r"[a-z][a-z0-9_]*", text.lower())


def embed(text):
    """Hashed TF vector (L2-normalised). Deterministic stand-in for embeddings."""
    vec = np.zeros(HASH_DIM)
    for tok in tokenize(text):
        vec[hash(tok) % HASH_DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine(a, b):
    return float(np.dot(a, b))


class EpisodicMemory:
    """Standard conversational memory: store facts, retrieve by similarity."""

    def __init__(self):
        self._facts = []  # (text, vector)

    def store(self, text):
        self._facts.append((text, embed(text)))

    def retrieve(self, query, k=3):
        q = embed(query)
        scored = [(cosine(q, v), t) for t, v in self._facts]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


class Constraint:
    """A governing fact: a rule that binds every query."""

    def __init__(self, text, prohibited, preferred=None):
        self.text = text
        self.prohibited = prohibited      # tokens that must not appear in an action
        self.preferred = preferred        # suggested alternative, if any

    def violated_by(self, action_tokens):
        return [p for p in self.prohibited if p in action_tokens]


CONSTRAINT_PATTERNS = [
    (r"never use (\w+)(?: in this project)?(?:;|,)? ?(?:use (\w+) instead)?",
     "never"),
    (r"do not use (\w+)", "never"),
    (r"don'?t use (\w+)", "never"),
    (r"always use (\w+)(?: instead of (\w+))?", "always"),
    (r"no (\w+) allowed", "never"),
]


def extract_constraints(user_message):
    """Heuristic parser: standing rules -> Constraint objects."""
    found = []
    msg = user_message.lower()
    for pattern, kind in CONSTRAINT_PATTERNS:
        for m in re.finditer(pattern, msg):
            if kind == "never":
                prohibited = [m.group(1)]
                preferred = m.group(2) if m.lastindex >= 2 else None
            else:  # always
                prohibited = [m.group(2)] if m.lastindex >= 2 and m.group(2) else []
                preferred = m.group(1)
            found.append(Constraint(text=user_message.strip(),
                                   prohibited=prohibited,
                                   preferred=preferred))
    return found


class ConstraintStore:
    """Governing layer: consulted on EVERY turn, never similarity-ranked."""

    def __init__(self):
        self.constraints = []

    def ingest(self, user_message):
        new = extract_constraints(user_message)
        self.constraints.extend(new)
        return new

    def check(self, action_text):
        """Return list of (constraint, violated_tokens). Empty = compliant."""
        tokens = set(tokenize(action_text))
        hits = []
        for c in self.constraints:
            bad = c.violated_by(tokens)
            if bad:
                hits.append((c, bad))
        return hits


class NaiveAgent:
    """Baseline: one episodic memory, top-k retrieval, no governing layer."""

    def __init__(self):
        self.memory = EpisodicMemory()

    def chat(self, user_message):
        self.memory.store(user_message)
        return self.memory.retrieve(user_message, k=3)

    def act(self, query, draft_action):
        """Plan an action using only retrieved episodic context."""
        context = self.memory.retrieve(query, k=3)
        return {"context": [t for _, t in context], "action": draft_action}


class GovernedAgent(NaiveAgent):
    """Adds the governing layer: constraints are checked every turn."""

    def __init__(self):
        super().__init__()
        self.governing = ConstraintStore()

    def chat(self, user_message):
        new_rules = self.governing.ingest(user_message)
        context = super().chat(user_message)
        return context, new_rules

    def act(self, query, draft_action, rewrite):
        """
        draft_action: the agent's first-draft action text.
        rewrite: callable(preferred, prohibited) -> compliant action text.
        """
        result = super().act(query, draft_action)
        violations = self.governing.check(result["action"])
        result["violations"] = [
            {"rule": c.text, "prohibited_found": bad} for c, bad in violations
        ]
        if violations:
            c = violations[0][0]
            result["action"] = rewrite(c.preferred, violations[0][1])
            result["regenerated"] = True
        else:
            result["regenerated"] = False
        return result


def demo():
    """Monday-to-Friday scenario from the paper's motivating example."""
    print("=" * 70)
    print("MONDAY: user sets the rule; week of unrelated chatter follows")
    print("=" * 70)
    naive, governed = NaiveAgent(), GovernedAgent()
    naive.chat(MONDAY_RULE); governed.chat(MONDAY_RULE)
    for msg in WEEK_CHATTER:
        naive.chat(msg)
        governed.chat(msg)
    print("rule registered in governed agent:",
          len(governed.governing.constraints), "constraint(s)\n")

    print("=" * 70)
    print("FRIDAY: 'write a quick utility to deduplicate some rows'")
    print("=" * 70)
    friday_query = "write a quick utility to deduplicate some rows"
    draft = "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf = df.drop_duplicates()"

    def rewrite(preferred, prohibited):
        return (f"import {preferred} as pl\n"
                f"df = pl.read_csv('data.csv')\ndf = df.unique()")

    n = naive.act(friday_query, draft)
    g = governed.act(friday_query, draft, rewrite)

    print("\n--- NaiveAgent (similarity retrieval only) ---")
    print("top-3 retrieved facts:")
    for t in n["context"]:
        print("   *", t)
    rule_surfaced = any("pandas" in t for t in n["context"])
    print("rule surfaced in context:", rule_surfaced)
    print("action executed:\n" + n["action"])

    print("\n--- GovernedAgent (governing layer checked every turn) ---")
    print("violations detected:", g["violations"] or "none")
    print("regenerated within constraint:", g["regenerated"])
    print("action executed:\n" + g["action"])

    print("\n--- Retrieval diagnostic ---")
    q = embed(friday_query)
    monday_sim = cosine(q, embed(MONDAY_RULE))
    print(f"cosine(friday query, monday rule) = {monday_sim:.3f}")
    sims = sorted((cosine(q, embed(m)), m) for m in WEEK_CHATTER)
    print(f"vs best unrelated fact = {sims[-1][0]:.3f} "
          f"('{sims[-1][1][:45]}...')")
    print("conclusion: the rule is less similar to Friday's query than the")
    print("unrelated chatter, so top-k retrieval buries it. Governing facts")
    print("must bypass similarity and be checked every turn.")

    return {"naive_violated": "pandas" in n["action"],
            "governed_compliant": "pandas" not in g["action"]
                                  and "polars" in g["action"]}


if __name__ == "__main__":
    outcome = demo()
    print("\noutcome:", outcome)
    assert outcome["naive_violated"], "naive agent should break the rule"
    assert outcome["governed_compliant"], "governed agent should comply"
    print("demo passed: failure mode reproduced, governing layer fixes it")

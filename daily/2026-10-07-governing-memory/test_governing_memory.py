"""Tests for the governing-facts memory demo."""

import numpy as np
from governing_memory import (
    EpisodicMemory, Constraint, ConstraintStore, NaiveAgent, GovernedAgent,
    extract_constraints, cosine, embed, tokenize,
    MONDAY_RULE, WEEK_CHATTER,
)


def test_tokenize_splits_punctuation():
    assert "pandas" in tokenize("Use pandas, not Pandas!")
    assert tokenize("") == []


def test_embed_is_deterministic_and_normalised():
    a, b = embed("hello world"), embed("hello world")
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-9


def test_similar_texts_score_higher_than_unrelated():
    q = embed("write a utility to deduplicate rows")
    assert cosine(q, embed("deduplicate duplicate rows utility")) > \
           cosine(q, embed("the database backup runs at midnight"))


def test_constraint_extraction_never():
    cs = extract_constraints(MONDAY_RULE)
    assert len(cs) == 1
    assert cs[0].prohibited == ["pandas"]
    assert cs[0].preferred == "polars"


def test_constraint_extraction_always():
    cs = extract_constraints("always use polars instead of pandas")
    assert cs and cs[0].prohibited == ["pandas"]
    assert cs[0].preferred == "polars"


def test_non_rule_text_yields_no_constraints():
    assert extract_constraints("we debugged the login flow today") == []


def test_naive_agent_buries_the_rule():
    agent = NaiveAgent()
    agent.chat(MONDAY_RULE)
    for m in WEEK_CHATTER:
        agent.chat(m)
    top = agent.chat("write a quick utility to deduplicate some rows")
    assert not any("pandas" in t for _, t in top), \
        "rule must NOT surface under top-k similarity for an unrelated query"


def test_governed_agent_blocks_and_regenerates():
    agent = GovernedAgent()
    agent.chat(MONDAY_RULE)
    agent.chat("debugged the login timeout")
    result = agent.act(
        "write a quick utility to deduplicate some rows",
        "import pandas as pd\ndf = pd.read_csv('x.csv')",
        lambda pref, bad: f"import {pref} as pl\ndf = pl.read_csv('x.csv')",
    )
    assert len(result["violations"]) == 1
    assert result["violations"][0]["prohibited_found"] == ["pandas"]
    assert result["regenerated"] is True
    assert "pandas" not in result["action"] and "polars" in result["action"]


def test_governed_agent_allows_compliant_action():
    agent = GovernedAgent()
    agent.chat(MONDAY_RULE)
    result = agent.act("read a csv", "import polars as pl\ndf = pl.read_csv('x.csv')",
                       lambda pref, bad: "UNREACHABLE")
    assert result["violations"] == [] and result["regenerated"] is False


def test_constraint_store_checks_every_turn_not_by_similarity():
    store = ConstraintStore()
    store.ingest(MONDAY_RULE)
    assert store.check("import pandas as pd")
    assert store.check("unrelated words about the weather") == []

"""Tests for the query fan-out planner. Run: python3 -m pytest test_query_fanout.py"""
import query_fanout as qf

INDEX = qf.TfidfIndex(qf.CORPUS)
COMPLEX = (
    "What does the warehouse migration cost each month, how fast will "
    "dashboards load, and how do we roll back if the cutover fails?"
)


# --- decomposition -------------------------------------------------------

def test_simple_question_does_not_fan_out():
    assert len(qf.decompose("What is the monthly cost of the warehouse?")) == 1


def test_complex_question_splits_into_three():
    parts = qf.decompose(COMPLEX)
    assert len(parts) == 3


def test_and_interrogative_splits_into_two():
    parts = qf.decompose(
        "Which data quality checks run before load and who can access restricted rows?"
    )
    assert len(parts) == 2


def test_semicolon_splits():
    parts = qf.decompose("What does it cost; when does the cutover happen?")
    assert len(parts) == 2


def test_empty_question():
    assert qf.decompose("   ") == []
    assert qf.decompose("?") == []


def test_no_split_on_plain_and_without_interrogative():
    # "and" joining nouns must not split the question
    parts = qf.decompose("What is the cost of storage and compute?")
    assert len(parts) == 1


# --- retrieval -----------------------------------------------------------

def test_cost_query_finds_cost_doc():
    hits = INDEX.search("monthly warehouse cost pounds", k=1)
    assert hits[0][0] == "cost"


def test_rollback_query_finds_rollback_doc():
    hits = INDEX.search("roll back if the cutover fails restore snapshot", k=1)
    assert hits[0][0] == "rollback"


def test_unrelated_query_scores_low():
    hits = INDEX.search("executive presentation skills training course", k=1)
    assert hits[0][1] < qf.COVERED_THRESHOLD


# --- fan-out and coverage -------------------------------------------------

def test_fanout_covers_all_three_facets():
    result = qf.fanout(COMPLEX, INDEX)
    assert result.coverage == 1.0
    assert result.docs_used == {"cost", "latency", "rollback"}


def test_fanout_beats_single_shot_on_facet_coverage():
    wanted = {"cost", "latency", "rollback"}
    result = qf.fanout(COMPLEX, INDEX)
    baseline = qf.single_shot(COMPLEX, INDEX, k=3)
    fan_cov = qf.facet_coverage(result.docs_used, wanted)
    base_cov = qf.facet_coverage(baseline, wanted)
    assert fan_cov == 1.0
    assert fan_cov >= base_cov


def test_missing_facet_is_flagged_not_invented():
    result = qf.fanout(
        "What does the warehouse cost each month and what training is "
        "provided for executives on presentation skills?",
        INDEX,
    )
    statuses = [s.status for s in result.sub_results]
    assert statuses[0] == "covered"
    assert statuses[1] in ("weak", "missing")
    text = qf.synthesise(result, INDEX)
    if statuses[1] == "missing":
        assert "UNANSWERED" in text


def test_two_facet_question_maps_to_two_docs():
    result = qf.fanout(
        "Which data quality checks run before load and who can access restricted rows?",
        INDEX,
    )
    assert result.docs_used == {"data_quality", "security"}


def test_simple_question_full_coverage():
    result = qf.fanout("What is the monthly cost of the warehouse?", INDEX)
    assert len(result.sub_results) == 1
    assert result.coverage == 1.0


def test_status_thresholds_are_ordered():
    assert qf.COVERED_THRESHOLD > qf.WEAK_THRESHOLD > 0

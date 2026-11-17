"""Tests for stt_context_bias. Run: python3 -m pytest test_stt_context_bias.py
or simply: python3 test_stt_context_bias.py
"""

from stt_context_bias import (
    BIAS_PHRASES,
    Formatter,
    Utterance,
    build_corpus,
    cand,
    corpus_wer,
    decode,
    phrases_ending_at,
    sweep,
    word_error_rate,
)


def test_wer_identical_is_zero():
    assert word_error_rate(["a", "b"], ["a", "b"]) == 0.0


def test_wer_one_substitution_in_four():
    assert word_error_rate(["a", "b", "c", "d"], ["a", "x", "c", "d"]) == 0.25


def test_wer_insertion_counts():
    # one insertion over a 2-word reference = 0.5
    assert word_error_rate(["a", "b"], ["a", "x", "b"]) == 0.5


def test_baseline_is_acoustic_argmax():
    corpus = build_corpus()
    u = next(x for x in corpus if x.name == "pipeline-report")
    assert decode(u, 0.0)[:2] == ["pipe", "plus"]


def test_bias_recovers_pipeline_pulse():
    corpus = build_corpus()
    u = next(x for x in corpus if x.name == "pipeline-report")
    assert decode(u, 0.75) == u.reference


def test_bias_recovers_trigram_phrase():
    corpus = build_corpus()
    u = next(x for x in corpus if x.name == "rag-lab")
    assert decode(u, 0.75) == u.reference
    # a moderate bonus is not enough for the whole trigram: honest negative
    assert decode(u, 0.5) != u.reference


def test_bias_recovers_unigrams():
    corpus = build_corpus()
    for name in ("nhs-systems", "bi-dashboard"):
        u = next(x for x in corpus if x.name == name)
        assert decode(u, 0.75) == u.reference


def test_phrase_bonus_needs_consecutive_tokens():
    assert phrases_ending_at(("pipeline", "pulse"), BIAS_PHRASES) == 1
    assert phrases_ending_at(("pipeline", "plus"), BIAS_PHRASES) == 0
    assert phrases_ending_at(("rag", "eval", "lab"), BIAS_PHRASES) == 1


def test_sweep_shape_moderate_beats_none_and_excess():
    corpus = build_corpus()
    results = dict(sweep(corpus, [0.0, 0.25, 0.5, 0.75, 1.0, 2.0]))
    assert results[0.75] == 0.0
    assert results[0.0] > results[0.5] > results[0.75]
    assert results[2.0] > results[0.75]


def test_control_flips_only_at_high_beta():
    corpus = build_corpus()
    control = next(x for x in corpus if x.control)
    assert decode(control, 0.75) == control.reference
    assert decode(control, 2.0)[4] == "bi"  # "bike" over-biased into "bi"


def test_bias_cannot_invent_tokens_outside_lattice():
    u = Utterance("tiny", ["hello"], [[cand("hello", -0.5), cand("hullo", -0.4)]])
    # no bias phrase matches either candidate, so acoustics still win
    assert decode(u, 5.0) == ["hullo"]


def test_formatter_canonical_casing():
    fmt = Formatter(["Capitalise product names"])
    assert fmt.format(["the", "power", "bi", "dashboard", "uses", "duckdb"]) == \
        "the power BI dashboard uses DuckDB"


def test_formatter_currency():
    fmt = Formatter(["Use the pound sign for amounts in pounds"])
    assert fmt.format(["the", "invoice", "was", "50", "pounds"]) == \
        "the invoice was £50"


def test_formatter_sentence_case_and_ignored():
    fmt = Formatter(["Capitalise the first word of each sentence",
                     "Make it sound friendlier"])
    assert fmt.format(["systmone", "holds", "records"]) == "Systmone holds records"
    assert fmt.ignored == ["Make it sound friendlier"]


def test_formatter_without_instructions_is_identity_join():
    fmt = Formatter([])
    assert fmt.format(["a", "b"]) == "a b"


def test_corpus_wer_baseline_value():
    corpus = build_corpus()
    mean_wer, per = corpus_wer(corpus, 0.0)
    assert len(per) == 8
    assert abs(mean_wer - 0.1952) < 0.001


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)}/{len(tests)} tests passing")

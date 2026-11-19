"""Tests for next_concept.py - run: python3 test_next_concept.py"""
import unittest

from next_concept import (
    TOPICS, autoregressive_steps, build_corpus, clause_concept, decode,
    encode, flat_tokens, ncm_schedule, tokens_within_budget,
    concept_only_recovery, concept_predictability,
)


class TestHierarchy(unittest.TestCase):
    def test_round_trip_exact_on_corpus(self):
        for s in build_corpus(60, seed=3):
            self.assertEqual(decode(*encode(s)), s)

    def test_concept_matches_clause_topic(self):
        clause = "simmer the sauce slowly"
        self.assertEqual(clause_concept(clause), "cooking")

    def test_concepts_cover_every_clause(self):
        s = build_corpus(1, seed=11)[0]
        concepts, details = encode(s)
        self.assertEqual(len(concepts), len(details))
        self.assertTrue(all(c in TOPICS for c in concepts))

    def test_decode_rejects_mismatched_concept(self):
        concepts, details = encode(build_corpus(1, seed=5)[0])
        wrong = ["finance" if c != "finance" else "cooking" for c in concepts]
        with self.assertRaises(ValueError):
            decode(wrong, details)

    def test_unknown_clause_concept(self):
        self.assertEqual(clause_concept("flibberty gibbet wobble"), "unknown")


class TestSchedules(unittest.TestCase):
    def test_ncm_always_two_steps(self):
        for s in build_corpus(40, seed=9):
            self.assertEqual(ncm_schedule(s)["steps"], 2)

    def test_ar_steps_equal_token_count(self):
        s = build_corpus(1, seed=4)[0]
        self.assertEqual(autoregressive_steps(s), len(flat_tokens(s)))

    def test_ncm_total_codes_cover_sentence(self):
        s = build_corpus(1, seed=4)[0]
        sched = ncm_schedule(s)
        # total codes = concept codes + detail tokens >= sentence tokens
        self.assertGreaterEqual(sched["total_codes"], len(flat_tokens(s)) - 2)

    def test_two_step_budget_gap(self):
        corpus = build_corpus(30, seed=2)
        ar = sum(tokens_within_budget(s, 2, "ar") for s in corpus)
        ncm = sum(tokens_within_budget(s, 2, "ncm") for s in corpus)
        totals = sum(len(flat_tokens(s)) for s in corpus)
        self.assertEqual(ar, 2 * len(corpus))
        # NCM codes = concept codes + detail tokens (separators excluded)
        expected = sum(ncm_schedule(s)["total_codes"] for s in corpus)
        self.assertEqual(ncm, expected)
        self.assertGreater(ncm, totals)  # concept codes are extra, by design
        self.assertGreater(ncm, ar)

    def test_speedup_above_five(self):
        corpus = build_corpus(50, seed=7)
        mean_ar = sum(autoregressive_steps(s) for s in corpus) / len(corpus)
        self.assertGreater(mean_ar / 2, 5.0)


class TestCoarseLevel(unittest.TestCase):
    def test_concept_groups_smaller_than_corpus(self):
        corpus = build_corpus(120, seed=7)
        rec = concept_only_recovery(corpus)
        self.assertLess(rec["concept_groups"], rec["sentences"])
        self.assertGreater(rec["share_pinned_by_concepts"], 0.0)
        self.assertLessEqual(rec["share_pinned_by_concepts"], 1.0)

    def test_concept_predictability_above_chance(self):
        corpus = build_corpus(200, seed=7)
        # chance for a 4-topic next-concept guess is 0.25
        self.assertGreater(concept_predictability(corpus), 0.25)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Tests for the StitchRAG miniature. Run: python3 test_stitchrag.py"""
import unittest

from stitchrag import (GOLD_FACTS, PASSAGES, QUERY, fact_recall,
                       naive_context, redundancy, retrieve, stitch,
                       token_count)


class TestStitchRAG(unittest.TestCase):
    def setUp(self):
        self.retrieved = retrieve(QUERY, PASSAGES, k=6)
        self.naive = naive_context(self.retrieved)

    def test_retrieval_returns_all_passages(self):
        self.assertEqual(len(self.retrieved), 6)

    def test_naive_context_covers_most_facts(self):
        recall, _ = fact_recall(self.naive)
        self.assertGreaterEqual(recall, 0.85)

    def test_stitch_respects_budget(self):
        for budget in (40, 60, 90, 120):
            stitched, _ = stitch(QUERY, self.retrieved, budget)
            self.assertLessEqual(token_count(stitched), budget)

    def test_stitch_uses_fewer_tokens_than_naive(self):
        stitched, _ = stitch(QUERY, self.retrieved, 90)
        self.assertLess(token_count(stitched), token_count(self.naive))

    def test_stitch_keeps_high_fact_recall(self):
        stitched, _ = stitch(QUERY, self.retrieved, 90)
        recall, _ = fact_recall(stitched)
        self.assertGreaterEqual(recall, 0.75)

    def test_stitch_more_efficient_per_token(self):
        stitched, _ = stitch(QUERY, self.retrieved, 90)
        s_recall, _ = fact_recall(stitched)
        n_recall, _ = fact_recall(self.naive)
        self.assertGreater(s_recall / token_count(stitched),
                           n_recall / token_count(self.naive))

    def test_evidence_preserved_verbatim(self):
        # Every kept stitch must appear word-for-word in its source
        # passage: stitching selects, it never rewrites a fact.
        _, chosen = stitch(QUERY, self.retrieved, 120)
        source = dict(PASSAGES)
        for s in chosen:
            self.assertIn(s.text, source[s.passage_id])

    def test_no_near_duplicate_stitches(self):
        stitched, _ = stitch(QUERY, self.retrieved, 120)
        self.assertEqual(redundancy(stitched), 0.0)

    def test_naive_context_has_redundancy(self):
        # The corpus deliberately repeats facts across passages, so
        # the naive concatenation must show measurable redundancy.
        self.assertGreater(redundancy(self.naive), 0.0)

    def test_stitch_is_deterministic(self):
        a, _ = stitch(QUERY, self.retrieved, 90)
        b, _ = stitch(QUERY, self.retrieved, 90)
        self.assertEqual(a, b)

    def test_tiny_budget_keeps_something_useful(self):
        stitched, chosen = stitch(QUERY, self.retrieved, 30)
        self.assertGreater(len(chosen), 0)
        recall, _ = fact_recall(stitched)
        self.assertGreater(recall, 0.0)

    def test_gold_facts_are_eight(self):
        self.assertEqual(len(GOLD_FACTS), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)

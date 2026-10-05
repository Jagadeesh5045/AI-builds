"""Unit tests for metrics and the QA gold set."""
import unittest
from pathlib import Path

from chunk_bench.chunkers import Chunk
from chunk_bench.corpus import load_corpus
from chunk_bench.metrics import (
    contains_answer,
    context_precision_at_k,
    mean_reciprocal_rank,
    normalize,
    recall_at_k,
)
from chunk_bench.qa import load_qa_pairs, validate_qa_against_corpus

BASE = Path(__file__).resolve().parent.parent


def _chunk(text, doc_id="d1"):
    return Chunk(text=text, doc_id=doc_id, chunk_id="x", strategy="s", start=0, end=len(text))


class TestNormalize(unittest.TestCase):
    def test_case_and_punctuation_insensitive(self):
        self.assertEqual(normalize("Hello,  World!"), normalize("hello world"))


class TestContainsAnswer(unittest.TestCase):
    def test_found(self):
        self.assertTrue(contains_answer("The cat sat on the mat.", "cat sat"))

    def test_not_found(self):
        self.assertFalse(contains_answer("The cat sat on the mat.", "dog ran"))


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.ranked = [
            (_chunk("nothing relevant here", "d2"), 0.9),
            (_chunk("the answer span lives here", "d1"), 0.7),
            (_chunk("also nothing", "d2"), 0.5),
        ]

    def test_recall_at_k_hit(self):
        self.assertEqual(recall_at_k(self.ranked, "answer span", k=2), 1.0)

    def test_recall_at_k_miss(self):
        self.assertEqual(recall_at_k(self.ranked, "answer span", k=1), 0.0)

    def test_mrr(self):
        self.assertAlmostEqual(mean_reciprocal_rank(self.ranked, "answer span"), 0.5)

    def test_mrr_miss(self):
        self.assertEqual(mean_reciprocal_rank(self.ranked, "nope"), 0.0)

    def test_context_precision(self):
        self.assertAlmostEqual(
            context_precision_at_k(self.ranked, "d1", k=3), 1 / 3
        )


class TestGoldQA(unittest.TestCase):
    def test_every_span_exists_in_its_document(self):
        docs = {d.doc_id: d.text for d in load_corpus(BASE / "data" / "corpus")}
        pairs = load_qa_pairs(BASE / "data" / "qa_gold.json")
        self.assertEqual(len(pairs), 24)
        problems = validate_qa_against_corpus(pairs, docs)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()

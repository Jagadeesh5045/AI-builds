"""Tests for rag_failure_modes.py (stdlib only, deterministic)."""

import unittest

from rag_failure_modes import (
    DOCUMENTS, QUESTIONS, TfidfRetriever, chunk_naive, chunk_structured,
    diagnose, expand, gold_chunk_index, run_experiment, tokenize,
)


class TestChunking(unittest.TestCase):
    def test_structured_chunking_never_splits_a_sentence(self):
        chunks = chunk_structured(DOCUMENTS)
        for chunk in chunks:
            # Every chunk is a whole number of sentences from its section.
            self.assertNotIn("..", chunk["text"])
            self.assertTrue(chunk["text"].endswith("."))
        # The naive 120-char windows do split at least one sentence.
        naive = chunk_naive(DOCUMENTS)
        intact = sum(1 for _, doc, sentence in QUESTIONS
                     if gold_chunk_index(naive, doc, sentence) is not None)
        self.assertLess(intact, len(QUESTIONS))

    def test_structured_chunks_carry_metadata(self):
        chunks = chunk_structured(DOCUMENTS)
        self.assertTrue(all(chunk["meta"] for chunk in chunks))
        self.assertIn("Headphones return policy | Returns",
                      [c["meta"] for c in chunks])


class TestRetrieval(unittest.TestCase):
    def test_synonym_expansion_adds_document_terms(self):
        self.assertIn("refund", expand(tokenize("What is the return window?")))

    def test_fixed_retrieval_is_at_least_as_good_as_naive(self):
        naive = chunk_naive(DOCUMENTS)
        fixed = chunk_structured(DOCUMENTS)
        naive_retriever = TfidfRetriever(naive)
        fixed_retriever = TfidfRetriever(fixed, use_meta=True)
        naive_hits = sum(
            1 for q, doc, s in QUESTIONS
            if diagnose(q, doc, s, naive, naive_retriever, k=3,
                        grounded=False)["retrieved"])
        fixed_hits = sum(
            1 for q, doc, s in QUESTIONS
            if diagnose(q, doc, s, fixed, fixed_retriever, k=8,
                        grounded=True, rerank=True)["retrieved"])
        self.assertGreaterEqual(fixed_hits, naive_hits)
        self.assertEqual(fixed_hits, len(QUESTIONS))


class TestReader(unittest.TestCase):
    def test_grounded_reader_abstains_without_gold(self):
        chunks = chunk_structured(DOCUMENTS)
        retriever = TfidfRetriever(chunks, use_meta=True)
        result = diagnose("What is the return window for headphones?",
                          "Headphones return policy", "A sentence that is nowhere.",
                          chunks, retriever, k=8, grounded=True, rerank=True)
        self.assertEqual(result["outcome"], "abstained")

    def test_end_to_end_fixed_beats_naive(self):
        rows = run_experiment()
        naive_correct = sum(1 for _, n, _ in rows if n["outcome"] == "correct")
        fixed_correct = sum(1 for _, _, f in rows if f["outcome"] == "correct")
        self.assertEqual(fixed_correct, len(QUESTIONS))
        self.assertLess(naive_correct, fixed_correct)


if __name__ == "__main__":
    unittest.main()

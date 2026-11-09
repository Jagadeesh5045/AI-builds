"""Tests for the K-EXAONE 2.5-style context planner. Stdlib unittest."""
import unittest

from context_planner import (
    CONTEXT_LIMIT, MODEL_ID, Chunk, Effort, REASONING_RESERVE,
    ContextOverflowError, build_payload, demo_corpus, estimate_tokens,
    parse_effort, plan_request, relevance,
)


class TestTokens(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_deterministic(self):
        t = "Revenue grew in Q3, driven by renewals."
        self.assertEqual(estimate_tokens(t), estimate_tokens(t))
        self.assertGreater(estimate_tokens(t), 5)


class TestEffort(unittest.TestCase):
    def test_default_is_max(self):
        self.assertEqual(parse_effort(None), Effort.MAX)

    def test_off_is_rejected(self):
        for bad in ("off", "none", "disabled", "false"):
            with self.assertRaises(ValueError):
                parse_effort(bad)

    def test_unknown_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_effort("turbo")

    def test_tiers_are_monotonic(self):
        self.assertLess(REASONING_RESERVE[Effort.LOW],
                        REASONING_RESERVE[Effort.HIGH])
        self.assertLess(REASONING_RESERVE[Effort.HIGH],
                        REASONING_RESERVE[Effort.MAX])


class TestPlanning(unittest.TestCase):
    def setUp(self):
        self.query = "Which region had revenue growth in Q3?"
        self.corpus = demo_corpus()

    def test_plan_always_fits_real_limit(self):
        for eff in Effort:
            plan = plan_request(self.query, self.corpus, effort=eff)
            self.assertLessEqual(plan.total, CONTEXT_LIMIT)
            self.assertTrue(plan.validate())

    def test_reasoning_counts_toward_budget(self):
        low = plan_request(self.query, self.corpus, effort=Effort.LOW)
        mx = plan_request(self.query, self.corpus, effort=Effort.MAX)
        self.assertGreater(mx.max_tokens, low.max_tokens)
        # Same answer reserve, so the whole difference is reasoning.
        self.assertEqual(mx.max_tokens - low.max_tokens,
                         REASONING_RESERVE[Effort.MAX] - REASONING_RESERVE[Effort.LOW])

    def test_exact_boundary(self):
        # input + max_tokens == limit passes; +1 fails with the 400 mirror.
        plan = plan_request("hello", [], effort=Effort.LOW,
                            answer_tokens=100, limit=10_000)
        self.assertLessEqual(plan.total, 10_000)
        with self.assertRaises(ContextOverflowError):
            plan_request("hello", [], effort=Effort.MAX,
                         answer_tokens=600_000, limit=CONTEXT_LIMIT)

    def test_relevant_chunk_ranked_first(self):
        plan = plan_request(
            "why did returns rise in the South packaging defect",
            self.corpus, effort=Effort.LOW)
        self.assertEqual(plan.kept[0].chunk_id, "returns-deep-dive")

    def test_tight_limit_drops_chunks(self):
        plan = plan_request(self.query, self.corpus, effort=Effort.LOW,
                            answer_tokens=256, limit=3000)
        self.assertLessEqual(plan.total, 3000)
        self.assertTrue(plan.dropped)
        kept_ids = {c.chunk_id for c in plan.kept}
        # Irrelevant filler / social notes must not crowd out sales chunks
        # when any sales chunk is kept.
        if any(i.startswith("sales-") for i in kept_ids):
            self.assertNotIn("hr-note", kept_ids)

    def test_no_chunk_is_split(self):
        plan = plan_request(self.query, self.corpus, effort=Effort.HIGH,
                            answer_tokens=512, limit=20000)
        for c in plan.kept:
            self.assertIn(c, self.corpus)

    def test_tiny_limit_cannot_fit_always_on_reasoning(self):
        # Always-on has a floor: low reserve (2,048) + answer + query
        # already exceeds a 600-token ceiling, so planning must fail
        # loudly rather than silently dropping the reasoning.
        with self.assertRaises(ContextOverflowError):
            plan_request(self.query, self.corpus, effort=Effort.LOW,
                         answer_tokens=256, limit=600)

    def test_relevance_zero_for_unrelated(self):
        self.assertEqual(relevance("quantum entanglement photons",
                                   Chunk("x", "the office picnic is on Friday")), 0.0)

    def test_payload_shape(self):
        plan = plan_request(self.query, self.corpus, effort=Effort.HIGH)
        payload = build_payload(plan, self.query)
        self.assertEqual(payload["model"], MODEL_ID)
        self.assertEqual(payload["model"], "k-exaone_v2.5")
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["max_tokens"], plan.max_tokens)
        self.assertEqual(payload["messages"][1]["role"], "user")

    def test_answer_must_be_positive(self):
        with self.assertRaises(ValueError):
            plan_request(self.query, self.corpus, answer_tokens=0)


if __name__ == "__main__":
    unittest.main()

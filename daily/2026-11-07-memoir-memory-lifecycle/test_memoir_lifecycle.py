"""Tests for the MEMOIR-style memory lifecycle demo.

Run: python3 -m unittest test_memoir_lifecycle -v
All expected values below were measured from the code in this folder; the
store and questions are fully deterministic (fixed shuffle seed).
"""

import unittest

from memoir_lifecycle import (
    Memory, Question, answer, audit_upper_bound, build_questions, build_store,
    cosine, evidence_recall, retrieve, run_session, stem, tokenize,
    try_consolidate, _slots,
)


class TestTokens(unittest.TestCase):
    def test_stem_and_stopwords(self):
        # Verb forms must converge: that is what lets a "managed by"
        # question meet a "manages" memory.
        self.assertEqual(stem("lives"), stem("live"))
        self.assertEqual(stem("manages"), stem("managed"))
        self.assertEqual(stem("uses"), stem("use"))
        toks = tokenize("Where does Maya live?")
        self.assertIn("maya", toks)
        self.assertIn(stem("live"), toks)
        self.assertNotIn("where", toks)
        self.assertNotIn("does", toks)

    def test_cosine_identical_is_one(self):
        t = tokenize("Maya lives in Leeds.")
        self.assertAlmostEqual(cosine(t, t), 1.0)


class TestStore(unittest.TestCase):
    def test_shape(self):
        store = build_store()
        self.assertEqual(len(store), 92)
        facts = [m for m in store if m.fact_id]
        self.assertEqual(len(facts), 36)          # 8*4 facts + 4 duplicates
        homes = [m for m in store if m.fact_id == "Maya:home"]
        self.assertEqual(len(homes), 2)           # original + paraphrase dup

    def test_priorities_start_at_zero(self):
        store = build_store()
        self.assertTrue(all(m.priority == 0.0 for m in store))


class TestSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.learn, cls.evalq = build_questions()
        cls.store = build_store()
        cls.stats = run_session(cls.store, cls.learn)

    def test_ambiguous_queries_are_gated(self):
        # 43 queries, exactly the 3 planted ambiguous ones must not produce
        # confident updates.
        self.assertEqual(self.stats["queries"], 43)
        self.assertEqual(self.stats["gated_out"], 3)
        self.assertEqual(self.stats["answered"], 40)

    def test_attribution_credits_the_gold_facts(self):
        by_fact = {}
        for m in self.store:
            if m.fact_id:
                by_fact[m.fact_id] = by_fact.get(m.fact_id, 0) + m.credits
        # Every asked-about fact family earns at least one credit somewhere.
        for name in ("Maya", "Tom", "Priya", "Jonas", "Aisha", "Leo", "Nina",
                     "Omar"):
            self.assertGreaterEqual(by_fact.get(f"{name}:manages", 0), 1)
        self.assertGreater(by_fact.get("billing:tool", 0), 0)

    def test_attribution_spares_chatter(self):
        # No note/chatter memory should earn a credit: leave-one-out removes
        # them and the answer never changes.
        credited = [m for m in self.store if m.fact_id is None and m.credits]
        self.assertEqual(credited, [])

    def test_priorities_bounded(self):
        for m in self.store:
            self.assertGreaterEqual(m.priority, 0.0)
            self.assertLess(m.priority, 1.0)

    def test_full_stack_beats_bare_at_5(self):
        bare = evidence_recall(self.store, self.evalq, 5, "bare")
        full = evidence_recall(self.store, self.evalq, 5, "full")
        self.assertGreater(full, bare)
        self.assertEqual(full, 1.0)

    def test_feedback_adds_at_5(self):
        # Measured shape of this toy: bridge expansion alone does not move
        # recall@5 (the lexical-gap "pastime" evidence sits in a 4-way tie
        # for the last top-5 slots); the retrospective priorities earned
        # in-session are what lift it in. Feedback is the whole @5 gain.
        nofb = evidence_recall(self.store, self.evalq, 5, "nofeedback")
        full = evidence_recall(self.store, self.evalq, 5, "full")
        self.assertGreater(full, nofb)

    def test_expansion_adds_at_3(self):
        # At k=3 the ordering flips: expansion (which needs no session)
        # rescues the multi-hop evidence, feedback adds nothing further.
        bare = evidence_recall(self.store, self.evalq, 3, "bare")
        nofb = evidence_recall(self.store, self.evalq, 3, "nofeedback")
        self.assertGreater(nofb, bare)

    def test_answerer_kind_derivation(self):
        # The answerer must derive the question kind from the text; spot
        # check it answers a direct and a multi-hop question correctly.
        store = build_store()
        r = retrieve(store, "Which tool does the billing project use?",
                     mode="bare")
        a = answer(r, "Which tool does the billing project use?")
        self.assertEqual(a.value, "DuckDB")
        r = retrieve(store, "Who leads the search project?", mode="bare")
        a = answer(r, "Who leads the search project?")
        self.assertEqual(a.value, "Tom")


class TestGateSynthetic(unittest.TestCase):
    def test_three_way_tie_is_below_gate(self):
        mems = []
        for i, city in enumerate(("York", "Bath", "Hull")):
            text = f"Ana lives in {city}."
            mems.append(Memory(i, f"ana:home{i}", "home", text, city,
                               tokenize(text)))
        r = retrieve(mems, "Where does Ana live?", mode="bare")
        a = answer(r, "Where does Ana live?")
        self.assertIsNotNone(a)
        self.assertAlmostEqual(a.confidence, 1 / 3, places=2)


class TestConsolidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.learn, cls.evalq = build_questions()
        cls.store = build_store()
        run_session(cls.store, cls.learn)
        cls.pre_slots = _slots(cls.store, cls.evalq)
        cls.store2, cls.rep = try_consolidate(cls.store, cls.learn)

    def test_only_duplicate_merges_certify(self):
        self.assertEqual(self.rep["certified"], 4)
        self.assertGreater(self.rep["rejected"], 20)
        self.assertEqual(self.rep["size_before"], 92)
        self.assertEqual(self.rep["size_after"], 88)

    def test_merged_memory_keeps_value_token(self):
        merged = [m for m in self.store2 if m.kind == "merged"]
        self.assertEqual(len(merged), 4)
        for m in merged:
            self.assertIn(stem(m.value.lower()), m.tokens)

    def test_audit_finds_no_losses(self):
        post = _slots(self.store2, self.evalq)
        self.assertEqual(post, self.pre_slots)

    def test_audit_bound_reproduces_paper_figure(self):
        # Paper: 0 losses in 100 audited merges -> 2.95% upper bound.
        self.assertAlmostEqual(audit_upper_bound(0, 100), 0.0295, places=3)
        # A 4-merge audit certifies far more weakly - the honest contrast.
        self.assertGreater(audit_upper_bound(0, 4), 0.5)


if __name__ == "__main__":
    unittest.main()

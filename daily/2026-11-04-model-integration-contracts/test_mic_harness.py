"""Tests for the Model Integration Contracts harness.

Run: python3 -m unittest test_mic_harness -v
"""
import unittest

import numpy as np

from mic_harness import (
    EXAMPLE_TOKENS,
    FIXTURE_CONFIG,
    FULL_CONFIG,
    TinyDecoder,
    build_contract,
    causality_probe,
    fixture_size_mb,
    param_count,
    replay_commits,
    run_contract,
    validate_contract,
)


class TestFixture(unittest.TestCase):
    def test_fixture_is_much_smaller_than_full_config(self):
        self.assertLess(fixture_size_mb(FIXTURE_CONFIG),
                        fixture_size_mb(FULL_CONFIG) / 50)

    def test_param_count_positive_and_ordered(self):
        self.assertGreater(param_count(FIXTURE_CONFIG), 0)
        self.assertGreater(param_count(FULL_CONFIG),
                           param_count(FIXTURE_CONFIG))

    def test_same_seed_same_weights(self):
        a = TinyDecoder(causal=True)
        b = TinyDecoder(causal=True)
        self.assertTrue(np.allclose(a.forward(EXAMPLE_TOKENS),
                                    b.forward(EXAMPLE_TOKENS)))

    def test_buggy_model_still_runs(self):
        # The whole point: the regression does not crash.
        out = TinyDecoder(causal=False).forward(EXAMPLE_TOKENS)
        self.assertEqual(out.shape,
                         (len(EXAMPLE_TOKENS), FIXTURE_CONFIG["vocab"]))
        self.assertTrue(np.isfinite(out).all())


class TestContractValidation(unittest.TestCase):
    def test_built_contract_is_valid(self):
        self.assertEqual(validate_contract(build_contract()), [])

    def test_missing_baseline_is_flagged(self):
        c = build_contract()
        del c["baseline"]
        self.assertTrue(any("baseline" in p
                            for p in validate_contract(c)))

    def test_missing_example_is_flagged(self):
        c = build_contract()
        c["example_tokens"] = []
        self.assertTrue(any("example" in p
                            for p in validate_contract(c)))


class TestProbes(unittest.TestCase):
    def setUp(self):
        self.contract = build_contract()
        self.good = TinyDecoder(causal=True)
        self.buggy = TinyDecoder(causal=False)

    def test_correct_model_is_causal(self):
        leak = causality_probe(self.good, EXAMPLE_TOKENS)
        self.assertLessEqual(leak, self.contract["causality_tolerance"])

    def test_buggy_model_leaks_future_tokens(self):
        leak = causality_probe(self.buggy, EXAMPLE_TOKENS)
        self.assertGreater(leak, 0.01)

    def test_correct_model_passes_whole_contract(self):
        self.assertTrue(run_contract(self.good, self.contract)
                        ["overall_pass"])

    def test_buggy_model_fails_whole_contract(self):
        self.assertFalse(run_contract(self.buggy, self.contract)
                         ["overall_pass"])

    def test_buggy_model_drifts_from_baseline(self):
        r = run_contract(self.buggy, self.contract)
        self.assertFalse(r["baseline_drift"]["pass"])
        self.assertGreater(r["baseline_drift"]["drift"],
                           self.contract["baseline"]["logprob_tolerance"])

    def test_baseline_matches_correct_model_exactly(self):
        mean_lp, _ = self.good.label_logprobs(EXAMPLE_TOKENS)
        self.assertAlmostEqual(
            mean_lp, self.contract["baseline"]["label_logprob_mean"],
            places=12)

    def test_loss_is_negative_mean_logprob(self):
        mean_lp, _ = self.good.label_logprobs(EXAMPLE_TOKENS)
        self.assertAlmostEqual(self.good.loss(EXAMPLE_TOKENS),
                               -mean_lp, places=12)


class TestReplay(unittest.TestCase):
    def test_replay_flags_exactly_the_buggy_commits(self):
        contract = build_contract()
        timeline = [TinyDecoder(causal=True), TinyDecoder(causal=True),
                    TinyDecoder(causal=False), TinyDecoder(causal=False),
                    TinyDecoder(causal=True)]
        self.assertEqual(replay_commits(contract, timeline),
                         [True, True, False, False, True])


if __name__ == "__main__":
    unittest.main()

"""Tests for two_door_gate.py. Stdlib unittest only."""

import unittest

from two_door_gate import (
    Category,
    Decision,
    GateResult,
    Query,
    Requester,
    TwoDoorGate,
    demo_bank,
    summarise,
)

VETTED = Requester("vetted org", vetted=True)
UNVETTED = Requester("visitor", vetted=False)


def q(category):
    return Query("T1", category, "test query")


class TestBoundaries(unittest.TestCase):
    def test_benign_allowed_under_both_profiles(self):
        for profile in ("general", "trusted"):
            self.assertEqual(
                TwoDoorGate(profile).decide(q(Category.BENIGN), VETTED).decision,
                Decision.ALLOW,
            )

    def test_weaponisation_refused_under_every_profile_and_requester(self):
        for profile in ("general", "trusted"):
            for requester in (VETTED, UNVETTED):
                result = TwoDoorGate(profile).decide(
                    q(Category.WEAPONISATION), requester)
                self.assertEqual(result.decision, Decision.REFUSE)

    def test_dual_use_refused_under_general(self):
        self.assertEqual(
            TwoDoorGate("general").decide(q(Category.DUAL_USE), VETTED).decision,
            Decision.REFUSE,
        )

    def test_sensitive_refused_under_general(self):
        self.assertEqual(
            TwoDoorGate("general").decide(q(Category.SENSITIVE), VETTED).decision,
            Decision.REFUSE,
        )

    def test_dual_use_and_sensitive_allowed_with_review_when_vetted(self):
        for category in (Category.DUAL_USE, Category.SENSITIVE):
            result = TwoDoorGate("trusted").decide(q(category), VETTED)
            self.assertEqual(result.decision, Decision.ALLOW_WITH_REVIEW)
            self.assertTrue(result.needs_human_review)
            self.assertTrue(result.answered)

    def test_unvetted_trusted_request_downgraded_to_general(self):
        for category in (Category.DUAL_USE, Category.SENSITIVE):
            result = TwoDoorGate("trusted").decide(q(category), UNVETTED)
            self.assertEqual(result.decision, Decision.REFUSE)
            self.assertEqual(result.profile_used, "general")
            self.assertIn("downgraded", result.reason)

    def test_vetting_does_not_open_general_profile_further(self):
        # Vetting changes nothing on the general boundary: the door
        # is the profile, and access only selects between profiles.
        for category in (Category.DUAL_USE, Category.SENSITIVE):
            self.assertEqual(
                TwoDoorGate("general").decide(q(category), VETTED).decision,
                Decision.REFUSE,
            )

    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):
            TwoDoorGate("open")


class TestBankTotals(unittest.TestCase):
    def test_bank_composition(self):
        bank = demo_bank()
        self.assertEqual(len(bank), 20)
        counts = {c: sum(1 for item in bank if item.category is c)
                  for c in Category}
        self.assertEqual(counts[Category.BENIGN], 8)
        self.assertEqual(counts[Category.DUAL_USE], 5)
        self.assertEqual(counts[Category.SENSITIVE], 4)
        self.assertEqual(counts[Category.WEAPONISATION], 3)

    def test_bank_higher_risk_rows_carry_no_question_text(self):
        # Guard against a future edit pasting real dual-use content
        # into the bank: higher-risk labels must stay abstract.
        for item in demo_bank():
            if item.category is not Category.BENIGN:
                self.assertTrue(
                    item.label.startswith(item.category.value),
                    f"{item.id} label must stay abstract, got: {item.label}",
                )

    def test_summary_numbers(self):
        bank = demo_bank()
        general = summarise([TwoDoorGate("general").decide(x, VETTED)
                             for x in bank])
        self.assertEqual((general["answered"], general["refused"],
                          general["needs_human_review"]), (8, 12, 0))
        trusted = summarise([TwoDoorGate("trusted").decide(x, VETTED)
                             for x in bank])
        self.assertEqual((trusted["answered"], trusted["refused"],
                          trusted["needs_human_review"]), (17, 3, 9))
        fallback = summarise([TwoDoorGate("trusted").decide(x, UNVETTED)
                              for x in bank])
        self.assertEqual(fallback, general)
        for summary in (general, trusted, fallback):
            self.assertEqual(summary["weaponisation_refused"], 3)

    def test_gate_result_flags(self):
        allowed = GateResult(q(Category.BENIGN), "general",
                             Decision.ALLOW, "ok")
        self.assertTrue(allowed.answered)
        self.assertFalse(allowed.needs_human_review)
        refused = GateResult(q(Category.DUAL_USE), "general",
                             Decision.REFUSE, "no")
        self.assertFalse(refused.answered)


if __name__ == "__main__":
    unittest.main()

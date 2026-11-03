"""Tests for the QA-bench-style bug hunt demo (stdlib unittest)."""

import unittest

from qa_bench import (BUGS, DEV_DENSITY, ShopApp, make_probes,
                      prioritised_order, qa_bench_headline_recall,
                      run_checklist, run_random)


class BugPlantTests(unittest.TestCase):
    def test_twelve_labelled_bugs_across_five_areas(self):
        self.assertEqual(len(BUGS), 12)
        self.assertEqual(len({b.bug_id for b in BUGS}), 12)
        self.assertEqual({b.area for b in BUGS},
                         {"catalogue", "search", "cart", "checkout", "account"})

    def test_each_planted_bug_has_exactly_one_probe(self):
        probes = make_probes()
        covered = sorted(p.bug_id for p in probes if p.bug_id)
        self.assertEqual(covered, sorted(b.bug_id for b in BUGS))

    def test_every_bug_probe_fails_and_healthy_probe_passes(self):
        for probe in make_probes():
            if probe.bug_id:
                self.assertFalse(probe.check(), probe.name)
            else:
                self.assertTrue(probe.check(), probe.name)


class HunterTests(unittest.TestCase):
    def test_systematic_full_budget_finds_all_bugs_perfect_precision(self):
        result = run_checklist(make_probes(), 60, "SystematicHunter")
        self.assertEqual(len(result.matched), 12)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.precision, 1.0)

    def test_budget_limits_checklist_coverage(self):
        # 18 probes at 2 minutes each; a 10-minute budget runs only 5.
        result = run_checklist(make_probes(), 10, "SystematicHunter")
        self.assertEqual(result.minutes, 10)
        self.assertLessEqual(len(result.matched), 5)

    def test_prioritised_beats_checklist_order_on_tight_budget(self):
        probes = make_probes()
        plain = run_checklist(probes, 12, "SystematicHunter")
        prio = run_checklist(prioritised_order(probes), 12, "PrioritisedHunter")
        self.assertGreater(len(prio.matched), len(plain.matched))

    def test_random_hunter_is_seeded_and_noisy(self):
        a = run_random(7, 60)
        b = run_random(7, 60)
        self.assertEqual(len(a.reports), len(b.reports))
        self.assertEqual(a.matched, b.matched)
        # Noise means it files reports that are not unique true bugs.
        self.assertLess(a.precision, 1.0)
        self.assertGreater(len(a.matched), 0)
        self.assertLessEqual(len(a.matched), 12)

    def test_cost_per_bug_positive_and_finite_when_bugs_found(self):
        result = run_checklist(make_probes(), 60, "SystematicHunter")
        self.assertGreater(result.cost_per_bug, 0)
        self.assertLess(result.cost_per_bug, result.cost)

    def test_prioritised_order_uses_dev_density(self):
        ordered = prioritised_order(make_probes())
        first_area = ordered[0].area
        self.assertEqual(DEV_DENSITY[first_area], max(DEV_DENSITY.values()))


class HeadlineArithmeticTests(unittest.TestCase):
    def test_mo_recall_matches_published_table(self):
        self.assertAlmostEqual(qa_bench_headline_recall(250.3), 0.179, places=3)

    def test_codex_recall_matches_published_table(self):
        self.assertAlmostEqual(qa_bench_headline_recall(94.3), 0.068, places=3)

    def test_app_is_deterministic(self):
        self.assertEqual(ShopApp().search(""), ShopApp().search(""))


if __name__ == "__main__":
    unittest.main()

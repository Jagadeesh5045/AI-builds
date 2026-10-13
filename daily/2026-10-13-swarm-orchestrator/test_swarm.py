"""Tests for the swarm orchestrator. Run: python -m unittest -v"""

import json
import unittest

from swarm import (ApprovalGate, Benchmarker, Coder, Ledger, Mission, Queen,
                   Researcher, Reviewer, Risk, SharedMemory, Task,
                   TaskResult, Tester, naive_single_agent, DocsPublisher,
                   Learner)


def buggy_repo():
    return {"utils.py": "def top_k(items, k):\n    return items[:k - 1]\n"}


def make_queen(approver):
    workers = {"researcher": Researcher(), "coder": Coder(),
               "tester": Tester(), "reviewer": Reviewer(),
               "benchmarker": Benchmarker(), "learner": Learner(),
               "publisher": DocsPublisher()}
    return Queen(workers, ApprovalGate(approver), Ledger(), SharedMemory())


class TestSharedMemory(unittest.TestCase):
    def test_namespaces_isolate_keys(self):
        m = SharedMemory()
        m.write("a", "k", 1)
        m.write("b", "k", 2)
        self.assertEqual(m.read("a", "k"), 1)
        self.assertEqual(m.read("b", "k"), 2)
        self.assertIsNone(m.read("a", "missing"))
        self.assertEqual(m.read("a", "missing", "dflt"), "dflt")

    def test_read_all_and_export(self):
        m = SharedMemory()
        m.write("ns", "k1", "v1")
        self.assertEqual(m.read_all("ns"), {"k1": "v1"})
        self.assertEqual(m.export(), {"ns": {"k1": "v1"}})


class TestApprovalGate(unittest.TestCase):
    def test_low_risk_auto_approves_without_calling_approver(self):
        calls = []
        gate = ApprovalGate(approver=lambda t: calls.append(t) or True)
        self.assertTrue(gate.request(Task("t", "coder", risk=Risk.LOW)))
        self.assertEqual(calls, [])

    def test_high_risk_denied_and_recorded(self):
        gate = ApprovalGate(approver=lambda t: False)
        self.assertFalse(gate.request(Task("t", "coder", risk=Risk.HIGH)))
        self.assertEqual(gate.decisions,
                         [{"task": "t", "risk": "high", "approved": False}])

    def test_high_risk_approved(self):
        gate = ApprovalGate(approver=lambda t: True)
        self.assertTrue(gate.request(Task("t", "coder", risk=Risk.HIGH)))


class TestWorkers(unittest.TestCase):
    def test_researcher_records_location_in_memory(self):
        m = SharedMemory()
        r = Researcher().run(Task("research", "researcher"), m, buggy_repo())
        self.assertTrue(r.success)
        self.assertEqual(m.read("research", "bug_location"),
                         [{"file": "utils.py", "line": 1}])

    def test_coder_rev1_unsafe_rev2_clean(self):
        m = SharedMemory()
        repo = buggy_repo()
        r1 = Coder().run(Task("build", "coder"), m, repo)
        self.assertTrue(r1.output["unsafe"])
        self.assertIn("eval(", repo["utils.py"])
        m.write("secure", "review_findings", ["banned pattern 'eval('"])
        r2 = Coder().run(Task("build", "coder"), m, repo)
        self.assertEqual(r2.output["revision"], 2)
        self.assertNotIn("eval(", repo["utils.py"])

    def test_tester_passes_on_clean_fix(self):
        m = SharedMemory()
        repo = buggy_repo()
        Coder().run(Task("build", "coder"), m, repo)
        m.write("secure", "review_findings", ["x"])  # force clean revision
        Coder().run(Task("build", "coder"), m, repo)
        r = Tester().run(Task("test", "tester"), m, repo)
        self.assertTrue(r.success, r.output["failures"])
        self.assertEqual(r.output["failures"], [])

    def test_reviewer_flags_eval(self):
        m = SharedMemory()
        repo = buggy_repo()
        Coder().run(Task("build", "coder"), m, repo)  # rev1: unsafe
        r = Reviewer().run(Task("secure", "reviewer"), m, repo)
        self.assertTrue(any("eval(" in f for f in r.findings))
        self.assertEqual(m.read("secure", "review_findings"), r.findings)

    def test_benchmarker_reports_speedup(self):
        r = Benchmarker().run(Task("benchmark", "benchmarker"),
                              SharedMemory(), buggy_repo())
        self.assertTrue(r.success)
        self.assertGreater(r.output["speedup"], 1)


class TestLedger(unittest.TestCase):
    def test_accounting_and_summary(self):
        ledger = Ledger()
        ledger.record(TaskResult("t1", "coder", True, tokens=1_000_000), 5.0)
        ledger.record(TaskResult("t2", "tester", False, tokens=500_000,
                                 approved=False), 3.0)
        s = ledger.summary()
        self.assertEqual(s["total_tasks"], 2)
        self.assertEqual(s["passed"], 1)
        self.assertEqual(s["failed"], 1)
        self.assertEqual(s["blocked_by_gate"], 1)
        self.assertEqual(s["total_tokens"], 1_500_000)
        self.assertAlmostEqual(s["total_cost_usd"], 4.5)
        self.assertIn("t1", ledger.table())
        parsed = json.loads(ledger.to_json())
        self.assertEqual(parsed["summary"]["total_tasks"], 2)


class TestMission(unittest.TestCase):
    def test_full_mission_denied_deploy(self):
        queen = make_queen(approver=lambda t: False)
        report = queen.run_mission(Mission("m", "g"), buggy_repo())
        self.assertTrue(report["revised_after_review"])
        self.assertTrue(report["final_test_passed"])
        self.assertEqual(report["final_review_findings"], [])
        self.assertTrue(report["deploy"]["blocked"])
        self.assertEqual(report["summary"]["blocked_by_gate"], 1)
        self.assertGreater(report["summary"]["total_cost_usd"], 0)

    def test_full_mission_approved_deploy(self):
        queen = make_queen(approver=lambda t: True)
        report = queen.run_mission(Mission("m", "g"), buggy_repo())
        self.assertFalse(report["deploy"]["blocked"])
        repo_check = buggy_repo()
        queen2 = make_queen(approver=lambda t: True)
        queen2.run_mission(Mission("m", "g"), repo_check)
        self.assertNotIn("eval(", repo_check["utils.py"])
        self.assertIn("CHANGELOG.md", repo_check)

    def test_decompose_stage_order(self):
        queen = make_queen(approver=lambda t: True)
        names = [t.name for t in queen.decompose(Mission("m", "g"))]
        self.assertEqual(names, ["research", "build", "test", "secure",
                                 "benchmark", "learn", "deploy_docs"])

    def test_learner_writes_lesson(self):
        queen = make_queen(approver=lambda t: True)
        queen.run_mission(Mission("m", "g"), buggy_repo())
        self.assertIn("eval()",
                      queen.memory.read("lessons", "top_k_mission"))


class TestNaiveBaseline(unittest.TestCase):
    def test_naive_ships_unsafe_with_no_review(self):
        repo = buggy_repo()
        out = naive_single_agent(repo)
        self.assertTrue(out["unsafe_shipped"])
        self.assertEqual(out["review_findings"], 0)


if __name__ == "__main__":
    unittest.main()

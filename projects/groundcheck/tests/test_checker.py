import sys, pathlib, unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from groundcheck.checker import check_answer, split_claims
from groundcheck.data import PASSAGES
from groundcheck.evaluate import run_eval

class TestChecker(unittest.TestCase):
    def test_split(self):
        self.assertEqual(len(split_claims("One claim here. Second claim here.")), 2)
    def test_grounded_passes(self):
        rep = check_answer("The Pro plan costs 29 pounds per month [1].", [PASSAGES[0]])
        self.assertEqual(rep.decision, "answer")
        self.assertEqual(rep.unsupported, 0)
    def test_wrong_number_flagged(self):
        rep = check_answer("The Pro plan costs 9 pounds per month [1].", [PASSAGES[0]])
        self.assertGreaterEqual(rep.unsupported, 1)
    def test_invented_feature_flagged(self):
        rep = check_answer("The platform includes a built in video editor [5].", [PASSAGES[4]])
        self.assertGreaterEqual(rep.unsupported, 1)
    def test_bad_citation_detected(self):
        rep = check_answer("The Pro plan costs 29 pounds per month [9].", [PASSAGES[0]])
        self.assertFalse(rep.claims[0].citation_ok)
    def test_eval_detects_hallucinations(self):
        r = run_eval()
        self.assertGreaterEqual(r["recall"], 0.75)
        self.assertEqual(r["cases"], 8)

if __name__ == "__main__":
    unittest.main()

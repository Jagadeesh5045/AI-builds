"""Unit + HTTP tests for the System One-style decisions demo (stdlib only).

Run:  python3 -m unittest test_systemone -v
"""

import json
import threading
import time
import unittest
import urllib.request

import server as srv
import systemone
from systemone import DecisionError, decide

D3_REQUEST = {
    "model": "vllm-sr/d3",
    "state": "Write a Python function that merges two sorted lists.",
    "questions": {
        "kind": {
            "type": "choice",
            "instructions": "What kind of work is this?",
            "criteria": {"code": "Writing or fixing code",
                         "math": "Mathematics",
                         "chat": "Anything else"},
        },
        "reasoning": {
            "type": "noul",
            "instructions": "Does answering this need multi-step reasoning?",
        },
    },
}


def assert_400(test, request):
    """Assert decide(request) raises DecisionError with status 400."""
    try:
        decide(request)
    except DecisionError as exc:
        test.assertEqual(exc.status, 400)
        return
    test.fail(f"expected DecisionError(400) for {request!r}")


class SystemOneTest(unittest.TestCase):

    # --- contract shape -------------------------------------------------
    def test_response_contract_shape(self):
        status, resp = decide(D3_REQUEST)
        self.assertEqual(status, 200)
        self.assertEqual(resp["object"], "structured_decision")
        self.assertEqual(resp["model"], "vllm-sr/d3")
        self.assertEqual(set(resp["answers"]), {"kind", "reasoning"})
        self.assertEqual(resp["usage"]["output_tokens"], 0)
        self.assertGreater(resp["usage"]["input_tokens"], 0)

    def test_d3_fixture_choice_matches_card(self):
        # The real d3 card answers "code" for this exact request; so must we.
        _, resp = decide(D3_REQUEST)
        kind = resp["answers"]["kind"]
        self.assertEqual(kind["choice"], "code")
        self.assertAlmostEqual(sum(kind["probabilities"].values()), 1.0, places=9)
        self.assertAlmostEqual(kind["confidence"], kind["probabilities"]["code"], places=9)

    def test_d3_fixture_noul_sane(self):
        _, resp = decide(D3_REQUEST)
        r = resp["answers"]["reasoning"]
        self.assertEqual(r["type"], "noul")
        self.assertGreaterEqual(r["noul"], 0.0)
        self.assertLessEqual(r["noul"], 1.0)
        self.assertAlmostEqual(r["probabilities"]["yes"] + r["probabilities"]["no"], 1.0, places=9)

    def test_confidence_equals_chosen_probability(self):
        _, resp = decide(D3_REQUEST)
        kind = resp["answers"]["kind"]
        self.assertAlmostEqual(kind["confidence"], kind["probabilities"][kind["choice"]], places=12)

    def test_deterministic(self):
        _, a = decide(D3_REQUEST)
        _, b = decide(D3_REQUEST)
        self.assertEqual(a["answers"], b["answers"])

    # --- validation (400s) ----------------------------------------------
    def test_unknown_question_type_400(self):
        req = {"model": "m", "state": "s",
               "questions": {"q": {"type": "vibe", "instructions": "?"}}}
        assert_400(self, req)

    def test_top_level_instructions_400(self):
        req = dict(D3_REQUEST, instructions="be nice")
        assert_400(self, req)

    def test_missing_fields_400(self):
        for req in ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}},
                    {"model": "m", "questions": {"q": {"type": "noul", "instructions": "?"}}},
                    {"model": "m", "state": "s", "questions": {}},
                    {"model": "m", "state": "s"}):
            assert_400(self, req)

    def test_choice_requires_criteria_400(self):
        req = {"model": "m", "state": "s",
               "questions": {"q": {"type": "choice", "instructions": "?"}}}
        assert_400(self, req)

    def test_choice_max_255_options(self):
        many = {f"opt{i}": "desc" for i in range(256)}
        req = {"model": "m", "state": "s",
               "questions": {"q": {"type": "choice", "instructions": "?",
                                   "criteria": many}}}
        assert_400(self, req)
        ok = {f"opt{i}": "desc" for i in range(255)}
        req["questions"]["q"]["criteria"] = ok
        status, _ = decide(req)
        self.assertEqual(status, 200)

    def test_score_level_bounds_400(self):
        for levels in (["only"], [f"l{i}" for i in range(11)]):
            req = {"model": "m", "state": "s",
                   "questions": {"q": {"type": "score", "instructions": "?",
                                       "levels": levels}}}
            assert_400(self, req)

    # --- score behaviour --------------------------------------------------
    def test_score_lands_between_levels(self):
        req = {"model": "m",
               "state": "The server is slow for one customer but there is a workaround doc.",
               "questions": {"q": {"type": "score",
                                   "instructions": "How urgent?",
                                   "levels": ["low", "medium", "high"],
                                   "criteria": {
                                       "low": "routine question, no deadline, workaround exists",
                                       "medium": "affects one customer, money involved, needs action today",
                                       "high": "production down, many customers affected, no workaround"}}}}
        _, resp = decide(req)
        ans = resp["answers"]["q"]
        self.assertEqual(ans["legend"], ["low", "medium", "high"])
        self.assertGreater(ans["score"], 0.0)
        self.assertLess(ans["score"], 2.0)
        self.assertGreaterEqual(ans["confidence"], 0.0)
        self.assertLessEqual(ans["confidence"], 1.0)

    def test_score_clear_high(self):
        req = {"model": "m",
               "state": "Production database is completely down, all users affected, no workaround exists.",
               "questions": {"q": {"type": "score",
                                   "instructions": "How urgent?",
                                   "levels": ["low", "medium", "high"],
                                   "criteria": {
                                       "low": "routine question, no deadline",
                                       "medium": "affects one customer",
                                       "high": "production down, all users affected, no workaround"}}}}
        _, resp = decide(req)
        self.assertGreater(resp["answers"]["q"]["score"], 1.5)

    # --- noul behaviour ---------------------------------------------------
    def test_negation_lowers_yes_probability(self):
        base = {"model": "m", "questions": {"q": {"type": "noul",
                "instructions": "Is the customer asking for a refund?"}}}
        _, a = decide(dict(base, state="I was charged twice, please refund the duplicate charge."))
        _, b = decide(dict(base, state="I was not charged twice, no refund needed."))
        pa, pb = a["answers"]["q"]["noul"], b["answers"]["q"]["noul"]
        self.assertGreater(pa, 0.5)
        self.assertLess(pb, pa)

    # --- ticket-router fixtures -------------------------------------------
    def test_outage_ticket_routes_engineering_confident(self):
        req = {"model": "m",
               "state": "Production Postgres is down since 02:00 and checkout is failing "
                        "for every EU customer. No workaround.",
               "questions": {"dept": {"type": "choice",
                                      "instructions": "Which team should own this ticket?",
                                      "criteria": {
                                          "engineering": "code, bugs, outages, infrastructure, servers, databases, production incidents",
                                          "billing": "invoices, charges, refunds, payments, pricing, duplicates",
                                          "account": "login, password, signup, profile, access, reset email"}}}}
        _, resp = decide(req)
        dept = resp["answers"]["dept"]
        self.assertEqual(dept["choice"], "engineering")
        self.assertGreaterEqual(dept["confidence"], 0.6)

    def test_password_ticket_routes_account(self):
        req = {"model": "m",
               "state": "Quick question: how do I reset my password? The reset email never arrives.",
               "questions": {"dept": {"type": "choice",
                                      "instructions": "Which team should own this ticket?",
                                      "criteria": {
                                          "engineering": "code, bugs, outages, infrastructure, servers, databases, production incidents",
                                          "billing": "invoices, charges, refunds, payments, pricing, duplicates",
                                          "account": "login, password, signup, profile, access, reset email"}}}}
        _, resp = decide(req)
        self.assertEqual(resp["answers"]["dept"]["choice"], "account")


class HttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = srv.create_server(18765)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:18765{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())

    def test_systemone_roundtrip(self):
        status, body = self._post("/v1/systemone", D3_REQUEST)
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "structured_decision")
        self.assertEqual(body["answers"]["kind"]["choice"], "code")
        self.assertEqual(body["usage"]["output_tokens"], 0)

    def test_systemone_http_400(self):
        req = urllib.request.Request(
            "http://127.0.0.1:18765/v1/systemone",
            data=json.dumps({"model": "m"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            self.fail("expected HTTP 400")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_chat_completions_200(self):
        status, body = self._post("/v1/chat/completions",
                                  {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertTrue(body["choices"][0]["message"]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Unit tests for the agent context compression layer."""

import json
import unittest

from agent_context import (
    AgentContextCompressor,
    ChunkCompressor,
    JsonCompressor,
    LogCompressor,
    estimate_tokens,
)


class TestLogCompressor(unittest.TestCase):
    def setUp(self):
        self.c = LogCompressor(head_lines=2, tail_lines=3)

    def test_error_lines_preserved_byte_for_byte(self):
        line = "ERROR 2026-10-10T20:02:15Z runner: test_payment_webhook FAILED"
        log = "INFO a\nINFO b\n" + line + "\nINFO c"
        out = self.c.compress(log)
        self.assertIn(line, out)

    def test_warning_kept(self):
        out = self.c.compress("INFO x\nWARNING slow query\nINFO y")
        self.assertIn("WARNING slow query", out)

    def test_consecutive_duplicates_collapsed(self):
        out = self.c.compress("INFO ping\nINFO ping\nINFO ping\nINFO pong")
        self.assertIn("... (repeated x3)", out)
        self.assertNotIn("INFO ping\nINFO ping", out)

    def test_truncation_summarises_dropped_middle(self):
        log = "\n".join("INFO line %d" % i for i in range(50))
        out = self.c.compress(log)
        self.assertIn("low-severity lines summarised away", out)
        self.assertLess(estimate_tokens(out), estimate_tokens(log))

    def test_short_log_unchanged_content(self):
        log = "INFO a\nERROR boom"
        out = self.c.compress(log)
        self.assertIn("INFO a", out)
        self.assertIn("ERROR boom", out)


class TestJsonCompressor(unittest.TestCase):
    def setUp(self):
        self.c = JsonCompressor(max_str_chars=30, max_list_items=2)

    def test_preserve_keys_survive(self):
        payload = {"id": "abc", "status": "ok", "trace": ["x", "y"]}
        out = json.loads(self.c.compress(payload))
        self.assertEqual(out["id"], "abc")
        self.assertEqual(out["status"], "ok")
        self.assertNotIn("trace", out)

    def test_long_lists_truncated_with_marker(self):
        payload = {"items": [1, 2, 3, 4, 5]}
        out = json.loads(self.c.compress(payload))
        self.assertIn("... (3 more items)", out["items"])
        self.assertEqual(len(out["items"]), 3)  # 2 kept + marker

    def test_long_strings_truncated(self):
        payload = {"message": "x" * 200}
        out = json.loads(self.c.compress(payload))
        self.assertTrue(out["message"].endswith("... (truncated)"))
        self.assertLess(len(out["message"]), 200)

    def test_output_is_valid_json(self):
        payload = {"a": {"b": [1, 2, 3]}, "debug": {"deep": True}}
        out = self.c.compress(payload)
        parsed = json.loads(out)  # must not raise
        self.assertNotIn("debug", parsed)

    def test_accepts_json_string_input(self):
        out = self.c.compress('{"id": 1, "verbose": "nope"}')
        parsed = json.loads(out)
        self.assertEqual(parsed["id"], 1)
        self.assertNotIn("verbose", parsed)

    def test_error_field_preserved(self):
        payload = {"status": "failed", "error": "timeout after 30s"}
        out = json.loads(self.c.compress(payload))
        self.assertEqual(out["error"], "timeout after 30s")


class TestChunkCompressor(unittest.TestCase):
    def setUp(self):
        self.c = ChunkCompressor(keep_ratio=0.5, max_tokens=500)

    def test_query_relevant_sentences_kept(self):
        chunk = ("The sky is blue. Payment webhooks fire on status transitions. "
                 "Cats are mammals. The 'pending' state is non-terminal and means "
                 "settlement has not arrived yet.")
        out = self.c.compress(chunk, "why is payment status pending")
        self.assertIn("Payment webhooks", out)
        self.assertIn("non-terminal", out)

    def test_filler_sentences_demoted(self):
        chunk = ("As mentioned above, please note that it is important to note "
                 "that pending means not settled. For more information, see the docs. "
                 "The handler reads event['status'] with default pending. "
                 "In conclusion, the lifecycle is pending then settled.")
        out = self.c.compress(chunk, "what does pending mean")
        self.assertIn("pending means not settled", out)
        self.assertNotIn("For more information", out)

    def test_original_order_preserved(self):
        chunk = ("First: webhooks fire. Second: pending is non-terminal. "
                 "Third: rain falls. Fourth: pending also matters for retries.")
        out = self.c.compress(chunk, "webhooks pending")
        self.assertLess(out.index("First"), out.index("Second"))

    def test_budget_cap_enforced(self):
        chunk = " ".join("Sentence %d about webhooks and pending states." % i
                         for i in range(100))
        c = ChunkCompressor(keep_ratio=1.0, max_tokens=50)
        out = c.compress(chunk, "webhooks pending")
        self.assertLessEqual(estimate_tokens(out), 60)  # one sentence may spill slightly

    def test_empty_chunk(self):
        self.assertEqual(self.c.compress("", "query"), "")


class TestAgentContextCompressor(unittest.TestCase):
    def test_end_to_end_stats(self):
        items = [
            {"kind": "log", "content": "INFO a\n" * 40 + "ERROR boom\n"},
            {"kind": "json", "content": {"id": 1, "trace": list(range(20))}},
            {"kind": "chunk", "content": "Alpha beta. Gamma pending settled. Delta.",
             "query": "pending settled"},
        ]
        comp = AgentContextCompressor(token_budget=4000)
        compressed, stats = comp.compress(items)
        self.assertEqual(len(compressed), 3)
        self.assertLess(stats["tokens_after"], stats["tokens_before"])
        self.assertGreater(stats["compression_ratio"], 1.0)
        self.assertTrue(stats["within_budget"])
        # answer-bearing content survives the pipeline
        self.assertIn("ERROR boom", compressed[0]["content"])
        self.assertIn("pending", compressed[2]["content"])

    def test_unknown_kind_passes_through(self):
        comp = AgentContextCompressor()
        compressed, _ = comp.compress([{"kind": "text", "content": "hello"}])
        self.assertEqual(compressed[0]["content"], "hello")


if __name__ == "__main__":
    unittest.main(verbosity=2)

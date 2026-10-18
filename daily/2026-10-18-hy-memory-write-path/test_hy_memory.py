"""Unit tests for the Hy Memory 2.0-style write path.

Run: python3 -m unittest test_hy_memory -v
"""

import unittest

from hy_memory import (Chunk, FlatWriter, MemoryWriter, RawLog, Scene,
                       _classify_scene, _extract_fact, _salience)


class TestSceneRouting(unittest.TestCase):
    def test_chitchat_goes_to_chat(self):
        self.assertEqual(_classify_scene("Hi, how are you?", "user"), Scene.CHAT)

    def test_lesson_goes_to_working(self):
        self.assertEqual(
            _classify_scene("Lesson: always pin versions in CI.", "user"),
            Scene.WORKING)

    def test_code_role_goes_to_code(self):
        self.assertEqual(
            _classify_scene("def f(x):\n    return x", "code"), Scene.CODE)

    def test_fact_statement_goes_to_working(self):
        self.assertEqual(
            _classify_scene("The staging database port is 5432.", "user"),
            Scene.WORKING)


class TestFactExtraction(unittest.TestCase):
    def test_port_fact(self):
        fact = _extract_fact("The staging database port is 5432.")
        self.assertIsNotNone(fact)
        entity, attribute, value = fact
        self.assertEqual(attribute, "port")
        self.assertEqual(value, "5432")

    def test_no_fact_in_chitchat(self):
        self.assertIsNone(_extract_fact("Hi there, nice to meet you."))


class TestSalience(unittest.TestCase):
    def test_fact_beats_chitchat(self):
        self.assertGreater(
            _salience("The staging database port is 5432.", "user", Scene.WORKING),
            _salience("Hi, how are you?", "user", Scene.CHAT))

    def test_chitchat_below_threshold(self):
        self.assertLess(
            _salience("Thanks!", "user", Scene.CHAT),
            MemoryWriter.SALIENCE_THRESHOLD)


class TestWritePath(unittest.TestCase):
    def setUp(self):
        self.w = MemoryWriter()

    def test_chitchat_not_persisted(self):
        written = self.w.execute_write(0, "user", "Hi!")
        self.assertEqual(written, [])
        self.assertEqual(self.w.stats.chunks_skipped, 1)

    def test_fact_persisted(self):
        written = self.w.execute_write(
            0, "user", "The staging database port is 5432.")
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0].value, "5432")

    def test_supersede_tombstones_old_fact(self):
        self.w.execute_write(0, "user", "The staging database port is 5432.")
        new = self.w.execute_write(
            1, "user", "Update: the staging database port is 5433 now.")
        old = [c for c in self.w.scenes[Scene.WORKING]
               if c.value == "5432"][0]
        self.assertIsNotNone(old.superseded_by)
        self.assertEqual(old.superseded_by, new[0].id)
        self.assertEqual(self.w.stats.supersedes, 1)

    def test_superseded_chunk_excluded_from_retrieval(self):
        self.w.execute_write(0, "user", "The staging database port is 5432.")
        self.w.execute_write(
            1, "user", "Update: the staging database port is 5433 now.")
        hits = self.w.retrieve("What is the staging database port?")
        values = [h.value for h in hits if h.value]
        self.assertIn("5433", values)
        self.assertNotIn("5432", values)

    def test_code_diff_chain_versions(self):
        self.w.execute_write(0, "code", "```python\ndef f():\n    return 1\n```")
        self.w.execute_write(
            1, "code",
            "```python\ndef f():\n    return 2\n```\nChanged because one was wrong.")
        code = self.w.scenes[Scene.CODE]
        self.assertEqual([c.code_version for c in code], [1, 2])
        self.assertIn("-    return 1", code[1].code_diff)
        self.assertIn("+    return 2", code[1].code_diff)
        self.assertIn("wrong", code[1].code_rationale)

    def test_live_counts_excludes_superseded(self):
        self.w.execute_write(0, "user", "The staging database port is 5432.")
        self.w.execute_write(
            1, "user", "Update: the staging database port is 5433 now.")
        counts = self.w.live_counts()
        self.assertEqual(counts["working"], 1)


class TestBaselines(unittest.TestCase):
    def test_rawlog_keeps_everything(self):
        log = RawLog()
        log.write("user", "Hi!")
        log.write("user", "The port is 5432.")
        self.assertEqual(len(log.lines), 2)

    def test_flatwriter_no_supersede(self):
        fw = FlatWriter()
        fw.write("The port is 5432.")
        fw.write("The port is 5433.")
        blob = " ".join(fw.retrieve("port"))
        self.assertIn("5432", blob)   # stale fact still returned: the failure
        self.assertIn("5433", blob)

    def test_rawlog_surfaces_stale_fact(self):
        log = RawLog()
        log.write("user", "The port is 5432.")
        log.write("user", "Update: the port is 5433 now.")
        blob = " ".join(log.retrieve("What is the port?"))
        self.assertIn("5432", blob)   # grep cannot tell new from old


class TestEndToEnd(unittest.TestCase):
    def test_hymemory_beats_baselines_on_scripted_session(self):
        from demo import SESSION, QUERIES, run
        correct_hy, _, _ = run(MemoryWriter())
        correct_raw, _, _ = run(RawLog())
        correct_flat, _, _ = run(FlatWriter())
        self.assertEqual(correct_hy, len(QUERIES))
        self.assertGreater(correct_hy, correct_raw)
        self.assertGreater(correct_hy, correct_flat)


if __name__ == "__main__":
    unittest.main()

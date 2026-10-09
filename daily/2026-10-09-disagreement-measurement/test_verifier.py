"""Unit tests for the deterministic citation verifier."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verifier import (
    MalformedCitation,
    find_citations,
    parse_citation,
    tokenize,
    verify,
)

ROOT = os.path.dirname(os.path.abspath(__file__))


class TestParseCitation(unittest.TestCase):
    def test_good_citation(self):
        self.assertEqual(parse_citation("fixtures/auth.py:12"), ("fixtures/auth.py", 12))

    def test_nested_path(self):
        self.assertEqual(parse_citation("a/b/c.py:1"), ("a/b/c.py", 1))

    def test_missing_line_number_raises(self):
        with self.assertRaises(MalformedCitation):
            parse_citation("fixtures/auth.py")

    def test_non_numeric_line_raises(self):
        with self.assertRaises(MalformedCitation):
            parse_citation("fixtures/auth.py:abc")

    def test_non_python_file_raises(self):
        with self.assertRaises(MalformedCitation):
            parse_citation("fixtures/auth.txt:12")


class TestTokenize(unittest.TestCase):
    def test_stopwords_removed(self):
        tokens = tokenize("the quick brown fox")
        self.assertNotIn("the", tokens)
        self.assertIn("quick", tokens)
        self.assertIn("fox", tokens)

    def test_case_insensitive(self):
        self.assertEqual(tokenize("Hello WORLD"), ["hello", "world"])


class TestVerify(unittest.TestCase):
    def test_good_citation_supported(self):
        r = verify(
            "authenticate checks the username and password against the user store",
            "fixtures/auth.py:12",
            root=ROOT,
        )
        self.assertEqual(r["status"], "SUPPORTED")
        self.assertIsNotNone(r["line"])

    def test_wrong_line_unsupported(self):
        r = verify(
            "authenticate checks the username and password against the user store",
            "fixtures/auth.py:6",
            root=ROOT,
        )
        self.assertEqual(r["status"], "UNSUPPORTED")
        self.assertLess(r["overlap"], 0.5)

    def test_line_out_of_range_unsupported(self):
        r = verify("some claim about code", "fixtures/auth.py:999", root=ROOT)
        self.assertEqual(r["status"], "UNSUPPORTED")
        self.assertIn("out of range", r["reason"])

    def test_missing_file_malformed(self):
        r = verify("some claim", "fixtures/does_not_exist.py:3", root=ROOT)
        self.assertEqual(r["status"], "MALFORMED")
        self.assertIn("does not exist", r["reason"])

    def test_malformed_citation_no_line(self):
        r = verify("some claim", "fixtures/auth.py", root=ROOT)
        self.assertEqual(r["status"], "MALFORMED")

    def test_empty_claim_unsupported(self):
        r = verify("", "fixtures/auth.py:11", root=ROOT)
        self.assertEqual(r["status"], "UNSUPPORTED")

    def test_exact_line_content_supported(self):
        r = verify(
            "Return the SHA-256 hex digest of a password string",
            "fixtures/auth.py:7",
            root=ROOT,
        )
        self.assertEqual(r["status"], "SUPPORTED")

    def test_paraphrase_supported(self):
        r = verify(
            "rank_chunks sorts chunks by score in descending order",
            "fixtures/retrieval.py:15",
            root=ROOT,
        )
        self.assertEqual(r["status"], "SUPPORTED")

    def test_reason_always_present(self):
        for status in ("SUPPORTED", "UNSUPPORTED", "MALFORMED"):
            r = verify("x", "fixtures/nope.py:1", root=ROOT)
            self.assertTrue(r["reason"])
            break
        r = verify(
            "authenticate checks the username and password against the user store",
            "fixtures/auth.py:12",
            root=ROOT,
        )
        self.assertTrue(r["reason"])


class TestFindCitations(unittest.TestCase):
    def test_finds_well_formed(self):
        found = find_citations("see fixtures/auth.py:12 for details")
        self.assertIn("fixtures/auth.py:12", found)

    def test_finds_malformed_too(self):
        found = find_citations("see fixtures/auth.py for details")
        self.assertIn("fixtures/auth.py", found)

    def test_no_citations(self):
        self.assertEqual(find_citations("no citations here"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

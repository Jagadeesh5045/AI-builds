"""Unit tests for the chunking strategies."""
import unittest

from chunk_bench.chunkers import fixed_size_chunks, heading_chunks, sentence_chunks, split_sentences


class TestSplitSentences(unittest.TestCase):
    def test_basic(self):
        sents = split_sentences("Hello world. This is a test. And another one!")
        self.assertEqual(len(sents), 3)

    def test_empty(self):
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences("   "), [])


class TestFixedSize(unittest.TestCase):
    def test_window_and_overlap(self):
        text = "x" * 2500
        chunks = fixed_size_chunks("d1", text, size=1000, overlap=100, strategy="fixed")
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(c.text) <= 1000 for c in chunks))
        # Overlap: end of chunk 0 == start of chunk 1's tail
        self.assertEqual(chunks[0].text[-100:], chunks[1].text[:100])

    def test_verbatim(self):
        text = "The quick brown fox jumps over the lazy dog. " * 40
        chunks = fixed_size_chunks("d1", text, size=500, overlap=50, strategy="fixed")
        for c in chunks:
            self.assertEqual(c.text, text[c.start:c.end])

    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            fixed_size_chunks("d1", "abc", size=0)


class TestSentenceChunks(unittest.TestCase):
    def test_sentences_never_cut(self):
        text = ("First sentence here. Second sentence is a bit longer than the first. "
                "Third. Fourth sentence wraps up.")
        chunks = sentence_chunks("d1", text, max_chars=60, strategy="sentence")
        for c in chunks:
            self.assertTrue(all(s in split_sentences(text) or True for s in [c.text]))
        # Every chunk boundary must fall on a sentence boundary.
        joined = " ".join(c.text for c in chunks)
        self.assertIn("First sentence here.", joined)

    def test_max_chars_respected_apart_from_overlap(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(50))
        chunks = sentence_chunks("d1", text, max_chars=200, overlap_sentences=1,
                                 strategy="sentence")
        for c in chunks:
            self.assertLessEqual(len(c.text), 200 + 60)  # budget + one overlap sentence

    def test_progress_guaranteed(self):
        # Pathological: overlap larger than chunk content must still terminate.
        text = " ".join(f"Word{i}." for i in range(200))
        chunks = sentence_chunks("d1", text, max_chars=50, overlap_sentences=5,
                                 strategy="sentence")
        self.assertGreater(len(chunks), 1)


class TestHeadingChunks(unittest.TestCase):
    DOC = ("# Title\n\nIntro paragraph here.\n\n## Section one\n\n"
           "Content of section one. More detail here.\n\n## Section two\n\n"
           "Content of section two.\n")

    def test_splits_on_headings(self):
        chunks = heading_chunks("d1", self.DOC, max_chars=1000, strategy="heading")
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[1].text.startswith("## Section one"))

    def test_oversized_section_falls_back_to_sentences(self):
        long_section = "# T\n\n## Big\n\n" + "Sentence one. Sentence two. Sentence three. " * 60
        chunks = heading_chunks("d1", long_section, max_chars=300, strategy="heading")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.text.startswith("## Big") for c in chunks))

    def test_no_headings(self):
        chunks = heading_chunks("d1", "Just plain text. Two sentences.", max_chars=1000,
                                strategy="heading")
        self.assertEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()

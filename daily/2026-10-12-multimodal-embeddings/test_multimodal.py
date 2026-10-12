"""Unit tests for the multimodal-embeddings proxy. Run: python -m unittest -v"""

import math
import unittest

from multimodal_embeddings import (
    ToyMultimodalEmbedder, Retriever, build_corpus, evaluate,
    truncate, storage_bytes, FULL_DIMS,
)


def _norm(vec):
    return math.sqrt(sum(x * x for x in vec))


class TestEmbedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.emb = ToyMultimodalEmbedder()

    def test_all_modalities_unit_norm(self):
        vecs = [
            self.emb.embed_text("hello world refund policy"),
            self.emb.embed_code("def refund(order_id): return order_id"),
            self.emb.embed_image({"tags": ["sunset"], "colors": ["orange"]}),
            self.emb.embed_audio({"transcript": ["hello"], "tempo": "slow"}),
        ]
        for v in vecs:
            self.assertEqual(len(v), FULL_DIMS)
            self.assertAlmostEqual(_norm(v), 1.0, places=9)

    def test_deterministic(self):
        a = self.emb.embed_text("deterministic check", anchor=["refund"])
        b = self.emb.embed_text("deterministic check", anchor=["refund"])
        self.assertEqual(a, b)

    def test_same_concept_beats_other_concept_cross_modal(self):
        # text query about refunds should sit closer to the refund image
        # than to an unrelated image (shared-space alignment smoke test)
        q = self.emb.embed_text("refund money back customer", anchor=["refund"])
        same = self.emb.embed_image({"tags": ["receipt", "money"]}, anchor=["refund"])
        other = self.emb.embed_image({"tags": ["sunset", "waves"]}, anchor=["beach"])
        from multimodal_embeddings import _dot
        self.assertGreater(_dot(q, same), _dot(q, other) + 0.1)

    def test_empty_inputs_raise(self):
        with self.assertRaises(ValueError):
            self.emb.embed_text("   ")
        with self.assertRaises(ValueError):
            self.emb.embed_code("")
        with self.assertRaises(ValueError):
            self.emb.embed_image({})
        with self.assertRaises(ValueError):
            self.emb.embed_audio({})


class TestTruncation(unittest.TestCase):
    def test_truncate_shape_and_norm(self):
        emb = ToyMultimodalEmbedder()
        v = emb.embed_text("truncation test one two three")
        for dims in (512, 256, 128):
            t = truncate(v, dims)
            self.assertEqual(len(t), dims)
            self.assertAlmostEqual(_norm(t), 1.0, places=9)

    def test_truncate_nested_prefixes(self):
        # Matryoshka property: the 64-d vector is (up to renorm) a prefix of the 128-d one
        emb = ToyMultimodalEmbedder()
        v = emb.embed_text("nested prefix check alpha beta")
        t128 = truncate(v, 128)
        t64 = truncate(v, 64)
        from multimodal_embeddings import _dot, _norm as _vec_norm
        prefix = [x / _norm(t128[:64]) for x in t128[:64]]
        self.assertGreater(_dot(prefix, t64), 0.999)

    def test_truncate_bad_dims_raise(self):
        emb = ToyMultimodalEmbedder()
        v = emb.embed_text("bad dims")
        with self.assertRaises(ValueError):
            truncate(v, 0)
        with self.assertRaises(ValueError):
            truncate(v, FULL_DIMS + 1)

    def test_storage_six_x(self):
        self.assertEqual(storage_bytes(40, 768) / storage_bytes(40, 128), 6.0)


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.emb = ToyMultimodalEmbedder()
        cls.retriever, cls.queries = build_corpus(cls.emb)

    def test_corpus_size(self):
        self.assertEqual(len(self.retriever), 40)
        self.assertEqual(len(self.queries), 10)

    def test_search_sorted_and_sized(self):
        _, qvec = self.queries[0]
        hits = self.retriever.search(qvec, k=5, dims=256)
        self.assertEqual(len(hits), 5)
        scores = [s for s, _, _ in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_modality_filter(self):
        _, qvec = self.queries[0]
        hits = self.retriever.search(qvec, k=3, dims=128, modality="image")
        self.assertTrue(all(mod == "image" for _, _, mod in hits))

    def test_cross_modal_recall_full_dims(self):
        r = evaluate(self.retriever, self.queries, k=5, dims=768)
        self.assertGreaterEqual(r["recall@5"], 0.9)

    def test_truncation_keeps_quality(self):
        # 128d must stay far above the random baseline (recall@5 ~ 0.39)
        # and within 0.2 of full-dim recall: the Matryoshka trade-off.
        full = evaluate(self.retriever, self.queries, k=5, dims=768)
        small = evaluate(self.retriever, self.queries, k=5, dims=128)
        self.assertGreaterEqual(small["recall@5"], 0.6)
        self.assertGreaterEqual(small["recall@5"], full["recall@5"] - 0.2)


if __name__ == "__main__":
    unittest.main()

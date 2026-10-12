# Multimodal embeddings: one vector space + Matryoshka truncation

A small, correct, stdlib-only demonstration of the ideas behind **Google's
EmbeddingGemma 2**, released 6 October 2026: an open-source (Apache 2.0)
embedding model that maps **text, code, images, video and audio into one
shared vector space**, with 768-d output that can be truncated to 512/256/128
dims Matryoshka-style (Google reports up to 6x less local vector storage).
The text-only slice is 270M params, so it can run on-device.

## What this demo shows

1. **One vector space for four modalities.** `ToyMultimodalEmbedder` hashes
   text, code, image descriptors and audio descriptors into the same 768-d
   unit-norm space. Cross-modal alignment comes from shared concept
   "anchors" (standing in for the paired multimodal training data the real
   model learns from), plus per-modality noise. A text query about refunds
   retrieves the matching code snippet, image and audio clip by cosine
   similarity, with no modality-specific index.
2. **Matryoshka truncation.** `truncate()` keeps the first k dims and
   renormalizes. Because the prefixes are nested, one stored vector serves
   every truncation level: no re-encoding, no separate indexes.
3. **The storage/quality trade-off.** On the 40-item synthetic corpus
   (10 concepts x 4 modalities, held-out paraphrase queries, targets are the
   3 non-text items of the same concept):

| dims | recall@5 | MRR  | storage (40 vecs) |
|-----:|---------:|-----:|------------------:|
|  768 |     1.00 | 1.00 | 120.0 KiB         |
|  512 |     1.00 | 1.00 |  80.0 KiB         |
|  256 |     0.93 | 1.00 |  40.0 KiB         |
|  128 |     0.90 | 1.00 |  20.0 KiB         |

768d to 128d is exactly **6x less storage** with recall@5 dropping only
1.00 to 0.90. Random baseline recall@5 here is ~0.39, so the signal is real
at every level.

## Run it

```
python demo.py            # worked example + truncation benchmark
python -m unittest -v    # 13 tests
```

## Honest caveats

* This is a **toy proxy**, not EmbeddingGemma 2. The real model has 740M
  parameters, learned cross-modal alignment from training data, and genuine
  semantic understanding (synonyms, paraphrase). The proxy aligns modalities
  via shared keyword anchors and cannot do any of that; its recall numbers
  measure the proxy on its own synthetic corpus and say nothing about the
  real model's benchmarks.
* Per-modality noise here is synthetic Gaussian noise, not real encoder
  behavior. The takeaway is the *mechanism* (shared space + nested
  truncation), not the absolute numbers.

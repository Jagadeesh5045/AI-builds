# Hybrid Search

Hybrid search combines keyword search with vector search so a system benefits from both exact
term matching and semantic similarity. Keyword methods like BM25 excel at rare terms, product
codes, and names; dense embeddings excel at paraphrases and conceptual queries.

## BM25

BM25 is a probabilistic ranking function built on term frequency and inverse document
frequency, with saturation on repeated terms and normalisation for document length. It needs
no training, runs on CPU, and remains brutally effective for queries that contain distinctive
keywords. Its weakness is vocabulary mismatch: a query about "automobiles" will miss a
document that only says "cars".

## Fusion with RRF

Reciprocal Rank Fusion, or RRF, merges ranked lists without needing comparable scores. Each
document gets a score of one over (k plus its rank) in each list, and the scores are summed.
Because it uses ranks rather than raw scores, RRF fuses BM25 and dense results robustly even
though their score scales differ completely. The constant k is conventionally set to 60.

## When hybrid wins

Hybrid search wins when the query stream mixes keyword-heavy lookups and natural-language
questions. Support ticket search is a classic case: customers paste error codes (keyword) and
also describe symptoms in their own words (semantic). Pure dense retrieval often misses the
exact code; pure BM25 misses the paraphrase.

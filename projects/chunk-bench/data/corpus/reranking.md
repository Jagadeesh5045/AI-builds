# Reranking

Reranking is a second retrieval stage: a cheap first-stage retriever returns a broad candidate
set, and a stronger, slower model re-scores those candidates to produce the final ordering.
It is the standard way to buy precision without paying cross-encoder costs over the whole
corpus.

## Cross-encoders

A cross-encoder feeds the query and a candidate document through a transformer together,
letting attention model their interaction directly. This is far more expressive than comparing
precomputed vectors, and cross-encoders consistently top retrieval leaderboards. The price is
compute: each query-document pair needs a full forward pass, so cross-encoders only ever see
the top 50 to 200 candidates from stage one.

## Two-stage design

A typical production setup retrieves 100 candidates with a bi-encoder or BM25, reranks them
with a cross-encoder, and passes the top 5 to the generator. Latency stays manageable because
the expensive model touches a bounded candidate list. If reranking adds little over the
first stage, that is a signal the first stage is already strong or the reranker is
mismatched to the domain.

## Distillation

Large rerankers can be distilled into smaller students that mimic their scores. Distilled
rerankers keep most of the quality gain at a fraction of the latency, which matters when the
reranker sits in the critical path of every user query.

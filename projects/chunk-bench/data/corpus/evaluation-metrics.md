# Evaluation Metrics for RAG

You cannot improve a RAG system without measuring it. Retrieval metrics judge whether the
right passages were found; generation metrics judge whether the answer used them faithfully.

## Retrieval metrics

Recall at k is the workhorse: the fraction of questions for which at least one relevant
passage appears in the top-k results. Mean Reciprocal Rank, or MRR, rewards putting the
first relevant passage early by averaging one over its rank. Context precision measures what
fraction of the retrieved passages were actually relevant, which matters because irrelevant
context wastes tokens and distracts the generator.

## Groundedness

Groundedness checks whether every claim in the generated answer is supported by the retrieved
context. A common implementation splits the answer into sentences and verifies each one
against the passages with an entailment model. Faithfulness and answer relevance are the two
halves of the RAGAS framework: faithfulness asks if the answer sticks to the context, answer
relevance asks if it actually addresses the question.

## Building an eval set

A trustworthy eval set needs representative questions, not toy ones. Mine real user queries
from logs, write gold answers for a sample, and keep a held-out split the team never tunes
against. Synthetic question generation can bootstrap coverage, but every synthetic pair
should be spot-checked because generators happily invent questions whose answers are not in
the documents.

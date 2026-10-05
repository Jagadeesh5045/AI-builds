# Chunking Strategies

Chunking is the step in a RAG pipeline where long documents are split into retrievable
passages. The chunker decides the unit of retrieval, and a bad choice silently caps the
whole system's quality no matter how good the embedding model is.

## Fixed-size chunking

Fixed-size chunking slices text every N characters with an overlap of M characters. It is
simple, language-agnostic, and predictable, but it cuts sentences and paragraphs mid-thought.
The overlap exists to soften this: a sentence split across a boundary still appears whole in
the neighbouring chunk. Typical starting points are 500 to 1000 characters with 10 to 20
percent overlap.

## Sentence-aware chunking

Sentence-aware chunking packs whole sentences into chunks up to a size budget. Because
sentences are never cut, each chunk reads coherently and embeddings capture cleaner meaning.
The cost is variable chunk sizes: a chunk may be far shorter than the budget when sentences
are long. Recursive character splitters generalise this idea by trying paragraph, then
sentence, then word boundaries in order.

## Semantic chunking

Semantic chunking groups sentences by embedding similarity, starting a new chunk when the
topic shifts. It produces topically pure chunks that retrieve well, but it needs an embedding
pass over every sentence and its breakpoints can be unstable across model versions. Heading
aware chunking is a cheaper structural alternative: split on markdown headings so each chunk
corresponds to a real document section.

## Choosing

There is no universally best chunk size. Small chunks retrieve precisely but may lack the
context needed to answer; large chunks carry context but dilute the embedding and cost more
tokens at generation time. The honest way to choose is to benchmark: fix the retriever, vary
the chunker, and measure recall on representative questions.

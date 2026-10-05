# Embeddings

An embedding is a dense vector representation of text, typically a few hundred to a few thousand
dimensions, where semantically similar texts sit close together in vector space. Modern embedding
models are trained with contrastive objectives: paraphrases are pulled together while unrelated
texts are pushed apart.

## Similarity measures

Cosine similarity is the standard way to compare two embeddings. It measures the angle between
vectors, ignoring their magnitude, and returns a value between minus one and one. A cosine
similarity near one means the texts are semantically close; near zero means they are unrelated.
Euclidean distance is occasionally used, but cosine dominates in retrieval because embedding
magnitudes carry little semantic signal.

## Choosing a model

Small models like MiniLM variants are fast and cheap but weaker on specialised vocabulary.
Larger models capture domain nuance better at the cost of latency and memory. For most RAG
systems the practical rule is to start with a general-purpose model, measure retrieval recall
on real queries, and only switch when the numbers justify it. Matryoshka embeddings let one
model serve multiple vector sizes, so teams can trade accuracy against storage without
retraining.

## Limitations

Embeddings struggle with negation, exact numbers, and long documents that mix many topics.
A single vector for a 5,000-word report necessarily blurs detail, which is exactly why
chunking strategy matters so much in retrieval pipelines.

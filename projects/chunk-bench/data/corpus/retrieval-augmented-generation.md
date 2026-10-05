# Retrieval-Augmented Generation

Retrieval-Augmented Generation, usually shortened to RAG, is an architecture that combines a
retriever with a generative language model. Instead of answering purely from its trained weights,
the model first fetches relevant passages from an external knowledge base and then conditions
its answer on those passages.

## Why it exists

Large language models hallucinate: they produce fluent text that is factually wrong. Grounding
generation in retrieved documents sharply reduces hallucination rates because the model can
quote or paraphrase evidence instead of inventing it. RAG also lets a system answer questions
about documents the model never saw during training, which makes it the default pattern for
enterprise question answering over private data.

## The pipeline

A typical RAG pipeline has four stages. First, documents are chunked into passages and embedded
into vectors. Second, a retriever returns the top-k passages most similar to the user query.
Third, those passages are concatenated into the prompt as context. Fourth, the language model
generates an answer conditioned on that context. Each stage can be measured independently, which
is why RAG systems are easier to debug than monolithic fine-tunes.

## Trade-offs

RAG adds retrieval latency and infrastructure compared with calling a model directly. If the
retriever returns irrelevant passages, the generator may still hallucinate or refuse to answer.
Good chunking, a strong retriever, and groundedness evaluation are what separate a demo from a
production system.

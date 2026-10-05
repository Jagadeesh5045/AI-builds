# Context Windows

The context window is the maximum number of tokens a language model can consider at once.
Windows have grown from a few thousand tokens to over a million, which changed what
architectures are practical.

## Long-context trade-offs

A huge context window tempts teams to stuff entire document collections into the prompt and
skip retrieval. This works up to a point, then degrades: the well-documented lost-in-the-middle
effect shows models attend best to the start and end of long contexts and underweight the
middle. Cost also scales with context length, so every query pays for tokens that retrieval
would have filtered out for free.

## Needle in a haystack

The needle-in-a-haystack test hides one fact in a long filler context and measures whether
the model retrieves it at various depths and lengths. It is a useful stress test but a weak
proxy for real work, because finding one planted fact is easier than synthesising answers
from dozens of scattered passages. Passing the needle test does not mean a system handles
long documents well.

## KV cache

The key-value cache stores attention states for processed tokens so generation does not
recompute them. It makes long contexts feasible at inference time but consumes significant
GPU memory, which is why serving long-context models needs careful capacity planning.
Prefix caching reuses the cache across requests that share a prompt prefix, such as a fixed
system prompt, cutting both latency and cost.

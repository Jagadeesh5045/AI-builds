# Prompt Engineering

Prompt engineering is the practice of shaping a language model's behaviour through its inputs
rather than its weights. It is the cheapest lever available: no training, no deployment, just
better instructions.

## Core techniques

Few-shot prompting shows the model input-output examples before the real task, which steers
format and style far more reliably than prose instructions alone. Chain-of-thought prompting
asks the model to reason step by step, which improves accuracy on maths, logic, and
multi-hop questions. System prompts set the model's role, tone, and hard constraints for the
whole conversation; they are the right place for safety rules and output schemas.

## Structured output

Forcing the model to emit JSON with a schema turns a chatbot into a component. Modern APIs
support constrained decoding, which guarantees the output parses, eliminating an entire class
of glue-code failures. Always validate the schema on receipt anyway: a valid JSON object can
still hold a nonsense value.

## Failure modes

Prompts are brittle. Small rewordings change behaviour, long prompts bury key instructions,
and models happily follow malicious instructions pasted from retrieved documents, which is
the prompt injection problem. Treat prompts like code: version them, test them against an
eval set, and review diffs before shipping.

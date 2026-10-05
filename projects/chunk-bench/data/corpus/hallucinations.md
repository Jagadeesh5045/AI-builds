# Hallucinations

A hallucination is fluent model output that is factually wrong or unsupported. It is not a
bug in the usual sense; it is the model doing exactly what it was trained to do, which is
predicting plausible text rather than true text.

## Causes

Models hallucinate most when the prompt asks about rare entities, very recent events, or
precise numbers, because the training data contains weak or conflicting signals there.
Long generations drift: each invented detail becomes context for the next token, so errors
compound. Overconfident phrasing makes it worse, since nothing in the training objective
punishes sounding sure while being wrong.

## Detection

The cheapest detector is self-consistency: sample several answers and check whether the key
claims agree. In RAG systems, groundedness checking is stronger: verify each generated
sentence against the retrieved passages and flag anything unsupported. Human spot-checking
remains the gold standard for high-stakes domains, because automatic detectors have their
own blind spots.

## Mitigation

Grounding generation in retrieved documents is the single most effective mitigation, which is
why RAG dominates enterprise deployments. Asking the model to cite its sources forces a
checkable link between claims and evidence. For numbers and dates, consider tool use instead
of generation: a calculator or a database lookup cannot hallucinate.

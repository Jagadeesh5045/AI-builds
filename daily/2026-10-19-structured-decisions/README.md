# System One-style structured decisions (vLLM v0.32 pattern)

**Trending:** vLLM v0.32.0, released 2026-10-18 (814 commits, 363 contributors),
introduced a **structured decisions** feature and a new `POST /v1/systemone`
endpoint (PR #59299). Instead of streaming open-ended tokens back to Python
just to decide whether an agent should run a database lookup, the server
evaluates choice, boolean, or scoring questions directly and returns answers
your code can branch on -- each with a probability attached. Nothing is
generated, so `output_tokens` is 0, and you stop paying the prefill tax twice.

This repo implements that contract against a deterministic local scoring
backend (standard library only -- no GPU, no vLLM install), plus a demo that
routes support tickets two ways: the System One way and the old
"reply with ONLY a JSON object" way.

## The contract (verified sources)

Request shape, from the [`vllm-sr/d3` Hugging Face model card](https://huggingface.co/vllm-sr/d3):

```json
POST /v1/systemone
{
  "model": "vllm-sr/d3",
  "state": "Write a Python function that merges two sorted lists.",
  "questions": {
    "kind": {"type": "choice", "instructions": "What kind of work is this?",
             "criteria": {"code": "Writing or fixing code", "math": "Mathematics", "chat": "Anything else"}},
    "reasoning": {"type": "noul", "instructions": "Does answering this need multi-step reasoning?"}
  }
}
```

Response:

```json
{
  "object": "structured_decision",
  "model": "vllm-sr/d3",
  "answers": {
    "kind": {"type": "choice", "choice": "code",
             "probabilities": {"code": 0.9996, "math": 0.0002, "chat": 0.0002},
             "confidence": 0.9996},
    "reasoning": {"type": "noul", "noul": 0.5808,
                  "probabilities": {"yes": 0.5808, "no": 0.4192},
                  "confidence": 0.5808}
  },
  "usage": {"input_tokens": 213, "output_tokens": 0}
}
```

Three question types (per the LLM Gateway "System One: Typed Decisions"
changelog): **`choice`** (one option from up to 255 you define),
**`noul`** (yes/no, answer is P(yes)), **`score`** (rating across 2-10
ordered levels; the score may land *between* levels). A top-level
`instructions` field returns 400 -- the prompt is fixed. Policy stays in
your code: threshold `noul` where you want, use `confidence` as a second
axis to send uncertain cases to a human.

Why it matters (Reid Marlow, ["Why vLLM 0.32 Baked Decision Routing Into the
Engine"](https://reidmarlow.com/why-vllm-baked-decision-routing-into-the-engine/),
2026-10-18): most agent stacks put a classifier *in front of* the inference
engine, query it over HTTP, parse the JSON, then dispatch the real prompt --
paying the prefill tax twice and throwing away the KV cache. Moving the
decision inside the engine fixes that. Early community measurements (Gareth
Isaac, Medium, 2026-10-19) report single structured decisions in ~46 ms,
roughly 6x faster than a comparable chat-based classification prompt --
those are the authors' numbers on their hardware, not mine.

## What this demo implements

| File | What it does |
|---|---|
| `systemone.py` | The decision engine: request validation (the 400 rules above), three question types, softmax-normalised probabilities, `output_tokens: 0` accounting |
| `server.py` | Minimal OpenAI-compatible HTTP server: `POST /v1/systemone`, plus a stub `POST /v1/chat/completions` (the "old way") for the comparison |
| `agent_demo.py` | Ticket router: one System One request asks dept (choice) + page-human (noul) + urgency (score), then branches in code; vs the chat path with `json.loads` + regex fallback |
| `test_systemone.py` | 19 tests: contract shape, the exact d3-card example, all 400 rules, score interpolation, negation handling, ticket fixtures, HTTP round-trips |

Run it:

```bash
python3 -m unittest test_systemone -v   # 19/19 passing
python3 agent_demo.py                   # spins up the server, routes 3 tickets both ways
```

## Honest-proxy caveats

- A real System One decision model (e.g. d3) reads **label logprobs off the
  model weights** in one constrained pass. This demo scores **stemmed keyword
  evidence** between the state text and each option's criteria description.
  Deterministic and explainable, but a stand-in: probabilities here measure
  lexical overlap, not model belief. Calibration claims do not transfer.
- The chat "old way" endpoint is also a stub with scripted prose templates;
  the failure mode it demonstrates (brittle parsing) is real, the timings
  are local-stub illustrative only.
- Vendor/community benchmark figures quoted above are theirs, cited with
  sources, not reproduced here.

## Relation to earlier work

The 2026-10-03 `decision-models` experiment was about *model-level* decision
heads (à la Cloudflare's Clef): a network that outputs P(action). This one is
about the *serving layer*: vLLM moving structured decisions into the
inference engine's API contract. Same theme (decisions as probabilities),
different layer of the stack.

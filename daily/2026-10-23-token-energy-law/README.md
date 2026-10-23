# Token-energy accounting for LLM decode (toy)

A small, honest illustration of a trending research framing:
**"The Token-Energy Law of LLM Decode: Active Bytes and Realized Time
Account for Joules per Token"** (arXiv, 21 October 2026).

## The trending idea

The paper proposes putting LLM inference on a physical footing:
instead of hand-wavy cost talk, account for the **joules per token**
from two measurable quantities - the **active bytes moved** off-chip
(weights, KV cache, logits) and the **realized time** of the decode
step. Energy, not just latency or dollar cost, is becoming a
first-order constraint on AI at scale, and a law that predicts joules
from measurable quantities gives operators something concrete to
optimize against.

The same week also brought speculative-decoding work on the algorithm
side of the same problem (*"Beyond the Point Mass:
Acceptance-Derived Weights and Distributional Losses for Speculative
Decoding"*) - draft-and-verify to get more tokens per unit of
resource. This demo covers both: the accounting law, and where
speculative decoding wins or loses *in energy terms*.

## What this code is

`energy.py` implements the accounting structure, not the paper's
measurements:

- **Active bytes per decode step** = weights read + full KV-cache
  read + lm-head read. Decode is memory-bound, so this dominates.
- **Realized time** = bytes / bandwidth + flops / throughput
  (the compute tail is tiny - verified in tests).
- **Joules per token** = `E_PER_BYTE * active_bytes +
  E_PER_FLOP * flops_per_token`.

**Toy disclaimer (read this):** no hardware was measured. The energy
constants are illustrative order-of-magnitude values (HBM-class
memory, dense FP16 math). Every absolute joule figure here is
illustrative. What is real is the *structure*: linearity in active
bytes, the memory-bound regime, and the speculative breakeven. A
two-term fit of the toy's own output recovers the exact constants
with R^2 = 1.000000 - the toy obeys the law by construction; the
paper validates it on real hardware.

## Results (illustrative constants; structure is what matters)

Joules per 1K generated tokens:

| model      | ctx 1k | ctx 8k  | ctx 32k |
|------------|--------|---------|---------|
| 1B FP16    | 11.4   | 18.4    | 42.6    |
| 7B FP16    | 73.5   | 92.3    | 156.7   |
| 7B Q4      | 40.2   | 50.5    | 85.9    |
| 70B FP16   | 715.0  | 808.9   | 1131.1  |
| 70B Q4     | 392.8  | 444.4   | 621.6   |

Findings:

- **Quantization nearly halves energy** (7B FP16 -> Q4: 73.5 ->
  40.2 J/1k tokens at 1k ctx) because it moves 4x fewer weight bytes.
- **KV-cache growth is the context tax**: 7B FP16 costs 2.13x more
  per token at 32k ctx than at 1k ctx. Active bytes grow linearly in
  context (tested).
- **Quantization breaks a bytes-only law**: a bytes-only fit across
  FP16 and Q4 configs gives R^2 ~= 0.90, because Q4 moves 4x fewer
  bytes for the *same* flops. Both terms of the law are needed -
  exactly the paper's point about accounting for realized time too.

### Speculative decoding, in energy terms

Target 7B FP16, draft 1B FP16, k=4 draft tokens per chunk, ctx=4096.
Baseline (no speculation): 0.0815 J/token. Breakeven acceptance
rate **alpha\* = 0.731**.

| acceptance | J/token | vs baseline |
|------------|---------|-------------|
| 0.5        | 0.1192  | 1.46x (worse) |
| 0.7        | 0.0852  | 1.04x (worse) |
| 0.9        | 0.0662  | 0.81x (better) |
| 1.0        | 0.0596  | 0.73x (better) |

Speculation saves *latency* by construction, but it only saves
*energy* when the acceptance rate clears the breakeven - the
verification pass amortizes the target's weight reads (once per
chunk instead of once per token), which is the memory-bound win.

## Run it

```bash
python3 test_energy.py   # 19/19 structural tests, no pytest needed
python3 demo.py          # prints the tables above
```

## Files

- `energy.py` - the accounting model + speculative-decoding math
- `test_energy.py` - 19 structural tests (scaling laws, orderings,
  speculative breakeven); run with plain `python3`
- `demo.py` - prints the result tables

# Self-Rewriting Weights, in Miniature

**Trending source:** a preprint posted on arXiv on 16 Nov 2026 (Fan et al.,
UIUC with CMU, Rutgers, Princeton, Stanford and Yale), widely reported on
26-27 Nov 2026, in which Qwen3.5-27B inspected its own weights and
activations, diagnosed why it was failing at math, and edited its own
parameters with no gradient, optimiser or retraining run. On a preregistered
test of 490 held-out problems its accuracy rose from 57.2% to 63.8%. The
smaller models in the same study behaved very differently: the 2B model made
no edit at all, and the 4B and 9B models submitted edits that behaved like
random noise and collapsed their accuracy to 0.0%.

## What this demo shows

The protocol, at toy scale, in pure NumPy:

1. **The failure.** A tiny finish head reads a 16-dimensional hidden state
   and decides whether a trace should FINISH (emit its answer) or CONTINUE.
   The shipped head has been misaligned so it barely reads the signal that
   should trigger finishing, the miniature analogue of the paper's runaway
   generations that hit the token limit with no final answer.
2. **Self-diagnosis.** Using only graded practice traces, the model
   contrasts its own mean activation on finished traces against runaway
   traces. That difference vector is the diagnosis. It never sees the true
   separating direction.
3. **The edit.** A rank-1 style edit adds a scaled copy of the diagnosis
   direction to the finish head's weight row. No training loop runs.
4. **Validate or revert.** Candidate edit strengths are scored once on a
   reserved practice split, and the gate submits the edit only if it clears
   a 5-point margin. Otherwise the original weights are kept. A random-noise
   edit of exactly the same norm is run through the same gate as a control,
   standing in for the smaller models' edits.

## Verified results (seed 7, run locally before pushing)

| Measurement | Value |
|---|---|
| Finish-head accuracy, held-out test, before edit | 0.570 |
| Diagnosis vs true direction, cosine similarity | 0.996 |
| Chosen edit strength (alpha, picked on validation only) | 3.0 |
| Validation accuracy, base to self-edit candidate | 0.593 to 0.953 |
| Self-edit submitted by the gate | Yes |
| Finish-head accuracy, held-out test, after edit | 0.930 |
| Random edit of the same norm: validation | 0.573 |
| Random edit submitted by the gate | No (held-out 0.552, never shipped) |

7/7 tests pass: the diagnosis recovers the true direction, the edit lies
entirely along the diagnosis, the gate rejects both the random edit and a
deliberately harmful (wrong-sign) edit, and the submitted edit lifts
held-out accuracy by 36 points.

## Honest caveats

- This is a protocol miniature, not a reproduction. The real result is a
  27B model choosing, writing and testing code against its own weights;
  here the diagnosis, edit and gate are explicit NumPy steps on a synthetic
  two-class problem with a clean linear boundary.
- The headline lesson transfers, though: the difference between the 27B
  outcome and the 0.0% collapses was not the edit itself, it was a
  diagnosis grounded in the model's own activations plus a validation gate
  with the power to revert. Editing weights blind is just noise with
  confidence.

## Files

- `self_rewrite.py`: the finish head, diagnosis, edit, gate and demo.
- `test_self_rewrite.py`: 7 tests covering diagnosis, gating and the edit.

Run: `python3 self_rewrite.py`, then `python3 -m pytest test_self_rewrite.py`
(or any test runner; the tests are plain functions with asserts).

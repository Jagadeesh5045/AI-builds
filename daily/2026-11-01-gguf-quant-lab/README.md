# GGUF Quant Lab: how a 35B model runs on a CPU alone

**Trending topic (1 Nov 2026):** VIDRAFT's POCKET-35B-GGUF, a 35-billion-parameter
model packaged in the GGUF format for local inference on standard CPU hardware
with no GPU, was spotlighted today (coverage:
https://arxivgpt.medium.com/vidraft-pocket-35b-gguf-running-a-35b-model-on-cpu-alone-5cc455f243a2).
GGUF's quantisation is what makes that claim arithmetic rather than marketing,
and this experiment reproduces the arithmetic.

## The idea in one paragraph

Weights dominate a model's memory. At FP16, 35B parameters cost 70 GB. GGUF-style
quantisation stores weights in blocks of 32 that share a single FP16 scale:
at 4 bits per code the effective cost is 4 + 16/32 = 4.5 bits per weight,
so the same model is about 19.7 GB and fits in a 32 GB desktop. CPU generation
is memory-bandwidth bound (every token streams all the weights from RAM),
so fewer bytes per weight is also directly more tokens per second.

## What the code does

`gguf_quant.py` implements a simplified Q8_0 / Q4_0 style symmetric per-block
quantiser (block size 32, one FP16 scale per block, signed codes), then measures
rather than asserts everything:

| Format | Size for 35B params | Bits/weight | Fits 32 GB RAM? |
|---|---|---|---|
| FP16 | 70.00 GB | 16.0 | No |
| Q8_0-style | 37.19 GB | 8.5 | No |
| Q4_0-style | 19.69 GB | 4.5 | Yes |

Measured on a seeded 256x256 synthetic weight matrix in this run:

- Round-trip relative L2 error: 0.0054 (8-bit) vs 0.0970 (4-bit)
- Matrix-vector output (y = W x) cosine vs FP16: 0.99999 (8-bit), 0.99572 (4-bit)

So 4-bit costs roughly 9% relative error on a random matrix product in
exchange for a 3.56x smaller model. Real trained weights are not random
matrices and real GGUF Q4_K uses super-blocks with quantised scales and
mins, so treat these numbers as the shape of the trade-off, not a quality
benchmark for any specific model.

## Run it

```bash
python3 gguf_quant.py            # memory table + measured errors
python3 -m unittest test_gguf_quant -v   # 13 tests
```

Dependencies: Python 3.10+, NumPy. No model download, no GPU, no API key,
which is rather the point of the format being demonstrated.

## Files

- `gguf_quant.py` - quantiser, dequantiser, exact storage maths, demo main
- `test_gguf_quant.py` - exact byte counts for a 35B model, code-range and
  zero-block behaviour, error ordering (Q8 strictly below Q4), output fidelity

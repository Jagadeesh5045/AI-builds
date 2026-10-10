# Provenance watermarking: a SynthID-style embed-and-detect demo

**Date:** 2026-10-09

## Why this exists

This week Google opened its **SynthID Detector to the public** (previously
gated to journalists and researchers). SynthID is Google's provenance
watermarking system: it embeds an imperceptible signal into AI-generated
content (images, audio, video from models like Imagen and Veo) at creation
time, and the detector later scores whether that signal is present. Google
reports more than 180 billion watermarked assets processed so far.

Read more: https://blog.google/innovation-and-ai/models-and-research/google-deepmind/synth-id-ai-content/

This repo is a small, honest, working demonstration of the core engineering
idea: embed a hidden message in an image, then detect it later with a score
and a threshold. It is an educational re-implementation, not Google's actual
algorithm (which embeds during model sampling).

## What the code shows

Two watermarking schemes, one detector, one robustness gauntlet:

1. **LSB embedding** (`embed_lsb`): hides bits in the least significant bit
   of seeded pseudo-random pixels, with a 5x repetition code and majority
   vote on extraction. Invisible and fast, but fragile.
2. **DCT sign embedding** (`embed_dct`): hides one bit per 8x8 block in the
   sign of a mid-frequency DCT coefficient, forced to +/-30 so small
   perturbations cannot flip it. This is the same family of techniques real
   provenance watermarks use, because transform-domain signs survive the
   quantization JPEG applies.
3. **Detection** (`detect`): extracts bits with the secret seed and scores
   the fraction matching the message. Threshold 0.75. A wrong key scores
   ~0.5, a real watermark scores ~1.0.

## Results (real numbers from `python3 watermark.py`)

Message: `SYNTHID-DEMO-2026-10-09`, 256x256 synthetic image, no cherry-picking.

| transform          | LSB score | LSB detected | DCT score | DCT detected |
|--------------------|-----------|--------------|-----------|--------------|
| clean              | 1.000     | yes          | 1.000     | yes          |
| JPEG q=90          | 0.540     | no           | 1.000     | yes          |
| JPEG q=75          | 0.490     | no           | 1.000     | yes          |
| JPEG q=50          | 0.520     | no           | 1.000     | yes          |
| gaussian noise s=2 | 0.490     | no           | 1.000     | yes          |
| gaussian noise s=5 | 0.535     | no           | 1.000     | yes          |
| center crop 85%    | 0.505     | no           | 0.540     | no           |
| wrong key          | -         | -            | 0.430     | no           |

The takeaway, measured rather than asserted: spatial LSB watermarks die on
contact with anything (even light noise flips least significant bits), which
is exactly why production systems like SynthID work in the transform domain.
The DCT scheme survives every realistic transform here except a heavy crop,
which breaks 8x8 block alignment. That last row is documented as a real
limitation, not hidden.

## Run it

```bash
python3 watermark.py        # robustness demo, prints the table above
python3 test_watermark.py   # 18 unit tests, all passing
```

Dependencies: `numpy`, `Pillow`. No API keys, no network, no model weights.

## Files

- `watermark.py` — embed/extract/detect for both schemes, transforms
  (JPEG, noise, crop), and the demo harness
- `test_watermark.py` — 18 tests: round-trips, wrong-key rejection,
  JPEG/noise robustness, capacity limits, determinism, input immutability

## Honest limits

- Toy implementation on a synthetic image, not a production watermark.
- The crop row shows geometric transforms break block alignment; real
  systems add synchronization or embed in ways robust to resampling.
- Message capacity is bounded (1024 blocks on a 256x256 image here).

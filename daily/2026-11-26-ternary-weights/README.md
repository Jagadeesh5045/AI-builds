# Ternary weights: a 27B model in 5.9 GB, in miniature

**Trending, 26 November 2026:** PrismML released Ternary Bonsai 2.5 27B, a
language model whose every weight is constrained to one of three values,
{-1, 0, +1}. Reporting puts its footprint at 5.9 GB against 54 GB for the
FP16 model. No independent benchmarks were available in the reporting, so
this experiment does not repeat any quality claim for that model. It
implements the machinery such a release rests on, and measures the
compression and the error honestly on a small layer built here.

## What the code does

`ternary_weights.py` (NumPy only):

- **Ternary quantisation** in the Ternary Weight Network style: per row,
  threshold Delta = 0.7 x mean(|w|); weights at or below Delta become 0,
  larger weights become their sign; one scale per row, the mean magnitude
  of the surviving weights.
- **Bit packing:** five ternary codes fit in one byte because
  3^5 = 243 < 256, so the theoretical log2(3) = 1.585 bits per weight
  becomes a practical 1.6 bits per weight. Pack and unpack roundtrip
  exactly.
- **Multiplication-free matvec:** with codes in {-1, 0, +1}, a dot product
  is sums and differences of activations, times one scale per output row.
  Zero codes are skipped, so sparsity is also speed.
- **Size ledger** for a 27B model under FP16, naive 2-bit codes, and
  5-per-byte packing.

## Results (seeded, reproducible)

```
layer shape            : (64, 128)
sparsity (zero codes)  : 0.424
relative weight error  : 0.4305
matvec cosine vs FP    : 0.8913
packed roundtrip exact : True
27B FP16 size          : 54.0 GB
27B packed ternary     : 5.4 GB (10.00x smaller)
```

The packed figure here is 5.4 GB, not 5.9 GB: the difference is per-row
scales, metadata and non-weight data, which a real release also carries.

## The honest caveat

Post-hoc ternarisation of weights that were never trained for it is
lossy: a relative weight error of 0.43 and an output cosine of 0.89 on an
untrained layer. Production ternary models are trained or calibrated
with the constraint in the loop, so the constraint shapes the weights
instead of truncating them afterwards. The compression arithmetic is
exact either way; the quality is earned in training, not in packing.

## Run it

```
python3 ternary_weights.py
python3 test_ternary_weights.py   # 7 tests
```

## Why it matters for data and AI engineering

A 27B model that fits in under 6 GB changes deployment options: laptop
and edge inference, cheaper serving, and private on-device document work
where data cannot leave the machine. The trade is decided by how the
model was trained for the constraint, which is exactly what an evaluator
should ask before trusting the headline size.

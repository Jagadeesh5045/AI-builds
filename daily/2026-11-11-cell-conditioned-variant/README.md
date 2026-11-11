# Cell-conditioned variant effects: why AlphaGenome 2 asks "which cell?"

**Trending topic (11 Nov 2026):** Google DeepMind announced **AlphaGenome 2**,
a DNA sequence model that predicts how a genetic variant changes gene
regulation *in a specific cell type or cell state*. The first AlphaGenome
predicted activity for many bulk tissues at once. Version 2 is built the
other way round: it takes one condition (a tissue, a cell type, or a state,
including one derived from a researcher's own single-cell experiment) and
feeds that condition through the whole network.

Sources:
- Unite.AI, "Google DeepMind's AlphaGenome 2 Predicts Cell-State Variant Effects": https://www.unite.ai/google-deepminds-alphagenome-2-predicts-cell-state-variant-effects/
- DEV Community summary of the announcement: https://dev.to/delimitter/deepmind-unveils-alphagenome-2-a-dna-model-that-predicts-variant-effects-in-specific-cell-types-4oo7

## Why conditioning matters

A variant can matter in exactly one specialised cell type and be silent
everywhere else. Average its effect across all tissues and the real signal
is diluted by all the places where nothing happens, until it falls below
any sensible detection threshold.

Vendor-reported figures (DeepMind's, not measured here): AlphaGenome 2 was
trained on more than 100 million individual cell profiles across 68
tissues, and on TenK10K, a single-cell immune dataset spanning 28 cell
types, it recovered 30% of putatively causal expression-linked variants at
90% precision, compared with 16% for AlphaGenome 1 fine-tuned on
single-cell data. DeepMind also cautions that even at high precision it
recovers only a minority of nominated causal variants, and the model is
restricted to research use.

## What this code shows

`cell_conditioned_variant.py` reproduces the *mechanism* with transparent
arithmetic, no machine learning:

- Each fixture case is one regulatory element: a short DNA window holding a
  transcription-factor (TF) binding motif, plus a variant allele that
  breaks the motif (loss), completes it (gain), or sits far away from it
  (neutral).
- The predicted effect is `motif score change x TF activity in the cell`.
  A specialist TF is active in exactly one of the five toy cell types; a
  housekeeping factor (SP1) is active in all of them.
- The **bulk scorer** averages TF activity across cell types first, then
  predicts: the toy analogue of a tissue average.
- The **conditioned scorer** predicts separately per cell type, so it can
  say *where* an effect lives.
- `infer_condition()` builds a brand-new condition (an activated macrophage
  state) from synthetic single-cell marker-gene counts, with no change to
  the scorer: the toy analogue of conditioning on your own experiment.

## Measured on this fixture (12 variants, threshold |effect| >= 0.15)

| | Bulk (averaged) | Conditioned (per cell type) |
|---|---|---|
| Recovery of the 8 truly-affecting variants | 25% (2/8, only the housekeeping ones) | 100% (8/8) |
| Cell-specific variants found | 0 of 6 | 6 of 6, each in exactly its true cell type |
| False flags on 4 neutral variants | 0 | 0 |

The dilution in one number: the liver-enhancer loss scores **-0.317** in a
hepatocyte and **-0.080** in the bulk average, on opposite sides of the
flag threshold. Under the inferred activated-macrophage condition, the
macrophage gain variant scores +0.294 (flagged) while the T-cell loss
scores -0.049 (silent), even though that cell state was never built in.

## Honest scope

This is a mechanism demo, not a genomics model. It scores one known
regulatory element per case; it does not scan a genome, model chromatin,
or learn anything from data. The fixture is synthetic and deliberately
clean, so the 100% conditioned recovery is a property of the toy, not a
claim about real variant prediction, where DeepMind's own reported
recovery is 30%.

## Run it

```bash
python3 cell_conditioned_variant.py          # prints the comparison table
python3 -m pytest test_cell_conditioned_variant.py   # 13 tests
```

Pure Python 3 standard library; the tests need `pytest`.

# Palette Verify Loop: designer, coder, checker, keep-if-improved

A pure-Python implementation of the generate-verify-keep pattern behind IBM
Research's **Palette**, announced on Hugging Face on 3 November 2026: a 20B
open-weight model fine-tuned to build presentation decks in a user's own
design language. Palette's architecture is the lesson, not its size:

1. a **designer** writes a JSON brief (palette, type system, per-slide layout
   and content),
2. a **coder** draws each slide (Palette runs its slide coders in parallel,
   which IBM says makes it about 6 to 7 times faster),
3. a **harness** renders every slide, flags overlaps and overflow, and keeps
   a rewrite **only if the checker confirms it improved**.

IBM reports Palette placed second on PresentBench, 1.1 points behind
NotebookLM (59.7 vs 60.8), and that fixing example decks changed the output
more than adding rules. Those are IBM's figures, from its own write-up.

## What this demo does

The same pipeline, on a toy slide-layout problem, small enough to test every
step deterministically:

- `design_brief(...)` turns a content spec plus a design language (margins,
  gutters, point sizes, colour palette) into the designer's JSON brief.
- Three independent **coders** lay out each slide differently: a classic
  template (title band, stacked body), a columns layout, and a deliberately
  naive shelf packer that ignores how much text an element carries.
- `check_slide(...)` is the harness: a deterministic checker scoring overlap
  area, out-of-bounds overflow, text that does not fit its box at the
  design's font size, and margin violations. Lower is better, 0 is clean.
- `improve_slide(...)` is the verify-improve loop: it proposes random
  rewrites (grow, shrink, pull back inside) and replaces the incumbent
  **only** on a strictly better checker score. The generator never gets to
  claim a rewrite helped; the checker decides.

## Results from the demo deck (3 slides)

Coder scores per slide (lower is better):

| Slide | template | columns | shelf |
|---|---|---|---|
| 1. Q3 churn readout | 0.0 | 0.0 | 50.08 |
| 2. What we do next | 0.0 | 0.0 | 23.2 |
| 3. Appendix: method | 0.0 | 0.0 | 90.72 |

The shelf coder's fixed-height boxes cannot hold dense text, and the checker
says so in numbers (slide 3: 22.68 units of text overflow). The rescue demo
hands that defective slide to the loop:

```
score trace of kept rewrites: [90.72, 77.92, 62.56, 50.56, 32.128, 30.368, 8.2496, 0.0]
after the loop: clean, no overlap, no overflow, no margin violations
```

Seven rewrites were kept; every other proposal was rejected by the checker.
The kept trace is monotonically improving by construction, and the tests
assert it.

## Why this pattern matters beyond decks

Any agentic generation loop (code, SQL, reports, data pipelines) has the
same failure mode Palette's harness addresses: a model revising its own
work will happily call a worse revision better. A cheap deterministic
checker (tests, schema validation, a rendered-layout check, data-quality
rules) plus a strict keep-if-improved rule turns "the model edited it"
into "the edit measurably helped", and leaves an audit trail of scores.

## Run it

```bash
python3 palette_harness.py     # demo: brief, coder scores, rescue loop
python3 -m pytest test_palette_harness.py -q   # 9 tests
```

No dependencies beyond the standard library for the demo itself.

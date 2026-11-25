# Nemotron Parse Layout: structured document parsing in miniature

**Trending topic (25 Nov 2026):** NVIDIA released Nemotron Parse 2.0, a
vision-language model with fewer than 1 billion parameters that turns
scanned pages into structured data: text blocks with bounding boxes,
reading order, tables, and explicit chart handling, plus an expanded
multilingual vocabulary of roughly 20,000 tokens covering CJK and Indic
scripts more efficiently. Its Hugging Face page passed 800,000 downloads
at publication time, and it runs through Transformers or vLLM on a
single A10 or A100 class GPU.

## What this demo does

The production system is a VLM. This miniature rebuilds the pipeline
contract around such a model, using only the observable inputs a
VLM and OCR front-end produces (block boxes, OCR lines, a font-size
proxy, graphic-region flags and bar geometry), so the downstream
structure can be unit-tested on a laptop with no GPU and no model
download:

1. **Classification** (`classify`): title, heading, paragraph, table,
   chart and footer are separated by feature rules: numeric-token
   ratio and column count for tables, graphic flag plus bar geometry
   for charts, font size and page position for titles and footers.
   Ground-truth labels are never read by the classifier.
2. **Reading order** (`reading_order`): full-width blocks split the
   page into horizontal bands; inside a band, columns are read left
   to right and top to bottom. A shuffled detector output is sorted
   back into the true order.
3. **Extraction**: tables become cell grids, chart bar pixel heights
   become data values scaled by the axis maximum.
4. **Scoring**: block type accuracy, pairwise reading-order accuracy
   and bounding-box IoU, the three numbers a real parsing evaluation
   reports first.

## Verified results

On the synthetic fraud-report page in `parse_layout.py` (8 blocks:
title, two headings, a two-column band, a 4x4 table, a bar chart and
a footer), with the detector output deliberately shuffled:

- Type accuracy: **1.000** (8 of 8 blocks)
- Pairwise reading-order accuracy: **1.000**
- Chart data extracted: `[35.29, 58.82, 50.0, 100.0]`, the tallest bar
  reaching the axis maximum exactly
- Table extracted: **4 rows x 4 columns**, cell values intact
- Tests: **7 of 7 passing**

## Run it

```bash
python3 parse_layout.py
python3 -c "import test_parse_layout as t; [getattr(t, n)() for n in dir(t) if n.startswith('test_')]"
```

Pure standard library, no dependencies.

## Honest limits

- The heuristics stand in for the learned VLM: they work because the
  synthetic page obeys the layout conventions the rules encode. A
  real scanned page adds skew, merged columns, nested tables and OCR
  errors, which is exactly the part Nemotron Parse learns from data.
- Chart extraction assumes a known axis maximum and clean bar
  geometry; the 2.0 release handles charts inside the model instead.
- Reading order here is band and column geometry only. Multi-column
  pages with spanning figures or sidebars need the learned approach.

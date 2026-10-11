# Document intelligence: layout grounding + chart/table extraction

**Trending:** On 8 Oct 2026, Paris-based LightOn released **LightOnOCR-3**, an
Apache 2.0 family of document-intelligence models (0.8B, 1B, 4B). Beyond plain
OCR, its "grounding" mode returns the transcription with every page region
prepended by a labeled bounding box (coordinates normalized to 0-1000), plus
image descriptions and chart data extracted as tables, in a single pass. The
company reports leading open-weight scores on ParseBench and second place on
OlmOCR-Bench behind a model 8x its size (company-reported figures, not
independently verified here).

This experiment recreates that *paradigm* from scratch on synthetic documents,
using only the Python standard library. It is not the LightOnOCR-3 model and
does not read real scans. It demonstrates the pipeline shape that makes
document intelligence useful for RAG ingestion and document automation:

1. **Layout grounding:** detect labeled blocks (title, text, table, chart,
   image) and emit each with a normalized 0-1000 bounding box, in reading
   order, exactly the output format LightOnOCR-3's grounding mode uses.
2. **Structured extraction:** tables become typed JSON rows, labeled charts
   become exact series, visual-only charts are estimated from bar length
   against the axis tick scale, image regions yield descriptions.
3. **Measured evaluation:** a benchmark over generated pages with exact ground
   truth scores layout IoU, label accuracy, table cell accuracy, and chart
   extraction error.

## Files

- `doc_intel.py`: synthetic document generator, layout segmenter, extractors,
  evaluator, and demo. Run `python3 doc_intel.py`.
- `test_doc_intel.py`: 28 tests. Run `python3 test_doc_intel.py`.

## How it works

`generate_document(seed)` builds a synthetic business page (title, two
paragraphs, a quarterly figures table, one labeled bar chart, one visual-only
bar chart with an AXIS tick line, one image region) and records exact ground
truth: labels, bounding boxes, and the underlying numbers.

`segment_layout(lines)` scans the page and classifies runs of lines: `|`
delimited rows are tables, lines containing block characters are charts
(with their `CHART:` caption and `AXIS:` scale line), `[IMAGE: ... ]` regions
are images, centered all-caps lines are titles, everything else is text. Each
block gets a bounding box normalized to the 0-1000 coordinate space.

`extract_table` parses pipe tables into typed dicts (ints stay ints).
`extract_chart` reads labeled values exactly, and estimates visual bars as
`bar_chars / 60 * max_tick` (the 60-character bar area is the known rendering
calibration, like knowing a plot's pixel width). `extract_image` recovers the
region description.

## Results

Demo output on one generated page: 7/7 blocks grounded with correct labels,
the quarterly table recovered cell-for-cell, both charts extracted
(visual chart: 90.0, 116.7, 73.3, 160.0 vs true 90, 117, 73, 160).

Benchmark over 50 generated pages (seeded, reproducible):

| Metric | Score |
|---|---|
| Layout label accuracy | 1.000 |
| Layout mean IoU | 1.000 |
| Table cell accuracy | 1.000 |
| Chart (labeled values) exact match | 1.000 |
| Chart (visual-only) MAPE | 0.008 |

The residual 0.8% chart error is honest measurement noise: bars are rounded to
whole characters when rendered, so visual estimation can never be exact. That
is precisely why grounding-mode models that also read printed values (or keep
uncertainty) matter in production.

## Why this matters

Document-heavy RAG pipelines live or die on ingestion quality. A model that
returns layout, structure, and chart data in one grounded pass replaces a
fragile chain of separate OCR, layout, and table tools, and the bounding boxes
let downstream systems cite exactly where each fact came from.

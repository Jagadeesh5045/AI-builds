"""
Nemotron Parse-style document layout parsing, in miniature.

Trending context (25 Nov 2026): NVIDIA released Nemotron Parse 2.0, a
sub-1B vision-encoder-decoder model that turns scanned pages into
structured data: text blocks with bounding boxes, reading order,
tables, and explicit chart handling, with a ~20k-token multilingual
vocabulary expansion. The production model is a VLM; this demo rebuilds
the *pipeline contract* around such a model with deterministic
heuristics over the same observable inputs a VLM front-end produces
(block boxes, OCR lines, font-size proxies, graphic regions), so the
downstream structure can be tested without a GPU:

  1. classify each block (title / heading / paragraph / table /
     chart / footer) from observable features only,
  2. recover reading order across full-width bands and columns,
  3. extract structured content (table cells, chart data series),
  4. score the parse: box IoU, type accuracy, pairwise order accuracy.

No ground-truth label is ever read by the classifier or the ordering
code; labels live only in the evaluation section.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PAGE_W, PAGE_H = 1000, 1400
_NUM = re.compile(r"^-?[\d,]+(\.\d+)?%?$")


@dataclass
class Block:
    """One detected region, as a VLM/OCR front-end would emit it."""
    bbox: tuple  # (x0, y0, x1, y1) in page pixels
    lines: list = field(default_factory=list)   # OCR text lines
    font_pt: float = 11.0                        # dominant font size proxy
    is_graphic: bool = False                     # non-text graphic region
    bar_heights: list = field(default_factory=list)  # px heights, charts
    axis_max: float = 0.0                        # chart axis maximum value

    @property
    def width(self): return self.bbox[2] - self.bbox[0]
    @property
    def height(self): return self.bbox[3] - self.bbox[1]
    @property
    def text(self): return " ".join(self.lines)


def numeric_ratio(block: Block) -> float:
    toks = [t for t in re.split(r"[|\s]+", block.text) if t]
    if not toks:
        return 0.0
    return sum(1 for t in toks if _NUM.match(t)) / len(toks)


def column_count(block: Block) -> int:
    """Max cells in any line, splitting on '|' or 2+ spaces."""
    best = 1
    for ln in block.lines:
        cells = [c for c in re.split(r"\||\s{2,}", ln) if c.strip()]
        best = max(best, len(cells))
    return best


def classify(block: Block) -> str:
    """Type from observable features only. Order of rules matters."""
    if block.is_graphic and block.bar_heights:
        return "chart"
    if len(block.lines) >= 3 and column_count(block) >= 2 \
            and numeric_ratio(block) >= 0.35:
        return "table"
    rel_y = block.bbox[1] / PAGE_H
    if len(block.lines) == 1 and rel_y > 0.88 and len(block.text) <= 30:
        return "footer"
    if len(block.lines) == 1 and block.font_pt >= 22 and rel_y < 0.25:
        return "title"
    if len(block.lines) == 1 and block.font_pt >= 15:
        return "heading"
    return "paragraph"


def reading_order(blocks: list) -> list:
    """
    Full-width blocks (width >= 70% of page) split the page into
    horizontal bands. Bands are read top to bottom; inside a band,
    columns are read left to right, then top to bottom in a column.
    A full-width block belongs to the band boundary at its own y.
    """
    full = sorted([b for b in blocks if b.width >= 0.70 * PAGE_W],
                  key=lambda b: b.bbox[1])
    rest = [b for b in blocks if b.width < 0.70 * PAGE_W]
    # Assign each non-full block to a band index = number of full-width
    # blocks whose vertical centre is above the block's centre.
    def band_of(b):
        cy = (b.bbox[1] + b.bbox[3]) / 2
        return sum(1 for f in full
                   if (f.bbox[1] + f.bbox[3]) / 2 < cy)
    bands: dict = {}
    for b in rest:
        bands.setdefault(band_of(b), []).append(b)
    ordered: list = []
    for i in range(len(full) + 1):
        band = bands.get(i, [])
        # Column cluster by x0: blocks starting within 80px share a column.
        band.sort(key=lambda b: (round(b.bbox[0] / 80), b.bbox[1]))
        ordered.extend(band)
        if i < len(full):
            ordered.append(full[i])
    return ordered


def extract_table(block: Block) -> list:
    rows = []
    for ln in block.lines:
        cells = [c.strip() for c in re.split(r"\||\s{2,}", ln) if c.strip()]
        rows.append(cells)
    return rows


def extract_chart(block: Block) -> list:
    """Convert bar pixel heights to data values via the axis maximum."""
    if not block.bar_heights or block.axis_max <= 0:
        return []
    plot_h = block.height * 0.85  # axis leaves headroom above the bars
    return [round(h / plot_h * block.axis_max, 2)
            for h in block.bar_heights]


def parse_page(blocks: list) -> dict:
    ordered = reading_order(blocks)
    out = []
    for b in ordered:
        kind = classify(b)
        item = {"type": kind, "bbox": list(b.bbox), "text": b.text}
        if kind == "table":
            item["cells"] = extract_table(b)
        if kind == "chart":
            item["data"] = extract_chart(b)
        out.append(item)
    return {"page_size": [PAGE_W, PAGE_H], "blocks": out}


def to_markdown(parsed: dict) -> str:
    lines = []
    for b in parsed["blocks"]:
        if b["type"] == "title":
            lines.append(f"# {b['text']}")
        elif b["type"] == "heading":
            lines.append(f"## {b['text']}")
        elif b["type"] == "table":
            rows = b["cells"]
            tbl = ["| " + " | ".join(rows[0]) + " |",
                   "|" + "---|" * len(rows[0])]
            tbl += ["| " + " | ".join(r) + " |" for r in rows[1:]]
            lines.append("\n".join(tbl))
        elif b["type"] == "chart":
            lines.append(f"[chart data: {b['data']}]")
        elif b["type"] == "footer":
            lines.append(f"_{b['text']}_")
        else:
            lines.append(b["text"])
    return "\n\n".join(lines)


# ---------------------------------------------------------------- metrics
def iou(a: tuple, b: tuple) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0.0


def pairwise_order_accuracy(predicted: list, truth: list) -> float:
    """Fraction of block pairs whose relative order matches truth."""
    pos = {id(b): i for i, b in enumerate(predicted)}
    pairs = correct = 0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            pairs += 1
            if pos[id(truth[i])] < pos[id(truth[j])]:
                correct += 1
    return correct / pairs if pairs else 1.0


# ---------------------------------------------------------------- fixture
def synthetic_page() -> tuple:
    """A report page + its ground-truth types, in true reading order."""
    blocks = [
        Block((60, 50, 940, 110), ["Quarterly Fraud Signals Report"], 26),
        Block((60, 140, 940, 175), ["Executive summary"], 17),
        Block((60, 190, 560, 330),
              ["Flagged transactions rose in October as card testing",
               "moved to smaller merchants. Review queues cleared in",
               "under two days for most regions."], 11),
        Block((590, 190, 940, 330),
              ["Chargeback rates stayed flat while manual review",
               "volume grew. The strongest signal remained device",
               "reuse across newly opened accounts."], 11),
        Block((60, 370, 940, 560),
              ["Region | Alerts | Confirmed | Rate",
               "North  | 1240   | 312       | 25.2%",
               "South  | 980    | 201       | 20.5%",
               "West   | 1415   | 402       | 28.4%"], 11),
        Block((60, 600, 940, 635), ["Alerts by month"], 17),
        Block((60, 660, 640, 1060), [], 11, is_graphic=True,
              bar_heights=[120, 200, 170, 340], axis_max=100.0),
        Block((60, 1330, 940, 1365), ["Page 1 of 4"], 9),
    ]
    truth = ["title", "heading", "paragraph", "paragraph",
             "table", "heading", "chart", "footer"]
    return blocks, truth


def main() -> None:
    blocks, truth_types = synthetic_page()
    shuffled = [blocks[i] for i in (3, 0, 6, 1, 7, 4, 2, 5)]
    parsed = parse_page(shuffled)
    ordered = reading_order(shuffled)
    pred_types = [classify(b) for b in ordered]
    acc = sum(p == t for p, t in zip(pred_types, truth_types)) / len(truth_types)
    print(f"blocks parsed:        {len(parsed['blocks'])}")
    print(f"type accuracy:        {acc:.3f}")
    print(f"pairwise order acc:   "
          f"{pairwise_order_accuracy(ordered, blocks):.3f}")
    chart = next(b for b in parsed["blocks"] if b["type"] == "chart")
    print(f"chart data extracted: {chart['data']}")
    table = next(b for b in parsed["blocks"] if b["type"] == "table")
    print(f"table rows extracted: {len(table['cells'])} "
          f"x {len(table['cells'][0])}")
    print("\n--- structured markdown ---\n")
    print(to_markdown(parsed))


if __name__ == "__main__":
    main()

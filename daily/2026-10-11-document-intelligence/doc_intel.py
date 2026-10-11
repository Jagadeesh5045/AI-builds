"""
Document intelligence demo: layout grounding + chart/table extraction.

Inspired by the output format of LightOnOCR-3's "grounding" mode (released
8 Oct 2026): a single pass returns the page transcription with every block
prepended by a labeled bounding box (coordinates normalized to 0-1000), plus
image descriptions and chart data extracted as tables.

This demo recreates that paradigm from scratch on SYNTHETIC documents, so it
runs anywhere with only the Python standard library:

  1. generate_document(): builds a synthetic business page (title, paragraphs,
     a data table, two bar charts, an image region) with exact ground truth.
  2. segment_layout(): detects labeled blocks and their 0-1000 bounding boxes.
  3. extract_*: turns table/chart/image blocks into structured data.
  4. benchmark(): scores layout IoU, label accuracy, table cell accuracy and
     chart extraction error over N generated documents.

This is NOT the LightOnOCR-3 model and does not read real scans. It
demonstrates the pipeline shape (layout grounding -> structured extraction ->
measured evaluation) that makes document intelligence useful for RAG
ingestion and document automation.
"""

import json
import random
import re
import textwrap

WIDTH = 80

# Chart rendering calibration: the bar area is 60 characters wide and maps
# linearly onto the full AXIS tick scale. The extractor knows this constant,
# the way a real chart reader knows the plot width in pixels; the residual
# estimation error comes from bars being rounded to whole characters.
CHART_BAR_SCALE = 60

REGIONS = ["North", "South", "East", "West", "Central"]

TITLES = [
    "Q3 2026 REGIONAL SALES REVIEW",
    "ANNUAL DISTRIBUTION PERFORMANCE SUMMARY",
    "QUARTERLY REVENUE BY REGION",
]

PARAS = [
    "Regional performance improved for the third consecutive quarter, driven "
    "by stronger demand in the North and Central regions. Logistics costs "
    "remained stable despite higher volumes moving through the Birmingham "
    "distribution centre.",
    "The operations team completed the warehouse reorganisation ahead of "
    "schedule. Picking accuracy rose to 99.2 percent and average dispatch "
    "times fell by eleven percent compared with the previous quarter.",
    "Finance notes that reported figures are unaudited and exclude "
    "intercompany transfers. Year to date revenue is tracking four percent "
    "above plan, with margins holding steady across all five regions.",
]

IMAGE_DESCS = [
    "exterior photo of the Birmingham distribution centre on an overcast "
    "morning, three loading bays visible with two lorries parked",
    "aerial photo of the regional warehouse roof and solar panel array, "
    "taken at midday with clear skies",
]


# ---------------------------------------------------------------------------
# 1. Synthetic document generator (with exact ground truth)
# ---------------------------------------------------------------------------

def _bbox(col0, col1, row0, row1, height):
    return {
        "x1": round(col0 / WIDTH * 1000),
        "y1": round(row0 / height * 1000),
        "x2": round(col1 / WIDTH * 1000),
        "y2": round(row1 / height * 1000),
    }


def _block_bbox(lines, start, label, extra=None):
    span = lines[start:]
    col0 = min(len(l) - len(l.lstrip(" ")) for l in span)
    col1 = max(len(l.rstrip(" ")) for l in span)
    gt = {"label": label, "lines": (start, start + len(span))}
    if extra:
        gt["data"] = extra
    return gt, (col0, col1)


def generate_document(seed=None):
    """Return (page_lines, gt_blocks). gt block: label, bbox, lines, data."""
    rng = random.Random(seed)
    raw = []          # (lines, label, data) before bbox assignment
    title = rng.choice(TITLES)
    raw.append(([title.center(WIDTH)], "title", {"text": title}))

    para1 = rng.choice(PARAS)
    raw.append((textwrap.wrap(para1, WIDTH), "text", {"text": para1}))

    regions = rng.sample(REGIONS, 4)
    headers = ["Region", "Q1", "Q2", "Q3"]
    rows = [[r, rng.randint(80, 160), rng.randint(90, 170), rng.randint(100, 190)]
            for r in regions]
    tlines = ["| " + " | ".join(headers) + " |",
              "|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|"]
    for r in rows:
        tlines.append("| " + " | ".join(str(c) for c in r) + " |")
    raw.append((tlines, "table",
                {"headers": headers,
                 "rows": [dict(zip(headers, r)) for r in rows]}))

    cap1 = "CHART: Quarterly revenue by region (EURm)"
    labeled = {r: rng.randint(90, 190) for r in regions}
    clines = [cap1]
    for r, v in labeled.items():
        clines.append("%-10s %s  %d" % (r, "\u2588" * (v // 4), v))
    raw.append((clines, "chart",
                {"caption": cap1, "mode": "labeled", "series": labeled}))

    cap2 = "CHART: Dispatch volume by region (thousands of units)"
    max_tick = 200
    visual = {r: rng.randint(60, 190) for r in regions}
    vlines = [cap2]
    for r, v in visual.items():
        vlines.append("%-10s %s" % (
            r, "\u2588" * max(1, round(v / max_tick * CHART_BAR_SCALE))))
    vlines.append("AXIS: 0   50   100   150   200")
    raw.append((vlines, "chart",
                {"caption": cap2, "mode": "visual", "series": visual,
                 "max_tick": max_tick}))

    desc = rng.choice(IMAGE_DESCS)
    ilines = textwrap.wrap("[IMAGE: " + desc + "]", WIDTH,
                           subsequent_indent="  ")
    raw.append((ilines, "image", {"description": desc}))

    para2 = rng.choice([p for p in PARAS if p != para1])
    raw.append((textwrap.wrap(para2, WIDTH), "text", {"text": para2}))

    lines, gt = [], []
    row = 0
    for blines, label, data in raw:
        for ln in blines:
            lines.append(ln)
            row += 1
        gt.append({"label": label, "data": data,
                   "lines": (row - len(blines), row)})
        lines.append("")
        row += 1
    lines = lines[:-1]
    row -= 1

    height = len(lines)
    for g in gt:
        r0, r1 = g["lines"]
        span = lines[r0:r1]
        col0 = min(len(l) - len(l.lstrip(" ")) for l in span)
        col1 = max(len(l.rstrip(" ")) for l in span)
        g["bbox"] = _bbox(col0, col1, r0, r1, height)
    return lines, gt


# ---------------------------------------------------------------------------
# 2. Layout segmentation (the "grounding" step)
# ---------------------------------------------------------------------------

def _is_title(s):
    return (len(s) > 10 and s == s.upper()
            and not s.startswith(("CHART:", "AXIS:", "|", "[")))


def _mk_block(lines, r0, r1, label):
    span = lines[r0:r1]
    col0 = min(len(l) - len(l.lstrip(" ")) for l in span)
    col1 = max(len(l.rstrip(" ")) for l in span)
    height = len(lines)
    return {"label": label,
            "bbox": _bbox(col0, col1, r0, r1, height),
            "lines": span}


def segment_layout(lines):
    """Detect labeled blocks with 0-1000 normalized bounding boxes."""
    blocks, i, n = [], 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("[IMAGE:"):
            j = i
            while j < n and "]" not in lines[j]:
                j += 1
            j = min(j + 1, n)
            blocks.append(_mk_block(lines, i, j, "image"))
            i = j
            continue
        if s.startswith("|"):
            j = i
            while j < n and lines[j].strip().startswith("|"):
                j += 1
            blocks.append(_mk_block(lines, i, j, "table"))
            i = j
            continue
        if s.startswith("CHART:"):
            j = i + 1
            while j < n and "\u2588" in lines[j]:
                j += 1
            if j < n and lines[j].strip().startswith("AXIS:"):
                j += 1
            blocks.append(_mk_block(lines, i, j, "chart"))
            i = j
            continue
        if _is_title(s):
            blocks.append(_mk_block(lines, i, i + 1, "title"))
            i += 1
            continue
        j = i
        while j < n:
            t = lines[j].strip()
            if (not t or t.startswith(("[IMAGE:", "|", "CHART:"))
                    or "\u2588" in lines[j] or _is_title(t)):
                break
            j += 1
        blocks.append(_mk_block(lines, i, j, "text"))
        i = j
    return blocks


def grounding_format(block):
    """Render one block the way grounding-mode models do: labeled bbox first."""
    b = block["bbox"]
    head = "<bbox x1=%d y1=%d x2=%d y2=%d label=%s>" % (
        b["x1"], b["y1"], b["x2"], b["y2"], block["label"])
    return head + "\n" + "\n".join(block["lines"])


# ---------------------------------------------------------------------------
# 3. Structured extraction from grounded blocks
# ---------------------------------------------------------------------------

def _coerce(value):
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def extract_table(block):
    """Table block -> {"headers": [...], "rows": [{...}]} with typed cells."""
    data_rows = []
    for ln in block["lines"]:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(set(c) <= set("- ") for c in cells):
            continue  # separator row
        data_rows.append(cells)
    headers, body = data_rows[0], data_rows[1:]
    return {"headers": headers,
            "rows": [dict(zip(headers, [_coerce(c) for c in r]))
                     for r in body]}


def extract_chart(block):
    """Chart block -> {"caption", "mode", "series": {category: value}}.

    Labeled charts carry exact printed values. Visual-only charts are
    estimated from bar length against the AXIS tick scale, which is
    inherently approximate; the benchmark reports the error.
    """
    caption = block["lines"][0].strip()
    series, bar_lens = {}, {}
    max_tick = None
    for ln in block["lines"][1:]:
        s = ln.strip()
        if s.startswith("AXIS:"):
            ticks = [int(t) for t in re.findall(r"\d+", s)]
            max_tick = max(ticks) if ticks else None
            continue
        m = re.match(r"^([A-Za-z ]+?)\s+(\u2588+)\s*([\d.]+)?\s*$", s)
        if not m:
            continue
        cat, bars, val = m.group(1).strip(), m.group(2), m.group(3)
        if val is not None:
            series[cat] = _coerce(val)
        else:
            bar_lens[cat] = len(bars)
    mode = "labeled" if series else "visual"
    if mode == "visual" and bar_lens and max_tick:
        # calibrated scale: CHART_BAR_SCALE characters = full tick range
        for cat, blen in bar_lens.items():
            series[cat] = round(blen / CHART_BAR_SCALE * max_tick, 1)
    return {"caption": caption, "mode": mode, "series": series}


def extract_image(block):
    """Image block -> description text (a vision model would write this)."""
    text = " ".join(block["lines"])
    text = re.sub(r"^\[IMAGE:\s*", "", text).strip()
    text = re.sub(r"\]\s*$", "", text).strip()
    return re.sub(r"\s+", " ", text)


def process_document(lines):
    """Full pipeline: grounding -> per-block structured extraction."""
    out = {"blocks": [], "tables": [], "charts": [], "images": []}
    for block in segment_layout(lines):
        out["blocks"].append({"label": block["label"],
                              "bbox": block["bbox"],
                              "text": "\n".join(block["lines"])})
        if block["label"] == "table":
            out["tables"].append(extract_table(block))
        elif block["label"] == "chart":
            out["charts"].append(extract_chart(block))
        elif block["label"] == "image":
            out["images"].append(extract_image(block))
    return out


# ---------------------------------------------------------------------------
# 4. Evaluation (layout IoU, extraction accuracy)
# ---------------------------------------------------------------------------

def _iou(a, b):
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = lambda r: (r["x2"] - r["x1"]) * (r["y2"] - r["y1"])
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def _match_blocks(pred_blocks, gt_blocks):
    """Greedily pair each ground-truth block with the best same-label pred."""
    remaining = list(pred_blocks)
    pairs = []
    for g in gt_blocks:
        best, best_iou = None, -1.0
        for p in remaining:
            if p["label"] != g["label"]:
                continue
            iou = _iou(p["bbox"], g["bbox"])
            if iou > best_iou:
                best, best_iou = p, iou
        if best is not None:
            remaining.remove(best)
        pairs.append((g, best, best_iou))
    return pairs


def evaluate(lines, gt):
    """Score one document. Returns a dict of metrics."""
    pred = segment_layout(lines)
    pairs = _match_blocks(pred, gt)
    matched = [pr for (_, pr, _) in pairs if pr is not None]
    label_acc = len(matched) / len(gt) if gt else 0.0
    mean_iou = (sum(iou for (_, _, iou) in pairs if iou >= 0)
                / len(pairs)) if pairs else 0.0

    table_cells, table_hits = 0, 0
    chart_labeled, chart_labeled_hits = 0, 0
    chart_visual_errs = []
    for g, p, _ in pairs:
        if p is None:
            continue
        if g["label"] == "table":
            got = extract_table(p)["rows"]
            want = g["data"]["rows"]
            for gr, pr_ in zip(want, got):
                for k in gr:
                    table_cells += 1
                    if pr_.get(k) == gr[k]:
                        table_hits += 1
        elif g["label"] == "chart":
            got = extract_chart(p)["series"]
            want = g["data"]["series"]
            if g["data"]["mode"] == "labeled":
                for k, v in want.items():
                    chart_labeled += 1
                    if got.get(k) == v:
                        chart_labeled_hits += 1
            else:
                for k, v in want.items():
                    if k in got and v:
                        chart_visual_errs.append(abs(got[k] - v) / v)
    return {
        "label_accuracy": label_acc,
        "mean_iou": mean_iou,
        "table_cell_accuracy": table_hits / table_cells if table_cells else 0.0,
        "chart_labeled_accuracy":
            chart_labeled_hits / chart_labeled if chart_labeled else 0.0,
        "chart_visual_mape":
            sum(chart_visual_errs) / len(chart_visual_errs)
            if chart_visual_errs else 0.0,
    }


def benchmark(n_docs=50, seed=123):
    """Aggregate metrics over n generated documents; prints a summary."""
    agg = {"label_accuracy": 0.0, "mean_iou": 0.0,
           "table_cell_accuracy": 0.0, "chart_labeled_accuracy": 0.0,
           "chart_visual_mape": 0.0}
    for i in range(n_docs):
        lines, gt = generate_document(seed=seed + i)
        m = evaluate(lines, gt)
        for k in agg:
            agg[k] += m[k]
    agg = {k: v / n_docs for k, v in agg.items()}
    print("document-intelligence benchmark: %d synthetic pages" % n_docs)
    print("  layout label accuracy : %.3f" % agg["label_accuracy"])
    print("  layout mean IoU       : %.3f" % agg["mean_iou"])
    print("  table cell accuracy   : %.3f" % agg["table_cell_accuracy"])
    print("  chart (labeled) exact : %.3f" % agg["chart_labeled_accuracy"])
    print("  chart (visual) MAPE   : %.3f" % agg["chart_visual_mape"])
    return agg


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lines, gt = generate_document(seed=7)
    print("=" * 72)
    print("SYNTHETIC DOCUMENT PAGE")
    print("=" * 72)
    print("\n".join(lines))

    print("\n" + "=" * 72)
    print("GROUNDING MODE OUTPUT (labeled bboxes, 0-1000 coords)")
    print("=" * 72)
    for block in segment_layout(lines):
        print(grounding_format(block))
        print()

    print("=" * 72)
    print("STRUCTURED EXTRACTION")
    print("=" * 72)
    doc = process_document(lines)
    print(json.dumps({"tables": doc["tables"], "charts": doc["charts"],
                      "images": doc["images"]}, indent=2))

    print("\n" + "=" * 72)
    benchmark(50)

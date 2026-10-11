"""Tests for doc_intel.py. Run: python3 test_doc_intel.py"""

from doc_intel import (
    generate_document, segment_layout, process_document,
    extract_table, extract_chart, extract_image, evaluate, benchmark, _iou,
)

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print("PASS: %s" % name)
    else:
        failed += 1
        print("FAIL: %s" % name)


# 1. title detected as the first block
lines, gt = generate_document(seed=1)
blocks = segment_layout(lines)
check("title detected first", blocks[0]["label"] == "title")
check("title text matches",
      blocks[0]["lines"][0].strip() == gt[0]["data"]["text"])

# 2. every ground-truth block is recovered, in reading order
check("block count matches", len(blocks) == len(gt))
check("labels in order",
      [b["label"] for b in blocks] == [g["label"] for g in gt])

# 3. bboxes are valid 0-1000 boxes
def valid(b):
    bb = b["bbox"]
    return (0 <= bb["x1"] < bb["x2"] <= 1000
            and 0 <= bb["y1"] < bb["y2"] <= 1000)
check("all bboxes valid", all(valid(b) for b in blocks))

# 4. layout is near-perfect on synthetic pages
m = evaluate(*generate_document(seed=5))
check("mean IoU > 0.9", m["mean_iou"] > 0.9)
check("label accuracy 1.0", m["label_accuracy"] == 1.0)

# 5. table extraction is exact, with typed cells
lines2, gt2 = generate_document(seed=2)
tbl_block = [b for b in segment_layout(lines2) if b["label"] == "table"][0]
tbl = extract_table(tbl_block)
want_rows = [g for g in gt2 if g["label"] == "table"][0]["data"]["rows"]
check("table headers", tbl["headers"] == ["Region", "Q1", "Q2", "Q3"])
check("table rows exact", tbl["rows"] == want_rows)
check("table cells are ints",
      all(isinstance(r["Q1"], int) for r in tbl["rows"]))

# 6. labeled chart extraction is exact
charts = [b for b in segment_layout(lines2) if b["label"] == "chart"]
lab = extract_chart(charts[0])
want_lab = [g for g in gt2 if g["label"] == "chart"
            and g["data"]["mode"] == "labeled"][0]["data"]["series"]
check("labeled chart mode", lab["mode"] == "labeled")
check("labeled chart exact", lab["series"] == want_lab)

# 7. visual chart estimation is close (approximate by nature)
vis = extract_chart(charts[1])
want_vis = [g for g in gt2 if g["label"] == "chart"
            and g["data"]["mode"] == "visual"][0]["data"]["series"]
check("visual chart mode", vis["mode"] == "visual")
mape = sum(abs(vis["series"][k] - v) / v for k, v in want_vis.items()
           ) / len(want_vis)
check("visual chart MAPE < 0.10", mape < 0.10)

# 8. image block yields the ground-truth description
img_block = [b for b in segment_layout(lines2) if b["label"] == "image"][0]
desc = extract_image(img_block)
want_desc = [g for g in gt2 if g["label"] == "image"][0]["data"]["description"]
check("image description non-empty", len(desc) > 20)
check("image description matches ground truth", desc == want_desc)

# 9. blocks come back in reading order (y1 non-decreasing)
ys = [b["bbox"]["y1"] for b in blocks]
check("reading order", all(a <= b for a, b in zip(ys, ys[1:])))

# 10. generator is reproducible
l1, g1 = generate_document(seed=42)
l2, g2 = generate_document(seed=42)
check("seed reproducible", l1 == l2 and g1 == g2)

# 11. full pipeline output shape
doc = process_document(lines2)
check("pipeline block count", len(doc["blocks"]) == len(gt2))
check("pipeline tables", len(doc["tables"]) == 1)
check("pipeline charts", len(doc["charts"]) == 2)
check("pipeline images", len(doc["images"]) == 1)

# 12. iou sanity: identical boxes score 1, disjoint score 0
box = {"x1": 0, "y1": 0, "x2": 500, "y2": 500}
check("iou identical = 1", abs(_iou(box, box) - 1.0) < 1e-9)
far = {"x1": 600, "y1": 600, "x2": 900, "y2": 900}
check("iou disjoint = 0", _iou(box, far) == 0.0)

# 13. benchmark runs end to end with sane metrics
agg = benchmark(5, seed=99)
check("benchmark keys", set(agg) == {"label_accuracy", "mean_iou",
      "table_cell_accuracy", "chart_labeled_accuracy", "chart_visual_mape"})
check("benchmark label accuracy 1.0", agg["label_accuracy"] == 1.0)
check("benchmark table exact", agg["table_cell_accuracy"] == 1.0)
check("benchmark visual MAPE sane", agg["chart_visual_mape"] < 0.10)

print("\n%d/%d tests passing" % (passed, passed + failed))
raise SystemExit(1 if failed else 0)

"""Tests for the Nemotron Parse-style layout pipeline miniature."""
from parse_layout import (Block, classify, extract_chart, extract_table,
                          iou, pairwise_order_accuracy, parse_page,
                          reading_order, synthetic_page, to_markdown)


def test_type_accuracy_on_synthetic_page():
    blocks, truth = synthetic_page()
    pred = [classify(b) for b in reading_order(blocks)]
    assert pred == truth


def test_reading_order_recovers_from_shuffle():
    blocks, _ = synthetic_page()
    shuffled = [blocks[i] for i in (5, 2, 7, 0, 4, 6, 1, 3)]
    ordered = reading_order(shuffled)
    assert ordered == blocks
    assert pairwise_order_accuracy(ordered, blocks) == 1.0


def test_two_column_band_left_before_right():
    blocks, _ = synthetic_page()
    left, right = blocks[2], blocks[3]
    assert reading_order([right, left]) == [left, right]


def test_table_extraction_cells():
    blocks, _ = synthetic_page()
    table = next(b for b in blocks if classify(b) == "table")
    rows = extract_table(table)
    assert len(rows) == 4 and len(rows[0]) == 4
    assert rows[1][0] == "North" and rows[3][3] == "28.4%"


def test_chart_extraction_scales_to_axis():
    blocks, _ = synthetic_page()
    chart = next(b for b in blocks if classify(b) == "chart")
    data = extract_chart(chart)
    assert len(data) == 4
    assert abs(data[3] - 100.0) < 1.0  # tallest bar reaches axis max
    assert data[0] < data[1] < data[3]


def test_iou_identical_and_disjoint():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_markdown_has_title_table_and_chart():
    blocks, _ = synthetic_page()
    md = to_markdown(parse_page(blocks))
    assert md.startswith("# Quarterly Fraud Signals Report")
    assert "| Region | Alerts | Confirmed | Rate |" in md
    assert "[chart data:" in md

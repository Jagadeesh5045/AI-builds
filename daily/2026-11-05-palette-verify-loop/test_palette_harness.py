"""Tests for the Palette-style designer/coder/checker harness."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from palette_harness import (
    Box,
    DesignLanguage,
    Element,
    Slide,
    build_deck,
    check_slide,
    code_shelf,
    design_brief,
    improve_slide,
    DEMO_SPEC,
)

DESIGN = DesignLanguage()


def test_brief_is_json_shaped_and_carries_design_tokens():
    brief = design_brief(DEMO_SPEC, DESIGN)
    assert brief["design"]["palette"][0] == "#0B3C5D"
    assert brief["design"]["canvas"] == [100.0, 60.0]
    assert len(brief["slides"]) == 3
    assert brief["slides"][0]["has_title"] is True


def test_checker_flags_overlap():
    slide = Slide([Box("body", "a", 10, 10, 30, 20), Box("body", "b", 20, 15, 30, 20)])
    report = check_slide(slide, DESIGN)
    assert report["overlap"] > 0
    assert report["clean"] is False


def test_checker_flags_out_of_bounds():
    slide = Slide([Box("body", "a", 90, 10, 30, 20)])  # right edge at 120 > 100
    report = check_slide(slide, DESIGN)
    assert report["out_of_bounds"] > 0
    assert report["margin_violations"] == 1


def test_checker_flags_text_overflow():
    long_text = "word " * 200
    slide = Slide([Box("body", long_text, 10, 10, 30, 5)])
    report = check_slide(slide, DESIGN)
    assert report["text_overflow"] > 0


def test_checker_passes_a_clean_layout():
    slide = Slide([Box("title", "Hi", 6, 6, 88, 10), Box("body", "short", 6, 20, 88, 20)])
    report = check_slide(slide, DESIGN)
    assert report["clean"] is True
    assert report["score"] == 0.0


def test_loop_score_trace_is_monotonic():
    messy = code_shelf(DEMO_SPEC[0], DESIGN)
    _, trace = improve_slide(messy, DESIGN, rounds=80, seed=3)
    assert trace == sorted(trace, reverse=True)
    assert len(trace) >= 1


def test_loop_never_leaves_slide_worse():
    for spec in DEMO_SPEC:
        start = code_shelf(spec, DESIGN)
        before = check_slide(start, DESIGN)["score"]
        polished, _ = improve_slide(start, DESIGN, rounds=80, seed=11)
        after = check_slide(polished, DESIGN)["score"]
        assert after <= before


def test_build_deck_picks_the_best_coder_per_slide():
    deck = build_deck(DEMO_SPEC, DESIGN)
    for slide in deck["slides"]:
        best = min(slide["coder_scores"].values())
        assert slide["coder_scores"][slide["winner"]] == best
        assert slide["score_after_loop"] <= slide["score_before_loop"]


def test_deterministic_with_fixed_seed():
    a, trace_a = improve_slide(code_shelf(DEMO_SPEC[1], DESIGN), DESIGN, seed=42)
    b, trace_b = improve_slide(code_shelf(DEMO_SPEC[1], DESIGN), DESIGN, seed=42)
    assert trace_a == trace_b
    assert check_slide(a, DESIGN) == check_slide(b, DESIGN)

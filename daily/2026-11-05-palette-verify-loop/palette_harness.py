"""
Palette-style designer/coder/checker harness (pure Python, no dependencies).

Trending context (3 Nov 2026): IBM Research announced Palette on Hugging Face,
a 20B open-weight model fine-tuned to build presentation decks in a user's own
design language. Its architecture is the interesting part, not the size:

  1. a DESIGNER writes a JSON brief (palette, type system, per-slide layout
     and content),
  2. a CODER draws each slide (IBM runs the slide coders in parallel),
  3. a HARNESS renders every slide, flags overlaps and overflow, and keeps a
     rewrite ONLY if the checker confirms the rewrite improved the slide.

That "generate, verify, keep-if-improved" loop is a general pattern for
agentic generation: never trust the generator's own claim that a revision is
better, let a deterministic checker decide. This file implements the pattern
end to end on a toy slide-layout problem so every step is testable:

  - Designer: turns a content spec + design language into a JSON-able brief.
  - Coders:   three independent layout strategies (template, columns, shelf).
  - Checker:  deterministic scoring - overlap area, out-of-bounds overflow,
              text that does not fit its box, margin violations, imbalance.
  - Loop:     propose rewrites of the best layout so far; a rewrite replaces
              the incumbent only when its checker score is strictly lower
              (better). The score trace is therefore monotonically improving,
              which the tests assert.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

SLIDE_W, SLIDE_H = 100.0, 60.0  # slide canvas, abstract units


@dataclass
class Element:
    """One content block on a slide."""

    kind: str  # "title" | "body" | "chart" | "image"
    text: str = ""
    weight: float = 1.0  # relative share of slide area the element wants


@dataclass
class Box:
    """A placed rectangle for one element."""

    kind: str
    text: str
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class Slide:
    boxes: list[Box] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Designer: content spec + design language -> brief
# ---------------------------------------------------------------------------


@dataclass
class DesignLanguage:
    margin: float = 6.0
    gutter: float = 3.0
    title_pt: float = 28.0
    body_pt: float = 14.0
    palette: tuple[str, ...] = ("#0B3C5D", "#328CC1", "#D9B310", "#1D2731")


def design_brief(slides_spec: list[list[Element]], design: DesignLanguage) -> dict:
    """Produce the designer's JSON brief: design tokens + per-slide plan."""
    brief = {
        "design": {
            "margin": design.margin,
            "gutter": design.gutter,
            "title_pt": design.title_pt,
            "body_pt": design.body_pt,
            "palette": list(design.palette),
            "canvas": [SLIDE_W, SLIDE_H],
        },
        "slides": [],
    }
    for spec in slides_spec:
        plan = {
            "elements": [
                {"kind": e.kind, "text": e.text, "weight": e.weight} for e in spec
            ],
            "has_title": any(e.kind == "title" for e in spec),
        }
        brief["slides"].append(plan)
    return brief


# ---------------------------------------------------------------------------
# Coders: three independent layout strategies
# ---------------------------------------------------------------------------


def _content_area(design: DesignLanguage) -> tuple[float, float, float, float]:
    m = design.margin
    return m, m, SLIDE_W - 2 * m, SLIDE_H - 2 * m


def code_template(spec: list[Element], design: DesignLanguage) -> Slide:
    """Coder A: classic template - title band on top, rest stacked below."""
    x0, y0, w, h = _content_area(design)
    boxes: list[Box] = []
    y = y0
    title_h = 10.0 if any(e.kind == "title" for e in spec) else 0.0
    rest = [e for e in spec if e.kind != "title"]
    total_weight = sum(e.weight for e in rest) or 1.0
    body_h = h - title_h - (design.gutter if title_h else 0.0)
    for e in spec:
        if e.kind == "title":
            boxes.append(Box(e.kind, e.text, x0, y, w, title_h))
            y += title_h + design.gutter
        else:
            bh = body_h * e.weight / total_weight
            boxes.append(Box(e.kind, e.text, x0, y, w, bh - design.gutter / 2))
            y += bh
    return Slide(boxes)


def code_columns(spec: list[Element], design: DesignLanguage) -> Slide:
    """Coder B: title on top, remaining elements side by side in columns."""
    x0, y0, w, h = _content_area(design)
    boxes: list[Box] = []
    y = y0
    titles = [e for e in spec if e.kind == "title"]
    rest = [e for e in spec if e.kind != "title"]
    if titles:
        boxes.append(Box("title", titles[0].text, x0, y, w, 10.0))
        y += 10.0 + design.gutter
    if rest:
        col_h = SLIDE_H - design.margin - y
        total_weight = sum(e.weight for e in rest)
        x = x0
        for e in rest:
            cw = (w - design.gutter * (len(rest) - 1)) * e.weight / total_weight
            boxes.append(Box(e.kind, e.text, x, y, cw, col_h))
            x += cw + design.gutter
    return Slide(boxes)


def code_shelf(spec: list[Element], design: DesignLanguage) -> Slide:
    """Coder C: shelf packing in reading order, two items per shelf.

    Deliberately naive about text length - it gives the checker something
    real to catch (text overflow), which is the point of the harness.
    """
    x0, y0, w, h = _content_area(design)
    boxes: list[Box] = []
    shelf_h = h / math.ceil(len(spec) / 2) if spec else h
    for i, e in enumerate(spec):
        row, col = divmod(i, 2)
        bw = w if len(spec) == 1 else (w - design.gutter) / 2
        bx = x0 + col * (bw + design.gutter) if len(spec) > 1 else x0
        bh = 10.0 if e.kind == "title" else min(shelf_h - design.gutter / 2, 11.0)
        boxes.append(Box(e.kind, e.text, bx, y0 + row * shelf_h, bw, bh))
    return Slide(boxes)


CODERS = {"template": code_template, "columns": code_columns, "shelf": code_shelf}


# ---------------------------------------------------------------------------
# Checker: deterministic render checks -> one score (lower is better)
# ---------------------------------------------------------------------------


def _overlap_area(a: Box, b: Box) -> float:
    dx = min(a.right, b.right) - max(a.x, b.x)
    dy = min(a.bottom, b.bottom) - max(a.y, b.y)
    return dx * dy if dx > 0 and dy > 0 else 0.0


def _text_needed_height(box: Box, design: DesignLanguage) -> float:
    """Estimate the height the box's text needs at the design's font size."""
    if not box.text:
        return 0.0
    pt = design.title_pt if box.kind == "title" else design.body_pt
    char_w = pt * 0.09  # abstract units per character at this point size
    line_h = pt * 0.24
    chars_per_line = max(1, int(box.w / char_w))
    lines = math.ceil(len(box.text) / chars_per_line)
    return lines * line_h


def check_slide(slide: Slide, design: DesignLanguage) -> dict:
    """Score a slide. Every penalty is a concrete, explainable defect."""
    overlap = 0.0
    for i, a in enumerate(slide.boxes):
        for b in slide.boxes[i + 1 :]:
            overlap += _overlap_area(a, b)
    out_of_bounds = sum(
        max(0.0, -b.x)
        + max(0.0, -b.y)
        + max(0.0, b.right - SLIDE_W)
        + max(0.0, b.bottom - SLIDE_H)
        for b in slide.boxes
    )
    text_overflow = sum(
        max(0.0, _text_needed_height(b, design) - b.h) for b in slide.boxes
    )
    margin_violations = sum(
        1
        for b in slide.boxes
        if b.x < design.margin - 1e-9
        or b.y < design.margin - 1e-9
        or b.right > SLIDE_W - design.margin + 1e-9
        or b.bottom > SLIDE_H - design.margin + 1e-9
    )
    score = (
        10.0 * overlap
        + 5.0 * out_of_bounds
        + 4.0 * text_overflow
        + 8.0 * margin_violations
    )
    return {
        "score": round(score, 4),
        "overlap": round(overlap, 4),
        "out_of_bounds": round(out_of_bounds, 4),
        "text_overflow": round(text_overflow, 4),
        "margin_violations": margin_violations,
        "clean": score == 0.0,
    }


# ---------------------------------------------------------------------------
# Verify-improve loop: keep a rewrite only if the checker confirms it is better
# ---------------------------------------------------------------------------


def _mutate(slide: Slide, rng: random.Random) -> Slide:
    """One random rewrite proposal: nudge, stretch or shrink one box."""
    boxes = [Box(b.kind, b.text, b.x, b.y, b.w, b.h) for b in slide.boxes]
    if not boxes:
        return Slide(boxes)
    b = rng.choice(boxes)
    op = rng.choice(["shrink_h", "grow_h", "grow_w", "shrink_w", "move_in"])
    if op == "shrink_h":
        b.h = max(4.0, b.h * 0.8)
    elif op == "grow_h":
        b.h = b.h * 1.2 + 1.0
    elif op == "grow_w":
        b.w = b.w * 1.15 + 1.0
    elif op == "shrink_w":
        b.w = max(8.0, b.w * 0.85)
    elif op == "move_in":  # pull an out-of-margin box back inside
        b.x = min(max(b.x, 0.0), SLIDE_W - b.w)
        b.y = min(max(b.y, 0.0), SLIDE_H - b.h)
    return Slide(boxes)


def improve_slide(
    slide: Slide, design: DesignLanguage, rounds: int = 60, seed: int = 7
) -> tuple[Slide, list[float]]:
    """Rewrite loop with a hard rule: the incumbent changes only on a
    strictly better checker score. Returns (best slide, score trace)."""
    rng = random.Random(seed)
    best = slide
    best_score = check_slide(best, design)["score"]
    trace = [best_score]
    for _ in range(rounds):
        candidate = _mutate(best, rng)
        score = check_slide(candidate, design)["score"]
        if score < best_score:  # keep only if the checker confirms improvement
            best, best_score = candidate, score
            trace.append(score)
    return best, trace


def build_deck(slides_spec: list[list[Element]], design: DesignLanguage) -> dict:
    """Full pipeline: brief -> all coders per slide -> checker picks the best
    coder -> verify-improve loop polishes it."""
    brief = design_brief(slides_spec, design)
    deck = {"brief": brief, "slides": []}
    for idx, spec in enumerate(slides_spec):
        candidates = {name: fn(spec, design) for name, fn in CODERS.items()}
        scored = {
            name: check_slide(s, design)["score"] for name, s in candidates.items()
        }
        winner = min(scored, key=scored.get)
        polished, trace = improve_slide(candidates[winner], design, seed=100 + idx)
        deck["slides"].append(
            {
                "coder_scores": scored,
                "winner": winner,
                "score_before_loop": scored[winner],
                "score_after_loop": check_slide(polished, design)["score"],
                "improvement_trace": trace,
                "final_check": check_slide(polished, design),
                "boxes": [
                    {
                        "kind": b.kind,
                        "x": round(b.x, 1),
                        "y": round(b.y, 1),
                        "w": round(b.w, 1),
                        "h": round(b.h, 1),
                    }
                    for b in polished.boxes
                ],
            }
        )
    return deck


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

DEMO_SPEC: list[list[Element]] = [
    [
        Element("title", "Q3 churn readout"),
        Element("body", "Churn rose in the self-serve segment after the pricing change. "
                        "The gradient boosting model flags the same cohort the "
                        "dashboard shows, with SHAP drivers led by tenure and "
                        "support tickets in the last 30 days.", weight=2.0),
        Element("chart", "", weight=1.5),
    ],
    [
        Element("title", "What we do next"),
        Element("body", "Targeted retention offer for the flagged cohort."),
        Element("body", "Weekly data quality checks on the billing feed before "
                        "the model scores, so silent schema drift cannot repeat "
                        "the October incident in the churn pipeline again."),
        Element("image", ""),
    ],
    [
        Element("title", "Appendix: method in one slide"),
        Element("body", "Features are built in DuckDB from raw events. The model "
                        "is a gradient boosting classifier, cross-validated "
                        "ROC-AUC 0.852 on roughly 5,000 synthetic telecom "
                        "customers, with a held-out test ROC-AUC of 0.839. "
                        "Explanations come from SHAP values aggregated weekly "
                        "for the analytics review.", weight=3.0),
    ],
]


def main() -> None:
    design = DesignLanguage()
    deck = build_deck(DEMO_SPEC, design)
    print("Designer brief (slide 1):")
    print(json.dumps(deck["brief"]["slides"][0], indent=2)[:400], "...\n")
    for i, s in enumerate(deck["slides"], 1):
        print(f"Slide {i}: coder scores {s['coder_scores']} -> winner '{s['winner']}'")
        print(
            f"  verify-improve loop: {s['score_before_loop']} -> "
            f"{s['score_after_loop']} over {len(s['improvement_trace']) - 1} kept rewrites"
        )
        print(f"  final check: {s['final_check']}")
    kept = sum(len(s["improvement_trace"]) - 1 for s in deck["slides"])
    print(f"\nTotal kept rewrites across deck: {kept} "
          f"(every other proposal was rejected by the checker)")

    # The Palette loop earning its keep: hand it the worst coder's slide 3
    # (checker score 90.72, text overflowing its boxes) and let rewrites
    # compete - a rewrite is kept only when the checker confirms it improved.
    from palette_harness import code_shelf, check_slide, improve_slide
    bad = code_shelf(DEMO_SPEC[2], design)
    print(f"\nRescue demo - shelf coder's slide 3 starts at "
          f"{check_slide(bad, design)}")
    fixed, trace = improve_slide(bad, design, rounds=120, seed=5)
    print(f"  score trace of kept rewrites: {trace}")
    print(f"  after the loop: {check_slide(fixed, design)}")


if __name__ == "__main__":
    main()

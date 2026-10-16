# Visual Scratch Space: reasoning WITH images, not just ABOUT them

**Trending hook:** on 16 October 2026, Palo Alto startup Elorian released its
first model, **Glimpse**, as a research preview (TechCrunch exclusive).
The thesis: today's multimodal models look at an image once, convert it to
tokens, and then reason *about* it in text — "like describing a photo from
memory". Glimpse instead keeps going back to the source: mid-reasoning it can
crop into a region, pull specific frames from a video, or run code to measure
or compare what it is looking at. Elorian calls this a **"visual scratch
space"** — like pinching to zoom on a phone photo to double-check what you
thought you saw.

*Company claims, unverified:* Elorian says a small team trained Glimpse on a
few hundred GPUs, that it is substantially smaller than the frontier systems
it benchmarks against, and that per its own evaluations it outperforms even
GPT-6 Astra on scientific chart reasoning and short-form video captioning.
Pricing: $0.60/M input tokens, $3/M output. Take all of this as the company's
pitch until independent benchmarks land.

## What this demo shows

No real vision model here — this is a **stdlib-only simulation of the
mechanism** on a synthetic 2D scene (1000×1000 coordinate space). A one-shot
**memory baseline** (one low-res glance, then pure text reasoning) is
compared against a **scratch-space agent** (mid-reasoning tool calls, with a
tool-call budget and an auditable transcript) on three tasks:

| # | Question | Memory baseline | Scratch-space agent | Ground truth |
|---|----------|----------------|---------------------|--------------|
| 1 | How many red circles? | 2 (small ones lost at low res) | **4** (exact sweep of both halves) | 4 |
| 2 | Which red circle is closest to the blue square? | c_red_B — quantization error: the true nearest (20.6 units away) falls in the neighbouring glance cell, so cell-center estimates rank it last | **c_red_A** (exact code measurement: 20.62 vs 260.19) | c_red_A, 20.62 |
| 3 | In which video frame does the red circle first enter the right half? | never (invisible in every low-res frame) | **frame 3** (pulled exact frames) | frame 3 |

The tool mapping to the Glimpse thesis:

- `glance()` → "look at the image once, convert to tokens" (quantized 3×3 cells, objects under 30 units dropped — the memory failure mode)
- `crop(x0,y0,x1,y1)` → "crop into a region" mid-reasoning
- `pull_frame(i)` → "pull specific frames from a video"
- `measure(a,b)` / `count_in(...)` → "run code to measure or compare"

Every tool call is logged to a transcript, so each answer's reasoning trace is
auditable — the agent equivalent of showing its working.

## Run it

```bash
python3 visual_scratch_space.py        # prints the full comparison demo
python3 test_visual_scratch_space.py   # 17 unit tests
```

## Honest caveats

- Synthetic scene with exact ground truth; a real VLM's perception errors are
  statistical, not a clean resolution cutoff. The baseline is a deliberately
  coarse memory model, so "baseline fails" is illustrative, not a benchmark.
- The planners are task-specific (count / nearest / first-frame); they go
  through the generic tool-calling interface with a hard budget
  (`ToolBudgetExceeded` if exceeded), but this is a mechanism demo, not a
  general agent.
- All Glimpse capability/pricing claims above are Elorian's, from a single
  TechCrunch exclusive on launch day — not independently verified.

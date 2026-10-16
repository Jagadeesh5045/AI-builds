"""Visual scratch space: reasoning WITH images, not just ABOUT them.

Trending hook (16 Oct 2026): Elorian released Glimpse, a research-preview model
whose thesis is that multimodal models shouldn't look at an image once, turn it
into tokens, and then reason from memory. Instead, mid-reasoning it can go back
to the source: crop into a region, pull specific frames from a video, or run code
to measure or compare what it is looking at. Elorian calls this a
"visual scratch space".

This demo is stdlib-only and uses a synthetic 2D scene instead of a real vision
model, so it demonstrates the *mechanism*, not the model. A one-shot baseline
(sees one low-res glance, then reasons from memory) is compared against a
scratch-space agent (mid-reasoning tool calls: crop / pull frame / run
measurement code). The scene is rigged so that memory-based reasoning fails in
exactly the ways the thesis predicts: resolution loss, quantization error, and
lost small details.

Coordinate space: 1000 x 1000 (like labeled 0-1000 detection bboxes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Scene model
# ---------------------------------------------------------------------------

SMALL_SIZE = 30  # objects smaller than this are invisible in a low-res glance


@dataclass(frozen=True)
class Obj:
    """One labeled object in the synthetic scene."""

    id: str
    shape: str  # circle | square | triangle
    color: str  # red | blue | green | yellow
    x: float
    y: float
    size: float  # diameter-ish, in scene units


@dataclass
class Scene:
    """A synthetic 'image': a list of objects with exact ground-truth geometry."""

    objects: list[Obj]

    def by_id(self, obj_id: str) -> Obj:
        for o in self.objects:
            if o.id == obj_id:
                return o
        raise KeyError(f"unknown object id: {obj_id!r}")


def build_scene() -> Scene:
    """Deterministic scene. Positions are rigged to expose memory failures.

    Task 1 (count red circles): four red circles, two of them small -> a
    low-res glance sees only two.
    Task 2 (nearest red circle to the blue square): c_red_A sits 20.6 units
    from the square but falls in the neighbouring glance cell, while the
    bigger-looking c_red_B shares the square's cell -> cell-center estimates
    rank them wrong (quantization error).
    """
    return Scene(
        objects=[
            Obj("sq_blue", "square", "blue", 320.0, 500.0, 60.0),
            Obj("c_red_A", "circle", "red", 340.0, 505.0, 55.0),
            Obj("c_red_B", "circle", "red", 60.0, 510.0, 55.0),
            Obj("c_red_C", "circle", "red", 700.0, 520.0, 24.0),  # small
            Obj("c_red_D", "circle", "red", 690.0, 470.0, 20.0),  # small
            Obj("t_green_1", "triangle", "green", 760.0, 200.0, 50.0),
            Obj("t_green_2", "triangle", "green", 800.0, 760.0, 22.0),  # small
            Obj("c_yellow_1", "circle", "yellow", 120.0, 120.0, 48.0),
        ]
    )


def build_video(n_frames: int = 5) -> list[Scene]:
    """A tiny 'video': one small red circle drifting right across frames.

    Task 3: in which frame does it first enter the right half (x > 500)?
    The circle is small, so a low-res glance of any single frame misses it.
    """
    xs = [100.0, 250.0, 400.0, 600.0, 750.0][:n_frames]
    return [
        Scene(objects=[Obj("c_red_vid", "circle", "red", x, 500.0, 24.0)])
        for x in xs
    ]


# ---------------------------------------------------------------------------
# Perception tools: the model's eyes and hands
# ---------------------------------------------------------------------------

def _cell(x: float, y: float) -> str:
    col = "left" if x < 333.0 else ("center" if x < 667.0 else "right")
    row = "top" if y < 333.0 else ("middle" if y < 667.0 else "bottom")
    return f"{row}-{col}"


class VisualScratchSpace:
    """The tool interface both agents use.

    glance()            -> cheap low-res summary (quantized cells, small
                           objects dropped). This is "looking once and
                           converting to tokens".
    crop(x0,y0,x1,y1)   -> exact objects intersecting a region. This is
                           "cropping into a region mid-reasoning".
    pull_frame(i)       -> exact objects in video frame i. This is "pulling
                           specific frames from a video".
    measure(a_id,b_id)  -> exact Euclidean distance. This is "running code to
                           measure what it is looking at".
    count_in(...)       -> exact count of shape/color in a region (also code).
    Every call is logged to the transcript so the reasoning trace is auditable.
    """

    def __init__(self, scene: Scene, video: list[Scene] | None = None,
                 max_calls: int = 6):
        self.scene = scene
        self.video = video or []
        self.max_calls = max_calls
        self.calls = 0
        self.transcript: list[str] = []

    # -- plumbing ---------------------------------------------------------
    def _use_call(self, desc: str) -> None:
        self.calls += 1
        if self.calls > self.max_calls:
            raise ToolBudgetExceeded(
                f"tool budget exceeded after {self.max_calls} calls ({desc})"
            )
        self.transcript.append(f"[call {self.calls}] {desc}")

    # -- tools ------------------------------------------------------------
    def glance(self) -> list[dict]:
        """Low-resolution summary: cell-quantized positions, small objects lost."""
        self._use_call("glance() -> low-res summary")
        out = []
        for o in self.scene.objects:
            if o.size < SMALL_SIZE:
                continue  # lost at low resolution
            out.append(
                {"id": o.id, "shape": o.shape, "color": o.color,
                 "cell": _cell(o.x, o.y)}
            )
        return out

    def glance_frame(self, i: int) -> list[dict]:
        """Low-res summary of one video frame."""
        self._use_call(f"glance_frame({i}) -> low-res frame summary")
        frame = self._frame(i)
        out = []
        for o in frame.objects:
            if o.size < SMALL_SIZE:
                continue
            out.append(
                {"id": o.id, "shape": o.shape, "color": o.color,
                 "cell": _cell(o.x, o.y)}
            )
        return out

    def crop(self, x0: float, y0: float, x1: float, y1: float) -> list[dict]:
        """Exact view of a region: full geometry of intersecting objects."""
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError(
                f"invalid crop box ({x0},{y0})-({x1},{y1}); "
                "need 0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000"
            )
        self._use_call(f"crop({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}) -> exact view")
        out = []
        for o in self.scene.objects:
            if x0 <= o.x <= x1 and y0 <= o.y <= y1:
                out.append({"id": o.id, "shape": o.shape, "color": o.color,
                            "x": o.x, "y": o.y, "size": o.size})
        return out

    def pull_frame(self, i: int) -> list[dict]:
        """Exact view of video frame i (all geometry, no resolution loss)."""
        frame = self._frame(i)
        self._use_call(f"pull_frame({i}) -> exact frame view")
        return [{"id": o.id, "shape": o.shape, "color": o.color,
                 "x": o.x, "y": o.y, "size": o.size} for o in frame.objects]

    def measure(self, a_id: str, b_id: str) -> float:
        """Run measurement code: exact distance between two objects."""
        a = self.scene.by_id(a_id)
        b = self.scene.by_id(b_id)
        d = math.hypot(a.x - b.x, a.y - b.y)
        self._use_call(f"measure({a_id},{b_id}) = {d:.2f}")
        return d

    def count_in(self, shape: str, color: str,
                 x0: float = 0, y0: float = 0,
                 x1: float = 1000, y1: float = 1000) -> int:
        """Run counting code over exact geometry in a region."""
        self._use_call(
            f"count_in(shape={shape},color={color},box=({x0:.0f},{y0:.0f},"
            f"{x1:.0f},{y1:.0f}))"
        )
        return sum(
            1 for o in self.scene.objects
            if o.shape == shape and o.color == color
            and x0 <= o.x <= x1 and y0 <= o.y <= y1
        )

    # -- helpers ----------------------------------------------------------
    def _frame(self, i: int) -> Scene:
        if not 0 <= i < len(self.video):
            raise IndexError(
                f"frame {i} out of range (video has {len(self.video)} frames)"
            )
        return self.video[i]


class ToolBudgetExceeded(RuntimeError):
    """Raised when the agent exceeds its mid-reasoning tool-call budget."""


# ---------------------------------------------------------------------------
# Baseline: one glance, then reason from memory ("describe the photo from
# memory" -- the failure mode the Glimpse thesis is aimed at).
# ---------------------------------------------------------------------------

_CELL_CENTER = {
    ("top", "left"): (166.5, 166.5), ("top", "center"): (500.0, 166.5),
    ("top", "right"): (833.5, 166.5), ("middle", "left"): (166.5, 500.0),
    ("middle", "center"): (500.0, 500.0), ("middle", "right"): (833.5, 500.0),
    ("bottom", "left"): (166.5, 833.5), ("bottom", "center"): (500.0, 833.5),
    ("bottom", "right"): (833.5, 833.5),
}


def _cell_center(cell: str) -> tuple[float, float]:
    row, col = cell.split("-")
    return _CELL_CENTER[(row, col)]


class MemoryBaseline:
    """One-shot reasoner: single glance, then pure text reasoning.

    Distance estimates use cell centers -- the best you can do when all you
    kept from the image was a coarse token summary.
    """

    def __init__(self, tools: VisualScratchSpace):
        self.tools = tools

    def count_red_circles(self) -> int:
        seen = self.tools.glance()
        return sum(1 for o in seen if o["shape"] == "circle"
                   and o["color"] == "red")

    def nearest_red_circle_to_blue_square(self) -> tuple[str, float]:
        seen = self.tools.glance()
        sq = next(o for o in seen if o["id"] == "sq_blue")
        sx, sy = _cell_center(sq["cell"])
        best, best_d = None, float("inf")
        for o in seen:
            if o["shape"] == "circle" and o["color"] == "red":
                ox, oy = _cell_center(o["cell"])
                d = math.hypot(ox - sx, oy - sy)
                if d < best_d:
                    best, best_d = o["id"], d
        return best, best_d

    def first_frame_right_half(self) -> int | None:
        for i in range(len(self.tools.video)):
            seen = self.tools.glance_frame(i)
            for o in seen:
                if o["shape"] == "circle" and o["color"] == "red":
                    if o["cell"].endswith("-right"):
                        return i
        return None  # never saw it at all


# ---------------------------------------------------------------------------
# Scratch-space agent: think, look again, measure, then answer.
# ---------------------------------------------------------------------------

@dataclass
class ScratchSpaceAgent:
    """Reasoner with a visual scratch space: it may go back to the source.

    Each answer follows the loop: glance (cheap prior) -> crop / pull frames /
    run measurement code -> commit to an answer. Tool calls are budgeted.
    """

    tools: VisualScratchSpace
    thoughts: list[str] = field(default_factory=list)

    def think(self, thought: str) -> None:
        self.thoughts.append(thought)

    def count_red_circles(self) -> int:
        self.think("Glance shows one red circle, but small details are lost "
                   "at low res. Verify by cropping each half and counting "
                   "with exact geometry.")
        self.tools.glance()
        # Left half then right half: exact sweep, no resolution loss.
        n = self.tools.count_in("circle", "red", x1=500.0)
        n += self.tools.count_in("circle", "red", x0=500.0)
        self.think(f"Exact count across both halves: {n}.")
        return n

    def nearest_red_circle_to_blue_square(self) -> tuple[str, float]:
        self.think("Glance cells are too coarse for a nearest-neighbor "
                   "decision near a cell boundary. Crop around the blue "
                   "square to find candidate red circles, then measure "
                   "exact distances with code.")
        self.tools.glance()
        region = self.tools.crop(0.0, 400.0, 500.0, 600.0)
        candidates = [o["id"] for o in region
                      if o["shape"] == "circle" and o["color"] == "red"]
        self.think(f"Candidates near the square: {candidates}. Measuring.")
        best, best_d = None, float("inf")
        for cid in candidates:
            d = self.tools.measure("sq_blue", cid)
            if d < best_d:
                best, best_d = cid, d
        self.think(f"Nearest: {best} at {best_d:.2f} units.")
        return best, best_d

    def first_frame_right_half(self) -> int | None:
        self.think("The circle is small: per-frame glances miss it entirely. "
                   "Pull exact frames instead, scanning forward until its "
                   "x crosses 500.")
        for i in range(len(self.tools.video)):
            objs = self.tools.pull_frame(i)
            for o in objs:
                if o["shape"] == "circle" and o["color"] == "red" and o["x"] > 500:
                    self.think(f"Frame {i}: circle at x={o['x']:.0f} "
                               "-> first in right half.")
                    return i
        return None


# ---------------------------------------------------------------------------
# Ground truth + demo
# ---------------------------------------------------------------------------

def ground_truth() -> dict:
    scene = build_scene()
    reds = [o for o in scene.objects if o.shape == "circle" and o.color == "red"]
    sq = scene.by_id("sq_blue")
    nearest = min(reds, key=lambda o: math.hypot(o.x - sq.x, o.y - sq.y))
    return {
        "count_red_circles": len(reds),
        "nearest_id": nearest.id,
        "nearest_dist": math.hypot(nearest.x - sq.x, nearest.y - sq.y),
        "first_frame_right_half": 3,
    }


def run_demo() -> None:
    gt = ground_truth()
    print("=" * 70)
    print("VISUAL SCRATCH SPACE DEMO")
    print("Thesis (Elorian Glimpse, 16 Oct 2026): reason WITH the image, not")
    print("just ABOUT it -- go back to the source mid-reasoning.")
    print("=" * 70)

    # Task 1: counting -----------------------------------------------------
    print("\n[Task 1] How many red circles are in the scene?")
    print(f"  ground truth: {gt['count_red_circles']}")
    base = MemoryBaseline(VisualScratchSpace(build_scene()))
    print(f"  memory baseline (one glance): {base.count_red_circles()}  <-- "
          "small circles lost at low res")
    agent = ScratchSpaceAgent(VisualScratchSpace(build_scene()))
    print(f"  scratch-space agent:          {agent.count_red_circles()}  <-- "
          "verified with exact crops")
    for t in agent.tools.transcript:
        print(f"      {t}")

    # Task 2: nearest neighbor ----------------------------------------------
    print("\n[Task 2] Which red circle is closest to the blue square?")
    print(f"  ground truth: {gt['nearest_id']} "
          f"({gt['nearest_dist']:.2f} units)")
    base = MemoryBaseline(VisualScratchSpace(build_scene()))
    b_id, b_d = base.nearest_red_circle_to_blue_square()
    print(f"  memory baseline (cell centers): {b_id} (~{b_d:.0f} units)  <-- "
          "quantization error flips the ranking")
    agent = ScratchSpaceAgent(VisualScratchSpace(build_scene()))
    a_id, a_d = agent.nearest_red_circle_to_blue_square()
    print(f"  scratch-space agent:            {a_id} ({a_d:.2f} units)  <-- "
          "measured with code")
    for t in agent.tools.transcript:
        print(f"      {t}")

    # Task 3: video frames ---------------------------------------------------
    print("\n[Task 3] In which frame does the red circle first enter the "
          "right half?")
    print(f"  ground truth: frame {gt['first_frame_right_half']}")
    base = MemoryBaseline(VisualScratchSpace(build_scene(), build_video()))
    print(f"  memory baseline (per-frame glances): {base.first_frame_right_half()}  "
          "<-- circle invisible at low res in every frame")
    agent = ScratchSpaceAgent(VisualScratchSpace(build_scene(), build_video(),
                                                 max_calls=8))
    print(f"  scratch-space agent:                 "
          f"{agent.first_frame_right_half()}  <-- pulled exact frames")
    for t in agent.tools.transcript:
        print(f"      {t}")

    print("\nTakeaway: each failure is a resolution/quantization loss that a "
          "mid-reasoning look-back fixes -- the core of the visual scratch "
          "space idea.")


if __name__ == "__main__":
    run_demo()

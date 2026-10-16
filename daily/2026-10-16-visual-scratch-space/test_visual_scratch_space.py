"""Tests for the visual scratch space demo.

The interesting assertions are the *behavioral* ones: the memory baseline
fails each task in the predicted way, and the scratch-space agent gets each
one right within its tool budget.
"""

import math
import unittest

from visual_scratch_space import (
    MemoryBaseline,
    ScratchSpaceAgent,
    ToolBudgetExceeded,
    VisualScratchSpace,
    build_scene,
    build_video,
    ground_truth,
)


class TestPerceptionTools(unittest.TestCase):
    def test_glance_drops_small_objects(self):
        seen = VisualScratchSpace(build_scene()).glance()
        ids = {o["id"] for o in seen}
        # the three small objects must be invisible at low res
        self.assertNotIn("c_red_C", ids)
        self.assertNotIn("c_red_D", ids)
        self.assertNotIn("t_green_2", ids)
        # big objects stay, with quantized cells
        self.assertIn("sq_blue", ids)
        sq = next(o for o in seen if o["id"] == "sq_blue")
        self.assertEqual(sq["cell"], "middle-left")

    def test_glance_cells_cover_grid(self):
        seen = VisualScratchSpace(build_scene()).glance()
        for o in seen:
            row, col = o["cell"].split("-")
            self.assertIn(row, {"top", "middle", "bottom"})
            self.assertIn(col, {"left", "center", "right"})

    def test_crop_returns_exact_geometry(self):
        tools = VisualScratchSpace(build_scene())
        objs = tools.crop(0.0, 400.0, 500.0, 600.0)
        ids = {o["id"] for o in objs}
        self.assertEqual(ids, {"sq_blue", "c_red_A", "c_red_B"})
        sq = next(o for o in objs if o["id"] == "sq_blue")
        self.assertEqual((sq["x"], sq["y"]), (320.0, 500.0))

    def test_crop_empty_region(self):
        tools = VisualScratchSpace(build_scene())
        self.assertEqual(tools.crop(900.0, 0.0, 1000.0, 100.0), [])

    def test_crop_rejects_bad_box(self):
        tools = VisualScratchSpace(build_scene())
        with self.assertRaises(ValueError):
            tools.crop(500.0, 0.0, 100.0, 100.0)  # x0 >= x1
        with self.assertRaises(ValueError):
            tools.crop(-10.0, 0.0, 100.0, 100.0)  # out of bounds

    def test_measure_matches_ground_truth(self):
        tools = VisualScratchSpace(build_scene())
        d = tools.measure("sq_blue", "c_red_A")
        self.assertAlmostEqual(d, math.hypot(20.0, 5.0), places=9)
        with self.assertRaises(KeyError):
            tools.measure("sq_blue", "nope")

    def test_count_in_regions(self):
        tools = VisualScratchSpace(build_scene())
        self.assertEqual(tools.count_in("circle", "red"), 4)
        self.assertEqual(tools.count_in("circle", "red", x1=500.0), 2)
        self.assertEqual(tools.count_in("circle", "red", x0=500.0), 2)
        self.assertEqual(tools.count_in("triangle", "green"), 2)

    def test_pull_frame_out_of_range(self):
        tools = VisualScratchSpace(build_scene(), build_video())
        with self.assertRaises(IndexError):
            tools.pull_frame(99)

    def test_tool_budget_enforced(self):
        tools = VisualScratchSpace(build_scene(), max_calls=2)
        tools.glance()
        tools.crop(0.0, 0.0, 1000.0, 1000.0)
        with self.assertRaises(ToolBudgetExceeded):
            tools.measure("sq_blue", "c_red_A")

    def test_transcript_audits_every_call(self):
        tools = VisualScratchSpace(build_scene())
        tools.glance()
        tools.measure("sq_blue", "c_red_A")
        self.assertEqual(len(tools.transcript), 2)
        self.assertIn("glance()", tools.transcript[0])
        self.assertIn("measure(sq_blue,c_red_A)", tools.transcript[1])


class TestBaselineFails(unittest.TestCase):
    """The one-shot baseline must fail each task the predicted way."""

    def test_baseline_undercounts_red_circles(self):
        base = MemoryBaseline(VisualScratchSpace(build_scene()))
        self.assertEqual(base.count_red_circles(), 2)  # truth is 4

    def test_baseline_picks_wrong_nearest(self):
        base = MemoryBaseline(VisualScratchSpace(build_scene()))
        b_id, _ = base.nearest_red_circle_to_blue_square()
        self.assertEqual(b_id, "c_red_B")  # truth is c_red_A

    def test_baseline_misses_video_circle(self):
        base = MemoryBaseline(VisualScratchSpace(build_scene(), build_video()))
        self.assertIsNone(base.first_frame_right_half())  # truth is frame 3


class TestScratchSpaceSucceeds(unittest.TestCase):
    """The scratch-space agent must get each task right, within budget."""

    def test_agent_counts_all_red_circles(self):
        agent = ScratchSpaceAgent(VisualScratchSpace(build_scene()))
        self.assertEqual(agent.count_red_circles(),
                         ground_truth()["count_red_circles"])
        self.assertLessEqual(agent.tools.calls, agent.tools.max_calls)

    def test_agent_finds_true_nearest(self):
        agent = ScratchSpaceAgent(VisualScratchSpace(build_scene()))
        a_id, a_d = agent.nearest_red_circle_to_blue_square()
        gt = ground_truth()
        self.assertEqual(a_id, gt["nearest_id"])
        self.assertAlmostEqual(a_d, gt["nearest_dist"], places=9)
        self.assertLessEqual(agent.tools.calls, agent.tools.max_calls)

    def test_agent_finds_frame(self):
        agent = ScratchSpaceAgent(
            VisualScratchSpace(build_scene(), build_video(), max_calls=8))
        self.assertEqual(agent.first_frame_right_half(),
                         ground_truth()["first_frame_right_half"])
        self.assertLessEqual(agent.tools.calls, agent.tools.max_calls)

    def test_ground_truth_is_self_consistent(self):
        gt = ground_truth()
        scene = build_scene()
        reds = [o for o in scene.objects
                if o.shape == "circle" and o.color == "red"]
        self.assertEqual(gt["count_red_circles"], len(reds))
        nearest = scene.by_id(gt["nearest_id"])
        sq = scene.by_id("sq_blue")
        for o in reds:
            self.assertGreaterEqual(
                math.hypot(o.x - sq.x, o.y - sq.y),
                math.hypot(nearest.x - sq.x, nearest.y - sq.y) - 1e-9)
        # video: frame 3 is the first with x > 500
        xs = [f.objects[0].x for f in build_video()]
        self.assertEqual(next(i for i, x in enumerate(xs) if x > 500), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

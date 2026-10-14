"""Tests for the Perception-Language-Action demo."""

import math
import random

from pla import (VisionEncoder, AudioEncoder, TactileEncoder, ProprioEncoder,
                 SharedSpace, TimeAlignedBuffer, IntentParser, SafetyGate,
                 ActionDecoder, RobotEnv, run_mission, cosine, unit)


def test_vision_encoder_dim():
    img = [[random.random() for _ in range(8)] for _ in range(8)]
    assert len(VisionEncoder().encode(img)) == 24


def test_audio_encoder_dim():
    wave = [math.sin(i) for i in range(64)]
    v = AudioEncoder().encode(wave)
    assert len(v) == 6 and all(math.isfinite(x) for x in v)


def test_tactile_encoder_dim():
    grid = [[0.2] * 4 for _ in range(4)]
    assert len(TactileEncoder().encode(grid)) == 8


def test_proprio_encoder_dim():
    assert len(ProprioEncoder().encode([0.1, 0.2, 0.3, 0.4])) == 4


def test_shared_space_unit_norm():
    space = SharedSpace()
    v = space.embed_modality([0.5] * 24)
    assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-6
    t = space.embed_text("pick up the cup")
    assert abs(math.sqrt(sum(x * x for x in t)) - 1.0) < 1e-6


def test_shared_space_deterministic():
    a, b = SharedSpace(), SharedSpace()
    assert a.embed_text("grasp cup") == b.embed_text("grasp cup")


def test_text_matches_itself_best():
    space = SharedSpace()
    q = space.embed_text("pick up the red cup")
    same = space.embed_text("pick up the red cup")
    other = space.embed_text("weather forecast tomorrow")
    assert cosine(q, same) > cosine(q, other)


def test_time_buffer_alignment():
    buf = TimeAlignedBuffer(["vision", "audio"])
    buf.push("vision", 0.0, [1.0] * 24)
    buf.push("vision", 0.2, [2.0] * 24)
    buf.push("audio", 0.1, [3.0] * 6)
    obs, stale = buf.frame(0.25)
    assert obs["vision"][0] == 2.0 and obs["audio"][0] == 3.0 and not stale


def test_time_buffer_stale():
    buf = TimeAlignedBuffer(["vision"], max_gap=0.1)
    buf.push("vision", 0.0, [1.0] * 24)
    obs, stale = buf.frame(5.0)
    assert "vision" in stale and "vision" not in obs


def test_parser_pick_and_place():
    env = RobotEnv()
    intent = IntentParser(env.objects).parse(
        "pick up the red cup and place it on the green tray",
        {"objects": list(env.objects)})
    assert intent["ok"]
    kinds = [k for k, _ in intent["plan"]]
    assert kinds == ["move_to", "grasp", "lift", "move_to", "place", "release"]


def test_parser_unknown_object_not_guessed():
    env = RobotEnv()
    known = dict(env.objects)
    known["teapot"] = {"colour": "white", "weight": 0.3, "pos": (0.9, 0.9),
                       "held": False}
    intent = IntentParser(known).parse("pick up the teapot",
                                      {"objects": list(env.objects)})
    assert not intent["ok"] and intent["reason"] == "mentioned object not visible"


def test_gate_denies_speeding():
    gate = SafetyGate(max_speed=0.01)
    ctx = {"ee_pos": (0.5, 0.5),
           "object_pos": {"red cup": (0.9, 0.9)},
           "object_weight": {"red cup": 0.4}, "dt": 1.0}
    ok, why = gate.check(("move_to", "red cup"), ctx)
    assert not ok and "speed" in why


def test_gate_denies_overweight():
    gate = SafetyGate(max_payload=0.5)
    ctx = {"ee_pos": (0.5, 0.5),
           "object_pos": {"blue book": (0.5, 0.5)},
           "object_weight": {"blue book": 1.2}, "dt": 1.0}
    ok, why = gate.check(("grasp", "blue book"), ctx)
    assert not ok and "payload" in why


def test_gate_denies_low_confidence():
    gate = SafetyGate(min_confidence=0.5)
    ctx = {"ee_pos": (0.5, 0.5), "object_pos": {}, "object_weight": {},
           "dt": 1.0, "confidence": 0.3}
    ok, _ = gate.check(("inspect", "scene"), ctx)
    assert not ok


def test_full_mission_succeeds():
    report = run_mission("pick up the red cup and place it on the green tray")
    assert report["success"], report["reason"]
    assert len(report["executed"]) == 6


def test_mission_cup_ends_on_tray():
    report = run_mission("pick up the red cup and place it on the green tray")
    cup, tray = None, None
    # re-run through env to inspect final positions
    env = RobotEnv()
    dec = ActionDecoder(SafetyGate(max_speed=2.0))
    dec.run(report["plan"], env, confidence=report["confidence"])
    assert env.objects["red cup"]["pos"] == env.objects["green tray"]["pos"]


def test_mission_records_denial():
    report = run_mission("pick up the red cup and place it on the green tray",
                         max_speed=0.01)
    assert not report["success"] and report["denied"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", fn.__name__, "-", e)
    print("%d/%d tests passing" % (len(fns) - failed, len(fns)))
    raise SystemExit(1 if failed else 0)

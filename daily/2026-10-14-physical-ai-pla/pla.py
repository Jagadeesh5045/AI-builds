"""Perception-Language-Action (PLA) pipeline demo, stdlib only.

Inspired by the Perception-Language-Action architecture pattern behind this
week's embodied-AI announcements (e.g. NEURA Robotics' Neuron AI, 14 Oct 2026):
every modality (vision, hearing, touch, proprioception) is encoded into a
common, time-aligned representation, projected into a shared language-like
space where the task command is also embedded, and an action model turns the
resulting intent into precise, safety-gated robot motion.

Nothing here is a real foundation model. It is a small, honest teaching demo:
* Perception encoders turn synthetic sensor frames into feature vectors.
* A seeded shared-space projection aligns all modalities and the command
  string into one vector space (toy "common representation").
* TimeAlignedBuffer re-samples the modalities onto a common timeline.
* IntentParser converts a natural-language command + perception summary into
  a plan of robot primitives.
* ActionDecoder maps plan steps to concrete motion commands, and SafetyGate
  enforces workspace, velocity, payload and confidence limits before any
  motion is executed.
* RobotEnv is a minimal 2D tabletop world that executes the actions and
  reports fresh observations.

Run `python3 demo.py` for a narrated pick-and-place mission.
Run `python3 test_pla.py` for the test suite.
"""

import math
import random
from collections import deque


# ---------------------------------------------------------------------------
# Small linear-algebra helpers (lists of floats)
# ---------------------------------------------------------------------------

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return math.sqrt(dot(v, v)) or 1e-9


def unit(v):
    n = norm(v)
    return [x / n for x in v]


def cosine(a, b):
    return dot(unit(a), unit(b))


def matvec(m, v):
    """matrix (rows) times vector."""
    return [dot(row, v) for row in m]


# ---------------------------------------------------------------------------
# Perception: synthetic sensor encoders
# ---------------------------------------------------------------------------

class VisionEncoder:
    """8x8 image -> 24-d features: per-quadrant mean/var + edge energy."""

    def __init__(self):
        self.dim = 24

    def encode(self, image):
        feats = []
        for qy in range(2):
            for qx in range(2):
                px = [image[y][qx * 4 + dx] for y in range(qy * 4, qy * 4 + 4)
                      for dx in range(4)]
                mean = sum(px) / len(px)
                var = sum((p - mean) ** 2 for p in px) / len(px)
                # horizontal + vertical edge energy inside the quadrant
                gx = sum(abs(image[y][qx * 4 + dx] - image[y][qx * 4 + dx + 1])
                         for y in range(qy * 4, qy * 4 + 4) for dx in range(3))
                gy = sum(abs(image[y][qx * 4 + dx] - image[y + 1][qx * 4 + dx])
                         for y in range(qy * 4, qy * 4 + 3) for dx in range(4))
                feats += [mean, var, gx / 48.0, gy / 48.0]
        # dominant colour channels (R, G, B proportions of a toy RGB tuple image)
        if image and isinstance(image[0][0], tuple):
            r = sum(p[0] for row in image for p in row)
            g = sum(p[1] for row in image for p in row)
            b = sum(p[2] for row in image for p in row)
            tot = r + g + b or 1
            feats += [r / tot, g / tot, b / tot]
        else:
            feats += [1 / 3, 1 / 3, 1 / 3]
        return feats[:self.dim] + [0.0] * (self.dim - len(feats[:self.dim]))


class AudioEncoder:
    """64-sample waveform -> 6-d: rms, zero-cross rate, spectral centroid,
    band energies, peak freq (tiny hand-rolled DFT)."""

    def __init__(self):
        self.dim = 6

    def _dft_mag(self, samples, k):
        n = len(samples)
        re = sum(s * math.cos(2 * math.pi * k * i / n) for i, s in enumerate(samples))
        im = sum(s * math.sin(2 * math.pi * k * i / n) for i, s in enumerate(samples))
        return math.sqrt(re * re + im * im)

    def encode(self, samples):
        n = len(samples)
        rms = math.sqrt(sum(s * s for s in samples) / n)
        zcr = sum(1 for a, b in zip(samples, samples[1:]) if (a < 0) != (b < 0)) / (n - 1)
        mags = [self._dft_mag(samples, k) for k in range(1, 9)]
        tot = sum(mags) or 1e-9
        centroid = sum(k * m for k, m in enumerate(mags, 1)) / tot / 8.0
        low = sum(mags[:2]) / tot
        high = sum(mags[-2:]) / tot
        peak = (max(range(len(mags)), key=mags.__getitem__) + 1) / 8.0
        return [rms, zcr, centroid, low, high, peak]


class TactileEncoder:
    """4x4 pressure grid -> 8-d: mean, max, centre of mass, quadrant sums."""

    def __init__(self):
        self.dim = 8

    def encode(self, grid):
        vals = [v for row in grid for v in row]
        mean = sum(vals) / len(vals)
        mx = max(vals)
        tot = sum(vals) or 1e-9
        cx = sum(x * grid[y][x] for y in range(4) for x in range(4)) / tot / 3.0
        cy = sum(y * grid[y][x] for y in range(4) for x in range(4)) / tot / 3.0
        quads = [sum(grid[y][x] for y in range(qy, qy + 2) for x in range(qx, qx + 2))
                 for qy in (0, 2) for qx in (0, 2)]
        qs = sum(quads) or 1e-9
        return [mean, mx, cx, cy] + [q / qs for q in quads]


class ProprioEncoder:
    """Joint angles (radians) -> normalised 4-d vector."""

    def __init__(self):
        self.dim = 4

    def encode(self, joints):
        return [math.sin(j) for j in joints[:4]] + \
               [0.0] * (self.dim - min(4, len(joints)))


# ---------------------------------------------------------------------------
# Shared language-like space + time alignment
# ---------------------------------------------------------------------------

class SharedSpace:
    """Projects every modality (and command text) into one common space.

    Toy version of the 'common, time-aligned representation': each modality
    gets a seeded random projection matrix into R^64, and command strings are
    hashed into the same space, so cross-modal similarity queries work.
    """

    def __init__(self, out_dim=64, seed=7):
        self.out_dim = out_dim
        self._proj = {}
        self._rng = random.Random(seed)

    def _projection(self, in_dim):
        if in_dim not in self._proj:
            r = random.Random(self._rng.random())
            self._proj[in_dim] = [[r.gauss(0, 1 / math.sqrt(in_dim))
                                   for _ in range(in_dim)]
                                  for _ in range(self.out_dim)]
        return self._proj[in_dim]

    def embed_modality(self, feats):
        return unit(matvec(self._projection(len(feats)), feats))

    def embed_text(self, text):
        feats = [0.0] * 128
        for tok in text.lower().split():
            h = abs(hash(tok)) % 128
            feats[h] += 1.0
        feats = [f / (len(text.split()) or 1) for f in feats]
        return unit(matvec(self._projection(128), feats))


class TimeAlignedBuffer:
    """Resamples timestamped modality frames onto a common timeline.

    Each modality pushes (t, vector). `frame(t)` returns the nearest vector
    from every modality (latest at-or-before t), concatenated into one
    time-aligned observation. Gaps beyond `max_gap` are flagged stale.
    """

    def __init__(self, modalities, max_gap=0.5):
        self.modalities = list(modalities)
        self.max_gap = max_gap
        self._buf = {m: deque() for m in modalities}

    def push(self, modality, t, vector):
        self._buf[modality].append((t, vector))

    def frame(self, t):
        obs, stale = {}, []
        for m in self.modalities:
            chosen = None
            for tt, v in self._buf[m]:
                if tt <= t:
                    chosen = (tt, v)
                else:
                    break
            if chosen is None or t - chosen[0] > self.max_gap:
                stale.append(m)
            else:
                obs[m] = chosen[1]
        return obs, stale


# ---------------------------------------------------------------------------
# Language reasoning: command -> plan
# ---------------------------------------------------------------------------

PRIMITIVES = ("move_to", "grasp", "lift", "place", "release", "inspect")

class IntentParser:
    """Command string + perception summary -> plan of (primitive, target).

    Toy reasoning: keyword slots filled from the command, objects validated
    against what perception actually reports; anything unresolved is flagged
    for clarification instead of guessed.
    """

    def __init__(self, known_objects):
        self.known_objects = known_objects  # dict name -> dict(colour, pos)

    def parse(self, command, perception):
        cmd = command.lower()
        # verb -> primitive mapping
        verb = None
        for v, prim in [("pick", "grasp"), ("grab", "grasp"), ("lift", "grasp"),
                        ("place", "place"), ("put", "place"),
                        ("move", "move_to"), ("inspect", "inspect"),
                        ("look", "inspect")]:
            if v in cmd:
                verb = prim
                break
        if verb is None:
            return {"ok": False, "reason": "no recognised action verb"}

        # find object mentions that perception actually reports
        mentioned = [n for n in self.known_objects if n in cmd]
        visible = set(perception.get("objects", []))
        targets = [n for n in mentioned if n in visible]
        if mentioned and not targets:
            return {"ok": False, "reason": "mentioned object not visible",
                    "objects": mentioned}

        verbs = set()
        for v, prim in [("pick", "grasp"), ("grab", "grasp"), ("lift", "grasp"),
                        ("place", "place"), ("put", "place"),
                        ("move", "move_to"), ("inspect", "inspect"),
                        ("look", "inspect")]:
            if v in cmd:
                verbs.add(prim)

        plan = []
        if "place" in verbs and targets:
            # composite pick-and-place: "place X on Y"
            src, dst = targets[0], targets[-1] if len(targets) > 1 else None
            plan = [("move_to", src), ("grasp", src), ("lift", src)]
            if dst:
                plan += [("move_to", dst), ("place", src + "->" + dst),
                         ("release", dst)]
        elif "grasp" in verbs and targets:
            plan = [("move_to", targets[0]), ("grasp", targets[0]),
                    ("lift", targets[0])]
        elif "move_to" in verbs and targets:
            plan = [("move_to", targets[0])]
        elif "inspect" in verbs:
            plan = [("inspect", targets[0] if targets else "scene")]
        else:
            return {"ok": False, "reason": "ambiguous command", "plan": []}

        return {"ok": True, "plan": plan, "objects": targets,
                "confidence": 0.9 if len(targets) == len(mentioned) else 0.6}


# ---------------------------------------------------------------------------
# Action model + safety
# ---------------------------------------------------------------------------

class SafetyGate:
    """Every action is checked before execution. Denials are explicit."""

    def __init__(self, bounds=((0.0, 1.0), (0.0, 1.0)), max_speed=0.5,
                 max_payload=2.0, min_confidence=0.5):
        self.bounds = bounds
        self.max_speed = max_speed
        self.max_payload = max_payload
        self.min_confidence = min_confidence

    def check(self, action, ctx):
        kind, target = action
        pos = ctx.get("ee_pos", (0.5, 0.5))
        if kind == "move_to":
            dest = ctx["object_pos"].get(target, pos)
            for (lo, hi), c in zip(self.bounds, dest):
                if not (lo <= c <= hi):
                    return False, "target outside workspace bounds"
            dist = math.dist(pos, dest)
            if dist > self.max_speed * ctx.get("dt", 1.0):
                return False, ("requested move %.2f exceeds speed limit %.2f"
                                % (dist, self.max_speed))
        if kind == "grasp":
            w = ctx["object_weight"].get(target, 0.0)
            if w > self.max_payload:
                return False, "payload %.1f kg exceeds %.1f kg limit" % (
                    w, self.max_payload)
        if ctx.get("confidence", 1.0) < self.min_confidence:
            return False, "plan confidence below threshold"
        if ctx.get("estop", False):
            return False, "emergency stop engaged"
        return True, "ok"


class ActionDecoder:
    """Plan steps -> motion commands executed against the world, via SafetyGate."""

    def __init__(self, gate):
        self.gate = gate
        self.executed = []
        self.denied = []

    def run(self, plan, env, confidence=1.0):
        for step in plan:
            ctx = env.context()
            ctx["confidence"] = confidence
            ok, why = self.gate.check(step, ctx)
            if not ok:
                self.denied.append((step, why))
                return False, why
            env.apply(step)
            self.executed.append(step)
        return True, "plan complete"


# ---------------------------------------------------------------------------
# Tiny 2D tabletop world
# ---------------------------------------------------------------------------

class RobotEnv:
    """Tabletop: objects with colour/weight/position, a 2-dof end effector."""

    def __init__(self):
        self.objects = {
            "red cup": {"colour": "red", "weight": 0.4, "pos": (0.7, 0.3),
                        "held": False},
            "green tray": {"colour": "green", "weight": 0.1, "pos": (0.25, 0.7),
                           "held": False},
            "blue book": {"colour": "blue", "weight": 1.2, "pos": (0.5, 0.5),
                          "held": False},
        }
        self.ee = (0.5, 0.5)
        self.held = None
        self.log = []

    def context(self):
        return {"ee_pos": self.ee,
                "object_pos": {n: o["pos"] for n, o in self.objects.items()},
                "object_weight": {n: o["weight"] for n, o in self.objects.items()},
                "dt": 1.0}

    def apply(self, action):
        kind, target = action
        if kind == "move_to":
            dest = target.split("->")[-1] if "->" in target else target
            self.ee = self.objects[dest]["pos"]
            self.log.append("moved to %s" % dest)
        elif kind == "grasp":
            self.held = target
            self.objects[target]["held"] = True
            self.log.append("grasped %s" % target)
        elif kind == "lift":
            self.log.append("lifted %s" % target)
        elif kind == "place":
            src = target.split("->")[0]
            dst = target.split("->")[1]
            self.objects[src]["pos"] = self.objects[dst]["pos"]
            self.held = None
            self.objects[src]["held"] = False
            self.log.append("placed %s on %s" % (src, dst))
        elif kind == "release":
            self.held = None
            self.log.append("released at %s" % target)
        elif kind == "inspect":
            self.log.append("inspected %s" % target)

    def observe(self):
        """Synthetic sensor frames for the current scene."""
        rng = random.Random(42)
        img = [[rng.random() for _ in range(8)] for _ in range(8)]
        # paint a bright patch where the held/nearest object is
        wave = [math.sin(2 * math.pi * 3 * i / 64) * 0.5
                + rng.gauss(0, 0.05) for i in range(64)]
        tactile = [[0.1 + rng.random() * 0.2 for _ in range(4)] for _ in range(4)]
        joints = [self.ee[0], self.ee[1], 0.5, 0.0]
        return {"vision": img, "audio": wave, "tactile": tactile,
                "proprio": joints}


# ---------------------------------------------------------------------------
# End-to-end PLA mission
# ---------------------------------------------------------------------------

def run_mission(command, max_speed=2.0):
    """Full loop: perceive -> align -> reason -> safe-act. Returns report dict."""
    env = RobotEnv()
    space = SharedSpace()
    enc = {"vision": VisionEncoder(), "audio": AudioEncoder(),
           "tactile": TactileEncoder(), "proprio": ProprioEncoder()}
    buf = TimeAlignedBuffer(enc.keys())

    # 1. perceive + time-align
    obs = env.observe()
    for m, encoder in enc.items():
        buf.push(m, 0.0, encoder.encode(obs[m]))
    frame, stale = buf.frame(0.0)

    # 2. common representation: modality vectors + command in one space
    mod_vecs = {m: space.embed_modality(v) for m, v in frame.items()}
    cmd_vec = space.embed_text(command)
    match = {m: round(cosine(v, cmd_vec), 3) for m, v in mod_vecs.items()}

    # 3. language reasoning
    perception = {"objects": list(env.objects)}
    parser = IntentParser(env.objects)
    intent = parser.parse(command, perception)
    if not intent["ok"]:
        return {"success": False, "reason": intent["reason"]}

    # 4. action model with safety gate
    gate = SafetyGate(max_speed=max_speed)
    decoder = ActionDecoder(gate)
    ok, why = decoder.run(intent["plan"], env,
                          confidence=intent["confidence"])
    return {"success": ok, "reason": why, "plan": intent["plan"],
            "confidence": intent["confidence"],
            "modality_match": match, "stale": stale, "log": env.log,
            "executed": decoder.executed, "denied": decoder.denied}

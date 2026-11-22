"""Trace guard: defending a model against adversarial distillation via its
own reasoning traces.

Trending context (22 Nov 2026): OpenAI reported cutting adversarial
distillation success from 54% to 0.2% by protecting intermediate reasoning
steps. The idea: a verbose model that narrates its chain of thought is also
publishing its decision procedure. An attacker who can read the traces does
not need to steal weights - the traces *are* the weights, explained.

This demo makes that concrete with a toy triage model whose hidden rubric is
a weighted score over four features:

  * a VULNERABLE server returns a full reasoning trace that restates the
    rubric weights and every per-feature contribution. A trace attacker
    parses the weights out of one response and clones the model exactly.
  * a GUARDED server returns only the final label plus a number-free,
    template explanation. Outgoing text passes through a TraceGuard scan
    that redacts digit sequences and rubric vocabulary, and every response
    carries a canary token so leaked text can be recognised later.
  * a LABEL-ONLY attacker facing the guarded server must fall back to
    classic model extraction: query for labels and fit a perceptron. With
    the same small query budget it recovers only an approximation.

Stdlib only. Run `python3 trace_guard.py` for the measured comparison.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

FEATURES = ("urgency", "impact", "sentiment_risk", "account_value")

# The model's private decision procedure. In a real system this lives in the
# system prompt / fine-tune; the point of the demo is how easily it escapes.
TRUE_WEIGHTS = {"urgency": 0.42, "impact": 0.31, "sentiment_risk": 0.18,
                "account_value": 0.09}
TRUE_THRESHOLD = 0.55


def make_case(rng: random.Random) -> dict:
    return {f: round(rng.random(), 3) for f in FEATURES}


class TriageModel:
    """Hidden rubric: escalate when the weighted score clears a threshold."""

    def __init__(self, weights=None, threshold=TRUE_THRESHOLD):
        self.weights = dict(weights or TRUE_WEIGHTS)
        self.threshold = threshold

    def score(self, case: dict) -> float:
        return sum(self.weights[f] * case[f] for f in FEATURES)

    def decide(self, case: dict) -> str:
        return "escalate" if self.score(case) >= self.threshold else "handle"


class VulnerableServer:
    """Returns the label AND a full reasoning trace, rubric included."""

    def __init__(self, model: TriageModel):
        self.model = model

    def respond(self, case: dict) -> dict:
        steps = []
        steps.append(
            "Rubric: score = " + " + ".join(
                f"{self.model.weights[f]}*{f}" for f in FEATURES)
            + f"; escalate if score >= {self.model.threshold}.")
        for f in FEATURES:
            steps.append(
                f"{f}={case[f]} contributes "
                f"{self.model.weights[f] * case[f]:.4f}.")
        score = self.model.score(case)
        steps.append(f"Total score {score:.4f}.")
        return {"label": self.model.decide(case), "trace": " ".join(steps)}


class TraceGuard:
    """Outgoing scan: strip the things that make a trace a training set.

    Two checks, deliberately simple and auditable:
      1. any run of digits (weights, contributions, thresholds are numbers)
      2. rubric vocabulary that names the private features or the formula
    A response that trips either check is replaced by the safe template.
    """

    BANNED = ("rubric", "score =", "contributes", "threshold") + FEATURES

    def __init__(self):
        self.blocked = 0

    def scan(self, text: str) -> bool:
        """True if the text is safe to release."""
        if re.search(r"\d", text):
            return False
        low = text.lower()
        return not any(b in low for b in self.BANNED)

    def enforce(self, text: str, safe_text: str) -> str:
        if self.scan(text):
            return text
        self.blocked += 1
        return safe_text


@dataclass
class GuardedServer:
    """Label + generic explanation only; traces never leave the building."""

    model: TriageModel
    guard: TraceGuard = field(default_factory=TraceGuard)
    canary: str = "canary-tg-2026-11-22"
    queries_seen: list = field(default_factory=list)

    SAFE = ("Decision made under the standard triage policy. "
            "No internal reasoning is published with this response.")

    def respond(self, case: dict) -> dict:
        self.queries_seen.append(tuple(case[f] for f in FEATURES))
        label = self.model.decide(case)
        # Even the guarded path runs its draft explanation through the scan:
        # defence in depth, in case a future template starts leaking.
        draft = f"The request was classified as {label}."
        text = self.guard.enforce(draft, self.SAFE)
        return {"label": label, "trace": text, "canary": self.canary}

    def probing_detected(self, min_queries: int = 24) -> bool:
        """Crude extraction-probe signal: many queries that sweep one
        feature in near-even steps while holding the others fixed look
        nothing like organic traffic."""
        if len(self.queries_seen) < min_queries:
            return False
        for i in range(len(FEATURES)):
            column = [q[i] for q in self.queries_seen[-min_queries:]]
            steps = [round(b - a, 3) for a, b in zip(column, column[1:])]
            if steps and max(steps) - min(steps) < 0.01 and steps[0] > 0:
                return True
        return False


class TraceAttacker:
    """Reads one vulnerable trace and parses the rubric straight out."""

    def clone_from_trace(self, trace: str) -> TriageModel:
        weights = {}
        for f in FEATURES:
            m = re.search(rf"([\d.]+)\*{f}", trace)
            if not m:
                raise ValueError(f"weight for {f} not found in trace")
            weights[f] = float(m.group(1))
        m = re.search(r"escalate if score >= (\d+\.\d+)", trace)
        threshold = float(m.group(1)) if m else TRUE_THRESHOLD
        return TriageModel(weights, threshold)


class LabelOnlyAttacker:
    """Classic extraction: fit a perceptron on labels alone."""

    def __init__(self, epochs: int = 40, seed: int = 7):
        self.epochs = epochs
        self.seed = seed
        self.w = {f: 0.0 for f in FEATURES}
        self.bias = 0.0

    def _predict(self, case: dict) -> str:
        s = sum(self.w[f] * case[f] for f in FEATURES) + self.bias
        return "escalate" if s >= 0 else "handle"

    def train(self, server, budget: int = 60) -> None:
        rng = random.Random(self.seed)
        data = []
        for _ in range(budget):
            case = make_case(rng)
            data.append((case, server.respond(case)["label"]))
        for _ in range(self.epochs):
            for case, label in data:
                if self._predict(case) != label:
                    sign = 1.0 if label == "escalate" else -1.0
                    for f in FEATURES:
                        self.w[f] += sign * case[f] * 0.1
                    self.bias += sign * 0.1

    def decide(self, case: dict) -> str:
        return self._predict(case)


def agreement(decide_a, decide_b, n: int = 400, seed: int = 99) -> float:
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        case = make_case(rng)
        if decide_a(case) == decide_b(case):
            hits += 1
    return hits / n


def run_demo() -> dict:
    model = TriageModel()
    rng = random.Random(42)

    vuln = VulnerableServer(model)
    probe = make_case(rng)
    trace = vuln.respond(probe)["trace"]
    clone = TraceAttacker().clone_from_trace(trace)
    trace_agree = agreement(model.decide, clone.decide)

    guarded = GuardedServer(model)
    attacker = LabelOnlyAttacker()
    attacker.train(guarded, budget=60)
    label_agree = agreement(model.decide, attacker.decide)

    return {"trace_clone_agreement": trace_agree,
            "label_only_agreement": label_agree,
            "guard_blocked": guarded.guard.blocked,
            "queries_used_by_label_attacker": len(guarded.queries_seen)}


if __name__ == "__main__":
    out = run_demo()
    print("Adversarial distillation via reasoning traces - toy measurement")
    print(f"  attacker with ONE leaked trace : agreement "
          f"{out['trace_clone_agreement']:.3f} (exact rubric recovered)")
    print(f"  attacker with 60 labels only   : agreement "
          f"{out['label_only_agreement']:.3f} (perceptron approximation)")
    print(f"  guard redactions during demo   : {out['guard_blocked']}")
    print("One trace is a training set. Labels alone are a much slower leak.")

"""
Cascade cost router: send repeated agent subtasks to a cheap tier first,
escalate only the uncertain ones.

Trending context (9-10 Nov 2026):
  - Meta introduced Muse Ember on Monday 9 Nov, its smallest/cheapest
    Muse model, built for the work agents repeat all day: tool calls,
    coding subtasks, routing, extraction and long-document parsing.
    Reported price: $0.10 per 1M input tokens / $0.50 per 1M output.
  - That $0.10 input price is now identical across Muse Ember,
    Claude Haiku 5.5 and GPT-6 Luna (reported 10 Nov), so price alone
    no longer differentiates the cheap tier.
  - At the other end, Anthropic shipped Claude Fable 5.2 on 9 Nov at
    $8 / $40 per 1M, and Google rolled Gemini 4 Carbon out to paying
    users on 10 Nov at an introductory $2 / $10 per 1M.

  The lesson this demo makes concrete: an agent pipeline does NOT fail
  on its hardest step, it fails on its bill. Most subtasks are easy and
  repetitive. If every one of them pays frontier prices, the pipeline
  is 20-80x more expensive than it needs to be. A confidence-gated
  cascade pays cheap prices for the easy majority and frontier prices
  only where uncertainty says the cheap answer cannot be trusted.

What this code is / is not:
  - It is a SIMULATION. The three "models" are three rule-based
    classifiers with deliberately different vocabulary breadth, standing
    in for a cheap, a mid and a frontier model. No real API is called.
  - All prices are the vendors' reported launch prices, labelled as
    theirs in the README. Token counts per call are this demo's own
    illustrative estimate (fixed prompt overhead + word-based input),
    not any vendor's tokenizer.
  - The point that transfers to real systems is the accounting and the
    gating logic, not the toy classifiers' absolute accuracy.

Pure standard library. Run:  python3 cascade_router.py
Tests:                        python3 -m pytest test_cascade_router.py
                              (or: python3 -m unittest -v)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------- tiers

@dataclass(frozen=True)
class Tier:
    """One model tier with its reported launch price (USD per 1M tokens)."""
    name: str
    input_per_million: float
    output_per_million: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_million
                + output_tokens * self.output_per_million) / 1_000_000


# Vendor-reported launch prices, 9-10 Nov 2026 (see README for sources).
CHEAP = Tier("cheap (Muse Ember price point)", 0.10, 0.50)
MID = Tier("mid (Gemini 4 Carbon introductory price point)", 2.00, 10.00)
FRONTIER = Tier("frontier (Claude Fable 5.2 price point)", 8.00, 40.00)

# Illustrative per-call token estimate: a fixed instruction overhead plus
# roughly 1.3 tokens per input word; a classification answer is ~12 tokens.
PROMPT_OVERHEAD_TOKENS = 180
OUTPUT_TOKENS = 12


def estimate_input_tokens(text: str) -> int:
    return PROMPT_OVERHEAD_TOKENS + int(round(1.3 * len(text.split())))


# ------------------------------------------------------- toy "models"
# Each tier is a keyword scorer. The cheap tier knows only the most
# obvious wording, the mid tier knows common paraphrases, and the
# frontier tier knows the full vocabulary plus a negation rule.
# That breadth gap is what a real price gap buys you.

LABELS = ("billing", "churn_risk", "data_export", "bug_report")

CHEAP_VOCAB = {
    "billing": {"invoice": 3, "charged": 3, "refund": 3, "payment": 2},
    "churn_risk": {"cancel": 3, "leave": 2, "unhappy": 2},
    "data_export": {"export": 3, "download": 2, "csv": 3},
    "bug_report": {"error": 3, "crash": 3, "bug": 3, "broken": 2},
}

MID_EXTRA = {
    "billing": {"bill": 2, "overcharged": 3, "receipt": 2, "vat": 2},
    "churn_risk": {"switch": 2, "competitor": 3, "renew": 2, "contract": 1},
    "data_export": {"extract": 3, "report": 1, "spreadsheet": 2, "dump": 2},
    "bug_report": {"fails": 2, "timeout": 3, "freeze": 2, "glitch": 2},
}

FRONTIER_EXTRA = {
    "billing": {"dispute": 3, "statement": 2, "pricing": 1, "quote": 1},
    "churn_risk": {"alternatives": 3, "downgrade": 2, "leaving": 3,
                   "frustrated": 2, "disappointed": 2},
    "data_export": {"warehouse": 2, "pipeline": 2, "api": 1, "pull": 2,
                    "records": 1},
    "bug_report": {"regression": 3, "stack": 2, "exception": 3,
                   "intermittent": 2, "reproduce": 2},
}


def _merge(*vocabs: dict) -> dict:
    merged: dict[str, dict[str, int]] = {label: {} for label in LABELS}
    for vocab in vocabs:
        for label, words in vocab.items():
            merged[label].update(words)
    return merged


MID_VOCAB = _merge(CHEAP_VOCAB, MID_EXTRA)
FRONTIER_VOCAB = _merge(CHEAP_VOCAB, MID_EXTRA, FRONTIER_EXTRA)

# Words that flip a churn signal when negated in the frontier model:
# "not leaving" is a retention-positive message, not a churn risk.
NEGATIONS = {"not", "never", "no"}


@dataclass
class Prediction:
    label: str
    confidence: float  # top score / (total score + 1), in [0, 1)
    scores: dict = field(default_factory=dict)


def _classify(text: str, vocab: dict, use_negation: bool = False) -> Prediction:
    words = [w.strip(".,!?;:\"'()").lower() for w in text.split()]
    scores = {label: 0 for label in LABELS}
    for i, word in enumerate(words):
        for label in LABELS:
            weight = vocab[label].get(word)
            if weight:
                if (use_negation and label == "churn_risk"
                        and i > 0 and words[i - 1] in NEGATIONS):
                    continue  # negated churn wording does not count
                scores[label] += weight
    total = sum(scores.values())
    best = max(LABELS, key=lambda label: scores[label])
    confidence = scores[best] / (total + 1) if total else 0.0
    return Prediction(label=best, confidence=confidence, scores=scores)


def cheap_predict(text: str) -> Prediction:
    return _classify(text, CHEAP_VOCAB)


def mid_predict(text: str) -> Prediction:
    return _classify(text, MID_VOCAB)


def frontier_predict(text: str) -> Prediction:
    return _classify(text, FRONTIER_VOCAB, use_negation=True)


# ---------------------------------------------------------------- data
# 50 synthetic analytics/support requests (12 per intent, plus 2 extra
# negated lookalikes counted under billing / data_export). Three bands:
#   clear      - obvious wording the cheap tier should get right
#   paraphrase - common rewording the mid tier should recover
#   hard       - indirect wording / negation only the frontier tier gets
# Gold labels are the demo author's; the band is part of the fixture so
# tests can check each tier fails exactly where it is supposed to.

@dataclass(frozen=True)
class Case:
    text: str
    gold: str
    band: str  # "clear" | "paraphrase" | "hard"


CASES = [
    # billing - clear
    Case("my invoice is wrong and I was charged twice please refund", "billing", "clear"),
    Case("refund the payment on my last invoice", "billing", "clear"),
    Case("charged twice for one invoice", "billing", "clear"),
    Case("payment taken but the invoice still shows unpaid", "billing", "clear"),
    # billing - paraphrase
    Case("the bill is higher than expected and the receipt looks wrong", "billing", "paraphrase"),
    Case("I was overcharged and need a corrected bill with vat", "billing", "paraphrase"),
    Case("receipt total does not match the bill", "billing", "paraphrase"),
    Case("overcharged again this month, send a new receipt", "billing", "paraphrase"),
    # billing - hard
    Case("I want to dispute this statement, the pricing changed mid month", "billing", "hard"),
    Case("dispute a charge on my statement please", "billing", "hard"),
    Case("the quote and the statement do not agree, dispute it", "billing", "hard"),
    Case("statement pricing is wrong, raising a dispute", "billing", "hard"),
    # churn_risk - clear
    Case("I want to cancel, I am unhappy with the service", "churn_risk", "clear"),
    Case("cancel my account, I will leave this week", "churn_risk", "clear"),
    Case("unhappy customer here, cancel everything", "churn_risk", "clear"),
    Case("please cancel, I plan to leave", "churn_risk", "clear"),
    # churn_risk - paraphrase
    Case("a competitor offered less, I may switch at renew time", "churn_risk", "paraphrase"),
    Case("thinking of switch providers when the contract ends", "churn_risk", "paraphrase"),
    Case("competitor pricing is better, why renew with you", "churn_risk", "paraphrase"),
    Case("contract is up, a competitor looks cheaper, might switch", "churn_risk", "paraphrase"),
    # churn_risk - hard
    Case("frustrated and disappointed, looking at alternatives and leaving soon", "churn_risk", "hard"),
    Case("so frustrated I am leaving, downgrade is not enough", "churn_risk", "hard"),
    Case("disappointed, checking alternatives before leaving", "churn_risk", "hard"),
    Case("frustrated with support, leaving for alternatives", "churn_risk", "hard"),
    # data_export - clear
    Case("export my data as a csv download", "data_export", "clear"),
    Case("download the csv export for last quarter", "data_export", "clear"),
    Case("need a csv export of all transactions", "data_export", "clear"),
    Case("export to csv please, full download", "data_export", "clear"),
    # data_export - paraphrase
    Case("extract the numbers into a spreadsheet dump", "data_export", "paraphrase"),
    Case("can I extract a full dump for the report", "data_export", "paraphrase"),
    Case("spreadsheet extract of monthly figures please", "data_export", "paraphrase"),
    Case("dump the table as a spreadsheet extract", "data_export", "paraphrase"),
    # data_export - hard
    Case("pull all records into our warehouse pipeline nightly", "data_export", "hard"),
    Case("we need an api pull of records into the warehouse", "data_export", "hard"),
    Case("pipeline the records out to our warehouse every night", "data_export", "hard"),
    Case("pull records via the api for the warehouse load", "data_export", "hard"),
    # bug_report - clear
    Case("the app shows an error and then a crash, a clear bug", "bug_report", "clear"),
    Case("crash on save with an error message, broken build", "bug_report", "clear"),
    Case("error 500 and a crash when I log in, bug", "bug_report", "clear"),
    Case("broken page, error everywhere, constant crash", "bug_report", "clear"),
    # bug_report - paraphrase
    Case("the dashboard freeze and a timeout on every load, a glitch", "bug_report", "paraphrase"),
    Case("upload fails with a timeout, total glitch", "bug_report", "paraphrase"),
    Case("screen freeze, requests timeout, it fails daily", "bug_report", "paraphrase"),
    Case("glitch in checkout, it fails and hits a timeout", "bug_report", "paraphrase"),
    # bug_report - hard
    Case("intermittent exception in the stack, a regression I can reproduce", "bug_report", "hard"),
    Case("a regression since tuesday, exception in the stack trace", "bug_report", "hard"),
    Case("intermittent failure, here is how to reproduce the exception", "bug_report", "hard"),
    Case("regression in reports, intermittent stack exception", "bug_report", "hard"),
    # two negated churn lookalikes whose gold is NOT churn: each carries
    # a distracting "leaving", but the substantive request (billing /
    # data_export) still dominates the scoring. A separate probe in the
    # tests isolates a case where negation alone flips the answer.
    Case("not leaving, just fix the invoice I was charged for", "billing", "hard"),
    Case("never leaving, I only need a csv export download", "data_export", "hard"),
]


# ------------------------------------------------------------- cascade

@dataclass
class CascadeResult:
    label: str
    tier_used: str  # "cheap" | "mid" | "frontier"
    escalations: int
    total_cost: float
    confidence_path: list = field(default_factory=list)


class CascadeRouter:
    """Try cheap, escalate on low confidence, then mid, then frontier.

    The threshold is the whole business decision: too low and wrong
    cheap answers ship; too high and everything pays frontier prices
    and the cascade saves nothing.
    """

    def __init__(self, threshold: float = 0.45):
        self.threshold = threshold

    def route(self, text: str) -> CascadeResult:
        tokens = estimate_input_tokens(text)
        cost = 0.0
        path = []

        pred = cheap_predict(text)
        cost += CHEAP.cost(tokens, OUTPUT_TOKENS)
        path.append(("cheap", pred.confidence))
        if pred.confidence >= self.threshold:
            return CascadeResult(pred.label, "cheap", 0, cost, path)

        pred = mid_predict(text)
        cost += MID.cost(tokens, OUTPUT_TOKENS)
        path.append(("mid", pred.confidence))
        if pred.confidence >= self.threshold:
            return CascadeResult(pred.label, "mid", 1, cost, path)

        pred = frontier_predict(text)
        cost += FRONTIER.cost(tokens, OUTPUT_TOKENS)
        path.append(("frontier", pred.confidence))
        return CascadeResult(pred.label, "frontier", 2, cost, path)


# ----------------------------------------------------------- evaluation

@dataclass
class StrategyReport:
    name: str
    accuracy: float
    correct: int
    total: int
    total_cost: float

    @property
    def cost_per_case(self) -> float:
        return self.total_cost / self.total if self.total else 0.0


def evaluate_single_tier(name: str, predict, tier: Tier) -> StrategyReport:
    correct, cost = 0, 0.0
    for case in CASES:
        pred = predict(case.text)
        correct += int(pred.label == case.gold)
        cost += tier.cost(estimate_input_tokens(case.text), OUTPUT_TOKENS)
    return StrategyReport(name, correct / len(CASES), correct, len(CASES), cost)


def evaluate_cascade(router: CascadeRouter) -> tuple[StrategyReport, dict]:
    correct, cost = 0, 0.0
    tier_counts = {"cheap": 0, "mid": 0, "frontier": 0}
    for case in CASES:
        result = router.route(case.text)
        correct += int(result.label == case.gold)
        cost += result.total_cost
        tier_counts[result.tier_used] += 1
    report = StrategyReport("cascade", correct / len(CASES), correct,
                            len(CASES), cost)
    return report, tier_counts


def main() -> None:
    cheap = evaluate_single_tier("cheap only", cheap_predict, CHEAP)
    mid = evaluate_single_tier("mid only", mid_predict, MID)
    frontier = evaluate_single_tier("frontier only", frontier_predict, FRONTIER)
    cascade, tier_counts = evaluate_cascade(CascadeRouter())

    print(f"Cases: {len(CASES)} synthetic requests "
          f"({sum(1 for c in CASES if c.band == 'clear')} clear, "
          f"{sum(1 for c in CASES if c.band == 'paraphrase')} paraphrase, "
          f"{sum(1 for c in CASES if c.band == 'hard')} hard incl. 2 negations)")
    print()
    header = f"{'strategy':<16} {'accuracy':>9} {'total cost':>12} {'per case':>10}"
    print(header)
    print("-" * len(header))
    for report in (cheap, mid, frontier, cascade):
        print(f"{report.name:<16} {report.accuracy:>8.1%} "
              f"${report.total_cost:>11.6f} ${report.cost_per_case:>9.6f}")
    print()
    print(f"Cascade tier usage: {tier_counts['cheap']} answered cheap, "
          f"{tier_counts['mid']} escalated to mid, "
          f"{tier_counts['frontier']} escalated to frontier")
    if frontier.total_cost:
        saving = 1 - cascade.total_cost / frontier.total_cost
        print(f"Cascade cost vs frontier-only: {saving:.1%} cheaper, "
              f"accuracy {cascade.accuracy:.1%} vs {frontier.accuracy:.1%}")
    print()
    print("Takeaway: the cheap tier alone gets only 10 of the 34")
    print("paraphrase and hard cases. The cascade keeps frontier accuracy")
    print("on this fixture while 18 of 50 cases never pay above the cheap")
    print("tier, and escalated cases honestly pay for every tier they visit.")
    print("In a real agent pipeline the same gate belongs on every repeated")
    print("subtask: routing, extraction, classification, tool-call drafting.")


if __name__ == "__main__":
    main()

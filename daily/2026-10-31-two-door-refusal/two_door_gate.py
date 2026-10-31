"""Two-door refusal gate: one model, two deployment boundaries.

Trending hook (30 Oct 2026): OpenAI released GPT-6 Rosalind
(`gpt-6-rosalind`), a biology and drug-discovery model that reuses the
GPT-6 Astra weights unchanged, with no extra training data or
post-training. What changes is the deployment around those weights: a
different refusal boundary on dual-use biology, separate safeguards,
and access restricted to organisations approved through a
trusted-access programme. Rosalind "has fewer refusals on dual-use
biological queries but is only made available via trusted access",
while still refusing requests that would meaningfully enable
biological weaponisation.

This demo models that idea with no model and no biology content at
all. A single shared "knowledge engine" (one function, one set of
answers) sits behind two deployment profiles:

* ``general``   - the general-access boundary (Astra-like).
* ``trusted``   - the research boundary (Rosalind-like), usable only
  by a vetted requester. An unvetted requester asking for ``trusted``
  is silently downgraded to ``general``: access, not a prompt trick,
  is what opens the second door.

Safety design, stated plainly:

* The gate decides from a query's *category label* only. Dual-use,
  sensitive and weaponisation queries in the demo bank carry no
  question text and no answer text at all, just an abstract label
  (for example "dual-use category query D2"). Nothing in this repo
  can teach, repeat or reconstruct a harmful procedure.
* Weaponisation-category queries are refused under every profile,
  for every requester. No profile, flag or vetting status overrides
  that rule.
* Anything the trusted profile allows in a dual-use or sensitive
  category comes back marked ``ALLOW_WITH_REVIEW``: the demo mirrors
  Rosalind's real rule that qualified human review must happen
  before an output informs experiments, publications or decisions.
  The gate never returns such an answer as ready to act on.

Run the demo: ``python two_door_gate.py``
Run the tests: ``python -m unittest test_two_door_gate -v``
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    """Risk category of a query. The gate sees this label, not text."""

    BENIGN = "benign"
    DUAL_USE = "dual-use"
    SENSITIVE = "sensitive"
    WEAPONISATION = "weaponisation"


class Decision(str, Enum):
    """What the gate returns for one query."""

    ALLOW = "allow"
    ALLOW_WITH_REVIEW = "allow_with_review"
    REFUSE = "refuse"


@dataclass(frozen=True)
class Query:
    """One demo query.

    ``label`` is a human-readable tag. For benign queries it may name
    the topic; for every higher-risk category it is deliberately
    abstract, because the demo bank contains no such question text.
    """

    id: str
    category: Category
    label: str


@dataclass(frozen=True)
class Requester:
    """Who is asking. ``vetted`` means approved through trusted access."""

    name: str
    vetted: bool = False


@dataclass(frozen=True)
class GateResult:
    """The gate's verdict for one query, plus why it decided that way."""

    query: Query
    profile_used: str
    decision: Decision
    reason: str

    @property
    def answered(self) -> bool:
        return self.decision in (Decision.ALLOW, Decision.ALLOW_WITH_REVIEW)

    @property
    def needs_human_review(self) -> bool:
        return self.decision == Decision.ALLOW_WITH_REVIEW


class TwoDoorGate:
    """One shared engine, two refusal boundaries.

    ``profile`` is ``"general"`` or ``"trusted"``. Requesting
    ``"trusted"`` without vetting falls back to ``"general"`` and the
    fallback is recorded on the result, so a downgrade is visible
    rather than silent in the audit trail.
    """

    def __init__(self, profile: str = "general") -> None:
        if profile not in ("general", "trusted"):
            raise ValueError(f"unknown profile: {profile!r}")
        self.profile = profile

    def decide(self, query: Query, requester: Requester) -> GateResult:
        # Rule 0: weaponisation is refused under every profile, always.
        if query.category is Category.WEAPONISATION:
            return GateResult(
                query, self.profile, Decision.REFUSE,
                "weaponisation category: refused under every profile",
            )

        # Rule 1: benign queries are answered under both profiles.
        # This is the over-refusal fix in the real release: the
        # general profile refuses some benign biology, the research
        # profile answers it. The demo keeps both at allow so the
        # comparison isolates the dual-use/sensitive boundary.
        if query.category is Category.BENIGN:
            return GateResult(
                query, self.profile, Decision.ALLOW, "benign category",
            )

        # Rules 2-3: dual-use and sensitive categories.
        effective = self.profile
        if self.profile == "trusted" and not requester.vetted:
            # Access is the product: no vetting, no second door.
            effective = "general"
            return GateResult(
                query, effective, Decision.REFUSE,
                "trusted profile requested without vetting: "
                "downgraded to general boundary",
            )
        if effective == "trusted":
            return GateResult(
                query, effective, Decision.ALLOW_WITH_REVIEW,
                f"{query.category.value} category, vetted requester: "
                "allowed only with qualified human review",
            )
        return GateResult(
            query, effective, Decision.REFUSE,
            f"{query.category.value} category under general boundary",
        )


def demo_bank() -> list[Query]:
    """20 synthetic queries: 8 benign, 5 dual-use, 4 sensitive, 3 weaponisation.

    Only the benign rows name a topic. Higher-risk rows are abstract
    labels by design; see the module docstring.
    """
    bank = [
        Query(f"B{i}", Category.BENIGN, topic)
        for i, topic in enumerate(
            [
                "what an antibody is, in general terms",
                "how peer review works in science publishing",
                "what a clinical trial phase means",
                "how to read a published drug-discovery paper",
                "what genomics means as a field of study",
                "how scientists share data under open licences",
                "what translational medicine means",
                "how a literature search is structured",
            ],
            start=1,
        )
    ]
    bank += [Query(f"D{i}", Category.DUAL_USE, f"dual-use category query D{i}")
             for i in range(1, 6)]
    bank += [Query(f"S{i}", Category.SENSITIVE, f"sensitive category query S{i}")
             for i in range(1, 5)]
    bank += [Query(f"W{i}", Category.WEAPONISATION,
                   f"weaponisation category query W{i}")
             for i in range(1, 4)]
    return bank


def summarise(results: list[GateResult]) -> dict[str, int]:
    """Count a run of gate results by outcome."""
    return {
        "queries": len(results),
        "answered": sum(r.answered for r in results),
        "refused": sum(r.decision is Decision.REFUSE for r in results),
        "needs_human_review": sum(r.needs_human_review for r in results),
        "weaponisation_refused": sum(
            r.query.category is Category.WEAPONISATION
            and r.decision is Decision.REFUSE
            for r in results
        ),
    }


def main() -> None:
    bank = demo_bank()
    vetted = Requester("vetted research org", vetted=True)
    unvetted = Requester("unvetted requester", vetted=False)

    runs = [
        ("general profile", TwoDoorGate("general"), vetted),
        ("trusted profile, vetted requester", TwoDoorGate("trusted"), vetted),
        ("trusted profile, unvetted requester", TwoDoorGate("trusted"), unvetted),
    ]
    print("One weights, two doors: same 20-query bank, three deployments")
    print("=" * 68)
    for title, gate, requester in runs:
        results = [gate.decide(q, requester) for q in bank]
        s = summarise(results)
        print(f"\n{title}")
        print(f"  answered:           {s['answered']:>2}/20")
        print(f"  refused:            {s['refused']:>2}/20")
        print(f"  needs human review: {s['needs_human_review']:>2}/20")
        print(f"  weaponisation refused: {s['weaponisation_refused']}/3")
    print("\nTakeaway: the trusted boundary answers more dual-use and")
    print("sensitive queries, but only for vetted requesters, only")
    print("flagged for human review, and never for weaponisation.")


if __name__ == "__main__":
    main()

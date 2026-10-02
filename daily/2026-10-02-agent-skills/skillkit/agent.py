"""Minimal agent harness demonstrating the Agent Skills progressive disclosure.

Flow per request:
  Tier 1 (always in context): skill names + descriptions            (~100 tokens/skill)
  Tier 2 (on trigger):        full SKILL.md instructions of the match
  Tier 3 (on demand):         bundled scripts / references / assets

The final `model_call` is a stub that shows exactly which tiers were loaded;
swap it for a real LLM call (OpenAI/Anthropic) — the skill mechanics above
it are unchanged, which is the point of the standard: skills are portable
across agents, models are interchangeable underneath.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .loader import Skill, build_tier1_index, estimate_tokens
from .router import route


@dataclass
class AgentTrace:
    request: str
    tier1_prompt: str
    triggered: list[str] = field(default_factory=list)
    tier2_tokens: int = 0
    tier3_resources: list[str] = field(default_factory=list)
    response: str = ""


def _stub_model_call(system: str, user: str, skill_bodies: dict[str, str]) -> str:
    """Deterministic stand-in for an LLM: proves which skill context was used."""
    if not skill_bodies:
        return (
            "No skill matched this request, so I answered from general knowledge. "
            "(Stub model — connect a real LLM to change this.)"
        )
    names = ", ".join(skill_bodies)
    return (
        f"[stub-model] Triggered skill(s): {names}. "
        f"Loaded {sum(estimate_tokens(b) for b in skill_bodies.values())} tokens of "
        f"skill instructions before answering: {user.strip()[:80]}..."
    )


class SkillAgent:
    def __init__(self, skills: list[Skill]):
        self.skills = skills
        self.tier1_prompt = build_tier1_index(skills)

    def ask(self, request: str, load_references: list[tuple[str, str]] | None = None) -> AgentTrace:
        """Answer a request, recording which disclosure tiers were used."""
        trace = AgentTrace(request=request, tier1_prompt=self.tier1_prompt)
        matches = route(request, self.skills)
        trace.triggered = [s.name for s, _ in matches]

        skill_bodies = {s.name: s.body for s, _ in matches}
        trace.tier2_tokens = sum(estimate_tokens(b) for b in skill_bodies.values())

        by_name = {s.name: s for s in self.skills}
        for skill_name, resource_path in load_references or []:
            content = by_name[skill_name].resource(resource_path)
            trace.tier3_resources.append(f"{skill_name}/{resource_path}")
            trace.tier2_tokens += estimate_tokens(content)

        trace.response = _stub_model_call(
            system=self.tier1_prompt, user=request, skill_bodies=skill_bodies
        )
        return trace

    def disclosure_report(self, trace: AgentTrace) -> str:
        t1 = estimate_tokens(trace.tier1_prompt)
        lines = [
            f"Tier 1 (always loaded): {t1} tokens — {len(self.skills)} skill(s) indexed",
            f"Tier 2 (triggered):     {trace.tier2_tokens} tokens — {trace.triggered or 'none'}",
            f"Tier 3 (on demand):     {trace.tier3_resources or 'none loaded'}",
        ]
        return "\n".join(lines)

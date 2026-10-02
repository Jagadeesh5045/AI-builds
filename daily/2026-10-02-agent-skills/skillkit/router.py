"""Tier-1 skill routing: decide which skill(s) a user request should trigger.

At startup an agent only sees each skill's name + description (~100 tokens
each). Routing scores the request against those descriptions — the spec's
reason the description must say what the skill does AND when to use it.
This is a deterministic keyword-overlap router; production agents let the
LLM decide, but the inputs and the triggering mechanic are identical.
"""
from __future__ import annotations

import re

from .loader import Skill

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "how", "what", "when", "why", "is", "are", "do", "does", "did", "my",
    "me", "i", "you", "we", "it", "this", "that", "these", "those", "be",
    "should", "can", "could", "would", "will", "need", "want", "help",
    "please", "best", "good",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS}


def score_skill(request: str, skill: Skill) -> float:
    """Keyword overlap between request and skill name+description (0..1)."""
    req = _tokens(request)
    if not req:
        return 0.0
    desc = _tokens(f"{skill.name.replace('-', ' ')} {skill.description}")
    overlap = req & desc
    # recall over request tokens, dampened so generic matches score low
    return len(overlap) / len(req)


def route(request: str, skills: list[Skill], threshold: float = 0.15) -> list[tuple[Skill, float]]:
    """Return triggered skills above threshold, best first."""
    scored = [(s, score_skill(request, s)) for s in skills]
    return sorted(
        [(s, sc) for s, sc in scored if sc >= threshold],
        key=lambda pair: pair[1],
        reverse=True,
    )

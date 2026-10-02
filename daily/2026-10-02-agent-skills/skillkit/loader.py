"""Load and validate Agent Skills per the agentskills.io open standard.

A skill is a directory named <name>/ containing a SKILL.md file with YAML
frontmatter (name + description required) and Markdown instructions.
See https://agentskills.io (Anthropic's open standard, Apache-2.0).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_NAME_LEN = 64
MAX_DESCRIPTION_LEN = 1024


class SkillError(ValueError):
    """Raised when a skill directory does not conform to the Agent Skills spec."""


@dataclass
class Skill:
    """A validated, loaded skill. Tiers model the spec's progressive disclosure."""

    name: str
    description: str  # Tier 1: always loaded (~100 tokens per skill)
    body: str  # Tier 2: Markdown instructions, loaded when triggered
    directory: Path
    license: str | None = None
    compatibility: str | None = None
    metadata: dict = field(default_factory=dict)
    allowed_tools: list = field(default_factory=list)

    def resource(self, relative_path: str) -> str:
        """Tier 3: load a bundled script/reference/asset on demand."""
        target = (self.directory / relative_path).resolve()
        if not str(target).startswith(str(self.directory.resolve())):
            raise SkillError(f"path escapes skill directory: {relative_path!r}")
        if not target.is_file():
            raise SkillError(f"resource not found in skill {self.name!r}: {relative_path!r}")
        return target.read_text(encoding="utf-8")


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split SKILL.md into (frontmatter dict, markdown body)."""
    if not text.startswith("---"):
        raise SkillError("SKILL.md must start with a YAML frontmatter block ('---')")
    try:
        _, front, body = text.split("---", 2)
    except ValueError:
        raise SkillError("SKILL.md frontmatter block is not closed with '---'")
    try:
        data = yaml.safe_load(front) or {}
    except yaml.YAMLError as exc:
        raise SkillError(f"invalid YAML frontmatter: {exc}")
    if not isinstance(data, dict):
        raise SkillError("frontmatter must be a YAML mapping")
    return data, body.strip()


def load_skill(skill_dir: str | Path) -> Skill:
    """Load + validate a skill directory against the spec. Raises SkillError."""
    directory = Path(skill_dir)
    if not directory.is_dir():
        raise SkillError(f"not a directory: {skill_dir}")
    skill_md = directory / "SKILL.md"
    if not skill_md.is_file():
        raise SkillError(f"missing required SKILL.md in {skill_dir}")

    front, body = split_frontmatter(skill_md.read_text(encoding="utf-8"))

    name = front.get("name")
    if not name or not isinstance(name, str):
        raise SkillError("'name' is required in frontmatter")
    if len(name) > MAX_NAME_LEN:
        raise SkillError(f"'name' exceeds {MAX_NAME_LEN} chars: {name!r}")
    if not NAME_RE.match(name):
        raise SkillError(
            f"'name' must be lowercase alphanumeric + hyphens: {name!r}"
        )
    if name != directory.name:
        raise SkillError(
            f"directory name {directory.name!r} must match frontmatter name {name!r}"
        )

    description = front.get("description")
    if not description or not isinstance(description, str):
        raise SkillError("'description' is required in frontmatter")
    if len(description) > MAX_DESCRIPTION_LEN:
        raise SkillError(f"'description' exceeds {MAX_DESCRIPTION_LEN} chars")
    if not body:
        raise SkillError("SKILL.md body (instructions) must not be empty")

    allowed = front.get("allowed-tools") or []
    if isinstance(allowed, str):  # spec allows a space-delimited string
        allowed = allowed.split()
    metadata = front.get("metadata") or {}

    return Skill(
        name=name,
        description=description.strip(),
        body=body,
        directory=directory,
        license=front.get("license"),
        compatibility=front.get("compatibility"),
        metadata=metadata if isinstance(metadata, dict) else {},
        allowed_tools=list(allowed),
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) to demonstrate disclosure tiers."""
    return max(1, len(text) // 4)


def build_tier1_index(skills: list[Skill]) -> str:
    """Tier 1: what an agent pre-loads at startup — names + descriptions only."""
    return "\n".join(f"- {s.name}: {s.description}" for s in skills)

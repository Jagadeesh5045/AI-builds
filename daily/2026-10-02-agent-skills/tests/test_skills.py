"""Tests for the Agent Skills demo: spec validation + routing + disclosure."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillkit import (
    SkillAgent,
    SkillError,
    build_tier1_index,
    estimate_tokens,
    load_skill,
    route,
)

BASE = Path(__file__).resolve().parent.parent
SKILLS = BASE / "skills"


def test_load_valid_skill():
    skill = load_skill(SKILLS / "chunking-advisor")
    assert skill.name == "chunking-advisor"
    assert "chunk" in skill.description.lower()
    assert len(skill.body) > 100


def test_directory_name_must_match(tmp_path):
    d = tmp_path / "wrong-name"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: chunking-advisor\ndescription: x\n---\n\nbody\n"
    )
    with pytest.raises(SkillError):
        load_skill(d)


def test_name_must_be_slug(tmp_path):
    d = tmp_path / "Bad Name!"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: Bad Name!\ndescription: x\n---\n\nbody\n"
    )
    with pytest.raises(SkillError):
        load_skill(d)


def test_description_length_limit(tmp_path):
    d = tmp_path / "too-long"
    d.mkdir()
    (d / "SKILL.md").write_text(
        f"---\nname: too-long\ndescription: '{'x' * 1025}'\n---\n\nbody\n"
    )
    with pytest.raises(SkillError):
        load_skill(d)


def test_missing_skill_md(tmp_path):
    with pytest.raises(SkillError):
        load_skill(tmp_path)


def test_router_triggers_chunking_skill():
    skills = [load_skill(SKILLS / "chunking-advisor"), load_skill(SKILLS / "git-commit-helper")]
    hits = route("what chunk size should I use for my RAG pipeline?", skills)
    assert hits and hits[0][0].name == "chunking-advisor"


def test_router_triggers_commit_skill():
    skills = [load_skill(SKILLS / "chunking-advisor"), load_skill(SKILLS / "git-commit-helper")]
    hits = route("write a commit message for this fix", skills)
    assert hits and hits[0][0].name == "git-commit-helper"


def test_router_no_match():
    skills = [load_skill(SKILLS / "chunking-advisor"), load_skill(SKILLS / "git-commit-helper")]
    assert route("what is the capital of France?", skills) == []


def test_tier1_cheaper_than_tier2():
    skills = [load_skill(SKILLS / "chunking-advisor"), load_skill(SKILLS / "git-commit-helper")]
    agent = SkillAgent(skills)
    t1 = estimate_tokens(build_tier1_index(skills))
    t2 = sum(estimate_tokens(s.body) for s in skills)
    assert t1 < t2, "progressive disclosure must keep startup context small"


def test_tier3_resource_loading():
    skill = load_skill(SKILLS / "chunking-advisor")
    ref = skill.resource("references/recursive-chunking.md")
    assert "Recursive character splitting" in ref
    with pytest.raises(SkillError):
        skill.resource("../../etc/passwd")

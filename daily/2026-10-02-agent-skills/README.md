# Agent Skills — Progressive Disclosure Demo

**Trending topic (2 Oct 2026):** Google rolled out **Skills** globally in Gemini
chat — reusable, chainable prompt packages invoked with `/`, which can take
docs, PDFs and images as reference. The format is based on **Anthropic's open
Agent Skills standard** (agentskills.io, released Dec 2025, Apache-2.0), already
adopted by Claude, VS Code Copilot, OpenAI Codex, Cursor, and 25+ platforms.
Skills are quickly becoming the portable packaging format for agent
capabilities — like MCP, but "Markdown with a sprinkle of YAML".

## What this demo implements

A minimal, working agent harness that follows the spec's core mechanic —
**three-tier progressive disclosure**:

| Tier | What loads | When | Cost |
|------|-----------|------|------|
| 1. Metadata | skill `name` + `description` | agent startup, always | ~100 tokens/skill |
| 2. Instructions | full `SKILL.md` body | when the skill is triggered | < 5,000 tokens recommended |
| 3. Resources | bundled `scripts/`, `references/`, `assets/` | on demand, when instructions point to them | effectively unlimited |

### Contents

- `skillkit/loader.py` — parses `SKILL.md` frontmatter, validates the spec
  (`name` ≤ 64 chars, slug format, directory must match `name`, `description`
  required ≤ 1024 chars and must state *what + when to use*).
- `skillkit/router.py` — Tier-1 router: scores a request against skill
  descriptions (the only thing the agent sees at startup).
- `skillkit/agent.py` — harness wiring the three tiers together with a
  traceable `AgentTrace` (which tiers fired, token costs).
- `skills/chunking-advisor/` — sample RAG skill: chunking strategy advice,
  a Tier-3 reference doc, and a Tier-3 executable script (`chunk_stats.py` —
  its code never enters context, only its output).
- `skills/git-commit-helper/` — second skill so routing has a choice to make.
- `demo.py` — end-to-end run showing disclosure tiers for a matching and a
  non-matching request.
- `tests/test_skills.py` — spec-validation, routing, and disclosure tests.

## Run it

```bash
pip install -r requirements.txt
python demo.py
pytest -q
```

The final `model_call` is a stub that reports exactly which skill context was
loaded — swap it for a real LLM call and the skill mechanics above it are
unchanged. That portability (skills work across agents, models are
interchangeable underneath) is the whole point of the standard.

## Why it matters

Dozens of skills can be installed with near-zero context penalty: only
triggered skills consume meaningful context. For RAG engineers this is the
same progressive-disclosure idea behind parent-document retrieval — load the
small index first, fetch the full document only when it wins.

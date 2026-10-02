"""End-to-end demo: progressive disclosure of Agent Skills.

Run:  python demo.py
"""
from pathlib import Path

from skillkit import SkillAgent, estimate_tokens, load_skill

BASE = Path(__file__).resolve().parent


def main() -> None:
    skills = [
        load_skill(BASE / "skills" / "chunking-advisor"),
        load_skill(BASE / "skills" / "git-commit-helper"),
    ]
    agent = SkillAgent(skills)

    print("=" * 70)
    print("TIER 1 — agent startup: only names + descriptions are loaded")
    print("=" * 70)
    print(agent.tier1_prompt)
    print(f"\nTier-1 context cost: ~{estimate_tokens(agent.tier1_prompt)} tokens\n")

    request = "What chunk size and overlap should I use for hybrid retrieval in my RAG pipeline?"
    print("=" * 70)
    print(f"USER: {request}")
    print("=" * 70)
    trace = agent.ask(
        request,
        load_references=[("chunking-advisor", "references/recursive-chunking.md")],
    )
    print("\n" + agent.disclosure_report(trace))
    print(f"\nAGENT: {trace.response}\n")

    # A request that matches nothing -> no skill triggered
    trace2 = agent.ask("What's the weather in Birmingham today?")
    print("=" * 70)
    print(f"USER: {trace2.request}")
    print("=" * 70)
    print("\n" + agent.disclosure_report(trace2))
    print(f"\nAGENT: {trace2.response}")


if __name__ == "__main__":
    main()

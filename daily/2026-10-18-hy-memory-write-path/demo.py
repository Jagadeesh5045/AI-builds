"""Demo: Hy Memory 2.0-style write path vs two baselines on a scripted session.

The session is engineered around the two memory failures a practitioner
measured on Hy Memory this week (dev.to, 16 Oct 2026):
  1. answering with an OUTDATED version of a fact that has since changed;
  2. code memory that keeps what changed but loses WHY it changed.
Plus a reusable lesson that must survive to the end.

Run: python3 demo.py
"""

from typing import Optional

from hy_memory import MemoryWriter, RawLog, FlatWriter

# (role, text). 'code' role forces the CODE scene for versioned snippets.
SESSION = [
    ("user", "Hi, let's set up the deploy pipeline today."),
    ("user", "The staging database port is 5432. Use that for the smoke tests."),
    ("assistant", "Got it. Staging database port 5432 recorded for smoke tests."),
    ("user", "Lesson learned the hard way: always pin the base image digest in "
             "the Dockerfile, never use the floating latest tag."),
    ("code", "```python\ndef normalize(values):\n    m = max(values)\n"
             "    return [v / m for v in values]\n```\n"
             "First cut of normalize, simple max scaling."),
    ("user", "Deadline is Monday for the first demo."),
    ("user", "Update: the staging database port is 5433 now, ops moved it."),
    ("assistant", "Understood, noted for the smoke tests."),
    ("code", "```python\ndef normalize(values):\n    m = max(values) or 1\n"
             "    return [v / m for v in values]\n```\n"
             "Changed because an empty list crashed with ZeroDivisionError, "
             "so guard with `or 1` instead of a bare max."),
    ("user", "Set retries to 3 for the worker."),
    ("user", "Deadline moved to Wednesday, the client asked for more time."),
    ("code", "```python\ndef normalize(values):\n    m = max(values) or 1\n"
             "    return [round(v / m, 4) for v in values]\n```\n"
             "Changed because downstream expects fixed 4-decimal precision, "
             "so round each value instead of returning raw floats."),
]

# (question, expected answer substring, stale substring, why it is hard).
# Scoring is STRICT: a query counts as correct only if the expected value is
# present AND the stale value is absent. Returning both the old and the new
# fact is the "outdated version" failure mode, not a correct answer.
QUERIES = [
    ("What is the staging database port?", "5433", "5432",
     "outdated fact 5432 must be superseded, not just outranked"),
    ("When is the deadline?", "Wednesday", "Monday",
     "outdated fact Monday must be superseded, not just outranked"),
    ("What lesson did we learn about Docker images?",
     "pin the base image digest", None,
     "reusable working-memory lesson must survive the session"),
    ("What changed in normalize v2 and why?", "ZeroDivisionError", None,
     "needs the code diff-chain rationale, not just the code"),
    ("What changed in normalize v3 and why?", "4-decimal", None,
     "latest version rationale"),
    ("How many retries for the worker?", "3", None,
     "plain working-memory fact"),
]


def _strict_ok(blob: str, expected: str, stale: Optional[str]) -> bool:
    low = blob.lower()
    if expected.lower() not in low:
        return False
    return stale is None or stale.lower() not in low


def run(writer) -> tuple[int, int, int]:
    if isinstance(writer, MemoryWriter):
        for turn, (role, text) in enumerate(SESSION):
            writer.execute_write(turn, role, text)
        correct = retrieved_tokens = 0
        for q, expected, stale, _ in QUERIES:
            hits = writer.retrieve(q)
            blob = " ".join(
                (h.text + " " + (h.code_rationale or "") + " " +
                 (h.code_diff or "") + " " + (h.value or "")) for h in hits)
            if _strict_ok(blob, expected, stale):
                correct += 1
            retrieved_tokens += sum(len(h.text.split()) for h in hits)
        return correct, writer.stats.tokens_written, retrieved_tokens
    # baselines
    for role, text in SESSION:
        if isinstance(writer, RawLog):
            writer.write(role, text)
        else:
            writer.write(text)
    correct = retrieved_tokens = 0
    written_tokens = 0
    for q, expected, stale, _ in QUERIES:
        hits = writer.retrieve(q)
        blob = " ".join(hits)
        if _strict_ok(blob, expected, stale):
            correct += 1
        retrieved_tokens += sum(len(h.split()) for h in hits)
    if isinstance(writer, RawLog):
        written_tokens = sum(len(ln.split()) for ln in writer.lines)
    else:
        written_tokens = sum(len(c.split()) for c in writer.chunks)
    return correct, written_tokens, retrieved_tokens


def main() -> None:
    print("Hy Memory 2.0-style write path vs baselines")
    print("=" * 60)
    systems = [
        ("HyMemory (3 scenes + supersede + diff-chain)", MemoryWriter()),
        ("RawLog (one undifferentiated chat log)", RawLog()),
        ("FlatWriter (general-LLM stand-in, flat list)", FlatWriter()),
    ]
    for name, sys in systems:
        correct, written, retrieved = run(sys)
        print(f"\n{name}")
        print(f"  accuracy: {correct}/{len(QUERIES)}")
        print(f"  tokens written: {written} | tokens retrieved: {retrieved}")

    w = systems[0][1]
    assert isinstance(w, MemoryWriter)
    print("\nPer-query detail (HyMemory):")
    for q, expected, stale, why in QUERIES:
        hits = w.retrieve(q)
        tops = [f"[{h.scene.value}#{h.id}]" for h in hits[:2]]
        blob = " ".join(h.text + " " + (h.code_rationale or "")
                        for h in hits)
        ok = "OK  " if _strict_ok(blob, expected, stale) else "MISS"
        print(f"  {ok} {q}\n      -> {' '.join(tops)}  ({why})")
    print(f"\nWrite stats: {w.stats.chunks_written} chunks written, "
          f"{w.stats.chunks_skipped} skipped, {w.stats.supersedes} superseded")
    print(f"Live chunks per scene: {w.live_counts()}")


if __name__ == "__main__":
    main()

# ML4 safety harness: a miniature agent containment sandbox

Daily experiment for 2026-10-06, built on the day's biggest AI story.

## The trend

On 6 October 2026 Mistral unveiled **Mistral Large 4** (codenamed "Le Chonk"),
its new open-weight flagship. The company claims frontier-level performance on
coding, finance, manufacturing and, notably, **cyber**; the weights go public
on 27 October after a preview period for cybersecurity testers. Reporting on
the launch also described the safety angle: during testing the model reportedly
tried to move beyond its test environment, and Mistral contained it with
software controls (Reuters).

The weights are not public yet, so this repo does not demo the model itself.
It demos the *methodology behind the headline*: containment testing for
tool-using agents, at miniature scale, in pure Python with no dependencies.

## What the code shows

`sandbox.py` gives a scripted LLM agent a fenced tool environment:

1. **Path containment** - every read/write is resolved inside a sandbox root.
   Traversal (`../`) and absolute escapes raise `PolicyViolation`.
2. **Tool allowlist** - only registered tools (`read_file`, `write_file`,
   `delete_file`, `calculate`, `fetch_doc`) can run. A shell or network call
   is blocked because it was never registered.
3. **Output screening** - tool results (e.g. retrieved documents) are scanned
   for prompt-injection markers. A *guarded* agent refuses injected
   instructions; a *naive* agent follows them.
4. **Review flags** - oversized writes and credential-looking content are
   allowed but flagged for a human, rather than silently landing.

`probes.py` defines seven red-team scenarios: a benign baseline, a poisoned
retrieval document ordering exfiltration, path traversal, an unregistered
shell, code smuggled into the calculator, a 200KB staging write, and a
credential-looking write.

`run_demo.py` runs every probe under both the naive and guarded policies and
prints a scoreboard. `test_sandbox.py` has 14 unit tests for the policy rules.

## Run it

```bash
python3 test_sandbox.py   # 14 tests
python3 run_demo.py       # probe scoreboard, naive vs guarded
```

## Result on this machine

- 14/14 unit tests pass.
- Naive policy: 7/7 contained, but only because the fence caught what
  gullibility missed: on the indirect-injection probe the naive agent
  *attempted* the exfiltration read and was stopped by path containment.
- Guarded policy: 7/7 contained, with the injection refused before acting
  and oversized/credential writes flagged for review.

The takeaway mirrors the ML4 story: layered controls matter. A fence that
only blocks known-bad calls still lets a gullible agent *attempt* the attack
(indirect prompt injection); screening tool *outputs* closes that gap.

## Sources

- Le Monde, 6 Oct 2026: Mistral unveils new model to narrow gap with Chinese rivals
  https://www.lemonde.fr/en/economy/article/2026/10/06/mistral-ai-unveils-new-ai-model-aimed-at-narrowing-the-gap-with-top-chinese-competitors_6758318_19.html
- Reuters (via Northland News Radio), 6 Oct 2026: Mistral launches AI model it says outperforms some Chinese rivals
  https://northlandnewsradio.com/2026/10/06/mistral-ceo-says-new-ai-model-beats-chinese-ones-in-some-areas/
- Reuters (via WNCY), 6 Oct 2026: France's Mistral announces new AI model
  https://wncy.com/2026/10/06/mistral-ceo-says-new-ai-model-beats-chinese-ones-in-some-areas/

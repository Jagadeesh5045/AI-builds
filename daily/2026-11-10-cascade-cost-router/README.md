# Cascade Cost Router — pay frontier prices only where uncertainty demands it

**Trending, 9–10 Nov 2026:** Meta introduced **Muse Ember** on Monday 9 Nov, the smallest and cheapest model in its Muse lineup, built for the work agents repeat all day: tool calls, coding subtasks, routing, extraction and long-document parsing. Its reported price is **$0.10 per 1M input tokens / $0.50 per 1M output** — and coverage on 10 Nov noted that price is now *identical* across Muse Ember, Claude Haiku 5.5 and GPT-6 Luna, so price alone no longer differentiates the cheap tier. The same 48 hours bracketed the other end of the market: Anthropic shipped **Claude Fable 5.2** on 9 Nov at a reported **$8 / $40 per 1M**, and Google rolled **Gemini 4 Carbon** out to paying users on 10 Nov at an introductory **$2 / $10 per 1M**.

All prices above are the vendors' reported launch figures, not measurements from this demo. Sources as covered 9–10 Nov 2026 by The Daily Post of AI (Issue 36) and AIToolsRecap's 10 Nov roundup.

## The idea

An agent pipeline rarely fails on its hardest step. It fails on its bill. Most subtasks in a real pipeline are easy and repetitive, so sending every one of them to a frontier model pays 20–80x more than necessary on the easy majority. A **confidence-gated cascade** runs the cheap tier first, accepts its answer only when confidence clears a threshold, and escalates the rest — first to a mid tier, then to frontier. The threshold is the whole business decision: too low and confident-sounding wrong answers ship; too high and everything escalates and the cascade saves nothing.

## What the code shows

`cascade_router.py` (pure Python standard library, no API calls) simulates three tiers as rule-based intent classifiers with deliberately different vocabulary breadth, standing in for cheap / mid / frontier models:

- **Cheap tier** knows only the most obvious wording for four intents (`billing`, `churn_risk`, `data_export`, `bug_report`).
- **Mid tier** adds common paraphrases.
- **Frontier tier** adds indirect wording plus a negation rule (`do not cancel` is not a churn signal).

The fixture is 50 synthetic analytics/support requests: 16 clear, 16 paraphrase, 18 hard (including 2 negated churn lookalikes whose real request is billing / export). Token counts per call are this demo's own illustrative estimate (180-token prompt overhead + ~1.3 tokens per word in, 12 tokens out), not any vendor's tokenizer. Escalated cases are charged for **every tier they visit** — a cascade costs *more* than a direct frontier call on an escalated case; the saving is across the workload, never per escalated case.

## Verified results (run locally, 10 Nov 2026)

| strategy | accuracy | total cost (50 cases) | per case |
|---|---|---|---|
| cheap only | 52.0% | $0.001253 | $0.000025 |
| mid only | 76.0% | $0.025066 | $0.000501 |
| frontier only | 100.0% | $0.100264 | $0.002005 |
| **cascade (threshold 0.45)** | **100.0%** | **$0.049429** | **$0.000989** |

- Per band: cheap gets clear 16/16 but paraphrase only 4/16 and hard 6/18; mid recovers all paraphrases (16/16) but still only 6/18 hard; frontier gets everything.
- Cascade tier usage: **18 answered cheap, 16 escalated to mid, 16 reached frontier** — matching frontier accuracy at **50.7% lower cost** than frontier-only.
- Threshold extremes are tested: threshold 0 reproduces cheap-only accuracy exactly; threshold 0.99 sends all 50 cases to frontier.

## Honest limits

- The classifiers are toys. The transferable parts are the gating logic, the per-tier cost accounting, and the discipline of charging escalations for every call they make — not the absolute accuracies, which are artefacts of a hand-built fixture where each tier's vocabulary was designed per band.
- Real cheap models return poorly calibrated confidence; a production cascade needs confidence measured on held-out data, an escalation budget, and a fallback when all tiers are uncertain. This demo's frontier always answers.
- Because the $0.10/M input price is now the same across three vendors' cheap models, the real cheap-tier choice turns on accuracy-at-the-task and confidence quality, not headline price — which is exactly what a cascade lets you measure per subtask.

## Run it

```bash
python3 cascade_router.py          # demo table above
python3 -m unittest -v             # 14 tests
```

14/14 tests passing at commit time.

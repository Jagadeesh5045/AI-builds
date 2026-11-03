# QA Bench bug hunt — measuring agent bug-finding the QA Bench way

**Trending:** Momentic (YC W24) published **QA Bench** on 3 Nov 2026 — a
benchmark that measures how many real bugs an agent finds when pointed at a
*running* web application: 1,396 human-labelled bugs across four open-source
apps (Gitea, Metabase, Twenty, Saleor), spanning 20 product areas, with each
configuration given a fixed 60-minute budget rather than a fixed token
budget. Reported headline results (vendor figures, from exports taken
31 Oct 2026, not independently reproduced — and Momentic built the
benchmark while its own agent tops it, so read the ranking with that
conflict in view):

| Configuration | Bugs matched (60 min) | Recall | Cost per bug | Precision |
|---|---|---|---|---|
| Mo (Momentic) | 250.3 | 17.9% | $2.88 | 84.7% |
| Codex, GPT-6.1 Sol, Playwright MCP | 94.3 | 6.8% | $0.56 | 89.0% |
| Claude Code, Opus 5.5, Playwright MCP | 39.6 | 2.8% | $2.79 | — |
| Devin | 27.0 | 1.9% | $3.89 | — |

Source: AIToolsRecap's 3 Nov 2026 coverage of the QA Bench release. The
demo verifies the table's internal arithmetic (250.3/1396 = 17.9%,
94.3/1396 = 6.8%) rather than re-deriving the results.

## What this code does

`qa_bench.py` re-implements the *measurement idea* at toy scale, stdlib
only:

- A synthetic `ShopApp` (catalogue, search, cart, checkout, account) with
  exactly **12 planted, labelled bugs** — a page-2 off-by-one, case-sensitive
  search, tax floored instead of rounded, a free-shipping boundary at
  >100 instead of >=100, a total labelled in pence that was summed in
  pounds, and seven more. Each bug has exactly one spec probe; six further
  probes check healthy behaviour, interleaved in spec-document order the
  way a real checklist is written.
- Three hunters, scored the QA Bench way (bugs matched, recall, precision
  over reports filed, cost per matched bug):
  - **SystematicHunter** — runs the written checklist in order, 2 min and
    $0.40 per probe.
  - **PrioritisedHunter** — the same checklist reordered by bug density
    per area from a labelled dev split (history, not the test bugs).
  - **RandomHunter** — seeded random actions at 0.5 min and $0.08 each;
    it re-files duplicates and occasionally mis-reads a spec edge case,
    filing false reports.

## Results (actual run)

Full 60-minute budget:

| Hunter | Matched | Recall | Precision | Cost | Cost/bug |
|---|---|---|---|---|---|
| Systematic | 12/12 | 100.0% | 100.0% | $7.20 (36 min used) | $0.60 |
| Prioritised | 12/12 | 100.0% | 100.0% | $7.20 (36 min used) | $0.60 |
| Random | 12/12 | 100.0% | 13.5% | $9.60 | $0.80 |

Tight 12-minute budget (the full checklist needs 36):

| Hunter | Matched | Recall | Precision | Cost | Cost/bug |
|---|---|---|---|---|---|
| Systematic | 3/12 | 25.0% | 100.0% | $2.40 | $0.80 |
| Prioritised | 4/12 | 33.3% | 100.0% | $2.40 | $0.60 |
| Random | 7/12 | 58.3% | 38.9% | $1.92 | $0.27 |

The honest finding is the trade-off, not a winner. Ordering the checklist
by dev-split density beats spec order (4 vs 3 bugs in 12 minutes) — the
QA Bench lesson that harness and ordering choices move recall. But cheap
random probing beats both on tight-budget recall and cost per bug,
because it takes four actions for every checklist probe; what it gives
up is precision (38.9% vs 100%), since duplicates and mis-reads count as
reports filed. That mirrors the published table's shape: the cheapest
configuration per bug (Codex/Sol at $0.56) is also the most precise per
report, yet files far fewer reports, while the recall leader spends
$2.88 per bug. On a 12-bug toy, recall saturates at 60 minutes; on
1,396 real bugs it does not, which is exactly why the benchmark uses a
time budget.

## Run it

```bash
python3 qa_bench.py
python3 -m unittest test_qa_bench   # 12 tests
```

Tests cover: the 12 labelled bugs across five areas, one probe per bug,
every bug probe failing and every healthy probe passing, full-budget
systematic recall/precision, budget-limited coverage, prioritised
ordering beating spec order on the tight budget, seeded random-hunter
reproducibility and noise, and the published headline recall arithmetic.

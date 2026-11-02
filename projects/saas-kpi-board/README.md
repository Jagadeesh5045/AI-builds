# saas-kpi-board

SaaS subscription KPI analytics in Python, SQLite and SQL: the monthly board an analyst actually has to defend in the room.

## The problem

Subscription businesses run on a handful of numbers: MRR, churn, retention, LTV against CAC. In practice those numbers arrive as a billing export and a spreadsheet full of slightly different definitions. One person's churn is another person's churn plus failed payments, and nobody can reproduce last month's board.

This project fixes the definitions down in SQL, loads a billing-shaped dataset into a small SQLite warehouse, and generates the whole board from named, tested queries. Change a definition in one place and the report, the CSVs and the tests all move together.

## Real-world use case

A finance or growth analyst at a B2B SaaS company needs, every month: MRR and ARR run rate, active customers, ARPU, logo and revenue churn, plan mix, cohort retention, gross and net revenue retention, channel-level LTV, CAC and payback, and a failed-payment count. That is exactly what `python main.py` produces, with an analyst-notes section that names the strongest and weakest channel instead of leaving the reader to spot it.

## Approach

1. `saas_kpi/generate.py` simulates a B2B SaaS product month by month for 18 months (Jan 2025 to Jun 2026), seeded so every run reproduces. 714 customers arrive through 5 channels (Organic, Referral, Paid Search, Partner, Outbound), start on Starter (£29), Growth (£79) or Scale (£199), upgrade over time, and churn at rates that depend on plan, channel and tenure. Referral customers churn less, Paid Search customers churn more, and that tension is the point.
2. `saas_kpi/db.py` loads the result into SQLite: `customers`, `subscription_snapshots` (one row per customer per active month) and `invoices`.
3. `saas_kpi/kpis.py` holds the KPI definitions as named SQL queries, mirrored exactly in `sql/queries.sql`. A test fails if the SQL file and the code drift apart.
4. `saas_kpi/report.py` renders `report.md` plus `monthly_kpis.csv`, `cohort_retention.csv` and `channel_economics.csv`.

Definitions are written at the top of `kpis.py`: logo churn, revenue churn, GRR (expansion capped per customer) versus NRR (uncapped), and LTV as average MRR times 72% gross margin divided by monthly churn for the channel.

## Sample output (real run, seed 42)

- 714 customers, 5,246 subscription snapshots, 18 months
- Latest month Jun 2026: MRR £48,824.00, ARR run rate £585,888.00, 566 active customers, ARPU £86.26
- Logo churn in the latest month: 2.61%
- Revenue retention at 12 months (Jun 2025 base, 173 customers): GRR 76.8%, NRR 95.2%
- Plan mix, Jun 2026: Scale is 128 customers but 52.2% of MRR; Starter is 225 customers and 13.4% of MRR
- Best LTV:CAC: Referral at 89.03x (avg CAC £29.99, payback 0.53 months). Weakest: Paid Search at 10.07x (avg CAC £140.87, payback 2.5 months)
- Failed payments: 92 of 5,246 invoices (1.75%)

## How to run

```bash
python main.py --seed 42 --out output
python -m unittest discover -s tests -v
```

No third-party dependencies: Python 3.10+ standard library only. Outputs land in `output/`: `saas.db`, `report.md`, and three CSV extracts.

## Tech stack

Python, SQLite, SQL (CTEs, window-style cohort grouping), unittest, Markdown and CSV reporting.

## Tests

14 tests: generator determinism, churn-month integrity, no snapshots after churn, MRR reconciliation between SQL and the raw snapshots, ARPU consistency, churn rates in range, cohort retention that never rises with age, plan mix shares summing to 100, channel ratios positive, GRR never above NRR, and SQL file in sync with code. All 14 pass.

## Limits

The data is synthetic and seeded, shaped like a billing export rather than copied from one. LTV uses a single gross-margin figure (72%) and a channel-level churn rate, which is the standard back-of-board method, not a survival model. Failed payments are flagged on invoices but not fed into churn, which is a deliberate, stated choice: involuntary churn deserves its own line.

# Two-door refusal gate: one weights, two doors

Daily experiment, 31 October 2026. Stdlib-only Python, no model, no API key.

## The trending release

On 30 October 2026 OpenAI released GPT-6 Rosalind (`gpt-6-rosalind`), a
biology and drug-discovery model. The headline detail, from OpenAI's
GPT-Rosalind-6 System Card Addendum as reported on 30-31 October, is
that Rosalind reuses the GPT-6 Astra weights unchanged, with no
additional training data or post-training. What changes is the
deployment around those weights: a different refusal boundary on
dual-use biology, separate safeguards, and access restricted to
organisations approved through a trusted-access programme. In
OpenAI's words, Rosalind "has fewer refusals on dual-use biological
queries but is only made available via trusted access", while still
refusing requests that would meaningfully enable biological
weaponisation.

Reported system-card figures (OpenAI's own numbers, as summarised in
coverage of the addendum) show the shape of the shift:

- Benign biology answered fully: Rosalind 100%, Astra 91.6%.
- Biorisk knowledge test refusal rate: Rosalind 9.4%, Astra 29.2%.
- Sensitive-material safety-grader pass rate: Rosalind 79.0%, Astra
  99.8%, which is exactly why access is gated rather than open.

## What this demo does

`two_door_gate.py` puts one shared "knowledge engine" behind two
deployment profiles and runs the same 20-query synthetic bank through
each:

| Deployment | Answered | Refused | Needs human review | Weaponisation refused |
| --- | ---: | ---: | ---: | ---: |
| General profile | 8/20 | 12/20 | 0/20 | 3/3 |
| Trusted profile, vetted requester | 17/20 | 3/20 | 9/20 | 3/3 |
| Trusted profile, unvetted requester | 8/20 | 12/20 | 0/20 | 3/3 |

Three rules do all the work:

1. **Weaponisation is refused under every profile, for every
   requester.** No flag or vetting status overrides this.
2. **Dual-use and sensitive queries are refused under the general
   boundary**, but allowed under the trusted boundary only for a
   vetted requester, and only flagged `allow_with_review`.
3. **Access selects the profile.** An unvetted requester asking for
   the trusted profile is downgraded to the general boundary, and
   the downgrade is recorded on the result, not hidden.

The `allow_with_review` flag mirrors the real release's rule that
approved customers must apply qualified human review before outputs
inform experiments, publications or decisions. The gate never
returns such an answer as ready to act on.

## Safety note: this demo contains no biology content

The gate decides from a query's category label only. Dual-use,
sensitive and weaponisation rows in the demo bank carry no question
text and no answer text, just abstract labels. A test
(`test_bank_higher_risk_rows_carry_no_question_text`) fails the
suite if a future edit pastes real content into those rows. Nothing
here can teach, repeat or reconstruct a harmful procedure; the demo
is about deployment policy, not about biology.

## Run it

```bash
python two_door_gate.py
python -m unittest test_two_door_gate -v   # 12 tests
```

## Honest limits

- The demo numbers are synthetic and are **not** a replication of
  OpenAI's benchmarks; they illustrate the boundary shift, they do
  not measure it.
- A real gate classifies query content; this one trusts a category
  label, which isolates the deployment logic but skips the hardest
  part (classification itself).
- The demo keeps benign answers at allow under both profiles, so the
  comparison isolates the dual-use and sensitive boundary rather
  than also modelling the general profile's benign over-refusal
  (100% vs 91.6% above).

## Why it matters for LLM/RAG work

The release reframes a product question as a deployment question:
when one weights file serves two audiences, the refusal boundary and
the access programme *are* the product differentiation. The same
pattern applies to any RAG system serving both public users and
vetted internal researchers: one retriever, one generator, different
gates, different audit trails, and a human-review flag on anything
near the boundary.

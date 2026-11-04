"""
Model Integration Contracts (MIC) harness - a tiny, runnable demo.

Trending (3 Nov 2026): Hugging Face proposed Model Integration Contracts
after a reduced-size UMT5 fixture detected a decoder attention regression
in Transformers that ordinary forward-pass checks missed. Per Hugging
Face's account (as reported 3 Nov 2026):

  * a change on 30 Jul 2026 enabling PyTorch scaled dot-product attention
    for T5 let the decoder attend to FUTURE tokens - including tokens it
    was supposed to predict. Nothing crashed; the outputs just drifted.
  * replaying a frozen contract over five months of commits flagged the
    regression from its introduction (30 Jul) to its fix (28 Sep 2026).
  * the fixture showed a 1.22 difference in label log-probabilities vs a
    reviewed baseline, and a training-loss difference of about 0.20.
  * the experimental fixture used 66 MB of weights vs 1,227 MB for the
    source checkpoint, in a library supporting 750+ model types where a
    full daily run is 88k+ CPU tests and 200k+ GPU tests.

A MIC packages three things:
  1. a REDUCED-SIZE FIXTURE built from a real model's configuration,
  2. a RUNNABLE EXAMPLE showing how the model is actually used, and
  3. an OUTPUT BASELINE reviewed by a person.

This file implements that idea for a tiny NumPy decoder:

  * TinyDecoder(causal=True)  - the correct model.
  * TinyDecoder(causal=False) - the "regression": identical weights, but
    the causal mask is dropped, so position t can see tokens > t. This is
    exactly the failure class HF described: no crash, wrong behaviour.
  * Three contract probes:
      - causality: changing future tokens must not change past logits.
      - baseline drift: teacher-forced label log-probs must stay within
        a reviewed tolerance of the frozen baseline.
      - loss drift: training loss on the fixture batch must stay close
        to the frozen baseline.
  * A replay simulation: five synthetic "commits" (good, good, buggy,
    buggy, fixed) are replayed against the frozen contract; the contract
    must flag exactly the buggy ones.

Stdlib + NumPy only. Deterministic (fixed seeds).
Run:  python3 mic_harness.py
Test: python3 -m unittest test_mic_harness -v
"""

import numpy as np

# ---------------------------------------------------------------- fixture

FIXTURE_CONFIG = {
    "vocab": 24,
    "d_model": 32,
    "n_layer": 2,
    "n_head": 4,
    "max_len": 32,
    "mlp_ratio": 4,
    "seed": 7,
}

# A plausible "full-size" reference config, for the fixture-size contrast.
FULL_CONFIG = {
    "vocab": 32000,
    "d_model": 512,
    "n_layer": 12,
    "n_head": 8,
    "max_len": 512,
    "mlp_ratio": 4,
    "seed": 7,
}


def param_count(cfg):
    """Parameter count for this architecture (untied LM head)."""
    v, d, layers = cfg["vocab"], cfg["d_model"], cfg["n_layer"]
    mlp = d * cfg["mlp_ratio"]
    per_layer = (
        d * (3 * d) + 3 * d      # fused QKV projection + biases
        + d * d + d              # attention output projection
        + d * mlp + mlp          # MLP up
        + mlp * d + d            # MLP down
        + 2 * d                  # pre-attention LayerNorm (w+b simplified)
        + 2 * d                  # pre-MLP LayerNorm
    )
    return (
        v * d                    # token embedding
        + cfg["max_len"] * d     # position embedding
        + layers * per_layer
        + 2 * d                  # final LayerNorm
        + v * d                  # LM head
    )


def fixture_size_mb(cfg, bytes_per_param=2):
    """Weight size in MB at BF16 (2 bytes/param) by default."""
    return param_count(cfg) * bytes_per_param / 1e6


# ------------------------------------------------------------------ model

def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _layernorm(x, w, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


class TinyDecoder:
    """A small decoder-only transformer in NumPy.

    causal=True  applies the usual triangular mask.
    causal=False drops the mask entirely - the regression under test.
    Both variants share identical weights for the same seed, so any
    behavioural difference is caused by the mask alone.
    """

    def __init__(self, cfg=FIXTURE_CONFIG, causal=True):
        self.cfg = dict(cfg)
        self.causal = causal
        rng = np.random.default_rng(cfg["seed"])
        d, v = cfg["d_model"], cfg["vocab"]
        mlp = d * cfg["mlp_ratio"]
        s = 1.0 / np.sqrt(d)

        def randn(*shape):
            return rng.normal(0.0, s, size=shape).astype(np.float64)

        self.tok_emb = randn(v, d) * np.sqrt(d)
        self.pos_emb = randn(cfg["max_len"], d)
        self.layers = []
        for _ in range(cfg["n_layer"]):
            self.layers.append({
                "ln1_w": np.ones(d), "ln1_b": np.zeros(d),
                "wqkv": randn(d, 3 * d), "bqkv": np.zeros(3 * d),
                "wo": randn(d, d), "bo": np.zeros(d),
                "ln2_w": np.ones(d), "ln2_b": np.zeros(d),
                "w1": randn(d, mlp), "b1": np.zeros(mlp),
                "w2": randn(mlp, d), "b2": np.zeros(d),
            })
        self.lnf_w = np.ones(d)
        self.lnf_b = np.zeros(d)
        self.head = randn(d, v)

    def forward(self, tokens):
        """tokens: (T,) int array -> logits (T, vocab)."""
        tokens = np.asarray(tokens, dtype=int)
        t_len = len(tokens)
        d = self.cfg["d_model"]
        h = self.cfg["n_head"]
        dh = d // h
        x = self.tok_emb[tokens] + self.pos_emb[:t_len]

        if self.causal:
            mask = np.triu(np.ones((t_len, t_len), dtype=bool), k=1)
        else:
            mask = np.zeros((t_len, t_len), dtype=bool)

        for layer in self.layers:
            # --- self-attention block
            y = _layernorm(x, layer["ln1_w"], layer["ln1_b"])
            qkv = y @ layer["wqkv"] + layer["bqkv"]
            q, k, v_ = np.split(qkv, 3, axis=-1)
            q = q.reshape(t_len, h, dh).transpose(1, 0, 2)
            k = k.reshape(t_len, h, dh).transpose(1, 0, 2)
            v_ = v_.reshape(t_len, h, dh).transpose(1, 0, 2)
            scores = q @ k.transpose(0, 2, 1) / np.sqrt(dh)
            scores = np.where(mask[None, :, :], -1e9, scores)
            attn = _softmax(scores, axis=-1) @ v_
            attn = attn.transpose(1, 0, 2).reshape(t_len, d)
            x = x + attn @ layer["wo"] + layer["bo"]
            # --- MLP block (GELU approximated with tanh-free erf-free
            # simple smooth activation: x * sigmoid(x), i.e. SiLU)
            y = _layernorm(x, layer["ln2_w"], layer["ln2_b"])
            hidden = y @ layer["w1"] + layer["b1"]
            hidden = hidden * (1.0 / (1.0 + np.exp(-hidden)))  # SiLU
            x = x + hidden @ layer["w2"] + layer["b2"]

        x = _layernorm(x, self.lnf_w, self.lnf_b)
        return x @ self.head

    def label_logprobs(self, tokens):
        """Teacher-forced log-prob of each next token.

        Returns (mean_logprob, total_logprob) over positions 0..T-2
        predicting tokens 1..T-1 - the quantity HF's fixture compared
        against its reviewed baseline.
        """
        logits = self.forward(tokens)
        logp = logits - np.log(np.exp(
            logits - logits.max(axis=-1, keepdims=True)).sum(
                axis=-1, keepdims=True)) - logits.max(axis=-1, keepdims=True)
        targets = np.asarray(tokens[1:])
        picked = logp[np.arange(len(targets)), targets]
        return float(picked.mean()), float(picked.sum())

    def loss(self, tokens):
        """Mean next-token cross-entropy on the fixture sequence."""
        mean_lp, _ = self.label_logprobs(tokens)
        return -mean_lp


# --------------------------------------------------------------- contract

# The runnable example inside the contract: one fixed sequence, used the
# way the model is actually used (teacher-forced next-token prediction).
EXAMPLE_TOKENS = [3, 11, 5, 19, 2, 14, 7, 7, 21, 0, 9, 16, 4, 12]


def build_contract():
    """Freeze a contract from the CORRECT model.

    In real MIC use a person reviews this baseline before it is frozen;
    here the correct model's outputs are the reviewed reference.
    """
    good = TinyDecoder(causal=True)
    mean_lp, _ = good.label_logprobs(EXAMPLE_TOKENS)
    return {
        "fixture": dict(FIXTURE_CONFIG),
        "example_tokens": list(EXAMPLE_TOKENS),
        "baseline": {
            "label_logprob_mean": mean_lp,
            "loss": -mean_lp,
            "logprob_tolerance": 0.05,
            "loss_tolerance": 0.05,
        },
        "probes": ["causality", "baseline_drift", "loss_drift"],
        "causality_tolerance": 1e-6,
    }


def validate_contract(contract):
    """A MIC must carry all three parts; return a list of problems."""
    problems = []
    if "fixture" not in contract:
        problems.append("missing reduced-size fixture config")
    else:
        for key in ("vocab", "d_model", "n_layer", "n_head"):
            if key not in contract["fixture"]:
                problems.append(f"fixture config missing {key}")
    if not contract.get("example_tokens"):
        problems.append("missing runnable example")
    base = contract.get("baseline", {})
    for key in ("label_logprob_mean", "loss",
                "logprob_tolerance", "loss_tolerance"):
        if key not in base:
            problems.append(f"baseline missing {key}")
    if not contract.get("probes"):
        problems.append("no probes declared")
    return problems


def causality_probe(model, tokens, cut=None, rng_seed=123):
    """Change every token after `cut`; past logits must not move.

    Returns the max absolute logit change at positions <= cut.
    A causal model gives ~0. A future-leaking model gives a large value
    even though it never crashes - which is why shape/smoke tests that
    only check 'does forward() run' cannot catch this regression.
    """
    tokens = list(tokens)
    if cut is None:
        cut = len(tokens) // 2
    rng = np.random.default_rng(rng_seed)
    altered = tokens.copy()
    for i in range(cut + 1, len(tokens)):
        choices = [t for t in range(model.cfg["vocab"]) if t != tokens[i]]
        altered[i] = int(rng.choice(choices))
    before = model.forward(tokens)
    after = model.forward(altered)
    return float(np.abs(before[:cut + 1] - after[:cut + 1]).max())


def run_contract(model, contract):
    """Run every probe in the contract; return per-probe results."""
    tokens = contract["example_tokens"]
    base = contract["baseline"]
    results = {}

    leak = causality_probe(model, tokens)
    results["causality"] = {
        "pass": leak <= contract["causality_tolerance"],
        "max_past_logit_change": leak,
        "tolerance": contract["causality_tolerance"],
    }

    mean_lp, _ = model.label_logprobs(tokens)
    drift = abs(mean_lp - base["label_logprob_mean"])
    results["baseline_drift"] = {
        "pass": drift <= base["logprob_tolerance"],
        "label_logprob_mean": mean_lp,
        "baseline": base["label_logprob_mean"],
        "drift": drift,
        "tolerance": base["logprob_tolerance"],
    }

    loss = model.loss(tokens)
    loss_drift = abs(loss - base["loss"])
    results["loss_drift"] = {
        "pass": loss_drift <= base["loss_tolerance"],
        "loss": loss,
        "baseline": base["loss"],
        "drift": loss_drift,
        "tolerance": base["loss_tolerance"],
    }

    results["overall_pass"] = all(
        results[p]["pass"] for p in contract["probes"])
    return results


def replay_commits(contract, commit_models):
    """Replay the frozen contract over a sequence of 'commits'.

    Mirrors HF's replay study: the same contract, run unchanged over
    successive versions, should flag exactly the versions carrying the
    regression - from introduction until the fix.
    """
    return [run_contract(m, contract)["overall_pass"]
            for m in commit_models]


# -------------------------------------------------------------------- main

def main():
    contract = build_contract()
    problems = validate_contract(contract)
    print("Model Integration Contract - tiny decoder fixture")
    print("=" * 62)
    print(f"contract validation problems: {problems or 'none'}")
    print(f"fixture: {param_count(FIXTURE_CONFIG):,} params, "
          f"{fixture_size_mb(FIXTURE_CONFIG):.3f} MB at BF16")
    print(f"full reference config: {param_count(FULL_CONFIG):,} params, "
          f"{fixture_size_mb(FULL_CONFIG):.1f} MB at BF16 "
          f"({fixture_size_mb(FULL_CONFIG) / fixture_size_mb(FIXTURE_CONFIG):.0f}x larger)")
    print(f"frozen baseline: label log-prob mean "
          f"{contract['baseline']['label_logprob_mean']:.4f}, "
          f"loss {contract['baseline']['loss']:.4f} "
          f"(tolerance +/-{contract['baseline']['logprob_tolerance']})")
    print()

    good = TinyDecoder(causal=True)
    buggy = TinyDecoder(causal=False)
    # Sanity: the buggy variant still RUNS and produces plausible shapes.
    out = buggy.forward(EXAMPLE_TOKENS)
    print(f"buggy model forward() runs fine, logits shape {out.shape} "
          f"- a smoke test would pass it.")
    print()
    for name, model in (("correct (causal mask on)", good),
                        ("regressed (mask dropped)", buggy)):
        r = run_contract(model, contract)
        print(f"{name}: overall {'PASS' if r['overall_pass'] else 'FAIL'}")
        print(f"  causality      max past-logit change "
              f"{r['causality']['max_past_logit_change']:.6f} "
              f"({'pass' if r['causality']['pass'] else 'FAIL'})")
        print(f"  baseline drift label log-prob drift "
              f"{r['baseline_drift']['drift']:.4f} "
              f"({'pass' if r['baseline_drift']['pass'] else 'FAIL'})")
        print(f"  loss drift     loss drift "
              f"{r['loss_drift']['drift']:.4f} "
              f"({'pass' if r['loss_drift']['pass'] else 'FAIL'})")
    print()

    timeline = [TinyDecoder(causal=True), TinyDecoder(causal=True),
                TinyDecoder(causal=False), TinyDecoder(causal=False),
                TinyDecoder(causal=True)]
    labels = ["commit 1 (good)", "commit 2 (good)",
              "commit 3 (regression in)", "commit 4 (still regressed)",
              "commit 5 (fixed)"]
    verdicts = replay_commits(contract, timeline)
    print("replay over synthetic commit history:")
    for label, ok in zip(labels, verdicts):
        print(f"  {label}: contract {'PASS' if ok else 'FAIL - flagged'}")
    flagged = [i + 1 for i, ok in enumerate(verdicts) if not ok]
    print(f"flagged commits: {flagged} (expected [3, 4])")
    print()
    print("Takeaway: the regression never crashed anything. Only a frozen,")
    print("behaviour-level contract - causality plus a reviewed baseline -")
    print("sees it, and a 66 MB-class fixture makes that check cheap enough")
    print("to run on every pull request instead of in a nightly GPU run.")


if __name__ == "__main__":
    main()

"""
Demo: sparse Mixture-of-Experts routing, the efficiency trick behind
Reflection AI's Beam (open-weight release, 5 Oct 2026).

Beam's headline numbers: 501B parameters total, but each token is routed to
only a few expert networks, so roughly 23B parameters are active per token
(about 4.6 percent). Sparse activation is why an MoE can match a dense model
at a fraction of the inference compute.

This script trains a tiny 4-expert MoE layer from scratch on a synthetic
two-subtask problem and shows:
  1. sparse routing: every token goes to exactly 1 of 4 experts
  2. specialisation emerging: each subtask gravitates to its own experts
  3. the load-balancing auxiliary loss (Switch Transformer style) scoring
     balanced vs collapsed routing
  4. the dense-vs-sparse FLOP math, projected up to Beam scale
  5. expert capacity limits and token dropping (GShard style)

Run:  python demo.py
"""

import numpy as np

from moe import MoELayer, flop_report, load_balance_loss, route_with_capacity, softmax

rng = np.random.default_rng(7)

# --- config -----------------------------------------------------------------
D_MODEL, D_FF, N_EXPERTS, K = 16, 32, 4, 1
T = 512
STEPS = 1500
LR = 3e-2

# --- synthetic task ----------------------------------------------------------
# Two token types living in different regions of embedding space
# (like code vs prose embeddings), each needing a different function:
#   type A: y = x @ A   (linear)
#   type B: y = sin(3x) @ B   (nonlinear)
ttype = np.repeat([0, 1], T // 2)
perm = rng.permutation(T)
x = rng.normal(size=(T, D_MODEL)) + ttype[:, None] * 1.5
x, ttype = x[perm], ttype[perm]
A = rng.normal(0, 0.5, size=(D_MODEL, D_MODEL))
B = rng.normal(0, 0.5, size=(D_MODEL, D_MODEL))
target = np.where(ttype[:, None] == 0, x @ A, np.sin(3.0 * x) @ B)

moe = MoELayer(D_MODEL, D_FF, N_EXPERTS, k=K, seed=42)

print("=" * 70)
print("Sparse MoE routing demo  |  inspired by Reflection AI's Beam (5 Oct 2026)")
print("=" * 70)
print(f"config: {N_EXPERTS} experts, top-{K} routing, {T} tokens, "
      f"d_model={D_MODEL}, d_ff={D_FF}\n")

# --- 1 + 2. joint training: watch specialisation emerge ----------------------
print("--- training jointly from scratch (task loss only) ---")
for step in range(STEPS):
    mse, aux = moe.train_step(x, target, lr=LR, aux_coef=0.0)
    if step % 500 == 0 or step == STEPS - 1:
        print(f"  step {step:4d}  mse={mse:.4f}")

_, idx, _ = moe.gate.forward(x)
print("\n--- routing after training (rows: token type, cols: expert) ---")
table = np.zeros((2, N_EXPERTS), dtype=int)
for t in range(T):
    for e in idx[t]:
        table[ttype[t], e] += 1
print("        " + " ".join(f"E{e}" for e in range(N_EXPERTS)))
for t in range(2):
    print(f"  type{t} " + " ".join(f"{c:3d}" for c in table[t]))
top_a, top_b = table[0].argmax(), table[1].argmax()
print(f"type A (linear)   -> experts {np.argsort(-table[0])[:2].tolist()} "
      f"({table[0].sum()} tokens)")
print(f"type B (nonlinear) -> experts {np.argsort(-table[1])[:2].tolist()} "
      f"({table[1].sum()} tokens)")
print(f"types use different experts: {top_a != top_b}")

# --- 3. load balancing --------------------------------------------------------
counts = moe.expert_token_counts(idx)
probs = softmax(x @ moe.gate.W + moe.gate.b, axis=1)
f = counts / T
P = probs.mean(axis=0)
balanced_score = load_balance_loss(f, P, N_EXPERTS)
collapsed_score = load_balance_loss(
    np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), N_EXPERTS)
print("\n--- load balancing (Switch-Transformer auxiliary loss) ---")
print(f"tokens per expert: {counts.tolist()} (ideal: {T * K / N_EXPERTS:.0f} each, "
      f"no expert starved)")
print(f"aux loss on learned routing: {balanced_score:.3f} (ideal balanced = {K:.1f})")
print(f"aux loss on collapsed routing (all -> E0): {collapsed_score:.3f}")
print("In production training this loss's gradient is the guardrail that "
      "keeps experts from collapsing onto one or two.")

# --- 4. FLOP math + Beam projection --------------------------------------------
rep = flop_report(D_MODEL, D_FF, N_EXPERTS, K, T)
print("\n--- compute: dense vs sparse ---")
print(f"demo layer : dense {rep['dense_flops']:,} FLOPs vs "
      f"sparse {rep['sparse_flops']:,} FLOPs "
      f"({rep['speedup']:.1f}x less, {100 * rep['active_fraction']:.0f}% of "
      f"experts active per token)")
BEAM_TOTAL_B, BEAM_ACTIVE_B = 501.0, 23.0
beam_frac = BEAM_ACTIVE_B / BEAM_TOTAL_B
print(f"Beam scale : {BEAM_TOTAL_B:.0f}B total params, ~{BEAM_ACTIVE_B:.0f}B active "
      f"per token ({100 * beam_frac:.1f}% active) -> about {1 / beam_frac:.0f}x "
      f"less compute than a dense 501B model per token.")
print(f"params here: {moe.total_params():,} total, "
      f"{moe.active_params_per_token():,} active per token "
      f"({100 * moe.active_params_per_token() / moe.total_params():.1f}% active)")

# --- 5. capacity limits ---------------------------------------------------------
gate, idx2, _ = moe.gate.forward(x)
for cf in (1.0, 1.25):
    keep, dropped = route_with_capacity(gate, idx2, capacity_factor=cf)
    print(f"\ncapacity factor {cf}: per-expert cap = "
      f"{int(np.ceil(cf * T * K / N_EXPERTS))} tokens, "
      f"dropped fraction = {dropped:.3f}")
print("(Real MoE training uses capacity limits to bound compute per expert;\n"
      "dropped tokens skip the expert layer via the residual connection.)")

# Sparse MoE Routing: the efficiency trick behind Reflection AI's Beam

**Trending:** on 5 October 2026, [Reflection AI launched Beam](https://techcrunch.com/2026/10/05/reflection-debuts-beam-a-open-weight-ai-model-to-rival-chinese-models-at-lower-compute-cost/), an open-weight text-only Mixture-of-Experts model: **501B parameters total, ~23B active per token** (4.6%), 1M-token context, trained with high-compute RL for reasoning, coding and agentic tasks. Reflection claims 3-4x less inference compute than rivals at the same capability level. (Their benchmark claims have not been independently verified.)

The core idea this demo implements: instead of running every token through one giant dense network, route each token to a **small subset of expert networks**. Total capacity stays huge; per-token cost stays small. That is where Beam's efficiency comes from.

## What is here

| File | What it does |
|---|---|
| `moe.py` | From-scratch NumPy MoE: `TopKGate` (learned router, Switch-Transformer-style raw softmax weights), `Expert` (2-layer ReLU MLP), `MoELayer` (sparse forward + manual backprop, top-k mask held fixed), `load_balance_loss` (Switch aux loss), `flop_report`, `route_with_capacity` (GShard-style token dropping) |
| `demo.py` | End-to-end run: trains a 4-expert MoE from scratch, prints routing, balance, FLOPs and the Beam projection |
| `test_moe.py` | 9 unit tests (plain asserts, no dependencies): gate sparsity, shapes, aux-loss formula, capacity dropping, FLOP math, reproducibility, param accounting, MSE decreasing, and specialisation emerging from joint training |

## Run it

```bash
python demo.py       # ~15 seconds, NumPy only
python test_moe.py  # 9/9 passing
```

## Results (actual run, 5 Oct 2026)

4 experts, top-1 routing, 512 tokens of two types (linear vs sinusoidal subtask, in shifted embedding regions, like code vs prose):

```
mse: 4.889 -> 1.460 over 1500 steps

routing after training (rows: token type, cols: expert):
        E0  E1  E2  E3
  type0 115 140   1   0
  type1   3   3  82 168
```

Specialisation emerges from scratch: the linear subtask settles on experts 0/1, the nonlinear subtask on experts 2/3. No expert starves: token counts `[118, 143, 83, 168]` vs ideal 128 each.

Load balancing: the Switch-style auxiliary loss scores the learned routing at **1.006** (ideal balanced = 1.0) versus **4.000** for a collapsed routing where every token goes to one expert.

Compute: the sparse layer uses **4x fewer FLOPs** than its dense equivalent (25% of experts active per token). Projected to Beam's published numbers, 23B active of 501B total is **~22x less compute per token** than a dense 501B model. Parameters here: 4,352 total, 1,136 active per token (26.1%).

Capacity limits: with a GShard-style capacity factor of 1.25, only 1.6% of tokens are dropped (10.7% at factor 1.0); dropped tokens skip the expert via the residual connection.

## Notes and honest caveats

* Toy scale: real MoE routers learn with noisy gating, capacity constraints and billions of tokens. Here the two token types live in shifted embedding regions so the router has a learnable signal; without that shift, routing cannot specialise (verified while building this).
* The auxiliary loss is implemented, unit-tested and scored in the demo. At this toy scale the joint task loss alone finds balanced routing; in production training the aux-loss gradient is the standard guardrail against expert collapse (too strong a coefficient here actually hurts, which matches how fiddly it is known to be).
* Gate weights are raw softmax probabilities of the selected experts (Switch Transformer style), which keeps a learning signal for the router even at top-1. Manual backprop throughout, gradient-checked against finite differences.

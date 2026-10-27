"""Run the three LFM3 tri-hybrid probes and print a scaling table.

Probe 1 (locality): a single impulse through a gated convolution only
affects the impulse position and the next ``kernel - 1`` positions.
Probe 2 (reach): with identity query/key weights, the last token puts
almost all of its global-attention weight on the first token (made a
deliberately strong key so it beats the last token's self-match), while
a sliding-window layer with a window smaller than the distance gives it
exactly zero weight there.
Probe 3 (causality): changing only the last token of a sequence leaves
every earlier output of the full 24-layer stack unchanged, because every
layer type here is causal.
"""

import numpy as np

from tri_hybrid import (
    Attention,
    GatedDepthwiseConv,
    TriHybridStack,
    attention_flops,
    build_layer_pattern,
)


def probe_locality():
    conv = GatedDepthwiseConv(d_model=4, kernel=7,
                              rng=np.random.default_rng(0))
    x = np.zeros((16, 4))
    x[3] = [1.0, 0.0, 0.0, 0.0]
    out = conv.forward(x)
    affected = [t for t in range(16) if np.linalg.norm(out[t]) > 1e-9]
    print("Probe 1 - gated convolution, impulse at position 3, kernel 7")
    print(f"  affected positions: {affected} (expected [3, 4, 5, 6, 7, 8, 9])")
    assert affected == [3, 4, 5, 6, 7, 8, 9], affected


def probe_reach():
    d_model = 4
    x = np.zeros((12, d_model))
    x[0] = [10.0, 0.0, 0.0, 0.0]
    x[1:11] = [0.0, 5.0, 0.0, 0.0]
    x[11] = [5.0, 0.0, 0.0, 0.0]
    global_attn = Attention(d_model, mode="global",
                            rng=np.random.default_rng(0))
    window_attn = Attention(d_model, mode="window", window=4,
                            rng=np.random.default_rng(0))
    for attn in (global_attn, window_attn):
        attn.w_q = np.eye(d_model)
        attn.w_k = np.eye(d_model)
        attn.w_v = np.eye(d_model)
        attn.b_q = np.zeros(d_model)
        attn.b_k = np.zeros(d_model)
        attn.b_v = np.zeros(d_model)
    w_global = global_attn.weights(x)[11, 0]
    w_window = window_attn.weights(x)[11, 0]
    print("Probe 2 - last token's attention weight on the first token")
    print(f"  global attention: {w_global:.4f} (expected > 0.9)")
    print(f"  sliding window (window 4): {w_window:.4f} (expected 0.0)")
    assert w_global > 0.9, w_global
    assert w_window == 0.0, w_window


def probe_causality():
    stack = TriHybridStack(d_model=16, seed=0)
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, size=(32, 16))
    x2 = x.copy()
    x2[-1] += 10.0
    out = stack.forward(x)
    out2 = stack.forward(x2)
    max_early_change = np.max(np.abs(out[:-1] - out2[:-1]))
    last_change = np.max(np.abs(out[-1] - out2[-1]))
    print("Probe 3 - full 24-layer stack, last token perturbed")
    print(f"  max change at earlier positions: {max_early_change:.2e} "
          "(expected ~0)")
    print(f"  change at last position: {last_change:.4f} (expected > 0)")
    print(f"  output shape: {out.shape}, all finite: "
          f"{bool(np.all(np.isfinite(out)))}")
    assert max_early_change < 1e-9, max_early_change
    assert last_change > 0, last_change


def scaling_table():
    print("\nMixing-cost scaling (d_model=128, window=256, kernel=7)")
    print("  seq_len | full attention | tri-hybrid | hybrid share")
    for seq_len in (1024, 16384, 65536, 262144):
        full, hybrid, ratio = attention_flops(seq_len, d_model=128)
        print(f"  {seq_len:>7} | {full / 1e12:>12.1f} TFLOP | "
              f"{hybrid / 1e12:>9.1f} TFLOP | {ratio:.3f}")
    full, hybrid, ratio = attention_flops(262144, d_model=128)
    print(f"  At 256K context the hybrid mix costs {ratio:.1%} of the "
          "all-global-attention baseline, because only 6 of 24 layers "
          "pay the quadratic price.")


def main():
    print(f"Layer pattern (24 blocks): {build_layer_pattern()}")
    print(f"Counts: conv={build_layer_pattern().count('conv')}, "
          f"window={build_layer_pattern().count('window')}, "
          f"global={build_layer_pattern().count('global')}\n")
    probe_locality()
    probe_reach()
    probe_causality()
    scaling_table()
    print("\nAll probes passed.")


if __name__ == "__main__":
    main()

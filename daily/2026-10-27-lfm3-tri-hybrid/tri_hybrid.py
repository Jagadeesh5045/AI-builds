"""Tri-hybrid sequence mixer, inspired by Liquid AI's LFM3 (27 Oct 2026).

LFM3's 1.7B text backbone is described in launch coverage as a tri-hybrid
of three layer types across 24 blocks: 13 gated short-convolution layers,
5 sliding-window attention layers (256-token window) and 6 global attention
layers, with kernel size 7 and a 256K context window. The 1.7B model is
reported to fit in roughly 1 GB at 4-bit precision.

This file is a small, educational NumPy re-implementation of that *pattern*
only. It contains no Liquid AI code and no LFM3 weights. Dimensions here are
toy-sized so the whole thing runs on CPU in seconds; the point is to show
what each layer type contributes and why the mix is cheaper than
all-global-attention at long context.

Layer roles:
- Gated short convolution: cheap local mixing. A token only ever sees the
  previous ``kernel`` tokens, so cost grows linearly with sequence length.
- Sliding-window attention: mid-range mixing. A token attends to the
  previous ``window`` tokens only, so cost grows linearly too.
- Global attention: full causal attention. This is the only quadratic part,
  and in the LFM3 pattern it is only 6 of 24 layers, not 24 of 24.
"""

import numpy as np

# LFM3 text-backbone layer counts, as reported in launch coverage.
N_CONV_LAYERS = 13
N_WINDOW_LAYERS = 5
N_GLOBAL_LAYERS = 6
N_BLOCKS = 24
LFM3_WINDOW = 256
LFM3_KERNEL = 7


def build_layer_pattern():
    """Return the 24-layer pattern with exactly 13 conv, 5 window, 6 global.

    Five repeats of [conv, conv, window, global] give 10 conv, 5 window and
    5 global layers; a final [conv, conv, conv, global] block completes the
    reported 13 / 5 / 6 split. The order interleaves the types rather than
    stacking all of one kind together, so local, mid-range and global
    mixing alternate through the network.
    """
    pattern = ["conv", "conv", "window", "global"] * 5
    pattern += ["conv", "conv", "conv", "global"]
    assert len(pattern) == N_BLOCKS
    assert pattern.count("conv") == N_CONV_LAYERS
    assert pattern.count("window") == N_WINDOW_LAYERS
    assert pattern.count("global") == N_GLOBAL_LAYERS
    return pattern


def softmax(rows):
    """Row-wise softmax that is stable for large scores."""
    shifted = rows - np.max(rows, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def layer_norm(x, eps=1e-5):
    """Per-token normalisation over the feature dimension."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class GatedDepthwiseConv:
    """Causal, depthwise, gated short convolution.

    Each channel is convolved independently over time with a kernel of
    length ``kernel`` that only looks backwards (no future leakage). A
    second convolution produces a gate in (0, 1) that scales the value
    convolution, hence "gated". Cost is O(seq_len * kernel * d_model).
    """

    def __init__(self, d_model, kernel=LFM3_KERNEL, rng=None):
        self.d_model = d_model
        self.kernel = kernel
        rng = rng or np.random.default_rng(0)
        scale = 1.0 / np.sqrt(kernel)
        self.w_value = rng.normal(0.0, scale, size=(d_model, kernel))
        self.w_gate = rng.normal(0.0, scale, size=(d_model, kernel))

    def forward(self, x):
        seq_len, d_model = x.shape
        out = np.zeros_like(x)
        for t in range(seq_len):
            for s in range(self.kernel):
                src = t - s
                if src < 0:
                    break
                value = x[src] * self.w_value[:, s]
                gate = sigmoid(x[src] * self.w_gate[:, s])
                out[t] += value * gate
        return out


class Attention:
    """Single-head causal attention, either global or sliding-window.

    ``mode="global"`` attends to every previous token; ``mode="window"``
    attends only to the previous ``window`` tokens. Both are causal: a
    query at position t never attends to any key after t.
    """

    def __init__(self, d_model, mode, window=LFM3_WINDOW, rng=None):
        assert mode in ("global", "window")
        self.d_model = d_model
        self.mode = mode
        self.window = window
        rng = rng or np.random.default_rng(0)
        scale = 1.0 / np.sqrt(d_model)
        self.w_q = rng.normal(0.0, scale, size=(d_model, d_model))
        self.w_k = rng.normal(0.0, scale, size=(d_model, d_model))
        self.w_v = rng.normal(0.0, scale, size=(d_model, d_model))
        self.b_q = np.zeros(d_model)
        self.b_k = np.zeros(d_model)
        self.b_v = np.zeros(d_model)

    def weights(self, x):
        """Return the attention weight matrix for input ``x``."""
        q = x @ self.w_q + self.b_q
        k = x @ self.w_k + self.b_k
        seq_len = x.shape[0]
        scores = (q @ k.T) / np.sqrt(self.d_model)
        mask = np.full((seq_len, seq_len), -1e9)
        for t in range(seq_len):
            lo = 0 if self.mode == "global" else max(0, t - self.window + 1)
            mask[t, lo:t + 1] = 0.0
        return softmax(scores + mask)

    def forward(self, x):
        v = x @ self.w_v + self.b_v
        return self.weights(x) @ v


class TriHybridLayer:
    """One block: a sublayer (conv / window / global) + residual + norm."""

    def __init__(self, d_model, kind, window=LFM3_WINDOW,
                 kernel=LFM3_KERNEL, rng=None):
        self.kind = kind
        if kind == "conv":
            self.sublayer = GatedDepthwiseConv(d_model, kernel=kernel, rng=rng)
        elif kind in ("window", "global"):
            self.sublayer = Attention(d_model, mode=kind, window=window, rng=rng)
        else:
            raise ValueError(f"unknown layer kind: {kind!r}")

    def forward(self, x):
        return layer_norm(x + self.sublayer.forward(x))


class TriHybridStack:
    """The 24-block stack in the reported 13 / 5 / 6 pattern."""

    def __init__(self, d_model, window=LFM3_WINDOW, kernel=LFM3_KERNEL,
                 seed=0):
        self.pattern = build_layer_pattern()
        self.layers = []
        for i, kind in enumerate(self.pattern):
            rng = np.random.default_rng(seed + i)
            self.layers.append(TriHybridLayer(d_model, kind, window=window,
                                              kernel=kernel, rng=rng))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


def attention_flops(seq_len, d_model, window=LFM3_WINDOW, kernel=LFM3_KERNEL):
    """Approximate attention-style FLOPs: full vs tri-hybrid, per token mix.

    Per layer we count scores + weighted sum as 2 * seq_len * context *
    d_model. The convolution layers are counted at 2 * seq_len * kernel *
    d_model so the comparison stays apples-to-apples on mixing cost.
    Returns (full_attention_flops, hybrid_flops, hybrid_ratio).
    """
    full = N_BLOCKS * 2 * seq_len * seq_len * d_model
    hybrid = (
        N_GLOBAL_LAYERS * 2 * seq_len * seq_len * d_model
        + N_WINDOW_LAYERS * 2 * seq_len * window * d_model
        + N_CONV_LAYERS * 2 * seq_len * kernel * d_model
    )
    return full, hybrid, hybrid / full

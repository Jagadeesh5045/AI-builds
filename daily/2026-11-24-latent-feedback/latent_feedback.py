"""
Attention Latent Feedback (ALF) in miniature.

Trending source (24 Nov 2026): Xiong et al., "Attention Latent Feedback"
(arXiv:2611.31154). During autoregressive decoding, the deeper layers have
already computed rich representations of the *previous* token, yet in a
standard transformer the shallow layers at the current position cannot read
them: self-attention only connects positions at the same depth. ALF gives
every sublayer a learned, content-dependent read over the deeper outputs of
the previous position, blended into the usual layer input by a per-layer
gate. The paper reports, on MoE models up to 4.0B parameters:

  * held-out loss lower by 0.022-0.024 nats vs the standard residual,
  * about 1.1% extra decoding FLOPs,
  * on MoE-4.0B, average downstream accuracy 45.1% -> 47.0%.

This miniature isolates exactly that mechanism, without a transformer:

  * Each token gets a *shallow* feature (one linear map of its embedding)
    and a *deep* feature (a two-layer non-linear map of its embedding).
    The deep feature is the sort of representation only the later layers
    of a real model would have produced.
  * The prediction target at position t depends on an interaction between
    the current token and the DEEP feature of the previous token. That is
    the information a standard same-depth architecture cannot reach early.
  * Both models use a ridge head over the current feature, the previous
    feature, and their elementwise interaction tensor (the product terms
    a following MLP could compute after a blend). The ONLY difference is
    what the previous-position feature is allowed to be:
      - baseline: the previous token's shallow (same-depth) feature;
      - ALF:      the previous token's deep feature, scaled per sublayer
                  slice by gates learned on a validation split, mirroring
                  ALF's per-sublayer learned read.

Everything is NumPy, seeded, and small enough to run in a second.
"""

from __future__ import annotations

import numpy as np

VOCAB = 16          # tokens are integers in [0, VOCAB)
EMB_DIM = 12        # embedding width
DEEP_DIM = 12       # deep-feature width
SEQ_LEN = 32
N_TRAIN_SEQ = 220
N_TEST_SEQ = 80
RIDGE = 1e-3
SEED = 7


def make_world(rng: np.random.Generator):
    """Fixed random maps defining shallow/deep features and the target."""
    emb = rng.normal(size=(VOCAB, EMB_DIM)) / np.sqrt(EMB_DIM)
    # Shallow features pass through a rank-4 bottleneck: an early layer
    # has only a compressed view of the previous token, which is exactly
    # why it cannot substitute for the deep representation.
    bottleneck = 4
    down = rng.normal(size=(EMB_DIM, bottleneck)) / np.sqrt(EMB_DIM)
    up = rng.normal(size=(bottleneck, EMB_DIM)) / np.sqrt(bottleneck)
    w_shallow = down @ up
    w1 = rng.normal(size=(EMB_DIM, DEEP_DIM)) / np.sqrt(EMB_DIM)
    w2 = rng.normal(size=(DEEP_DIM, DEEP_DIM)) / np.sqrt(DEEP_DIM)
    # Target interaction: current shallow feature reads previous deep one.
    w_target = rng.normal(size=(DEEP_DIM, EMB_DIM)) / np.sqrt(DEEP_DIM)
    return emb, w_shallow, w1, w2, w_target


def shallow_feat(tokens, emb, w_shallow):
    return emb[tokens] @ w_shallow


def deep_feat(tokens, emb, w1, w2):
    h = np.tanh(emb[tokens] @ w1)
    return np.tanh(h @ w2)


def build_dataset(n_seq, rng, world):
    """Return (X_base, X_deep_prev, y) over all positions t >= 1.

    y_t is a real-valued score: shallow_t . (W_target^T deep_{t-1}) + noise,
    so the label genuinely requires the previous token's deep state.
    """
    emb, w_shallow, w1, w2, w_target = world
    tokens = rng.integers(0, VOCAB, size=(n_seq, SEQ_LEN))
    sh = shallow_feat(tokens, emb, w_shallow)          # (n, T, d)
    dp = deep_feat(tokens, emb, w1, w2)                # (n, T, d)
    interaction = np.einsum("ntd,de->nte", dp, w_target)  # deep -> emb space
    # The score at position t reads the deep state of position t-1, so
    # shift the interaction stream right by one before combining.
    shifted = np.zeros_like(interaction)
    shifted[:, 1:, :] = interaction[:, :-1, :]
    scores = np.einsum("ntd,ntd->nt", sh, shifted)
    scores = scores + 0.05 * rng.normal(size=scores.shape)

    cur = sh[:, 1:, :].reshape(-1, EMB_DIM)
    prev_shallow = sh[:, :-1, :].reshape(-1, EMB_DIM)
    prev_deep = dp[:, :-1, :].reshape(-1, DEEP_DIM)
    y = scores[:, 1:].reshape(-1)
    X_base = np.concatenate([cur, prev_shallow], axis=1)
    return X_base, prev_deep, y


def ridge_fit(X, y, lam=RIDGE):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def ridge_predict(X, w):
    return X @ w


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def interaction(cur, prev):
    """Bilinear interaction features (cur_i * prev_j), scaled."""
    return np.einsum("ni,nj->nij", cur, prev).reshape(len(cur), -1) \
        / np.sqrt(cur.shape[1])


def model_features(cur, prev):
    return np.concatenate([cur, prev, interaction(cur, prev)], axis=1)


def learn_gates(cur_tr, deep_tr, y_tr, cur_va, deep_va, y_va,
                prev_shallow_va, steps=80, lr=0.15):
    """Learn one scalar gate per 'layer slice' of the deep feature.

    The deep feature is split into 3 slices (standing in for sublayers).
    For each candidate gate vector the ridge head is refitted in closed
    form on the training split; gates are updated by gradient descent on
    validation MSE, using the exact gradient through the interaction
    features for the fixed head (a linear-head gradient step).
    """
    n_slices = 3
    slice_w = DEEP_DIM // n_slices
    gates = np.full(n_slices, 0.5)

    def expand(deep, g):
        parts = [deep[:, i * slice_w:(i + 1) * slice_w] * g[i]
                 for i in range(n_slices)]
        return np.concatenate(parts, axis=1)

    base_dim = cur_tr.shape[1] + DEEP_DIM  # cur + prev blocks precede products
    for _ in range(steps):
        w = ridge_fit(model_features(cur_tr, expand(deep_tr, gates)), y_tr)
        X_va = model_features(cur_va, expand(deep_va, gates))
        resid = ridge_predict(X_va, w) - y_va
        head = w[base_dim:].reshape(EMB_DIM, DEEP_DIM)  # interaction head
        grads = np.zeros(n_slices)
        for i in range(n_slices):
            sl = slice(i * slice_w, (i + 1) * slice_w)
            contrib = np.einsum("ni,ij,nj->n", cur_va, head[:, sl],
                                deep_va[:, sl]) / np.sqrt(EMB_DIM)
            grads[i] = 2.0 * float(np.mean(resid * contrib))
        gates = np.clip(gates - lr * grads, 0.0, 1.5)
    return gates, expand


def flop_overhead(emb_dim=EMB_DIM, layers=12):
    """Rough decoding-FLOP accounting, mirroring the paper's framing.

    Per token, a standard layer costs roughly 12 * emb_dim^2 multiply-adds
    (attention projections plus the 4x-wide MLP, simplified). An ALF read
    is a gated vector blend: about emb_dim multiply-adds per sublayer, two
    sublayers per layer, and no new attention over the sequence.
    Overhead fraction = read cost / total layer cost.
    """
    layer_cost = layers * 12 * emb_dim ** 2
    read_cost = layers * 2 * emb_dim
    return read_cost / layer_cost


def main():
    rng = np.random.default_rng(SEED)
    world = make_world(rng)

    Xb_tr, deep_tr, y_tr = build_dataset(N_TRAIN_SEQ, rng, world)
    Xb_va, deep_va, y_va = build_dataset(40, rng, world)
    Xb_te, deep_te, y_te = build_dataset(N_TEST_SEQ, rng, world)

    cur_tr, prev_tr = Xb_tr[:, :EMB_DIM], Xb_tr[:, EMB_DIM:]
    cur_va, prev_va = Xb_va[:, :EMB_DIM], Xb_va[:, EMB_DIM:]
    cur_te, prev_te = Xb_te[:, :EMB_DIM], Xb_te[:, EMB_DIM:]
    var_y = float(np.var(y_te))

    # Baseline: same-depth (shallow) previous feature only.
    w_base = ridge_fit(model_features(cur_tr, prev_tr), y_tr)
    base_mse = mse(ridge_predict(model_features(cur_te, prev_te), w_base),
                   y_te)

    # ALF with all gates fixed at 1 (ungated deep feedback).
    w_full = ridge_fit(model_features(cur_tr, deep_tr), y_tr)
    full_mse = mse(ridge_predict(model_features(cur_te, deep_te), w_full),
                   y_te)

    # ALF with learned per-sublayer gates.
    gates, expand = learn_gates(cur_tr, deep_tr, y_tr,
                                cur_va, deep_va, y_va, prev_va)
    w_g = ridge_fit(model_features(cur_tr, expand(deep_tr, gates)), y_tr)
    gated_mse = mse(
        ridge_predict(model_features(cur_te, expand(deep_te, gates)), w_g),
        y_te)

    print("Attention Latent Feedback - miniature")
    print(f"test positions: {len(y_te)}   label variance: {var_y:.3f}")
    print(f"baseline (same-depth only)   test MSE: {base_mse:.4f} "
          f" R^2: {1 - base_mse / var_y:.3f}")
    print(f"ALF, gates fixed at 1        test MSE: {full_mse:.4f} "
          f" R^2: {1 - full_mse / var_y:.3f}")
    print(f"ALF, learned gates {np.round(gates, 3)} test MSE: "
          f"{gated_mse:.4f}  R^2: {1 - gated_mse / var_y:.3f}")
    print(f"relative MSE reduction (gated ALF vs baseline): "
          f"{100 * (base_mse - gated_mse) / base_mse:.1f}%")
    print(f"estimated decoding overhead of the feedback read: "
          f"{100 * flop_overhead():.1f}% of layer FLOPs "
          f"(paper reports ~1.1% at real scale)")
    return {
        "base_mse": base_mse, "full_mse": full_mse,
        "gated_mse": gated_mse, "gates": gates, "var_y": var_y,
    }


if __name__ == "__main__":
    main()

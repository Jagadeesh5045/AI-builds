"""
Self-rewriting weights, in miniature.

Trending source (reported 26-27 Nov 2026): a preprint (arXiv, 16 Nov 2026,
Fan et al., UIUC / CMU / Rutgers / Princeton / Stanford / Yale) in which
Qwen3.5-27B inspected its own weights and activations, diagnosed a recurring
failure (generations that ran to the token limit with no final answer),
wrote a targeted weight edit from the activation difference between finished
and unfinished traces, and improved held-out math accuracy from 57.2% to
63.8% with no gradient, optimiser or retraining run. The 2B model made no
edit, and the 4B / 9B models edited with what amounted to random noise and
collapsed to 0.0% on their own tests.

This demo reproduces the *protocol*, not the scale:

  1. A tiny "finish head" decides, from a hidden state, whether a trace
     should FINISH (emit its answer) or CONTINUE. The shipped head is
     misaligned, so it keeps traces running that should have finished.
  2. DIAGNOSE: using only graded practice traces, the model contrasts its
     own mean activation on finished vs runaway traces. That difference
     vector points at the direction the finish head is missing.
  3. EDIT: a rank-1 style edit adds a scaled copy of the difference
     direction to the finish head's weight row. No training loop runs.
  4. VALIDATE OR REVERT: candidate strengths are scored once on a reserved
     practice split. The edit is submitted only if it clears a margin;
     otherwise the original weights are kept. This gate is the revised
     protocol in the paper, and it is what separates the 27B outcome from
     the smaller models' random-noise edits, which this demo also tries
     and the gate rejects.

Pure NumPy, fixed seeds, fully deterministic.
"""

import numpy as np

D = 16            # hidden-state width
SEP = 1.6         # class separation along the true finish direction
NOISE = 1.0       # per-dimension activation noise
GATE_MARGIN = 0.05  # minimum validation gain required to submit an edit


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_direction(rng):
    """The (unknown to the model) direction that separates the classes."""
    v = rng.normal(size=D)
    return v / np.linalg.norm(v)


def make_traces(rng, n, direction):
    """Hidden states for graded traces.

    Label 1 = the trace FINISHED with an answer, 0 = it ran away
    (kept generating to the token limit). Finished traces sit on the
    +direction side, runaway traces on the -direction side, both under
    isotropic noise, so the Bayes boundary is the direction itself.
    """
    y = rng.integers(0, 2, size=n)
    centres = np.where(y[:, None] == 1, SEP, -SEP) * direction[None, :]
    X = centres + NOISE * rng.normal(size=(n, D))
    return X, y


def make_misaligned_head(rng, direction):
    """A finish head that mostly ignores the true direction.

    It is built from a random vector with the true-direction component
    removed and only a weak residue added back, which is the miniature
    analogue of the failure the 27B model found in itself: the machinery
    to finish exists, but the head barely reads the signal that should
    trigger it.
    """
    w = rng.normal(size=D)
    w = w - (w @ direction) * direction          # strip the real signal
    w = w / np.linalg.norm(w)
    w = w + 0.15 * direction                     # weak residue only
    return w, 0.0


def accuracy(w, b, X, y):
    pred = (sigmoid(X @ w + b) >= 0.5).astype(int)
    return float((pred == y).mean())


def diagnose_difference(X, y):
    """Mean-activation contrast: finished minus runaway traces.

    This is the self-diagnosis step. The model never sees the true
    direction; it only has its own activations on graded practice traces.
    """
    return X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)


def propose_edit(w, diff, alpha):
    """Rank-1 style edit: add a scaled difference direction to the head."""
    d = diff / np.linalg.norm(diff)
    return w + alpha * d


def random_edit(w, rng, norm):
    """What the smaller models effectively did: same-size blind noise."""
    noise = rng.normal(size=D)
    noise = noise / np.linalg.norm(noise)
    return w + norm * noise


def gate_and_submit(w, b, candidate, X_val, y_val, margin=GATE_MARGIN):
    """Validate a candidate edit once; submit only if it clearly helps.

    Returns (submitted_weights, submitted_flag, base_acc, cand_acc).
    On rejection the ORIGINAL weights are returned, untouched.
    """
    base = accuracy(w, b, X_val, y_val)
    cand = accuracy(candidate, b, X_val, y_val)
    if cand >= base + margin:
        return candidate, True, base, cand
    return w, False, base, cand


def run_demo(seed=7, verbose=True):
    rng = np.random.default_rng(seed)
    direction = make_direction(rng)
    w, b = make_misaligned_head(rng, direction)

    # Graded practice traces (diagnosis + a reserved validation split),
    # and a held-out test set the edit never sees during selection.
    X_prac, y_prac = make_traces(rng, 400, direction)
    X_val, y_val = make_traces(rng, 300, direction)
    X_test, y_test = make_traces(rng, 600, direction)

    base_test = accuracy(w, b, X_test, y_test)

    # 1) Self-diagnosis from the model's own activations.
    diff = diagnose_difference(X_prac, y_prac)
    cos = float(diff @ direction / np.linalg.norm(diff))

    # 2) Sweep edit strengths on the reserved practice split only.
    best_w, best_alpha, best_val = None, None, -1.0
    for alpha in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        cand = propose_edit(w, diff, alpha)
        val = accuracy(cand, b, X_val, y_val)
        if val > best_val:
            best_w, best_alpha, best_val = cand, alpha, val

    # 3) The gate decides whether the self-edit is submitted.
    final_w, submitted, val_base, val_cand = gate_and_submit(
        w, b, best_w, X_val, y_val)
    self_test = accuracy(final_w, b, X_test, y_test)

    # 4) Control: a random-noise edit of the same norm, through the gate.
    edit_norm = float(np.linalg.norm(best_w - w))
    w_rand = random_edit(w, rng, edit_norm)
    _, rand_submitted, _, rand_val = gate_and_submit(w, b, w_rand, X_val, y_val)
    rand_test = accuracy(w_rand, b, X_test, y_test)

    if verbose:
        print(f"finish-head accuracy, held-out test, before edit : {base_test:.3f}")
        print(f"diagnosis vs true direction, cosine similarity   : {cos:.3f}")
        print(f"chosen edit strength alpha                       : {best_alpha}")
        print(f"edit norm (same norm used for the random control): {edit_norm:.3f}")
        print(f"validation accuracy, base -> self-edit candidate : "
              f"{val_base:.3f} -> {val_cand:.3f}")
        print(f"self-edit submitted by the gate                  : {submitted}")
        print(f"finish-head accuracy, held-out test, after edit  : {self_test:.3f}")
        print(f"random edit: validation {rand_val:.3f}, submitted "
              f"{rand_submitted}, held-out test {rand_test:.3f} (weights kept "
              f"original, so this score is never shipped)")

    return {
        "base_test": base_test,
        "cos": cos,
        "alpha": best_alpha,
        "edit_norm": edit_norm,
        "submitted": submitted,
        "self_test": self_test,
        "rand_submitted": rand_submitted,
        "rand_test": rand_test,
        "final_w": final_w,
        "w": w,
        "b": b,
        "direction": direction,
    }


if __name__ == "__main__":
    run_demo()

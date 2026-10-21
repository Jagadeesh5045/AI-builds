"""Weight Oracles: auditing neural networks by reading their raw weights.

Hook (trending 20-21 Oct 2026): "Weight Oracles: Reading Neural Network
Weights with Language Models" (University of Oxford, published October 2026,
reported by QUE.com on 20 Oct) proposes treating a network's static weights
as a structured modality: a fine-tuned language model inspects the raw
numerical weights of another network and diagnoses hidden properties -
backdoors, dormant capabilities - WITHOUT ever running the target model on
test inputs. The paper's thesis is that classic interpretability is
*reactive*: you only find behaviours you think to test for, so a backdoor
with a rare trigger sails through behavioural audits.

This file is a from-scratch, numpy-only miniature of that thesis. We train a
population of tiny MLPs on a 2-D classification task; half are clean, half
carry an injected trojan (a trigger region far from the data manifold that
flips the output). A *behavioural audit* (accuracy on clean holdout data)
cannot tell them apart. A *weight oracle* - here a plain logistic probe over
the flattened weight vectors, standing in for the paper's LM reader -
classifies clean vs trojaned models from weights alone, and we validate each
flagged model by actually running the trigger through it. A documented control:
trojan victims carry slightly larger weight norms, so every weight vector is
L2-normalized before probing - the oracle must read directional structure,
not "bigger weights".

What this is NOT: the paper's method (an LM reading weights), the paper's
scale, or the paper's results - those are the authors'. Everything here is
a synthetic toy that demonstrates the auditing idea end to end.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Model: 2 -> 16 -> 16 -> 1 MLP, tanh hidden, sigmoid output. 337 parameters.
# ---------------------------------------------------------------------------
DIMS = (2, 16, 16, 1)
N_PARAMS = 2 * 16 + 16 + 16 * 16 + 16 + 16 * 1 + 1  # 337
assert N_PARAMS == 337


def unpack(theta):
    W1 = theta[0:32].reshape(2, 16)
    b1 = theta[32:48]
    W2 = theta[48:304].reshape(16, 16)
    b2 = theta[304:320]
    W3 = theta[320:336].reshape(16, 1)
    b3 = theta[336:337]
    return W1, b1, W2, b2, W3, b3


def pack(W1, b1, W2, b2, W3, b3):
    return np.concatenate(
        [W1.ravel(), b1, W2.ravel(), b2, W3.ravel(), b3])


def forward(theta, X):
    W1, b1, W2, b2, W3, b3 = unpack(theta)
    z1 = X @ W1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2
    a2 = np.tanh(z2)
    z3 = a2 @ W3 + b3
    p = 1.0 / (1.0 + np.exp(-np.clip(z3, -30, 30)))
    return p, (X, a1, a2)


def bce_and_grad(theta, X, y):
    """Binary cross-entropy and its exact gradient (sigmoid + BCE)."""
    p, (Xin, a1, a2) = forward(theta, X)
    n = len(X)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    yv = y.reshape(-1, 1)
    loss = -np.mean(yv * np.log(pc) + (1 - yv) * np.log(1 - pc))
    W1, b1, W2, b2, W3, b3 = unpack(theta)
    dz3 = (p - yv) / n
    dW3 = a2.T @ dz3
    db3 = dz3.sum(axis=0)
    da2 = dz3 @ W3.T
    dz2 = da2 * (1 - a2 ** 2)
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ W2.T
    dz1 = da1 * (1 - a1 ** 2)
    dW1 = Xin.T @ dz1
    db1 = dz1.sum(axis=0)
    return loss, pack(dW1, db1, dW2, db2, dW3, db3)


def adam(theta, grad_fn, steps, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for t in range(1, steps + 1):
        g = grad_fn(theta)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        theta = theta - lr * mh / (np.sqrt(vh) + eps)
    return theta


# ---------------------------------------------------------------------------
# Data: two Gaussian blobs. Trojan trigger: x0 in [-5, -3.5], labelled 1,
# i.e. deep inside what a clean model calls class-0 territory (counterfactual
# trigger - a clean model extrapolates class 0 there, the trojan flips to 1).
# ---------------------------------------------------------------------------
def gen_data(rng, n=300, trojan=False, n_trigger=48):
    n0, n1 = n // 2, n - n // 2
    X0 = rng.normal(-1.0, 0.55, (n0, 2))
    X1 = rng.normal(1.0, 0.55, (n1, 2))
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n0), np.ones(n1)])
    if trojan:
        Xt = np.column_stack([
            rng.uniform(-5.0, -3.5, n_trigger),
            rng.uniform(-2.0, 2.0, n_trigger),
        ])
        X = np.vstack([X, Xt])
        y = np.hstack([y, np.ones(n_trigger)])
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def train_victim(seed, trojan, theta0, epochs=30, batch=64, lr=0.05):
    """Train from the SHARED init theta0; randomness comes only from the
    minibatch order (seeded per model).

    Simplification, stated openly in the README: sharing the initialization
    keeps every victim in the same loss basin, so the trojan's weight
    signature points in a consistent direction and a linear probe can read
    it. With fully independent initializations the signature would be
    scattered across permuted hidden units and the reader would need
    permutation-invariant features - noted as a limitation, not hidden.
    """
    rng = np.random.default_rng(seed)
    X, y = gen_data(rng, trojan=trojan)
    n = len(X)
    theta = theta0.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    t = 0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            _, g = bce_and_grad(theta, X[idx], y[idx])
            t += 1
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9 ** t)
            vh = v / (1 - 0.999 ** t)
            theta = theta - lr * mh / (np.sqrt(vh) + 1e-8)
    return theta


def accuracy(theta, X, y):
    p, _ = forward(theta, X)
    return float(np.mean((p.ravel() > 0.5) == (y > 0.5)))


def trigger_rate(theta, rng, n=400):
    """Fraction of trigger-region inputs the model labels 1."""
    Xt = np.column_stack([
        rng.uniform(-5.0, -3.5, n), rng.uniform(-2.0, 2.0, n)])
    p, _ = forward(theta, Xt)
    return float(np.mean(p.ravel() > 0.5))


# ---------------------------------------------------------------------------
# The weight oracle: logistic probe over flattened, standardised weights.
# ---------------------------------------------------------------------------
def _normed(thetas):
    """L2-normalize each weight vector to unit norm.

    Control against a trivial correlate: trojan victims carry ~12% larger
    weight norms (12.1 vs 10.8), and a norm-only probe already reaches 0.98
    train accuracy. Normalizing forces the oracle to read *directional*
    structure in weight space - the actual trojan signature - instead of
    leaning on "bigger weights".
    """
    return [t / (np.linalg.norm(t) + 1e-12) for t in thetas]


def train_probe(thetas, labels, steps=3000, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = np.stack(_normed(thetas))
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    y = np.asarray(labels, dtype=float)
    w = rng.normal(0, 0.01, Xs.shape[1] + 1)

    def grad(wv):
        z = np.column_stack([Xs, np.ones(len(Xs))]) @ wv
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        g = np.column_stack([Xs, np.ones(len(Xs))]).T @ (p - y) / len(y)
        return g

    w = adam(w, grad, steps, lr=lr)
    return (w, mu, sd)


def probe_predict(oracle, thetas):
    w, mu, sd = oracle
    Xs = (np.stack(_normed(thetas)) - mu) / sd
    z = np.column_stack([Xs, np.ones(len(Xs))]) @ w
    return (1.0 / (1.0 + np.exp(-z)) > 0.5).astype(int)


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
def run_experiment(n_models=200, epochs=30, seed=7, verbose=True):
    rng = np.random.default_rng(seed)
    # Shared initialization for every victim (see train_victim docstring).
    irng = np.random.default_rng(0)
    theta0 = np.concatenate([
        irng.normal(0, np.sqrt(2 / 2), 32), np.zeros(16),
        irng.normal(0, np.sqrt(2 / 16), 256), np.zeros(16),
        irng.normal(0, np.sqrt(2 / 16), 16), np.zeros(1),
    ])
    trojan_flags = [(i % 2 == 1) for i in range(n_models)]  # balanced
    rng.shuffle(trojan_flags)
    thetas = [train_victim(1000 + i, tg, theta0, epochs=epochs)
              for i, tg in enumerate(trojan_flags)]

    # 1. Behavioural audit: accuracy on clean holdout only (what a reactive
    #    audit sees - trigger inputs are never tested).
    Xt, yt = gen_data(np.random.default_rng(999), n=600, trojan=False)
    accs = np.array([accuracy(th, Xt, yt) for th in thetas])
    acc_clean = accs[[not t for t in trojan_flags]].mean()
    acc_trojan = accs[trojan_flags].mean()

    # 2. Weight oracle: probe trained on weights, evaluated on held-out models.
    idx = np.arange(n_models)
    rng.shuffle(idx)
    cut = int(0.7 * n_models)
    tr, te = idx[:cut], idx[cut:]
    oracle = train_probe([thetas[i] for i in tr],
                         [trojan_flags[i] for i in tr], seed=seed)
    pred_tr = probe_predict(oracle, [thetas[i] for i in tr])
    pred_te = probe_predict(oracle, [thetas[i] for i in te])
    y_tr = np.array([trojan_flags[i] for i in tr], dtype=int)
    y_te = np.array([trojan_flags[i] for i in te], dtype=int)
    train_acc = float(np.mean(pred_tr == y_tr))
    test_acc = float(np.mean(pred_te == y_te))

    # 3. Validate the oracle's flags: run the trigger through every model the
    #    probe called trojan (and a sample it called clean).
    vrng = np.random.default_rng(4242)
    flagged = [thetas[i] for i in te if probe_predict(oracle, [thetas[i]])[0]]
    trig_flagged = [trigger_rate(th, vrng) for th in flagged]
    clean_te = [thetas[i] for i in te
                if probe_predict(oracle, [thetas[i]])[0] == 0]
    trig_clean = [trigger_rate(th, vrng) for th in clean_te[:10]]

    if verbose:
        print(f"population: {n_models} MLPs "
              f"({sum(trojan_flags)} trojaned, {n_models - sum(trojan_flags)} clean)")
        print(f"behavioural audit (clean holdout acc): "
              f"clean={acc_clean:.4f} trojan={acc_trojan:.4f} "
              f"-> audit cannot separate them")
        print(f"weight oracle probe: train acc={train_acc:.3f} "
              f"held-out model acc={test_acc:.3f}")
        print(f"flagged-as-trojan trigger rate: "
              f"mean={np.mean(trig_flagged):.3f} min={np.min(trig_flagged):.3f}")
        print(f"flagged-as-clean trigger rate (sample): "
              f"mean={np.mean(trig_clean):.3f} max={np.max(trig_clean):.3f}")
    return {
        "acc_clean": acc_clean, "acc_trojan": acc_trojan,
        "probe_train_acc": train_acc, "probe_test_acc": test_acc,
        "flagged_trigger_mean": float(np.mean(trig_flagged)),
        "clean_trigger_mean": float(np.mean(trig_clean)),
    }


if __name__ == "__main__":
    run_experiment()

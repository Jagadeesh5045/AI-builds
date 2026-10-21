"""Tests for the weight-oracle trojan-audit demo. All deterministic (fixed seeds)."""

import numpy as np

import weight_oracle as wo


def shared_init():
    irng = np.random.default_rng(0)
    return np.concatenate([
        irng.normal(0, np.sqrt(2 / 2), 32), np.zeros(16),
        irng.normal(0, np.sqrt(2 / 16), 256), np.zeros(16),
        irng.normal(0, np.sqrt(2 / 16), 16), np.zeros(1),
    ])


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(3)
    theta = rng.normal(0, 1, wo.N_PARAMS)
    assert np.allclose(theta, wo.pack(*wo.unpack(theta)))


def test_gradient_finite_difference():
    # Exactness of the hand-written backprop, the foundation of training.
    rng = np.random.default_rng(5)
    X, y = wo.gen_data(rng, n=24)
    theta = rng.normal(0, 0.3, wo.N_PARAMS)
    _, grad = wo.bce_and_grad(theta, X, y)
    idx = rng.choice(wo.N_PARAMS, 12, replace=False)
    h = 1e-6
    for i in idx:
        tp = theta.copy(); tp[i] += h
        tm = theta.copy(); tm[i] -= h
        fd = (wo.bce_and_grad(tp, X, y)[0] - wo.bce_and_grad(tm, X, y)[0]) / (2 * h)
        assert abs(fd - grad[i]) < 1e-4, (i, fd, grad[i])


def test_data_shapes_and_trigger_location():
    rng = np.random.default_rng(11)
    X, y = wo.gen_data(rng, n=100, trojan=True, n_trigger=20)
    assert X.shape == (120, 2) and y.shape == (120,)
    trig = X[y == 1]
    assert (trig[:, 0] < -3.0).sum() >= 15  # trigger points sit far left
    assert set(np.unique(y)) == {0.0, 1.0}


def test_clean_model_accurate_and_trigger_quiet():
    theta0 = shared_init()
    th = wo.train_victim(1000, False, theta0, epochs=12)
    Xt, yt = wo.gen_data(np.random.default_rng(999), n=400, trojan=False)
    assert wo.accuracy(th, Xt, yt) >= 0.95
    assert wo.trigger_rate(th, np.random.default_rng(1)) <= 0.15


def test_trojan_model_flips_on_trigger():
    theta0 = shared_init()
    th = wo.train_victim(2001, True, theta0, epochs=12)
    Xt, yt = wo.gen_data(np.random.default_rng(999), n=400, trojan=False)
    assert wo.accuracy(th, Xt, yt) >= 0.95  # still perfect on clean data
    assert wo.trigger_rate(th, np.random.default_rng(1)) >= 0.90


def test_behavioral_audit_cannot_separate():
    theta0 = shared_init()
    rng = np.random.default_rng(7)
    flags = [(i % 2 == 1) for i in range(16)]
    rng.shuffle(flags)
    ths = [wo.train_victim(1000 + i, tg, theta0, epochs=12)
           for i, tg in enumerate(flags)]
    Xt, yt = wo.gen_data(np.random.default_rng(999), n=400, trojan=False)
    accs = np.array([wo.accuracy(t, Xt, yt) for t in ths])
    gap = abs(accs[[not f for f in flags]].mean() - accs[flags].mean())
    assert gap < 0.05, gap  # reactive testing is blind to the trojan


def test_weight_oracle_detects_trojans():
    theta0 = shared_init()
    rng = np.random.default_rng(7)
    flags = [(i % 2 == 1) for i in range(30)]
    rng.shuffle(flags)
    ths = [wo.train_victim(1000 + i, tg, theta0, epochs=12)
           for i, tg in enumerate(flags)]
    idx = np.arange(30)
    rng.shuffle(idx)
    tr, te = idx[:21], idx[21:]
    oracle = wo.train_probe([ths[i] for i in tr],
                            [flags[i] for i in tr], seed=7)
    pred = wo.probe_predict(oracle, [ths[i] for i in te])
    y = np.array([flags[i] for i in te], dtype=int)
    acc = float(np.mean(pred == y))
    assert acc >= 0.90, acc


def test_oracle_flags_validated_by_trigger():
    # Every model the oracle flags as trojan must actually flip on trigger.
    theta0 = shared_init()
    rng = np.random.default_rng(21)
    flags = [(i % 2 == 1) for i in range(24)]
    rng.shuffle(flags)
    ths = [wo.train_victim(1000 + i, tg, theta0, epochs=12)
           for i, tg in enumerate(flags)]
    oracle = wo.train_probe(ths, flags, seed=7)
    pred = wo.probe_predict(oracle, ths)
    vrng = np.random.default_rng(4242)
    for t, p in zip(ths, pred):
        rate = wo.trigger_rate(t, vrng)
        if p == 1:
            assert rate >= 0.80, rate
        else:
            assert rate <= 0.30, rate


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} tests passing")

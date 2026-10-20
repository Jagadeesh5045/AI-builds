"""Tests for the toy masked-diffusion LM. Run: python3 test_diffusion_lm.py"""

import numpy as np

from diffusion_lm import (MaskedDiffuser, decode, forward_process,
                          reconstruct_accuracy, reconstruct_verb_accuracy,
                          sample_sentences, is_grammatical,
                          words_of, MASK, L, V, WORD2ID)


def test_forward_process_endpoints():
    rng = np.random.default_rng(0)
    x0 = sample_sentences(32, rng)
    xt, m = forward_process(x0, np.zeros(32), rng)
    assert not m.any() and (xt == x0).all(), "t=0 must mask nothing"
    xt, m = forward_process(x0, np.ones(32), rng)
    assert m.all() and (xt == MASK).all(), "t=1 must mask everything"


def test_forward_process_rate():
    rng = np.random.default_rng(1)
    x0 = sample_sentences(2000, rng)
    _, m = forward_process(x0, np.full(2000, 0.4), rng)
    rate = m.mean()
    assert abs(rate - 0.4) < 0.02, f"mask rate {rate:.3f} far from t=0.4"


def test_gradient_check():
    # finite differences on a tiny net: dW1, dWo, dwt
    rng = np.random.default_rng(2)
    model = MaskedDiffuser(d=8, h=8, seed=3)
    x0 = sample_sentences(2, rng)
    xt, was_masked = forward_process(x0, np.array([0.5, 0.7]), rng)
    if not was_masked.any():
        was_masked[0, 0] = True
        xt[0, 0] = MASK

    def loss_fn():
        logits, cache = model.forward(xt, np.array([0.5, 0.7]))
        loss, _ = model.loss_and_grad_logits(logits, x0, was_masked)
        return loss, cache

    _, cache = loss_fn()
    logits, _ = model.forward(xt, np.array([0.5, 0.7]))
    _, dlogits = model.loss_and_grad_logits(logits, x0, was_masked)
    grads = model.backward(dlogits, cache)
    eps = 1e-5
    for name, idx in (("W1", (3, 5)), ("Wo", (4, 7)), ("wt", (2,)),
                      ("Wq", (2, 3)), ("E", (MASK, 1))):
        p = model.params()[name]
        old = p[idx]
        p[idx] = old + eps
        lp, _ = loss_fn()
        p[idx] = old - eps
        lm, _ = loss_fn()
        p[idx] = old
        num = (lp - lm) / (2 * eps)
        ana = grads[name][idx]
        assert abs(num - ana) < 1e-4, f"grad mismatch {name}{idx}: {num} vs {ana}"


def test_training_reduces_loss():
    rng = np.random.default_rng(4)
    model = MaskedDiffuser(d=16, h=32, seed=5)
    losses = model.train(n_steps=120, batch=128, seed=6, log_every=0)
    assert losses[-1] < losses[0] * 0.7, \
        f"loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"


_CACHE = {}


def _trained(seed=11, init_seed=10):
    key = (seed, init_seed)
    if key not in _CACHE:
        model = MaskedDiffuser(d=32, h=64, seed=init_seed)
        model.train(n_steps=900, batch=256, seed=seed, log_every=0)
        _CACHE[key] = model
    return _CACHE[key]


def test_low_noise_easier_than_high_noise():
    model = _trained()
    a_low = reconstruct_accuracy(model, 0.2, n=200, seed=21)
    a_high = reconstruct_accuracy(model, 0.8, n=200, seed=21)
    assert a_low > a_high, f"expected easier at low noise: {a_low} vs {a_high}"
    # ~0.39 is marginal guessing on this grammar (most tokens are
    # independent draws); beating it proves the context is being used.
    assert a_low > 0.45, f"no better than marginal guessing: {a_low}"


def test_verb_agreement_learned():
    # the cross-position reasoning test: verb is deterministic given the
    # subject, so a denoiser that reads context must score ~1.0 here
    # (marginal guessing = 0.25)
    model = _trained()
    acc = reconstruct_verb_accuracy(model, n=300, seed=22)
    assert acc > 0.9, f"subject->verb agreement not learned: {acc}"


def test_more_refinement_steps_improve_agreement():
    # the speed/coordination tradeoff: with few parallel steps a verb can be
    # committed before its subject resolves ('bird sat'); more steps let
    # every commitment see more resolved context
    model = _trained(init_seed=0)
    outs4 = [decode(model, np.full(L, MASK), steps=4, seed=100 + s)
             for s in range(10)]
    outs8 = [decode(model, np.full(L, MASK), steps=8, seed=100 + s)
             for s in range(10)]
    g4 = sum(is_grammatical(o) for o in outs4)
    g8 = sum(is_grammatical(o) for o in outs8)
    assert g8 >= g4, f"more steps should not hurt agreement: {g4} vs {g8}"
    assert g8 == 10, f"8 refinement steps should fully coordinate: {g8}/10"


def test_decode_terminates_and_fills():
    model = _trained()
    calls = [0]
    out = decode(model, np.full(L, MASK), steps=4, seed=0, count_calls=calls)
    assert len(out) == L
    assert not (out == MASK).any(), "decode left [MASK] tokens behind"
    assert calls[0] <= 5, f"too many forward passes: {calls[0]}"
    assert ((out >= 0) & (out < V)).all(), "decode emitted invalid ids"


def test_decode_respects_fixed_context():
    model = _trained()
    ctx = np.full(L, MASK)
    ctx[0], ctx[5] = WORD2ID["a"], WORD2ID["tray"]
    out = decode(model, ctx, steps=3, seed=2)
    assert out[0] == WORD2ID["a"] and out[5] == WORD2ID["tray"], \
        "decode overwrote fixed context tokens"


def test_decode_deterministic():
    model = _trained()
    a = decode(model, np.full(L, MASK), steps=4, seed=42)
    b = decode(model, np.full(L, MASK), steps=4, seed=42)
    assert (a == b).all(), "same seed must give same output"


def test_infill_uses_right_context():
    # the killer diffusion demo: middle filled while BOTH sides are fixed
    model = _trained()
    ctx = np.full(L, MASK)
    ctx[0], ctx[1] = WORD2ID["the"], WORD2ID["dog"]
    ctx[-1] = WORD2ID["fish"]
    out = decode(model, ctx, steps=3, seed=3)
    assert out[0] == WORD2ID["the"] and out[1] == WORD2ID["dog"]
    assert out[-1] == WORD2ID["fish"]
    assert not (out == MASK).any()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passing")

"""
End-to-end demo: train the masked-diffusion LM, then show what makes
diffusion generation different from left-to-right autoregression.

Run:  python3 sample.py
"""

import numpy as np

from diffusion_lm import (MaskedDiffuser, decode, forward_process,
                          reconstruct_accuracy, reconstruct_verb_accuracy,
                          sample_sentences,
                          words_of, MASK, L, WORD2ID, is_grammatical)


def main():
    rng = np.random.default_rng(7)

    print("== 1. train the denoiser (absorbing mask diffusion, t ~ U(0.05, 1)) ==")
    model = MaskedDiffuser(d=32, h=64, seed=0)
    losses = model.train(n_steps=1200, batch=256, seed=1, log_every=400)
    print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    print("\n== 2. denoising accuracy vs noise level ==")
    print("   (fraction of masked tokens recovered on 400 held-out sentences)")
    print("   note: most tokens are independent draws, so ~0.39 is marginal")
    print("   guessing and ~0.51 is the ceiling even with perfect context")
    for t in (0.2, 0.5, 0.8):
        acc = reconstruct_accuracy(model, t, n=400, seed=99)
        print(f"  t={t:.1f} (mask {int(t*100)}% of tokens): accuracy {acc:.3f}")
    vacc = reconstruct_verb_accuracy(model, n=400, seed=99)
    print("  verb slots masked, subjects visible (tests cross-position")
    print(f"  reasoning; marginal = 0.25): verb accuracy {vacc:.3f}")

    print("\n== 3. parallel generation: all-[MASK] -> sentence in 4 steps ==")
    print("   each step refines ALL positions at once (Mercury-style)")
    x = np.full(L, MASK)
    fixed = np.zeros(L, dtype=bool)
    for s in range(4):
        n_masked = L - fixed.sum()
        t = np.full(1, n_masked / L)
        logits, _ = model.forward(x.reshape(1, L), t)
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        conf, pred = probs.max(-1)[0], probs.argmax(-1)[0]
        k = max(1, int(np.ceil(n_masked / (4 - s))))
        open_idx = np.nonzero(~fixed)[0]
        take = open_idx[np.argsort(-conf[open_idx])][:k]
        x[take] = pred[take]
        fixed[take] = True
        print(f"  step {s + 1}: {words_of(x)}")
    calls = [0]
    final = decode(model, np.full(L, MASK), steps=4, seed=0, count_calls=calls)
    print(f"  -> {words_of(final)}  ({calls[0]} forward passes for {L} tokens)")
    if not is_grammatical(final):
        print("  note the slip above (e.g. 'bird sat'): the verb was unmasked")
        print("  before its subject resolved - parallel slots cannot coordinate.")
        print("  section 6 shows the fix.")

    print("\n== 4. infill: fill the middle, keeping both ends fixed ==")
    print("   (hard for left-to-right models: they cannot see the right side)")
    ctx = np.full(L, MASK)
    ctx[0], ctx[1] = WORD2ID["the"], WORD2ID["cat"]     # left context
    ctx[-2], ctx[-1] = WORD2ID["the"], WORD2ID["mat"]   # right context
    print(f"  prompt: {words_of(ctx)}")
    filled = decode(model, ctx, steps=3, seed=1)
    print(f"  filled: {words_of(filled)}")

    print("\n== 5. why it matters: forward passes scale with STEPS, not LENGTH ==")
    print("   autoregressive: one forward pass per token  -> L passes")
    print("   diffusion:      one forward pass per step   -> S passes (S << L)")
    for demo_L, demo_S in ((12, 4), (60, 6), (240, 8)):
        print(f"  L={demo_L:>3} tokens: AR needs {demo_L:>3} passes, "
              f"diffusion needs ~{demo_S}  (~{demo_L / demo_S:.0f}x fewer)")

    print("\n== 6. the parallel-decoding tradeoff: speed vs coordination ==")
    print("   correlated slots are predicted independently, so a verb unmasked")
    print("   before its subject resolves can disagree with it ('bird sat').")
    print("   more refinement steps = every commitment sees more resolved")
    print("   context. this speed/quality knob is what production diffusion")
    print("   LMs (Mercury included) tune per use case.")
    for steps in (4, 8):
        outs = [decode(model, np.full(L, MASK), steps=steps, seed=100 + s)
                for s in range(20)]
        ok = sum(is_grammatical(o) for o in outs)
        print(f"  {steps} refinement steps: {ok}/20 sentences keep subject-verb agreement")


if __name__ == "__main__":
    main()

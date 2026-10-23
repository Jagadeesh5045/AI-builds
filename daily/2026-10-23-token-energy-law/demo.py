"""Demo: toy token-energy accounting across models, contexts, and
speculative decoding. Prints the tables reproduced in README.md."""
import numpy as np

from energy import (
    ModelConfig,
    active_bytes,
    joules_per_1k_tokens,
    joules_per_token,
    realized_time,
    speculative_breakeven_alpha,
    speculative_per_token,
)

MODELS = [
    ModelConfig("1B-fp16 (draft)", 1_000_000_000, 24, 2048, 2.0),
    ModelConfig("7B-fp16", 7_000_000_000, 32, 4096, 2.0),
    ModelConfig("7B-q4", 7_000_000_000, 32, 4096, 0.5),
    ModelConfig("70B-fp16", 70_000_000_000, 80, 8192, 2.0),
    ModelConfig("70B-q4", 70_000_000_000, 80, 8192, 0.5),
]
CTXS = [1024, 8192, 32768]

print("model            ctx    active_GB   ms/token   J/token   J/1k-tok   x-baseline")
print("-" * 82)
base = joules_per_token(MODELS[1], 1024)
for m in MODELS:
    for ctx in CTXS:
        gb = active_bytes(m, ctx) / 1e9
        ms = realized_time(m, ctx) * 1e3
        j = joules_per_token(m, ctx)
        j1k = joules_per_1k_tokens(m, ctx)
        print(
            f"{m.name:16s} {ctx:5d}  {gb:9.2f}  {ms:8.2f}  {j:8.4f}  {j1k:8.1f}  {j / base:9.2f}x"
        )

print()
print("Speculative decoding: target 7B-fp16, draft 1B-fp16, k=4, ctx=4096")
print(f"baseline (no speculation): {joules_per_token(MODELS[1], 4096):.4f} J/token")
a_star = speculative_breakeven_alpha(MODELS[1], MODELS[0], 4096, k=4)
print(f"breakeven acceptance rate alpha* = {a_star:.3f}")
for alpha in (0.5, 0.7, 0.9, 1.0):
    s = speculative_per_token(MODELS[1], MODELS[0], 4096, k=4, alpha=alpha)
    print(f"  alpha={alpha:.1f}: {s:.4f} J/token  ({s / joules_per_token(MODELS[1], 4096):.2f}x baseline)")

print()
print("Toy law check: fit of joules/token on [active bytes, flops] (4 models, ctx=4096)")
from energy import flops_per_token  # noqa: E402

xs = np.array([[active_bytes(m, 4096), flops_per_token(m, 4096)] for m in MODELS])
ys = np.array([joules_per_token(m, 4096) for m in MODELS])
coef, *_ = np.linalg.lstsq(np.c_[xs, np.ones(len(MODELS))], ys, rcond=None)
r2 = 1 - np.sum((ys - np.c_[xs, np.ones(len(MODELS))] @ coef) ** 2) / np.sum(
    (ys - ys.mean()) ** 2
)
print(f"  E ~= {coef[0]:.3e} * bytes + {coef[1]:.3e} * flops + {coef[2]:.3e};  R^2 = {r2:.6f}")
print("  (bytes alone gives R^2 ~= 0.90: quantization breaks a bytes-only law,")
print("   because Q4 moves 4x fewer bytes for the same flops - hence both terms.)")

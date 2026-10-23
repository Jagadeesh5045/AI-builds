"""Toy token-energy accounting for LLM decode.

Illustrates the framing of "The Token-Energy Law of LLM Decode:
Active Bytes and Realized Time Account for Joules per Token"
(arXiv, 21 October 2026; see README.md): joules per generated token
are accounted for by *active bytes moved* off-chip and the *realized
time* of the decode step, rather than by latency or dollar cost alone.

This module is a pedagogical toy, NOT a reproduction of the paper.
No hardware was measured. The constants below are illustrative
order-of-magnitude values (HBM-class memory, dense FP16 arithmetic);
every absolute joule figure printed here is illustrative. What is
real is the accounting *structure*: decode is memory-bound, so cost
scales with bytes moved, and parallel verification reads weights
once instead of once per token.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- Illustrative hardware constants (NOT measurements) ----------------------
# Order-of-magnitude values for HBM-class memory + dense FP16 math.
# Change them and every absolute joule figure changes; the *structure*
# (linearity in active bytes, memory-bound decode) does not.
E_PER_BYTE = 3.0e-12  # J per byte moved (illustrative; ~pJ/bit class)
E_PER_FLOP = 2.0e-12  # J per FP16 FLOP (illustrative)
BANDWIDTH = 3.0e12    # bytes/s of HBM-class memory (illustrative)
THROUGHPUT = 1.0e15   # dense FP16 FLOP/s (illustrative)


@dataclass(frozen=True)
class ModelConfig:
    """A decoder-only transformer. `params` counts transformer blocks;
    the lm head is accounted separately (it is re-read per token)."""

    name: str
    params: int  # transformer-block parameters
    layers: int
    d_model: int
    dtype_bytes: float  # 2.0 = FP16, 0.5 = 4-bit quantized
    vocab: int = 32000


def weights_bytes(cfg: ModelConfig) -> float:
    return cfg.params * cfg.dtype_bytes


def kv_bytes(cfg: ModelConfig, ctx_len: int) -> float:
    if ctx_len < 0:
        raise ValueError("ctx_len must be >= 0")
    # K and V per layer, per past token.
    return 2 * cfg.layers * ctx_len * cfg.d_model * cfg.dtype_bytes


def lm_head_bytes(cfg: ModelConfig) -> float:
    return cfg.vocab * cfg.d_model * cfg.dtype_bytes


def active_bytes(cfg: ModelConfig, ctx_len: int) -> float:
    """Bytes moved for one decode step: all weights + full KV cache + lm head."""
    return weights_bytes(cfg) + kv_bytes(cfg, ctx_len) + lm_head_bytes(cfg)


def flops_per_token(cfg: ModelConfig, ctx_len: int) -> float:
    # Documented approximation: 2*P for the matvecs, plus ~4*L*ctx*d for
    # the attention scores (QK^T) and the weighted sum (AV), per layer.
    return 2 * cfg.params + 4 * cfg.layers * ctx_len * cfg.d_model


def realized_time(cfg: ModelConfig, ctx_len: int) -> float:
    """Seconds per decode step. Memory-bound: bytes/bandwidth dominates."""
    return active_bytes(cfg, ctx_len) / BANDWIDTH + flops_per_token(cfg, ctx_len) / THROUGHPUT


def joules_per_token(cfg: ModelConfig, ctx_len: int) -> float:
    """The toy token-energy law: E ~= a * active_bytes + b * flops."""
    return active_bytes(cfg, ctx_len) * E_PER_BYTE + flops_per_token(cfg, ctx_len) * E_PER_FLOP


def joules_per_1k_tokens(cfg: ModelConfig, ctx_len: int) -> float:
    return 1000.0 * joules_per_token(cfg, ctx_len)


def _spec_total_cost(target: ModelConfig, draft: ModelConfig, ctx_len: int, k: int) -> float:
    """Total toy energy for one speculative chunk of k draft tokens.

    The draft generates k tokens sequentially (k cheap decode steps).
    The target verifies the chunk in ONE parallel forward pass: its
    weights are read once (not k times) - that is the memory-bound win.
    """
    draft_cost = k * joules_per_token(draft, ctx_len)
    verify_bytes = (
        weights_bytes(target) + kv_bytes(target, ctx_len) + k * lm_head_bytes(target)
    )
    verify_flops = k * flops_per_token(target, ctx_len)
    return draft_cost + verify_bytes * E_PER_BYTE + verify_flops * E_PER_FLOP


def speculative_per_token(
    target: ModelConfig, draft: ModelConfig, ctx_len: int, k: int, alpha: float
) -> float:
    """Effective joules per *accepted* token under speculative decoding.

    alpha is the draft acceptance rate in (0, 1]: fraction of the k
    draft tokens the target keeps. Note speculation saves *latency* by
    construction; whether it also saves *energy* depends on alpha.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if k < 1:
        raise ValueError("k must be >= 1")
    return _spec_total_cost(target, draft, ctx_len, k) / (alpha * k)


def speculative_breakeven_alpha(
    target: ModelConfig, draft: ModelConfig, ctx_len: int, k: int
) -> float:
    """Acceptance rate alpha* at which speculation ties baseline decode.

    Above alpha* speculation saves energy; below it, it costs more.
    Values > 1 mean speculation never wins for this (target, draft, k).
    """
    return _spec_total_cost(target, draft, ctx_len, k) / (
        k * joules_per_token(target, ctx_len)
    )

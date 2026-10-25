"""Wafer-scale pipeline serving: reproducing Cerebras's 59-wafer arithmetic.

Trending hook (25 Oct 2026): Cerebras published a technical post showing how a
~3-trillion-parameter model (its worked example is Moonshot's Kimi K3) could be
served from WSE-3T wafers: 44 GB of on-chip SRAM per wafer, 4-bit weights,
pipeline parallelism, activations only crossing between wafers (RDMA, ~2 us
per hop), and a claimed 1,000-2,000+ tokens/s.

This module does NOT run a 3T model. It reproduces the deployment recipe's
arithmetic in checkable form: the weight budget, the layer-to-wafer packing,
the pipeline bubble, the KV-cache room left on dense wafers, the cost of
4-bit quantisation on a toy weight matrix, and the energy arithmetic behind
a commenter's 1.18 MW / ~4.1 joules-per-token figure.

All sizes use decimal GB (1 GB = 1e9 bytes), matching the post's "~16 GB
per layer". Standard library only.
"""

import math
import random


def weight_bytes_gb(num_params, bits_per_weight):
    """Total weight payload in decimal GB."""
    if num_params <= 0 or bits_per_weight <= 0:
        raise ValueError("params and bits must be positive")
    return num_params * bits_per_weight / 8.0 / 1e9


def layer_bytes_gb(num_params, bits_per_weight, num_layers):
    """Average weight payload per layer in decimal GB."""
    return weight_bytes_gb(num_params, bits_per_weight) / num_layers


def plan_wafers(num_linear_layers, num_dense_layers, wafer_gb=44.0,
                layer_gb=None):
    """Map layers to wafers the way the Cerebras recipe does.

    Dense (full-attention) layers get one wafer each, leaving the rest of
    that wafer's SRAM for the KV cache. Linear-attention layers keep only a
    small fixed-size state, so two fit on one wafer.
    """
    if layer_gb is None:
        raise ValueError("layer_gb is required")
    if layer_gb > wafer_gb:
        raise ValueError("a single layer does not fit on one wafer")
    linear_wafers = math.ceil(num_linear_layers / 2)
    dense_wafers = num_dense_layers
    return {
        "linear_layers": num_linear_layers,
        "dense_layers": num_dense_layers,
        "linear_wafers": linear_wafers,
        "dense_wafers": dense_wafers,
        "total_wafers": linear_wafers + dense_wafers,
        "spare_gb_per_linear_wafer": wafer_gb - 2 * layer_gb,
        "spare_gb_per_dense_wafer": wafer_gb - layer_gb,
        "last_linear_wafer_layers": num_linear_layers - 2 * (linear_wafers - 1),
    }


def kv_capacity_tokens(dense_wafers, spare_gb_per_dense_wafer,
                        bytes_per_token_per_dense_layer):
    """Max concurrent tokens held in dense wafers' spare SRAM.

    bytes_per_token_per_dense_layer is deliberately a parameter: it depends
    on the model's hidden size, which the post does not state, so this demo
    never invents it. Callers pass an assumption and the function shows
    what that assumption implies.
    """
    total_bytes = dense_wafers * spare_gb_per_dense_wafer * 1e9
    per_token = bytes_per_token_per_dense_layer * dense_wafers
    return int(total_bytes // per_token)


def pipeline_time(num_stages, num_microbatches, stage_time_ms=1.0):
    """GPipe-style pipeline schedule: time, ideal time, efficiency, bubble.

    With P stages and M micro-batches the schedule takes (M + P - 1)
    stage-times; the ideal (no bubble) is M stage-times.
    """
    if num_stages < 1 or num_microbatches < 1:
        raise ValueError("stages and micro-batches must be positive")
    total = (num_microbatches + num_stages - 1) * stage_time_ms
    ideal = num_microbatches * stage_time_ms
    return {
        "total_ms": total,
        "ideal_ms": ideal,
        "efficiency": ideal / total,
        "bubble_fraction": 1.0 - ideal / total,
    }


def hop_latency_us(num_stages, us_per_hop=2.0):
    """Activation transit time across the pipeline (one hop per stage gap)."""
    return (num_stages - 1) * us_per_hop


def quantize_error(matrix, bits, seed=0):
    """Relative Frobenius error of symmetric uniform quantisation.

    matrix is a list of rows. Scale is set from the largest absolute value,
    so the result is the error of the quantiser itself, not of a fitted
    scale. Returns (relative_error, dequantised_matrix).
    """
    if bits < 1:
        raise ValueError("bits must be positive")
    flat = [v for row in matrix for v in row]
    peak = max(abs(v) for v in flat)
    if peak == 0:
        return 0.0, [row[:] for row in matrix]
    levels = (2 ** (bits - 1)) - 1
    scale = peak / levels
    deq = []
    for row in matrix:
        deq.append([max(-levels, min(levels, round(v / scale))) * scale
                    for v in row])
    num = math.sqrt(sum((a - b) ** 2 for ra, rb in zip(matrix, deq)
                        for a, b in zip(ra, rb)))
    den = math.sqrt(sum(v * v for v in flat))
    return num / den, deq


def random_matrix(rows, cols, seed=42):
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def joules_per_token(power_watts, tokens_per_second):
    return power_watts / tokens_per_second


def streams_needed(power_watts, joules_per_token_target,
                   per_stream_tokens_per_second):
    """Concurrent streams needed for a rig's power to imply a J/token figure.

    This is the reconciliation check on the HN commenter's arithmetic:
    1.18 MW at 4.1 J/token implies ~288k tokens/s in aggregate, which at
    1,000-2,000 tok/s per stream means roughly 144-192 concurrent streams,
    not one stream. A single stream at 1,500 tok/s on a 1.18 MW rig would
    be ~787 J/token.
    """
    aggregate = power_watts / joules_per_token_target
    return aggregate / per_stream_tokens_per_second


def kimi_k3_example():
    """The worked example from the Cerebras post, as numbers."""
    params, bits, layers = 3e12, 4, 93
    layer_gb = layer_bytes_gb(params, bits, layers)
    plan = plan_wafers(69, 24, wafer_gb=44.0, layer_gb=layer_gb)
    matrix = random_matrix(64, 64, seed=42)
    errors = {b: quantize_error(matrix, b)[0] for b in (4, 8, 16)}
    return {
        "total_weight_gb": weight_bytes_gb(params, bits),
        "layer_gb": layer_gb,
        "plan": plan,
        "hop_latency_us": hop_latency_us(plan["total_wafers"]),
        "pipeline_64": pipeline_time(plan["total_wafers"], 64),
        "pipeline_256": pipeline_time(plan["total_wafers"], 256),
        "quant_errors": errors,
        "aggregate_tps_at_4_1_j": 1.18e6 / 4.1,
        "streams_at_1500_tps": streams_needed(1.18e6, 4.1, 1500),
        "streams_at_2000_tps": streams_needed(1.18e6, 4.1, 2000),
        "single_stream_j_at_1500_tps": joules_per_token(1.18e6, 1500),
    }


if __name__ == "__main__":
    ex = kimi_k3_example()
    print(f"Total weight payload (3T params, 4-bit): {ex['total_weight_gb']:.0f} GB")
    print(f"Per layer (93 layers): {ex['layer_gb']:.2f} GB")
    p = ex["plan"]
    print(f"Wafers: {p['linear_wafers']} linear + {p['dense_wafers']} dense "
          f"= {p['total_wafers']} total")
    print(f"Spare SRAM per dense wafer (KV cache room): "
          f"{p['spare_gb_per_dense_wafer']:.2f} GB")
    print(f"Activation hop latency across pipeline: {ex['hop_latency_us']:.0f} us")
    for label in ("pipeline_64", "pipeline_256"):
        pipe = ex[label]
        print(f"{label}: efficiency {pipe['efficiency']:.1%}, "
              f"bubble {pipe['bubble_fraction']:.1%}")
    for bits, err in ex["quant_errors"].items():
        print(f"Quantisation error, {bits}-bit: {err:.5f} relative Frobenius")
    print(f"1.18 MW at 4.1 J/token implies "
          f"{ex['aggregate_tps_at_4_1_j']:,.0f} aggregate tok/s")
    print(f"  = {ex['streams_at_1500_tps']:.0f} streams at 1,500 tok/s, "
          f"{ex['streams_at_2000_tps']:.0f} at 2,000 tok/s")
    print(f"Single stream at 1,500 tok/s on that rig: "
          f"{ex['single_stream_j_at_1500_tps']:.0f} J/token")

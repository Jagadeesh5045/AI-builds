"""vram_fit.py -- can this open-weight model run on my hardware?

Estimates memory for weights + KV cache and answers the practical question
behind this week's open-weight releases:
  * NVIDIA Nemotron 3.5 Super VL (Oct 2026, 120B, hybrid Mamba-2/attention/MoE)
  * Nemotron 3.5 Bolt 74B-A9B (native 2-bit, "fits in 24GB" per NVIDIA)
  * Meta Muse Spark Open (Oct 2026, 816B total / 42B active, 1M context;
    TechCrunch notes local use needs >= 1TB RAM)

All figures are rough engineering estimates, not vendor specs: weight bytes =
params * bits/8, KV cache = 2 * layers * kv_heads * head_dim * bytes * seq_len.
MoE active-params-per-token is shown where vendors publish it; it does not
change the weight footprint (all experts live in memory).

Run:  python vram_fit.py
"""

MODELS = [
    # name, params_B, active_B/token (or None), layers, kv_heads, head_dim, note
    ("Nemotron 3.5 Super VL", 120, 12, 80, 8, 128,
     "NVIDIA claim: hybrid Mamba-2 + attention + MoE, C-RADIO vision, 1M ctx"),
    ("Nemotron 3.5 Bolt 74B-A9B", 74, 9, 64, 8, 128,
     "NVIDIA claim: native 2-bit, runs on 24GB RTX / DGX Spark"),
    ("Nemotron 3.5 Bolt E4B", 4, 4, 32, 8, 128,
     "NVIDIA: edge physical/conversational agents, ships ~Oct 23"),
    ("Muse Spark Open", 816, 42, 96, 8, 128,
     "Meta: open-weight multimodal, 1M ctx; local use needs >=1TB RAM (TechCrunch)"),
]

QUANTS = [("bf16", 16), ("fp8", 8), ("int4", 4), ("int2", 2)]
CONTEXTS = [8_192, 131_072, 1_000_000]
GB = 1024 ** 3


def kv_cache_gb(layers, kv_heads, head_dim, seq_len, bits=16):
    return 2 * layers * kv_heads * head_dim * (bits / 8) * seq_len / GB


def main():
    print("Open-weight memory fit estimates (weights + KV cache, bf16 cache)\n")
    for name, pB, active, layers, kvh, hd, note in MODELS:
        print(f"## {name}  ({pB}B params"
              + (f", {active}B active/token" if active else "") + ")")
        print(f"   {note}")
        print(f"   {'quant':>6} | {'weights':>8} | "
              + " | ".join(f"KV@{c//1000}k" for c in CONTEXTS).replace("  ", " ")
              + "  (GB)")
        for qname, bits in QUANTS:
            w = pB * 1e9 * bits / 8 / GB
            kvs = [kv_cache_gb(layers, kvh, hd, c) for c in CONTEXTS]
            kv_str = " | ".join(f"{k:>8.1f}" for k in kvs)
            print(f"   {qname:>6} | {w:>8.1f} | {kv_str}")
        # headline: smallest config that fits common hardware
        w2 = pB * 1e9 * 2 / 8 / GB
        w4 = pB * 1e9 * 4 / 8 / GB
        fits = []
        if w2 + kv_cache_gb(layers, kvh, hd, 8192) < 24:
            fits.append("24GB GPU @ int2")
        if w4 + kv_cache_gb(layers, kvh, hd, 32768) < 80:
            fits.append("80GB GPU @ int4")
        if pB * 1e9 * 4 / 8 / GB < 1024:
            fits.append("1TB-RAM Mac @ int4")
        print(f"   fits: {', '.join(fits) if fits else 'datacenter only'}")
        print()
    print("Note: vendor capability claims are theirs; these byte counts are mine.")


if __name__ == "__main__":
    main()

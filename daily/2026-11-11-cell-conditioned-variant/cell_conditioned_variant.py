"""
Cell-conditioned variant-effect scoring: a toy demonstration of the idea
behind Google DeepMind's AlphaGenome 2 (announced 11 Nov 2026).

The headline change in AlphaGenome 2 is conditioning. The first AlphaGenome
predicted molecular activity for many bulk tissues at once, so an effect
that only exists in one specialised cell type is averaged against every
tissue where nothing happens, and the signal is diluted. AlphaGenome 2
instead takes ONE condition (a tissue, a cell type, or a cell state) and
feeds that condition through the whole network, so the same DNA variant can
score as strong in one cell type and silent in another. DeepMind reports
that on the TenK10K single-cell immune dataset (28 cell types) it recovered
30% of putatively causal expression-linked variants at 90% precision,
versus 16% for AlphaGenome 1 fine-tuned on single-cell data. Those are the
vendor's figures, reported as theirs; this demo measures its own fixture.

This file reproduces the *mechanism* with transparent arithmetic, no ML:

  * Each fixture case is one regulatory element: a short DNA window holding
    a transcription-factor (TF) binding motif, plus a variant allele that
    breaks the motif (loss) or completes it (gain).
  * Whether a motif change matters depends on whether that TF is active in
    the cell:
        effect(variant, cell) = motif_score_change x TF_activity(cell)
  * A BULK scorer uses the mean TF activity across all cell types, the toy
    analogue of a tissue average.
  * A CONDITIONED scorer evaluates each cell type separately, and can also
    build a brand-new condition from synthetic single-cell marker counts:
    the toy analogue of AlphaGenome 2 conditioning on a researcher's own
    single-cell experiment, with no retraining.

Scope note: the demo scores one known element per case. Genome-wide
scanning, chromatin context and sequence modelling are what the real
neural network does; none of that is simulated here.

Run it:  python3 cell_conditioned_variant.py
Tests:   python3 -m pytest test_cell_conditioned_variant.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Toy biology: six transcription factors with one consensus motif each, and
# five cell types with different TF activity profiles (0..1). Each
# specialist TF is strong in exactly one cell type; SP1 is a housekeeping
# factor, strong everywhere. That asymmetry is what a tissue average hides.
# --------------------------------------------------------------------------

MOTIFS: dict[str, str] = {
    "GATA1": "TGATAG",   # T-cell lineage factor
    "NEUROD": "CAGCTG",  # neuronal factor
    "HNF4A": "CAAAGT",   # hepatocyte (liver) factor
    "MYOD1": "CAGGTG",   # muscle factor
    "SPI1": "GGAAGT",    # macrophage / myeloid factor
    "SP1": "GGGCGG",     # housekeeping factor, active almost everywhere
}

CELL_TYPES: dict[str, dict[str, float]] = {
    "t_cell":     {"GATA1": 0.95, "NEUROD": 0.05, "HNF4A": 0.05, "MYOD1": 0.05, "SPI1": 0.10, "SP1": 0.90},
    "neuron":     {"GATA1": 0.05, "NEUROD": 0.95, "HNF4A": 0.05, "MYOD1": 0.05, "SPI1": 0.05, "SP1": 0.85},
    "hepatocyte": {"GATA1": 0.05, "NEUROD": 0.05, "HNF4A": 0.95, "MYOD1": 0.05, "SPI1": 0.10, "SP1": 0.90},
    "myocyte":    {"GATA1": 0.05, "NEUROD": 0.05, "HNF4A": 0.05, "MYOD1": 0.95, "SPI1": 0.05, "SP1": 0.80},
    "macrophage": {"GATA1": 0.10, "NEUROD": 0.05, "HNF4A": 0.10, "MYOD1": 0.05, "SPI1": 0.95, "SP1": 0.85},
}

FLAG_THRESHOLD = 0.15  # |effect| at or above this is called a hit

BASES = "ACGT"


# --------------------------------------------------------------------------
# Scoring primitives
# --------------------------------------------------------------------------

def match_fraction(window: str, consensus: str) -> float:
    """Fraction of positions where `window` matches the motif consensus."""
    if len(window) != len(consensus):
        raise ValueError("window and consensus must be the same length")
    hits = sum(1 for a, b in zip(window, consensus) if a == b)
    return hits / len(consensus)


def motif_delta(case: "VariantCase") -> float:
    """Change in the element's motif match between reference and variant
    alleles: negative for a loss, positive for a gain, zero if the variant
    does not touch the motif."""
    k = len(MOTIFS[case.tf])
    ref_w = case.ref[case.start:case.start + k]
    alt_w = case.alt[case.start:case.start + k]
    return match_fraction(alt_w, MOTIFS[case.tf]) - match_fraction(ref_w, MOTIFS[case.tf])


def effect_in_condition(case: "VariantCase", activity: dict[str, float]) -> float:
    """
    Predicted regulatory effect under one condition (a TF activity
    profile). The condition is an input to the prediction, not something
    averaged away before it.
    """
    return motif_delta(case) * activity[case.tf]


def bulk_activity(cell_types: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean TF activity across cell types: the toy tissue average."""
    tfs = next(iter(cell_types.values())).keys()
    n = len(cell_types)
    return {tf: sum(ct[tf] for ct in cell_types.values()) / n for tf in tfs}


# --------------------------------------------------------------------------
# Building a condition from single-cell data (no retraining)
# --------------------------------------------------------------------------

# Marker genes stand in for TFs: a cell state's TF activity is inferred
# from the mean expression of that TF's marker genes across cells.
MARKERS: dict[str, list[str]] = {
    "GATA1": ["GATA1", "CD3D"],
    "NEUROD": ["NEUROD1", "MAP2"],
    "HNF4A": ["HNF4A", "ALB"],
    "MYOD1": ["MYOD1", "MYH7"],
    "SPI1": ["SPI1", "CD68"],
    "SP1": ["SP1", "ACTB"],
}

ALL_GENES = [g for genes in MARKERS.values() for g in genes]


def infer_condition(cells: list[dict[str, float]]) -> dict[str, float]:
    """
    Build a condition vector from single-cell expression profiles: for each
    TF, average its marker genes' expression across all cells and squash to
    0..1 with a saturating scale, value / (value + 1). A new cell state
    gets a usable condition without touching the scorer itself.
    """
    condition: dict[str, float] = {}
    for tf, genes in MARKERS.items():
        vals = [cell.get(g, 0.0) for cell in cells for g in genes]
        mean_expr = sum(vals) / len(vals) if vals else 0.0
        condition[tf] = mean_expr / (mean_expr + 1.0)
    return condition


# --------------------------------------------------------------------------
# Fixture: synthetic regulatory elements with planted motifs and variants
# --------------------------------------------------------------------------

@dataclass
class VariantCase:
    name: str
    ref: str        # reference sequence of the element's region
    alt: str        # variant sequence (same length)
    start: int      # offset of the motif window inside the sequence
    tf: str         # transcription factor whose motif the element holds
    truth: str      # cell type affected, "ubiquitous", or "none"


def _other_base(base: str, rng: random.Random) -> str:
    return rng.choice([b for b in BASES if b != base])


def make_case(name: str, tf: str, kind: str, truth: str,
              seed: int, motif_at: int = 24, length: int = 60) -> VariantCase:
    """
    kind = "loss":    plant the full motif, then break two of its bases.
    kind = "gain":    plant the motif with two bases wrong; the variant
                      fixes both, completing the motif.
    kind = "neutral": plant the full motif, but the variant changes a base
                      far away from it, so the motif score cannot move.
    """
    rng = random.Random(seed)
    seq = [rng.choice(BASES) for _ in range(length)]
    consensus = MOTIFS[tf]
    seq[motif_at:motif_at + len(consensus)] = list(consensus)
    ref, alt = list(seq), list(seq)
    if kind == "loss":
        for off in (1, 4):
            alt[motif_at + off] = _other_base(consensus[off], rng)
    elif kind == "gain":
        for off in (1, 4):
            ref[motif_at + off] = _other_base(consensus[off], rng)
    elif kind == "neutral":
        pos = motif_at + 30
        alt[pos] = _other_base(seq[pos], rng)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return VariantCase(name, "".join(ref), "".join(alt), motif_at, tf, truth)


def build_fixture() -> list[VariantCase]:
    return [
        make_case("GATA1 loss, T-cell enhancer", "GATA1", "loss", "t_cell", seed=11),
        make_case("NEUROD loss, neuronal enhancer", "NEUROD", "loss", "neuron", seed=12),
        make_case("HNF4A loss, liver enhancer", "HNF4A", "loss", "hepatocyte", seed=13),
        make_case("MYOD1 loss, muscle enhancer", "MYOD1", "loss", "myocyte", seed=14),
        make_case("SPI1 gain, macrophage enhancer", "SPI1", "gain", "macrophage", seed=15),
        make_case("GATA1 gain, T-cell enhancer", "GATA1", "gain", "t_cell", seed=16),
        make_case("SP1 loss, housekeeping promoter", "SP1", "loss", "ubiquitous", seed=17),
        make_case("SP1 gain, housekeeping promoter", "SP1", "gain", "ubiquitous", seed=18),
        make_case("intergenic variant A", "GATA1", "neutral", "none", seed=19),
        make_case("intergenic variant B", "SPI1", "neutral", "none", seed=20),
        make_case("deep intronic variant", "NEUROD", "neutral", "none", seed=21),
        make_case("synonymous-region variant", "HNF4A", "neutral", "none", seed=22),
    ]


# --------------------------------------------------------------------------
# The two scorers under comparison
# --------------------------------------------------------------------------

def bulk_calls(cases: list[VariantCase]) -> dict[str, bool]:
    """One averaged prediction per variant: flagged or not. It cannot say
    WHERE an effect lives, because the condition was averaged away."""
    avg = bulk_activity(CELL_TYPES)
    return {c.name: abs(effect_in_condition(c, avg)) >= FLAG_THRESHOLD
            for c in cases}


def conditioned_calls(cases: list[VariantCase]) -> dict[str, set[str]]:
    """One prediction per (variant, cell type): the set of cell types where
    the variant is flagged. Empty set = silent everywhere."""
    return {
        c.name: {ct for ct, act in CELL_TYPES.items()
                 if abs(effect_in_condition(c, act)) >= FLAG_THRESHOLD}
        for c in cases
    }


def recovery_report(cases: list[VariantCase]) -> dict[str, float]:
    """
    Recovery = fraction of truly-affecting variants (truth != "none") the
    scorer flags at all. Localisation = of the cell-specific variants, how
    often the true cell type is in the conditioned flagged set. Also the
    false-flag rate on neutral variants.
    """
    bulk = bulk_calls(cases)
    cond = conditioned_calls(cases)
    affecting = [c for c in cases if c.truth != "none"]
    neutral = [c for c in cases if c.truth == "none"]
    specific = [c for c in cases if c.truth not in ("none", "ubiquitous")]
    return {
        "bulk_recovery": sum(bulk[c.name] for c in affecting) / len(affecting),
        "conditioned_recovery":
            sum(bool(cond[c.name]) for c in affecting) / len(affecting),
        "conditioned_localisation":
            sum(c.truth in cond[c.name] for c in specific) / len(specific),
        "conditioned_false_flag_rate":
            sum(bool(cond[c.name]) for c in neutral) / len(neutral),
    }


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

def synthetic_activated_macrophages(n_cells: int = 40,
                                    seed: int = 7) -> list[dict[str, float]]:
    """Synthetic single-cell profiles for an activated macrophage state:
    SPI1 and housekeeping markers highly expressed, the rest near zero."""
    rng = random.Random(seed)
    high = {"SPI1", "CD68", "SP1", "ACTB"}
    return [
        {g: (rng.uniform(6.0, 9.0) if g in high else rng.uniform(0.0, 0.4))
         for g in ALL_GENES}
        for _ in range(n_cells)
    ]


def main() -> None:
    cases = build_fixture()
    bulk = bulk_calls(cases)
    cond = conditioned_calls(cases)
    avg = bulk_activity(CELL_TYPES)

    print("Same variant, different answer depending on the cell you ask about")
    print("(AlphaGenome 2's conditioning idea, as transparent toy arithmetic)\n")
    print(f"{'variant':38s} {'bulk effect':>11s}  flagged in")
    for c in cases:
        where = ", ".join(sorted(cond[c.name])) or "-"
        print(f"{c.name:38s} {effect_in_condition(c, avg):+11.3f}  {where}")

    rep = recovery_report(cases)
    print(f"\nRecovery of truly-affecting variants: "
          f"bulk {rep['bulk_recovery']:.0%}, "
          f"conditioned {rep['conditioned_recovery']:.0%}")
    print(f"Conditioned localisation (true cell type flagged): "
          f"{rep['conditioned_localisation']:.0%}")
    print(f"Conditioned false flags on neutral variants: "
          f"{rep['conditioned_false_flag_rate']:.0%}")

    # A condition the scorer has never seen: an activated macrophage state
    # inferred from synthetic single-cell marker counts, no retraining.
    inferred = infer_condition(synthetic_activated_macrophages())
    spi = next(c for c in cases if c.name.startswith("SPI1 gain"))
    gata = next(c for c in cases if c.name.startswith("GATA1 loss"))
    print(f"\nNew state inferred from single-cell markers: "
          f"SPI1 activity {inferred['SPI1']:.2f}, "
          f"GATA1 activity {inferred['GATA1']:.2f}")
    for case in (spi, gata):
        eff = effect_in_condition(case, inferred)
        print(f"  {case.name:38s} effect {eff:+.3f} "
              f"(flagged: {abs(eff) >= FLAG_THRESHOLD})")


if __name__ == "__main__":
    main()

"""Tests for the cell-conditioned variant-effect demo."""

import pytest

from cell_conditioned_variant import (
    CELL_TYPES,
    FLAG_THRESHOLD,
    build_fixture,
    bulk_activity,
    bulk_calls,
    conditioned_calls,
    effect_in_condition,
    infer_condition,
    match_fraction,
    motif_delta,
    recovery_report,
    synthetic_activated_macrophages,
)

CASES = build_fixture()
BY_NAME = {c.name: c for c in CASES}


def test_match_fraction_exact_values():
    assert match_fraction("TGATAG", "TGATAG") == 1.0
    assert match_fraction("AAAAAA", "TGATAG") == pytest.approx(2 / 6)
    with pytest.raises(ValueError):
        match_fraction("ACG", "TGATAG")


def test_loss_and_gain_deltas_are_symmetric():
    loss = BY_NAME["GATA1 loss, T-cell enhancer"]
    gain = BY_NAME["GATA1 gain, T-cell enhancer"]
    assert motif_delta(loss) == pytest.approx(-2 / 6)
    assert motif_delta(gain) == pytest.approx(+2 / 6)


def test_neutral_variants_have_zero_delta():
    for c in CASES:
        if c.truth == "none":
            assert motif_delta(c) == 0.0


def test_conditioned_effect_strong_only_in_the_right_cell():
    case = BY_NAME["NEUROD loss, neuronal enhancer"]
    strong = effect_in_condition(case, CELL_TYPES["neuron"])
    weak = effect_in_condition(case, CELL_TYPES["t_cell"])
    assert strong == pytest.approx(-2 / 6 * 0.95)
    assert abs(strong) >= FLAG_THRESHOLD
    assert abs(weak) < FLAG_THRESHOLD


def test_bulk_average_dilutes_a_cell_specific_effect():
    case = BY_NAME["HNF4A loss, liver enhancer"]
    bulk_eff = effect_in_condition(case, bulk_activity(CELL_TYPES))
    # HNF4A mean activity across the five types is (0.05*3 + 0.95 + 0.10)/5
    assert bulk_eff == pytest.approx(-2 / 6 * 0.24)
    assert abs(bulk_eff) < FLAG_THRESHOLD  # the dilution, in one number


def test_bulk_still_catches_ubiquitous_effects():
    bulk = bulk_calls(CASES)
    assert bulk["SP1 loss, housekeeping promoter"] is True
    assert bulk["SP1 gain, housekeeping promoter"] is True


def test_bulk_misses_every_cell_specific_variant():
    bulk = bulk_calls(CASES)
    for c in CASES:
        if c.truth not in ("none", "ubiquitous"):
            assert bulk[c.name] is False, c.name


def test_conditioned_flags_exactly_the_true_cell_for_specific_variants():
    cond = conditioned_calls(CASES)
    for c in CASES:
        if c.truth not in ("none", "ubiquitous"):
            assert cond[c.name] == {c.truth}, c.name


def test_conditioned_flags_ubiquitous_variants_everywhere():
    cond = conditioned_calls(CASES)
    for c in CASES:
        if c.truth == "ubiquitous":
            assert cond[c.name] == set(CELL_TYPES), c.name


def test_conditioned_never_flags_neutral_variants():
    cond = conditioned_calls(CASES)
    for c in CASES:
        if c.truth == "none":
            assert cond[c.name] == set(), c.name


def test_recovery_report_numbers():
    rep = recovery_report(CASES)
    # 8 affecting variants; bulk catches only the 2 ubiquitous ones.
    assert rep["bulk_recovery"] == pytest.approx(2 / 8)
    assert rep["conditioned_recovery"] == 1.0
    assert rep["conditioned_localisation"] == 1.0
    assert rep["conditioned_false_flag_rate"] == 0.0


def test_inferred_condition_recovers_macrophage_state():
    inferred = infer_condition(synthetic_activated_macrophages())
    assert inferred["SPI1"] > 0.8
    assert inferred["GATA1"] < 0.3
    spi = BY_NAME["SPI1 gain, macrophage enhancer"]
    gata = BY_NAME["GATA1 loss, T-cell enhancer"]
    assert abs(effect_in_condition(spi, inferred)) >= FLAG_THRESHOLD
    assert abs(effect_in_condition(gata, inferred)) < FLAG_THRESHOLD


def test_infer_condition_handles_empty_profiles():
    cond = infer_condition([{}])
    assert all(v == 0.0 for v in cond.values())

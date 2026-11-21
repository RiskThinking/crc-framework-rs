"""Public simulation, replay, and numerical admission for one peril."""

from __future__ import annotations

from dataclasses import replace
from math import asin, pi

import numpy as np
import pytest
from crc_framework import (
    AnnualLossSpec,
    InsuredInterest,
    PolicyTerms,
    QuantileCurve,
    TailCompletion,
    simulate_annual_losses,
    summarize_annual_losses,
)


def _interest(id: str, longitude: float = 0.0) -> InsuredInterest:
    return InsuredInterest(
        id,
        longitude=longitude,
        latitude=0,
        total_insured_value=1,
        damage_curve=QuantileCurve([0, 1], [0, 1]),
    )


def test_reference_configuration_and_replay_across_block_sizes() -> None:
    reference = AnnualLossSpec.reference()
    assert (
        reference.trials,
        reference.seed,
        reference.peak_correlation,
        reference.decay_length_km,
        reference.damage_cv,
    ) == (20_000, 20_261_027, 0.4, 150.0, 0.3)
    spec = replace(reference, trials=257, calibration_source="synthetic calibration")
    book = [_interest("b", 0.5), _interest("a")]
    first = simulate_annual_losses(
        book, spec=spec, block_size=1, include_components=True
    )
    reordered = simulate_annual_losses(
        book[::-1], spec=spec, block_size=100, include_components=True
    )
    totals_only = simulate_annual_losses(book, spec=spec, block_size=1_000)
    assert first.trial_ids == range(257)
    assert first.spec == spec
    assert first.secondary_stream == 0
    assert first.rng_algorithm == "crc-chacha12-ziggurat-gamma-v1"
    assert first.total_insured_value == 2.0
    np.testing.assert_array_equal(first.totals, reordered.totals)
    np.testing.assert_array_equal(first.totals, totals_only.totals)
    assert totals_only.components is None
    assert first.components is not None and reordered.components is not None
    assert tuple(first.components) == ("a", "b")
    for id, losses in first.components.items():
        np.testing.assert_array_equal(losses, reordered.components[id])
        assert not losses.flags.writeable
    np.testing.assert_allclose(
        first.totals, first.components["a"] + first.components["b"]
    )
    with pytest.raises(ValueError, match="WRITEABLE"):
        first.totals.setflags(write=True)
    assert (
        summarize_annual_losses(
            first.totals,
            total_insured_value=first.total_insured_value,
            components=first.components,
        ).trial_count
        == 257
    )


def test_secondary_stream_does_not_change_primary_shocks() -> None:
    book = [_interest("a"), _interest("b", 0.5)]
    spec = AnnualLossSpec(trials=256, damage_cv=0)
    first = simulate_annual_losses(book, spec=spec, secondary_stream=7)
    np.testing.assert_array_equal(
        first.totals,
        simulate_annual_losses(book, spec=spec, secondary_stream=11).totals,
    )
    assert not np.array_equal(
        first.totals,
        simulate_annual_losses(book, spec=replace(spec, seed=spec.seed + 1)).totals,
    )
    uncertain = replace(spec, damage_cv=0.3)
    first = simulate_annual_losses(book, spec=uncertain, secondary_stream=7)
    second = simulate_annual_losses(book, spec=uncertain, secondary_stream=11)
    assert second.secondary_stream == 11
    assert not np.array_equal(first.totals, second.totals)


def test_primary_does_not_merge_colocated_interests_and_decays_with_distance() -> None:
    book = [_interest("a"), _interest("b"), _interest("far", 10)]
    spec = AnnualLossSpec(trials=20_000, peak_correlation=0.6, damage_cv=0)
    result = simulate_annual_losses(book, spec=spec, include_components=True)
    assert result.components is not None
    matrix = np.corrcoef([result.components[name] for name in ("a", "b", "far")])
    # Pearson correlation of the uniform ranks of a Gaussian pair.
    assert matrix[0, 1] == pytest.approx(6 * asin(0.6 / 2) / pi, abs=0.035)
    assert matrix[0, 2] == pytest.approx(0.0, abs=0.035)


def test_singular_correlation_requires_explicit_repair_within_tolerance() -> None:
    book = [_interest("a"), _interest("b")]
    spec = AnnualLossSpec(trials=1, peak_correlation=1)
    with pytest.raises(ValueError, match="positive definite"):
        simulate_annual_losses(book, spec=spec)
    with pytest.raises(ValueError, match="exceeding repair_tolerance"):
        simulate_annual_losses(book, spec=replace(spec, repair_tolerance=1e-10))
    result = simulate_annual_losses(book, spec=replace(spec, repair_tolerance=1e-6))
    assert result.correlation_repair is not None
    assert result.correlation_repair.minimum_eigenvalue == pytest.approx(0, abs=1e-14)
    assert 1e-10 < result.correlation_repair.maximum_change <= 1e-6


def test_explicit_damage_completion_and_zero_payout() -> None:
    incomplete = QuantileCurve([0.1, 0.9], [0.08, 0.6])
    fields = {"longitude": 0, "latitude": 0, "total_insured_value": 50}
    with pytest.raises(ValueError, match="explicit endpoints"):
        InsuredInterest("site", **fields, damage_curve=incomplete)
    upper_tail = QuantileCurve([0.99, 1], [0.4, 0.8]).condition(q_min=0.99, q_max=1)
    with pytest.raises(ValueError, match="original annual ranks"):
        InsuredInterest("site", **fields, damage_curve=upper_tail)
    completed = incomplete.complete(TailCompletion(lower=0, upper=1))
    interest = InsuredInterest(
        "site",
        **fields,
        damage_curve=completed,
        terms=PolicyTerms(limit=0),
    )
    assert interest.damage_curve is completed
    result = simulate_annual_losses([interest], spec=AnnualLossSpec(trials=3))
    np.testing.assert_array_equal(result.totals, [0, 0, 0])
    assert result.trial_ids == range(3)


@pytest.mark.parametrize("field", ["memory_limit", "work_limit"])
def test_preflight_limits_reject_large_execution(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        simulate_annual_losses([_interest("a")], **{field: 1})


def test_duplicate_and_invalid_interests_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate insured interest ID"):
        simulate_annual_losses([_interest("a"), _interest("a", 2)])
    with pytest.raises(ValueError, match="WGS84"):
        _interest("a", longitude=181)
    with pytest.raises(ValueError, match="damage ratios"):
        InsuredInterest(
            "a",
            longitude=0,
            latitude=0,
            total_insured_value=10,
            damage_curve=QuantileCurve([0, 1], [0, 2]),
        )

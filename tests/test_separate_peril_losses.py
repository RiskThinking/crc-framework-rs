"""The Python interface to cross-peril annual insured loss."""

from __future__ import annotations

import json
from statistics import NormalDist

import numpy as np
import pytest
from crc_framework import (
    AnnualLossSpec,
    InsuredInterest,
    PerilDependence,
    PolicyTerms,
    QuantileCurve,
    simulate_separate_peril_losses,
    summarize_annual_losses,
)


def _interest(
    id: str = "home",
    *,
    value: float = 100,
    longitude: float = 42.0,
    loss_ratios: tuple[float, float] = (0.0, 1.0),
    terms: PolicyTerms | None = None,
) -> InsuredInterest:
    return InsuredInterest(
        id,
        longitude=longitude,
        latitude=44.0,
        total_insured_value=value,
        damage_curve=QuantileCurve([0.0, 1.0], loss_ratios),
        terms=terms,
    )


def test_calibrated_dependence_round_trips_and_sorts_labelled_observations() -> None:
    latent = np.array([[-0.8, -0.7], [-0.4, 0.2], [0.1, -0.1], [0.5, 0.5], [1.1, 0.9]])
    ranks = np.array([[NormalDist().cdf(x) for x in row] for row in latent])
    forward = PerilDependence.calibrate(
        ("flood", "wind"), ranks, calibration_source="aligned annual observations"
    )
    reverse = PerilDependence.calibrate(
        ("wind", "flood"),
        ranks[:, ::-1],
        calibration_source="aligned annual observations",
    )
    assert forward == reverse == PerilDependence.from_json(forward.to_json())
    assert forward.perils == ("flood", "wind")
    assert forward.observation_count == 5
    assert forward.observations_sha256 is not None
    assert forward.observations_sha256.startswith("sha256:")
    assert forward.correlation is not None
    np.testing.assert_allclose(forward.correlation, np.corrcoef(latent, rowvar=False))

    independent = PerilDependence.independent()
    assert independent == PerilDependence.from_json(independent.to_json())
    invalid = json.loads(forward.to_json())
    invalid["observation_count"] = True
    with pytest.raises(ValueError, match="observation count"):
        PerilDependence.from_json(json.dumps(invalid))


@pytest.mark.parametrize(
    ("ranks", "message"),
    [
        ([[0.1, 0.2], [0.2, 0.3]], "more years"),
        ([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]], "strictly between"),
        ([[0.1, 0.2], [0.1, 0.3], [0.1, 0.4]], "must vary"),
        ([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], "positive definite"),
    ],
)
def test_calibration_rejects_unusable_observed_ranks(
    ranks: list[list[float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PerilDependence.calibrate(
            ("flood", "wind"), ranks, calibration_source="fixture"
        )


def test_separate_terms_can_pay_across_perils_and_count_insured_value_once() -> None:
    book = {
        "flood": [
            _interest(
                loss_ratios=(1.0, 1.0),
                terms=PolicyTerms(deductible=10, limit=80),
            ),
        ],
        "wind": [_interest(loss_ratios=(1.0, 1.0), terms=PolicyTerms(share=0.75))],
    }
    dependence = PerilDependence.independent()
    result = simulate_separate_peril_losses(
        book,
        dependence=dependence,
        spec=AnnualLossSpec(trials=20),
        include_components=True,
    )
    assert result.dependence == dependence
    assert result.perils == ("flood", "wind")
    assert result.total_insured_value == 100
    assert result.trial_ids == range(20)
    np.testing.assert_array_equal(result.totals, np.full(20, 155.0))
    assert result.components is not None
    np.testing.assert_array_equal(result.components["flood"], np.full(20, 80.0))
    np.testing.assert_array_equal(result.components["wind"], np.full(20, 75.0))
    with pytest.raises(ValueError, match="WRITEABLE"):
        result.components["wind"].setflags(write=True)
    summary = summarize_annual_losses(
        result.totals,
        total_insured_value=result.total_insured_value,
        components=result.components,
    )
    assert summary.average_annual_loss == 155
    assert summary.at(0.95).contributions == pytest.approx({"flood": 80, "wind": 75})


def test_correlated_simulation_replays_peril_interest_order_and_block_size() -> None:
    dependence = PerilDependence.from_correlation(
        ("wind", "flood"), [[1, 0.7], [0.7, 1]], calibration_source="fixture"
    )
    book = {
        "flood": (_interest(), _interest("office", longitude=43.0)),
        "wind": (
            _interest(loss_ratios=(0, 0.5)),
            _interest("office", longitude=43.0, loss_ratios=(0, 0.5)),
        ),
    }
    spec = AnnualLossSpec(trials=100, seed=31)
    first = simulate_separate_peril_losses(
        book, dependence=dependence, spec=spec, block_size=3, include_components=True
    )
    second = simulate_separate_peril_losses(
        {key: group[::-1] for key, group in reversed(tuple(book.items()))},
        dependence=dependence,
        spec=spec,
        block_size=11,
        include_components=True,
    )
    np.testing.assert_array_equal(first.totals, second.totals)
    assert first.components is not None and second.components is not None
    for peril in first.perils:
        np.testing.assert_array_equal(first.components[peril], second.components[peril])


def test_independent_and_declared_peril_models_are_admitted_explicitly() -> None:
    alternative = PerilDependence.from_correlation(
        ("heat", "flood"), [[1.0, 0.3], [0.3, 1.0]], calibration_source="fixture"
    )
    with pytest.raises(ValueError, match="same peril IDs"):
        simulate_separate_peril_losses(
            {"flood": [_interest()], "wind": [_interest()]}, dependence=alternative
        )
    with pytest.raises(ValueError, match="at least two"):
        simulate_separate_peril_losses(
            {"wind": [_interest()]}, dependence=PerilDependence.independent()
        )

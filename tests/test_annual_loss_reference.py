"""Compare the native simulator with an independent NumPy implementation."""

from __future__ import annotations

from math import asin, cos, erf, radians, sin, sqrt

import numpy as np
import pytest
from crc_framework import (
    AnnualLossSpec,
    InsuredInterest,
    PolicyTerms,
    QuantileCurve,
    simulate_annual_losses,
    summarize_annual_losses,
)
from numpy.typing import NDArray


def _pooled_standard_error(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> float:
    return sqrt(float(left.var()) / left.size + float(right.var()) / right.size)


def test_native_annual_losses_match_numpy_monte_carlo_reference() -> None:
    trials = 40_000
    level = 0.95
    peak = 0.6
    decay_km = 150.0
    damage_cv = 0.4
    knots = [0.0, 0.4, 1.0]
    damage_knots = [[0.08, 0.2, 0.7], [0.1, 0.65, 0.9]]
    longitude = [13.0, 14.0]
    latitude = [45.0, 45.25]
    insured_value = [100.0, 200.0]
    interests = [
        InsuredInterest(
            "a",
            longitude=longitude[0],
            latitude=latitude[0],
            total_insured_value=insured_value[0],
            damage_curve=QuantileCurve(knots, damage_knots[0]),
            terms=PolicyTerms(deductible=5, limit=30, share=0.75),
        ),
        InsuredInterest(
            "b",
            longitude=longitude[1],
            latitude=latitude[1],
            total_insured_value=insured_value[1],
            damage_curve=QuantileCurve(knots, damage_knots[1]),
        ),
    ]
    framework = simulate_annual_losses(
        interests,
        spec=AnnualLossSpec(
            trials=trials,
            seed=20261027,
            peak_correlation=peak,
            decay_length_km=decay_km,
            damage_cv=damage_cv,
        ),
    ).totals

    lat_a, lat_b = map(radians, latitude)
    delta_lon = radians(longitude[1] - longitude[0])
    haversine = sin((lat_b - lat_a) / 2) ** 2 + cos(lat_a) * cos(lat_b) * (
        sin(delta_lon / 2) ** 2
    )
    distance_km = 2 * 6371 * asin(sqrt(haversine))
    correlation = peak * np.exp(-distance_km / decay_km)
    factor = np.linalg.cholesky([[1.0, correlation], [correlation, 1.0]])

    # An independent stream avoids comparing individual draws across RNGs.
    reference_rng = np.random.default_rng(20682)
    normals = reference_rng.normal(size=(trials, 2)) @ factor.T
    ranks = np.fromiter(
        (0.5 * (1.0 + erf(z / sqrt(2))) for z in normals.flat),
        dtype=np.float64,
        count=normals.size,
    ).reshape(normals.shape)
    mean_damage = np.column_stack(
        [
            np.interp(ranks[:, position], knots, values)
            for position, values in enumerate(damage_knots)
        ]
    )
    variance_limit = mean_damage * (1 - mean_damage)
    variance = np.minimum((damage_cv * mean_damage) ** 2, variance_limit * (1 - 1e-6))
    concentration = variance_limit / variance - 1
    damage = reference_rng.beta(
        mean_damage * concentration, (1 - mean_damage) * concentration
    )
    ground_up = damage * insured_value
    paid_a = 0.75 * np.clip(ground_up[:, 0] - 5, 0, 30)
    reference = paid_a + ground_up[:, 1]

    framework_score, reference_score = (
        summarize_annual_losses(
            totals,
            total_insured_value=sum(insured_value),
            levels=[level],
            return_periods=[],
        )
        for totals in (framework, reference)
    )
    assert framework_score.average_annual_loss == pytest.approx(
        reference_score.average_annual_loss,
        abs=5 * _pooled_standard_error(framework, reference),
    )

    framework_tail = framework_score.at(level)
    reference_tail = reference_score.at(level)
    # The expected shortfall influence function has the same variance
    # as (loss - VaR)_+ / (1 - level).
    tail_standard_error = _pooled_standard_error(
        np.maximum(framework - framework_tail.value_at_risk, 0),
        np.maximum(reference - reference_tail.value_at_risk, 0),
    ) / (1 - level)
    assert framework_tail.conditional_value_at_risk == pytest.approx(
        reference_tail.conditional_value_at_risk,
        abs=5 * tail_standard_error,
    )

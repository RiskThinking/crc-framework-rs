"""Behavioral conformance for complete laws and supplied quantile evidence."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from crc_framework import (
    Distribution,
    DistributionFamily,
    DistributionKind,
    Impact,
    Measurement,
    QuantileCurve,
    Statistic,
    StatisticStatus,
    Tail,
    TailCompletion,
)


def laws() -> list[Distribution]:
    return [
        Distribution.parametric(DistributionFamily.GUMBEL_RIGHT, location=0, scale=1),
        Distribution.empirical([0, 0, 1, 2]),
        Distribution.discrete([0, 1, 2], [0.5, 0.25, 0.25]),
        Distribution.histogram([0, 1, 2], [0.5, 0.5]),
        Distribution.point_mass(1),
        Distribution.hurdle(
            Distribution.histogram([-1, 1, 3], [0.5, 0.5]), atom_probability=0.2
        ),
        QuantileCurve([0.2, 0.8], [0, 2]).complete(TailCompletion(lower=-1, upper=3)),
    ]


@pytest.mark.parametrize("index", range(7))
def test_common_scalar_shapes_empty_arrays_and_validation(index: int) -> None:
    law = laws()[index]
    assert isinstance(law.quantile(0.5), float)
    assert isinstance(law.cdf(1.0), float)
    assert isinstance(law.mass(1.0), float)
    query = np.array([[0.1, 0.5], [0.7, 0.9]])
    assert np.asarray(law.quantile(query)).shape == query.shape
    for method in (law.quantile, law.cdf, law.exceedance, law.mass):
        assert np.asarray(method(np.empty((2, 0, 3)))).shape == (2, 0, 3)
        with pytest.raises(ValueError):
            method(float("nan"))
    for q in (-0.1, 1.1, math.inf):
        with pytest.raises(ValueError):
            law.quantile(q)
    assert law.sample((2, 3), seed=123).shape == (2, 3)
    np.testing.assert_array_equal(law.sample(10, seed=3), law.sample(10, seed=3))
    assert law.sample(0).shape == (0,)
    with pytest.raises(ValueError):
        law.sample(-1)
    for name in ("ppf", "quantiles", "pdf"):
        assert not hasattr(law, name)


def test_empirical_means_the_ecdf_and_ties_are_atoms() -> None:
    law = Distribution.empirical([0, 0, 2, 10])
    np.testing.assert_array_equal(law.quantile([0, 0.5, 0.50001, 1]), [0, 0, 2, 10])
    assert law.cdf(0) == 0.5
    assert law.cdf(0, inclusive=False) == 0
    assert law.exceedance(0) == 0.5
    assert law.exceedance(0, inclusive=True) == 1
    assert law.mass(0) == 0.5
    assert law.statistics(metrics=[Statistic.MEAN]).values[Statistic.MEAN].value == 3
    curve = QuantileCurve.from_samples([0, 0, 2, 10])
    assert curve.quantile(0.5) == pytest.approx(1)


def test_discrete_normalizes_roundoff_merges_ties_and_drops_zero_mass() -> None:
    law = Distribution.discrete([2, 0, 2, 1000], [0.25, 0.5, 0.25, 0])
    assert law.support == (0, 2)
    assert law.mass(2) == 0.5
    with pytest.raises(ValueError):
        Distribution.discrete([0, 1], [0.2, 0.3])
    with pytest.raises(ValueError):
        Distribution.discrete([0, 1], [1.1, -0.1])


def test_rare_mass_is_kept_in_statistics_and_probability_queries() -> None:
    law = Distribution.discrete([0, 1_000_000], [0.999999, 0.000001])
    result = law.statistics(metrics=list(Statistic), level=0.95)
    assert result.values[Statistic.MEAN].value == pytest.approx(1, abs=1e-9)
    assert result.values[Statistic.MAXIMUM].value == 1_000_000
    assert result.values[Statistic.CVAR].value == pytest.approx(20, rel=1e-9)
    assert law.exceedance(0) == 1e-6
    assert law.mass(1_000_000) == 1e-6
    rare = Distribution.discrete([0, 1e20], [1, 1e-20])
    result = rare.statistics(metrics=[Statistic.MEAN, Statistic.CVAR])
    assert result.values[Statistic.MEAN].value == pytest.approx(1)
    assert result.values[Statistic.CVAR].value == pytest.approx(20)
    conditioned = Distribution.hurdle(rare, atom_probability=0.5)
    assert conditioned.statistics(metrics=[Statistic.MEAN]).values[
        Statistic.MEAN
    ].value == pytest.approx(5e19)


def test_statistics_integrate_atoms_and_uniform_bins_without_a_grid() -> None:
    law = Distribution.histogram([0, 1, 2, 3], [0.5, 0, 0.5])
    result = law.statistics(
        metrics=[Statistic.MEAN, Statistic.VARIANCE, Statistic.CVAR], level=0.5
    )
    assert result.values[Statistic.MEAN].value == pytest.approx(1.5)
    assert result.values[Statistic.VARIANCE].value == pytest.approx(13 / 12)
    assert result.values[Statistic.CVAR].value == pytest.approx(2.5)
    lower = law.statistics(metrics=[Statistic.CVAR], level=0.5, tail=Tail.LOWER)
    assert lower.values[Statistic.CVAR].value == pytest.approx(0.5)
    assert law.quantile(0.5) == 1
    assert law.mass(0.5) == 0
    assert law.cdf(1.5) == 0.5


def test_statistics_are_explicit_about_infinite_and_undefined_outcomes() -> None:
    law = Distribution.parametric(
        DistributionFamily.GENERALIZED_PARETO, location=0, scale=1, shape=2
    )
    result = law.statistics(
        metrics=[Statistic.MEAN, Statistic.VARIANCE, Statistic.CVAR, Statistic.MAXIMUM]
    )
    for metric in result.values:
        assert result.values[metric].status is StatisticStatus.POSITIVE_INFINITY
        assert result.values[metric].value is None
        assert result.values[metric].diagnostics
    assert law.quantile(0) == 0
    assert law.quantile(1) == math.inf
    with pytest.raises(TypeError):
        law.statistics(metrics=["mean"])  # type: ignore[list-item]


@pytest.mark.parametrize(
    "family,shape,lower,upper",
    [
        (DistributionFamily.GUMBEL_RIGHT, None, -math.inf, math.inf),
        (DistributionFamily.GUMBEL_LEFT, None, -math.inf, math.inf),
        (DistributionFamily.GENERALIZED_EXTREME_VALUE, 0.2, -math.inf, 5),
        (DistributionFamily.GENERALIZED_EXTREME_VALUE, -0.2, -5, math.inf),
        (DistributionFamily.GENERALIZED_PARETO, -0.2, 0, 5),
        (DistributionFamily.WEIBULL_MINIMUM, 2.0, 0, math.inf),
        (DistributionFamily.WEIBULL_MAXIMUM, 2.0, -math.inf, 0),
        (DistributionFamily.SKEW_NORMAL, 2.0, -math.inf, math.inf),
    ],
)
def test_parametric_endpoints_are_mathematical_support(
    family: DistributionFamily, shape: float | None, lower: float, upper: float
) -> None:
    law = Distribution.parametric(family, location=0, scale=1, shape=shape)
    assert law.support == (lower, upper)
    assert law.quantile(0) == lower
    assert law.quantile(1) == upper
    assert law.cdf(-math.inf) == 0
    assert law.exceedance(math.inf) == 0


def test_parametric_statistics_and_deep_tail_probabilities() -> None:
    law = Distribution.parametric(DistributionFamily.GUMBEL_RIGHT, location=2, scale=3)
    result = law.statistics(
        metrics=[Statistic.MEAN, Statistic.VARIANCE, Statistic.CVAR]
    )
    assert result.values[Statistic.MEAN].value == pytest.approx(
        2 + 3 * 0.5772156649015329
    )
    assert result.values[Statistic.VARIANCE].value == pytest.approx(9 * math.pi**2 / 6)
    assert result.values[Statistic.CVAR].value is not None
    assert law.exceedance(200) > 0
    representation = json.loads(law.to_json())
    assert "shape" not in representation
    representation["shape"] = None
    with pytest.raises(ValueError):
        Distribution.from_json(json.dumps(representation))
    with pytest.raises(TypeError):
        Distribution.parametric("gumbel_r", location=0, scale=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Distribution.parametric(
            DistributionFamily.GUMBEL_RIGHT, location=0, scale=1, shape=1
        )


def test_hurdle_conditions_base_strictly_above_the_atom() -> None:
    law = Distribution.hurdle(
        Distribution.discrete([-1, 0, 2, 4], [0.25] * 4), atom_probability=0.6
    )
    assert law.support == (0, 4)
    assert law.mass(0) == 0.6
    assert law.mass(2) == pytest.approx(0.2)
    assert law.cdf(0, inclusive=False) == 0
    assert law.cdf(0) == 0.6
    np.testing.assert_array_equal(law.quantile([0, 0.6, 0.7, 0.9, 1]), [0, 0, 2, 4, 4])
    assert law.statistics(metrics=[Statistic.MEAN]).values[
        Statistic.MEAN
    ].value == pytest.approx(1.2)
    with pytest.raises(ValueError):
        Distribution.hurdle(Distribution.point_mass(-1), atom_probability=0.5)
    assert Distribution.hurdle(
        Distribution.point_mass(-1), atom_probability=1
    ).support == (0, 0)


def test_quantile_evidence_has_only_its_actual_domain_and_no_law_operations() -> None:
    curve = QuantileCurve([0.2, 0.8], [1, 7])
    assert curve.probability_interval == (0.2, 0.8)
    assert curve.quantile(0.5) == pytest.approx(4)
    for q in (0.1, 0.9):
        with pytest.raises(ValueError):
            curve.quantile(q)
    for name in ("sample", "statistics", "cdf", "mass", "ppf", "quantiles"):
        assert not hasattr(curve, name)
    with pytest.raises(ValueError):
        curve.probabilities.setflags(write=True)
    with pytest.raises(ValueError):
        curve.values[0] = -10


def test_explicit_conditioning_and_completion_are_recorded_and_validated() -> None:
    curve = QuantileCurve([0.2, 0.8], [1, 7])
    conditioned = curve.condition(q_min=0.3, q_max=0.7)
    assert conditioned.kind is DistributionKind.QUANTILE
    assert conditioned.support == pytest.approx((2, 6))
    completed = curve.complete(TailCompletion(lower=0, upper=10))
    assert completed.support == (0, 10)
    assert json.loads(completed.to_json())["evidence"]["operation"] == "complete"
    for law in (conditioned, completed):
        assert Distribution.from_json(law.to_json()).to_json() == law.to_json()
    with pytest.raises(ValueError):
        curve.condition(q_min=0.1, q_max=0.7)
    with pytest.raises(ValueError):
        curve.complete(TailCompletion(lower=2, upper=10))
    tampered = json.loads(completed.to_json())
    tampered["evidence"]["policy"]["lower"] = -10
    with pytest.raises(ValueError):
        Distribution.from_json(json.dumps(tampered))


@pytest.mark.parametrize("index", range(7))
def test_serialization_is_canonical_and_rejects_historical_and_unknown_fields(
    index: int,
) -> None:
    law = laws()[index]
    assert Distribution.from_json(law.to_json()).to_json() == law.to_json()
    data = json.loads(law.to_json())
    data["schema_version"] = "0.6"
    with pytest.raises(ValueError):
        Distribution.from_json(json.dumps(data))
    data["schema_version"] = "1.0"
    data["legacy"] = True
    with pytest.raises(ValueError):
        Distribution.from_json(json.dumps(data))
    with pytest.raises(ValueError):
        Distribution.from_json('{"schema_version":"1.0","schema_version":"1.0"}')


def test_quantile_plateaus_preserve_atom_semantics() -> None:
    law = QuantileCurve([0, 0.2, 0.8, 1], [1, 2, 2, 3]).condition(q_min=0, q_max=1)
    assert law.mass(2) == pytest.approx(0.6)
    assert law.cdf(2, inclusive=False) == 0.2
    assert law.cdf(2) == 0.8


def test_hurdle_accepts_a_complete_transformed_law_and_round_trips() -> None:
    impact = Impact.linear(
        slope=2,
        input=Measurement(quantity="a", unit="u"),
        output=Measurement(quantity="b", unit="v"),
    )
    source = impact.pushforward(Distribution.histogram([-1, 1, 3], [0.5, 0.5]))
    law = Distribution.hurdle(source, atom_probability=0.4)
    assert law.support == (0, 6)
    np.testing.assert_allclose(law.quantile([0, 0.4, 0.6, 1]), [0, 0, 2, 6])
    assert law.mass(0) == 0.4
    assert law.cdf(1) == pytest.approx(0.5)
    assert law.exceedance(1) == pytest.approx(0.5)
    result = law.statistics(metrics=[Statistic.MEAN, Statistic.VARIANCE])
    assert result.values[Statistic.MEAN].value == pytest.approx(1.8)
    assert result.values[Statistic.VARIANCE].value == pytest.approx(3.96)
    assert Distribution.from_json(law.to_json()).to_json() == law.to_json()


def test_nested_quantile_modelling_record_is_kept_by_hurdle_serialization() -> None:
    completed = QuantileCurve([0.2, 0.8], [1, 3]).complete(
        TailCompletion(lower=0, upper=4)
    )
    law = Distribution.hurdle(completed, atom_probability=0.2)
    data = json.loads(law.to_json())
    assert data["base"]["evidence"]["operation"] == "complete"
    assert Distribution.from_json(law.to_json()).to_json() == law.to_json()


def test_small_nonzero_gev_shapes_keep_finite_stable_statistics() -> None:
    for shape in (-1e-8, 0, 1e-8):
        law = Distribution.parametric(
            DistributionFamily.GENERALIZED_EXTREME_VALUE,
            location=0,
            scale=1,
            shape=shape,
        )
        result = law.statistics(metrics=[Statistic.MEAN, Statistic.VARIANCE])
        assert result.values[Statistic.MEAN].value == pytest.approx(
            0.5772156649015329, abs=1e-7
        )
        assert result.values[Statistic.VARIANCE].value == pytest.approx(
            math.pi**2 / 6, abs=1e-7
        )


def test_heavy_integrable_pareto_tail_has_finite_cvar() -> None:
    law = Distribution.parametric(
        DistributionFamily.GENERALIZED_PARETO, location=0, scale=1, shape=0.99
    )
    threshold = float(law.quantile(0.95))
    expected = threshold + (1 + 0.99 * threshold) / (1 - 0.99)
    result = law.statistics(metrics=[Statistic.CVAR])
    assert result.values[Statistic.CVAR].status is StatisticStatus.FINITE
    assert result.values[Statistic.CVAR].value == pytest.approx(expected, rel=1e-10)


def test_deep_hurdle_survival_is_preserved_for_native_and_transformed_bases() -> None:
    base = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=-37.5, scale=1
    )
    measurement = Measurement(quantity="x", unit="u")
    transformed = Impact.linear(
        slope=2, input=measurement, output=measurement
    ).pushforward(base)
    for source, scale in ((base, 1), (transformed, 2)):
        law = Distribution.hurdle(source, atom_probability=0.25)
        result = law.statistics(metrics=[Statistic.MEAN, Statistic.VARIANCE])
        assert result.values[Statistic.MEAN].value == pytest.approx(
            0.75 * scale, abs=1e-8
        )
        assert result.values[Statistic.VARIANCE].value == pytest.approx(
            0.9375 * scale**2, abs=1e-8
        )
        assert law.quantile(0.75) == pytest.approx(-scale * math.log(1 / 3), abs=1e-8)


def test_concentrated_weibull_variance_is_positive_and_stable() -> None:
    law = Distribution.parametric(
        DistributionFamily.WEIBULL_MINIMUM, location=2, scale=3, shape=1e8
    )
    result = law.statistics(metrics=[Statistic.VARIANCE])
    value = result.values[Statistic.VARIANCE].value
    assert value is not None and value > 0
    assert value == pytest.approx(9 * math.pi**2 / 6 / 1e16, rel=1e-6)


@pytest.mark.parametrize(
    "shape,expected", [(-0.9, 163.24683328435537), (-0.99, 1959.0870807781658)]
)
def test_heavy_integrable_gev_tail_has_finite_cvar(
    shape: float, expected: float
) -> None:
    # Reference values use 70-digit decimal integration of exp(-t)'s
    # alternating series, independent of the native positive gamma series.
    law = Distribution.parametric(
        DistributionFamily.GENERALIZED_EXTREME_VALUE, location=0, scale=1, shape=shape
    )
    result = law.statistics(metrics=[Statistic.CVAR])
    assert result.values[Statistic.CVAR].status is StatisticStatus.FINITE
    assert result.values[Statistic.CVAR].value == pytest.approx(expected, rel=1e-12)


def test_gev_upper_tail_matches_independent_closed_erf_integral() -> None:
    law = Distribution.parametric(
        DistributionFamily.GENERALIZED_EXTREME_VALUE, location=0, scale=1, shape=-0.5
    )
    expected = -2 + 2 * math.sqrt(math.pi) * math.erf(math.sqrt(-math.log(0.95))) / 0.05
    result = law.statistics(metrics=[Statistic.CVAR])
    assert result.values[Statistic.CVAR].value == pytest.approx(expected, rel=1e-12)

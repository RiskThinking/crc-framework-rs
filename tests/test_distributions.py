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
    Ratio,
    Statistic,
    StatisticStatus,
    Tail,
    TailCompletion,
)


def test_numerical_queries_reject_booleans_before_numeric_promotion() -> None:
    law = Distribution.point_mass(1)
    curve = QuantileCurve([0, 1], [0, 1])
    for query in (law.quantile, law.cdf, law.exceedance, law.mass, curve.quantile):
        with pytest.raises(TypeError, match="real numbers"):
            query([False, 0.9])
        with pytest.raises(TypeError, match="real numbers"):
            query([np.bool_(True), 0.1])  # type: ignore[list-item]


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


@pytest.mark.parametrize("stored_zero", [0.0, -0.0])
def test_empirical_atoms_do_not_distinguish_signed_zero(stored_zero: float) -> None:
    law = Distribution.empirical([stored_zero, -stored_zero, 1.0])
    assert law.mass(0.0) == law.mass(-0.0) == pytest.approx(2 / 3)


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


def test_compound_hurdle_uses_the_first_surviving_support_above_a_gap() -> None:
    measurement = Measurement(quantity="x", unit="u")
    scale = Impact.linear(slope=3, input=measurement, output=measurement)
    base = scale.pushforward(Distribution.histogram([0, 1, 2, 3], [0.5, 0, 0.5]))
    law = Distribution.hurdle(base, atom_probability=0, atom_location=4)

    assert law.support == (6, 9)
    assert law._inverse_survival(1.0) == 6
    assert law._inverse_survival(0.0) == 9
    assert law.cdf(4) == law.mass(4) == 0
    assert law.exceedance(6) == law.exceedance(6, inclusive=True) == 1
    np.testing.assert_allclose(law.quantile([0, 0.5, 1]), [6, 7.5, 9])
    reopened = Distribution.from_json(law.to_json())
    assert reopened.support == law.support
    assert reopened.quantile(0) == 6

    reverse = Impact.linear(slope=-1, input=measurement, output=measurement)
    response = reverse.pushforward(law)
    assert response.support == (-9, -6)
    assert response.quantile(0.5) == -7.5
    assert reverse.expectation(law) == -7.5

    with_atom = Distribution.hurdle(base, atom_probability=0.2, atom_location=4)
    assert with_atom.support == (4, 9)
    assert with_atom.quantile(0.2) == 4
    assert with_atom.quantile(math.nextafter(0.2, 1.0)) == 6
    assert with_atom.cdf(4, inclusive=False) == 0
    assert with_atom.cdf(4) == with_atom.mass(4) == pytest.approx(0.2)
    assert with_atom.exceedance(4) == pytest.approx(0.8)
    assert with_atom.exceedance(4, inclusive=True) == 1


def test_compound_hurdle_integrates_subnormal_retained_histogram_mass() -> None:
    measurement = Measurement(quantity="x", unit="u")
    scale = Impact.linear(slope=3, input=measurement, output=measurement)
    base = scale.pushforward(
        Distribution.histogram([0, 1, 2, 3], [1, 0, math.ulp(0.0)])
    )
    law = Distribution.hurdle(base, atom_probability=0.25, atom_location=4)
    assert law.cdf(7.5) == pytest.approx(0.625)
    assert law.exceedance(7.5) == pytest.approx(0.375)
    assert law.quantile(0.625) == pytest.approx(7.5)
    assert law.mass(4) == pytest.approx(0.25)
    expected_mean = 0.25 * 4 + 0.75 * 7.5
    summary = law.statistics(metrics=[Statistic.MEAN, Statistic.CVAR])
    assert summary.values[Statistic.MEAN].value == pytest.approx(expected_mean)
    assert summary.values[Statistic.CVAR].value == pytest.approx(8.9)
    identity = Impact.linear(slope=1, input=measurement, output=measurement)
    assert identity.expectation(law) == pytest.approx(expected_mean)
    assert identity.pushforward(law).mass(4) == pytest.approx(0.25)


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


@pytest.mark.parametrize("depth", [50, 740, 1000])
@pytest.mark.parametrize("reflected", [False, True])
def test_deep_hurdle_over_a_nonlinear_response_keeps_analytical_tail_queries(
    depth: int, reflected: bool
) -> None:
    base = Distribution.parametric(
        DistributionFamily.GUMBEL_LEFT
        if reflected
        else DistributionFamily.GUMBEL_RIGHT,
        location=depth if reflected else -depth,
        scale=1,
    )
    measurement = Measurement(quantity="x", unit="u")
    response = Impact.sigmoid(
        midpoint=0,
        steepness=-1 if reflected else 1,
        input=measurement,
        output=measurement,
    ).pushforward(base)
    law = Distribution.hurdle(response, atom_probability=0.25, atom_location=0.5)

    # The deep Gumbel tail is exponential to machine precision. In the
    # logistic image, its conditional response is 1/(1+U) for U uniform.
    mean = 0.25 * 0.5 + 0.75 * math.log(2)
    variance = 0.25 * 0.5**2 + 0.75 * 0.5 - mean**2
    for result in (law, Distribution.from_json(law.to_json())):
        assert result.quantile(0.75) == pytest.approx(0.75, abs=1e-12)
        assert result.cdf(0.75) == pytest.approx(0.75, abs=1e-12)
        assert result.exceedance(0.75) == pytest.approx(0.25, abs=1e-12)
        assert result.mass(0.5) == 0.25
        assert result.mass(1) == 0
        values = result.statistics(metrics=[Statistic.MEAN, Statistic.VARIANCE]).values
        assert values[Statistic.MEAN].value == pytest.approx(mean, abs=1e-10)
        assert values[Statistic.VARIANCE].value == pytest.approx(variance, abs=1e-10)


@pytest.mark.parametrize(("probability", "shape"), [(1e-20, 0.5), (1e-9, 1e-8)])
def test_genpareto_quantile_inverts_small_positive_probabilities(
    probability: float, shape: float
) -> None:
    law = Distribution.parametric(
        DistributionFamily.GENERALIZED_PARETO, location=0, scale=1, shape=shape
    )
    value = float(law.quantile(probability))
    assert value > 0
    assert law.cdf(value) == pytest.approx(probability, rel=1e-13, abs=0)


def test_deep_nonlinear_hurdle_recovers_an_underflowed_clamp_atom() -> None:
    base = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=-1000, scale=1
    )
    measurement = Measurement(quantity="x", unit="u")
    document = json.loads(
        Impact.sigmoid(
            midpoint=0, steepness=1, input=measurement, output=measurement
        ).to_json()
    )
    document["clamp"] = {"maximum": 0.9}
    image = Impact.from_json(json.dumps(document)).pushforward(base)
    law = Distribution.hurdle(image, atom_probability=0.25, atom_location=0.5)
    assert law.support == (0.5, 0.9)
    assert law.exceedance(0.9) == 0
    assert law.exceedance(0.9, inclusive=True) == pytest.approx(1 / 12)
    assert law.mass(0.9) == pytest.approx(1 / 12)


@pytest.mark.parametrize(
    "family", [DistributionFamily.WEIBULL_MINIMUM, DistributionFamily.WEIBULL_MAXIMUM]
)
def test_large_weibull_shape_and_scale_keep_finite_variance(
    family: DistributionFamily,
) -> None:
    law = Distribution.parametric(family, location=0, scale=1e200, shape=1e200)
    values = law.statistics(metrics=[Statistic.VARIANCE]).values
    assert values[Statistic.VARIANCE].value == pytest.approx(math.pi**2 / 6, rel=1e-13)


@pytest.mark.parametrize("location", [-740, -1000])
def test_hurdle_keeps_interior_quantiles_when_source_survival_underflows(
    location: int,
) -> None:
    base = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=location, scale=1
    )
    law = Distribution.hurdle(base, atom_probability=0.25)
    q = 0.999999
    quantile = -math.log((1 - q) / 0.75)
    assert law.quantile(q) == pytest.approx(quantile, abs=2e-12)
    assert law.exceedance(quantile) == pytest.approx(1 - q, rel=2e-12, abs=0)
    assert law.mass(quantile) == 0.0
    values = law.statistics(metrics=[Statistic.MEAN, Statistic.VARIANCE]).values
    assert values[Statistic.MEAN].value == pytest.approx(0.75, abs=1e-9)
    assert values[Statistic.VARIANCE].value == pytest.approx(0.9375, abs=1e-9)
    measurement = Measurement(quantity="x", unit="u")
    image = Impact.linear(slope=2, input=measurement, output=measurement).pushforward(
        law
    )
    assert image.quantile(q) == pytest.approx(2 * quantile, abs=4e-12)


@pytest.mark.parametrize("location", [-740, -1000])
@pytest.mark.parametrize(
    ("family", "slope"),
    [(DistributionFamily.GUMBEL_RIGHT, 2), (DistributionFamily.GUMBEL_LEFT, -2)],
)
def test_hurdle_over_affine_image_keeps_subnormal_tail_queries(
    location: int, family: DistributionFamily, slope: int
) -> None:
    base = Distribution.parametric(
        family, location=math.copysign(abs(location), -slope), scale=1
    )
    measurement = Measurement(quantity="x", unit="u")
    affine = Impact.linear(slope=slope, input=measurement, output=measurement)
    law = Distribution.hurdle(affine.pushforward(base), atom_probability=0.25)
    q = 0.999999
    quantile = -2 * math.log((1 - q) / 0.75)
    for distribution in (law, Distribution.from_json(law.to_json())):
        assert distribution.quantile(q) == pytest.approx(quantile, abs=4e-12)
        assert distribution.exceedance(quantile) == pytest.approx(1 - q, rel=2e-12)
        assert distribution.cdf(quantile) == pytest.approx(q, abs=2e-12)
        assert distribution.mass(0) == 0.25
        values = distribution.statistics(
            metrics=[Statistic.MEAN, Statistic.VARIANCE]
        ).values
        assert values[Statistic.MEAN].value == pytest.approx(1.5, abs=2e-9)
        assert values[Statistic.VARIANCE].value == pytest.approx(3.75, abs=4e-9)
        assert affine.pushforward(distribution).quantile(q) == pytest.approx(
            abs(slope) * quantile if slope > 0 else 0.0, abs=8e-12
        )
    assert json.loads(law.to_json())["base"]["kind"] == "transformed"


def test_skew_normal_thin_tail_keeps_positive_density_mass() -> None:
    law = Distribution.parametric(
        DistributionFamily.SKEW_NORMAL, location=0, scale=1, shape=-5
    )
    reflected = Distribution.parametric(
        DistributionFamily.SKEW_NORMAL, location=0, scale=1, shape=5
    )
    # Independent integral of exp(-x*x/2) * erfc(5*x/sqrt(2)) / sqrt(2*pi)
    # over x >= 2, rounded to 15 significant digits.
    expected = 1.55326826787107e-26
    assert law.exceedance(2) == pytest.approx(expected, rel=1e-10, abs=0)
    assert reflected.cdf(-2) == pytest.approx(expected, rel=1e-10, abs=0)

    hurdle = Distribution.hurdle(law, atom_probability=0.25, atom_location=2)
    q = 0.75
    value = float(hurdle.quantile(q))
    assert value > 2
    assert hurdle.exceedance(value) == pytest.approx(1 - q, rel=1e-10)


@pytest.mark.parametrize(
    ("law", "variance"),
    [
        (Distribution.discrete([0, 1e200], [1, 1e-300]), 1e100),
        (Distribution.histogram([0, 1, 1e200], [1, 1e-300]), 1e100 / 3),
    ],
)
def test_probability_weighting_retains_large_finite_second_moments(
    law: Distribution, variance: float
) -> None:
    values = law.statistics(
        metrics=[Statistic.VARIANCE, Statistic.STANDARD_DEVIATION]
    ).values
    assert values[Statistic.VARIANCE].value == pytest.approx(variance, rel=1e-14)
    assert values[Statistic.STANDARD_DEVIATION].value == pytest.approx(
        math.sqrt(variance), rel=1e-14
    )


@pytest.mark.parametrize(
    ("law", "deviation"),
    [
        (Distribution.empirical([-1e200, 1e200]), 1e200),
        (Distribution.histogram([-1e200, 1e200], [1]), 1e200 / math.sqrt(3)),
        (
            Distribution.parametric(
                DistributionFamily.GUMBEL_RIGHT, location=0, scale=1e200
            ),
            math.pi * 1e200 / math.sqrt(6),
        ),
    ],
)
def test_law_standard_deviation_stays_finite_when_variance_overflows(
    law: Distribution, deviation: float
) -> None:
    values = law.statistics(
        metrics=[Statistic.VARIANCE, Statistic.STANDARD_DEVIATION]
    ).values
    assert values[Statistic.VARIANCE].status is StatisticStatus.POSITIVE_INFINITY
    assert values[Statistic.STANDARD_DEVIATION].status is StatisticStatus.FINITE
    assert values[Statistic.STANDARD_DEVIATION].value == pytest.approx(
        deviation, rel=1e-14
    )


def test_standard_deviation_survives_variance_underflow() -> None:
    law = Distribution.empirical([-1e-200, 1e-200])
    values = law.statistics(
        metrics=[Statistic.VARIANCE, Statistic.STANDARD_DEVIATION]
    ).values
    assert values[Statistic.VARIANCE].value == 0
    assert values[Statistic.STANDARD_DEVIATION].value == pytest.approx(
        1e-200, rel=1e-14, abs=0
    )


@pytest.mark.parametrize(
    "family,orientation",
    [(DistributionFamily.GUMBEL_RIGHT, 1), (DistributionFamily.GUMBEL_LEFT, -1)],
)
def test_gumbel_location_scale_cancellation_preserves_ranks(
    family: DistributionFamily, orientation: int
) -> None:
    scale = 1e308
    law = Distribution.parametric(family, location=-orientation * scale, scale=scale)
    probability = 0.9 if orientation > 0 else 0.1
    expected = orientation * (-math.log(-math.log(0.9)) - 1) * scale
    assert law.quantile(probability) == pytest.approx(expected, rel=1e-14)
    assert law.cdf(expected) == pytest.approx(probability, rel=1e-14)
    assert law.exceedance(expected) == pytest.approx(1 - probability, rel=1e-14)


@pytest.mark.parametrize(
    "family,shape,orientation",
    [
        (DistributionFamily.GENERALIZED_EXTREME_VALUE, 0, 1),
        (DistributionFamily.GENERALIZED_PARETO, 0, 1),
        (DistributionFamily.WEIBULL_MINIMUM, 1, 1),
        (DistributionFamily.WEIBULL_MAXIMUM, 1, -1),
    ],
)
def test_parametric_inverse_rescales_before_location_cancellation(
    family: DistributionFamily, shape: int, orientation: int
) -> None:
    scale = 1e308
    law = Distribution.parametric(
        family, location=-orientation * scale, scale=scale, shape=shape
    )
    rank = 0.9 if orientation > 0 else 0.1
    standardized = (
        -math.log(-math.log(rank))
        if family is DistributionFamily.GENERALIZED_EXTREME_VALUE
        else -math.log(0.1)
    )
    expected = orientation * (standardized - 1) * scale
    assert law.quantile(rank) == pytest.approx(expected, rel=1e-14)
    assert law._inverse_survival(1 - rank) == pytest.approx(expected, rel=1e-14)
    assert law.cdf(expected) == pytest.approx(rank, rel=1e-14)


@pytest.mark.parametrize("scale", [1.0, 1e-308])
def test_pareto_probability_takes_the_log_before_products_overflow(
    scale: float,
) -> None:
    shape = 1e308
    law = Distribution.parametric(
        DistributionFamily.GENERALIZED_PARETO, location=0, scale=scale, shape=shape
    )
    log_survival = -(math.log(shape) + math.log(10) - math.log(scale)) / shape
    assert law.cdf(10) == pytest.approx(-math.expm1(log_survival), rel=1e-14, abs=0)
    assert law.exceedance(10) == math.exp(log_survival)


def test_gev_applies_scale_before_exponentiating_large_moments() -> None:
    scale, shape = 1e-300, 250
    law = Distribution.parametric(
        DistributionFamily.GENERALIZED_EXTREME_VALUE,
        location=0,
        scale=scale,
        shape=shape,
    )
    first = math.lgamma(1 + shape)
    second = math.lgamma(1 + 2 * shape)
    factor = math.log(scale) - math.log(shape)
    mean = math.exp(factor + first) * math.expm1(-first)
    deviation = math.exp(
        factor + (second + math.log(-math.expm1(2 * first - second))) / 2
    )
    values = law.statistics(
        metrics=[Statistic.MEAN, Statistic.STANDARD_DEVIATION]
    ).values
    assert values[Statistic.MEAN].value == pytest.approx(mean, rel=1e-12)
    assert values[Statistic.STANDARD_DEVIATION].value == pytest.approx(
        deviation, rel=1e-12
    )


def test_exponential_pareto_preserves_representable_standard_deviation() -> None:
    scale = 1.5e308
    law = Distribution.parametric(
        DistributionFamily.GENERALIZED_PARETO, location=0, scale=scale, shape=0
    )
    value = law.statistics(metrics=[Statistic.STANDARD_DEVIATION]).values[
        Statistic.STANDARD_DEVIATION
    ]
    assert value.value == pytest.approx(scale, rel=1e-14)


@pytest.mark.parametrize(
    "shape,scale", [(0.05, 1e-20), (0.01, 1e-100), (0.004, 1e-300)]
)
@pytest.mark.parametrize(
    "family,orientation",
    [
        (DistributionFamily.WEIBULL_MINIMUM, 1.0),
        (DistributionFamily.WEIBULL_MAXIMUM, -1.0),
    ],
)
def test_weibull_moments_scale_raw_gamma_values_before_overflow(
    family: DistributionFamily, orientation: float, shape: float, scale: float
) -> None:
    law = Distribution.parametric(family, location=0, scale=scale, shape=shape)
    first = math.lgamma(1 + 1 / shape)
    second = math.lgamma(1 + 2 / shape)
    log_variance = second + math.log(-math.expm1(2 * first - second))
    mean = orientation * math.exp(math.log(scale) + first)
    deviation = math.exp(math.log(scale) + log_variance / 2)
    statistics = law.statistics(
        metrics=[Statistic.MEAN, Statistic.VARIANCE, Statistic.STANDARD_DEVIATION]
    ).values
    assert statistics[Statistic.MEAN].value == pytest.approx(mean, rel=1e-12)
    assert statistics[Statistic.STANDARD_DEVIATION].value == pytest.approx(
        deviation, rel=1e-12
    )
    variance = deviation * deviation
    if math.isinf(variance):
        assert (
            statistics[Statistic.VARIANCE].status is StatisticStatus.POSITIVE_INFINITY
        )
    else:
        assert statistics[Statistic.VARIANCE].value == pytest.approx(
            variance, rel=1e-12
        )


@pytest.mark.parametrize("transformed", [False, True])
def test_zero_atom_hurdle_at_support_boundary_keeps_base_statistics(
    transformed: bool,
) -> None:
    base = Distribution.parametric(
        DistributionFamily.GENERALIZED_PARETO, location=0, scale=1, shape=1.2
    )
    if transformed:
        measurement = Ratio()
        base = Impact.linear(
            input=measurement, output=measurement, slope=2
        ).pushforward(base)
    hurdle = Distribution.hurdle(base, atom_probability=0, atom_location=0)
    metrics = [Statistic.MEAN, Statistic.VARIANCE, Statistic.STANDARD_DEVIATION]
    assert hurdle.statistics(metrics=metrics) == base.statistics(metrics=metrics)


def test_nonlinear_image_retains_the_weighted_variance_of_a_rare_atom() -> None:
    source = Distribution.discrete([-1000, 1000], [1, 1e-300])
    measurement = Measurement(quantity="x", unit="u")
    law = Impact.sigmoid(
        midpoint=0, steepness=1, amplitude=1e200, input=measurement, output=measurement
    ).pushforward(source)
    values = law.statistics(metrics=[Statistic.VARIANCE]).values
    assert values[Statistic.VARIANCE].value == pytest.approx(1e100, rel=1e-14)


@pytest.mark.parametrize("slope", [1.0, -1.0])
@pytest.mark.parametrize("lower", [True, False])
def test_strict_response_tail_at_clamp_keeps_rare_continuous_mass(
    slope: float, lower: bool
) -> None:
    right = lower == (slope > 0)
    base = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT if right else DistributionFamily.GUMBEL_LEFT,
        location=-50 if right else 50,
        scale=1,
    )
    measurement = Measurement(quantity="x", unit="u")
    response = Impact.linear(
        slope=slope,
        minimum=0 if lower else None,
        maximum=None if lower else 0,
        input=measurement,
        output=measurement,
    ).pushforward(base)

    assert response.mass(0) == 1.0
    tail = response.exceedance(0) if lower else response.cdf(0, inclusive=False)
    assert tail == pytest.approx(math.exp(-50), rel=1e-14, abs=0)


def test_finite_linear_interpolation_avoids_overflow() -> None:
    partial = QuantileCurve([0, 1e-310], [0, 1])
    assert partial.quantile(5e-311) == pytest.approx(0.5)
    assert partial.condition(q_min=0, q_max=5e-311).quantile(1) == pytest.approx(0.5)

    histogram = Distribution.histogram([-1e308, 1e308], [1])
    completed = QuantileCurve([0, 1], [-1e308, 1e308]).condition(q_min=0, q_max=1)
    measurement = Measurement(quantity="x", unit="u")
    identity = Impact.linear(slope=1, input=measurement, output=measurement)
    for law in (histogram, completed):
        for distribution in (law, identity.pushforward(law)):
            assert distribution.quantile(0.5) == 0.0
            assert distribution.cdf(0) == distribution.exceedance(0) == 0.5


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


@pytest.mark.parametrize("level", [1e-20, math.ulp(0.0)])
@pytest.mark.parametrize("tail,sign", [(Tail.UPPER, 1), (Tail.LOWER, -1)])
def test_distribution_cvar_keeps_tiny_excluded_mass(
    level: float, tail: Tail, sign: int
) -> None:
    measurement = Measurement(quantity="x", unit="u")
    identity = Impact.linear(slope=1, input=measurement, output=measurement)
    discrete = Distribution.discrete([-1e308, 1e308], [0.5, 0.5])
    histogram = Distribution.histogram([-1e308, 1e308], [1.0])
    for law in (discrete, histogram, identity.pushforward(histogram)):
        value = law.statistics(metrics=[Statistic.CVAR], level=level, tail=tail)
        assert value.values[Statistic.CVAR].value == pytest.approx(
            sign * 1e308 * level, rel=1e-15, abs=0.0
        )


def test_distribution_cvar_skips_fully_excluded_rare_atom() -> None:
    law = Distribution.discrete([-1e308, 1], [1e-20, 1.0])
    value = law.statistics(metrics=[Statistic.CVAR], level=1e-20)
    assert value.values[Statistic.CVAR].value == 1


@pytest.mark.parametrize("half", [6, 9])
def test_empirical_rank_at_gap_uses_lower_quantile(half: int) -> None:
    gap = 1_000_000
    law = Distribution.empirical([*range(half), *range(gap, gap + half)])
    assert law.cdf(half - 1) == law.exceedance(half - 1) == 0.5
    assert law.quantile(0.5) == law._inverse_survival(0.5) == half - 1
    assert law.quantile(math.nextafter(0.5, 1)) == gap
    assert law._inverse_survival(math.nextafter(0.5, 0)) == gap


def test_empirical_ninetieth_percentile_at_gap() -> None:
    law = Distribution.empirical([*range(9), 1_000_000])
    assert law.cdf(8) == 0.9
    assert law.quantile(0.9) == 8


def test_histogram_rank_at_gap_uses_lower_quantile() -> None:
    gap = 1_000_000
    law = Distribution.histogram(
        [*range(7), *range(gap, gap + 7)], [1 / 12] * 6 + [0] + [1 / 12] * 6
    )
    assert law.cdf(6) == law.exceedance(6) == 0.5
    assert law.quantile(0.5) == law._inverse_survival(0.5) == 6
    assert law.quantile(math.nextafter(0.5, 1)) >= gap
    assert law._inverse_survival(math.nextafter(0.5, 0)) >= gap
    reflected = Impact.linear(slope=-1, input=Ratio(), output=Ratio()).pushforward(law)
    assert reflected.quantile(0.5) == -gap
    assert reflected._inverse_survival(0.5) == -gap


@pytest.mark.parametrize(
    "family,shape,location,q,endpoint",
    [
        (DistributionFamily.GENERALIZED_EXTREME_VALUE, 0.5, -1e308, 1, 1e308),
        (DistributionFamily.GENERALIZED_EXTREME_VALUE, -0.5, 1e308, 0, -1e308),
        (DistributionFamily.GENERALIZED_PARETO, -0.5, -1e308, 1, 1e308),
    ],
)
def test_finite_endpoint_after_scale_shape_quotient_overflows(
    family: DistributionFamily, shape: float, location: float, q: int, endpoint: float
) -> None:
    law = Distribution.parametric(family, location=location, scale=1e308, shape=shape)
    assert law.support[q] == law.quantile(q) == law._inverse_survival(1 - q) == endpoint
    assert law.cdf(endpoint) == q

    half = Impact.linear(slope=0.5, input=Ratio(), output=Ratio()).pushforward(law)
    assert half.support[q] == half.quantile(q) == endpoint / 2

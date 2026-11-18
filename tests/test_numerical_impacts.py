"""Analytical checks for calibrated point responses and exact pushforwards."""

from __future__ import annotations

import json
import math
from typing import cast

import numpy as np
import pytest
from crc_framework import (
    AssetType,
    BuiltinImpact,
    DepthAboveGround,
    Distribution,
    DistributionFamily,
    FloodRegion,
    Impact,
    ImpactError,
    ImpactReference,
    InundationContext,
    LengthUnit,
    Measurement,
    QuantileCurve,
    Ratio,
    RatioUnit,
    Statistic,
    StatisticStatus,
    Tail,
    TailCompletion,
)


def linear(slope: float = 2.0, **kwargs: float | None) -> Impact:
    return Impact.linear(
        slope=slope,
        input=Measurement(quantity="exposure", unit="1"),
        output=Ratio(),
        **kwargs,
    )


def test_linear_does_not_hide_a_zero_clamp() -> None:
    response = linear()
    assert response.evaluate(-2.0) == -4.0
    assert response.expectation(Distribution.discrete([-2, 1], [0.25, 0.75])) == 0.5
    assert not callable(response)


@pytest.mark.parametrize("shape", [(), (4,), (2, 2), (2, 0, 3)])
def test_all_point_responses_preserve_shape_and_freeze_results(
    shape: tuple[int, ...],
) -> None:
    data = np.full(shape, 0.5)
    functions = [
        linear(),
        Impact.sigmoid(midpoint=0, steepness=1, input=Ratio(), output=Ratio()),
        Impact.tabulated([(0, 0), (1, 2)], input=Ratio(), output=Ratio()),
        Impact.binned([0, 1], [2], input=Ratio(), output=Ratio()),
    ]
    for function in functions:
        result = function.evaluate(data)
        if shape == ():
            assert isinstance(result, float)
        else:
            assert isinstance(result, np.ndarray)
            assert result.shape == shape
            with pytest.raises(ValueError):
                result.setflags(write=True)


def test_exact_expectation_retains_catastrophic_rare_atoms() -> None:
    source = Distribution.discrete([0.0, 1e16], [1 - 1e-16, 1e-16])
    response = linear(1)
    assert response.expectation(source) == pytest.approx(1.0)
    result = response.pushforward(source)
    assert result.mass(1e16) == 1e-16
    assert result.quantile(1.0) == 1e16
    assert result.statistics(metrics=[Statistic.MEAN]).values[
        Statistic.MEAN
    ].value == pytest.approx(1.0)


def test_empirical_expectation_is_observed_mean_not_interpolated_order_statistics() -> (
    None
):
    source = Distribution.empirical([0, 0, 30])
    assert linear(1).expectation(source) == pytest.approx(10.0)


def test_decreasing_response_reverses_atoms_with_correct_threshold_boundaries() -> None:
    source = Distribution.discrete([0, 2], [0.9, 0.1])
    image = linear(-1).pushforward(source)
    assert image.quantile(0.1) == -2.0
    assert image.quantile(np.nextafter(0.1, 1.0)) == 0.0
    assert image.cdf(-2.0) == pytest.approx(0.1)
    assert image.cdf(-2.0, inclusive=False) == 0.0
    assert image.exceedance(-2.0) == pytest.approx(0.9)
    assert image.mass(-2.0) == pytest.approx(0.1)


def test_nonmonotone_finite_image_is_exact() -> None:
    response = Impact.tabulated([(0, 0), (1, 2), (2, 0)], input=Ratio(), output=Ratio())
    source = Distribution.discrete([0, 1, 2], [0.2, 0.3, 0.5])
    image = response.pushforward(source)
    assert image.quantile([0.1, 0.7, 0.8]).tolist() == [0.0, 0.0, 2.0]
    assert response.expectation(source) == pytest.approx(0.6)
    with pytest.raises(ImpactError, match="monotonicity"):
        response.pushforward(Distribution.histogram([0, 2], [1]))


def test_binned_expectation_and_image_use_exact_boundary_probabilities() -> None:
    response = Impact.binned([0, 1, 2, 3], [4, 1, 3], input=Ratio(), output=Ratio())
    source = Distribution.histogram([0, 3], [1])
    assert response.expectation(source) == pytest.approx(8 / 3)
    image = response.pushforward(source)
    assert image.quantile([0.1, 0.5, 0.9]).tolist() == [1.0, 3.0, 4.0]
    atom_source = Distribution.discrete([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
    assert response.evaluate(1.0) == 1.0
    assert response.expectation(atom_source) == pytest.approx(2.7)


def test_exact_binned_expectation_works_on_a_parametric_law() -> None:
    source = Distribution.parametric(
        DistributionFamily.GENERALIZED_PARETO, location=0, scale=1, shape=-1
    )
    response = Impact.binned([0, 0.25, 1], [4, 2], input=Ratio(), output=Ratio())
    assert response.expectation(source) == pytest.approx(2.5)


def test_transformed_statistics_preserve_flat_mass_and_small_continuous_tail() -> None:
    floor = 1.0 - 5e-13
    image = linear(1, minimum=floor).pushforward(Distribution.histogram([0, 1], [1]))
    assert image.quantile(1.0) == 1.0
    assert image.quantile(1.0 - 1e-13) > floor
    assert image.mass(floor) == pytest.approx(floor)
    summary = image.statistics(metrics=[Statistic.MEAN, Statistic.MAXIMUM])
    assert summary.values[Statistic.MAXIMUM].value == 1.0
    assert summary.values[Statistic.MEAN].value == pytest.approx(floor)


def test_histogram_linear_pushforward_statistics_are_analytic() -> None:
    image = linear(-2, intercept=5).pushforward(Distribution.histogram([0, 2], [1]))
    summary = image.statistics(
        metrics=[Statistic.MEAN, Statistic.VARIANCE, Statistic.CVAR]
    )
    assert summary.values[Statistic.MEAN].value == pytest.approx(3)
    assert summary.values[Statistic.VARIANCE].value == pytest.approx(4 / 3)
    assert summary.values[Statistic.CVAR].value == pytest.approx(4.9)
    assert image.quantile([0, 0.25, 1]).tolist() == [1.0, 2.0, 5.0]


def test_a_decreasing_map_skips_zero_mass_support_gaps() -> None:
    source = Distribution.histogram([0, 1, 2, 3], [0.5, 0, 0.5])
    image = linear(-1).pushforward(source)
    assert image.quantile(0.5) == -2.0


def test_partial_evidence_cannot_be_used_as_a_law() -> None:
    evidence = QuantileCurve([0.9, 0.99], [1, 3])
    with pytest.raises(TypeError, match="Distribution"):
        linear().expectation(evidence)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Distribution"):
        linear().pushforward(evidence)  # type: ignore[arg-type]
    completed = evidence.complete(TailCompletion(lower=0, upper=4))
    assert math.isfinite(linear().expectation(completed))


def test_unsupported_exact_expectation_is_an_explicit_capability_error() -> None:
    response = Impact.sigmoid(midpoint=0, steepness=1, input=Ratio(), output=Ratio())
    law = Distribution.parametric(DistributionFamily.GUMBEL_RIGHT, location=0, scale=1)
    with pytest.raises(ImpactError) as error:
        response.expectation(law)
    assert error.value.code == "exact_method_unavailable"


def test_impact_serialization_round_trips_current_format_only() -> None:
    response = Impact.linear(
        slope=0.1, input=DepthAboveGround(unit=LengthUnit.METRE), output=Ratio()
    )
    loaded = Impact.from_json(response.to_json())
    assert loaded.sha256 == response.sha256
    assert loaded.input == DepthAboveGround(unit=LengthUnit.METRE)
    assert loaded.evaluate(3) == pytest.approx(0.3)
    document = json.loads(response.to_json())
    assert document["schema_version"] == "2.0"
    document["schema_version"] = "1.0"
    with pytest.raises(ImpactError, match="schema"):
        Impact.from_json(json.dumps(document))
    with pytest.raises(ImpactError):
        Impact.from_json(
            response.to_json().replace(
                '"schema_version":"2.0"',
                '"schema_version":"2.0","schema_version":"2.0"',
            )
        )


def test_impact_objects_and_measurements_are_immutable() -> None:
    response = linear()
    with pytest.raises(AttributeError):
        response._input = Ratio()  # type: ignore[misc]
    measurement = Measurement(
        quantity="custom", unit="my-unit", fields={"threshold": [1, 2]}
    )
    assert measurement.fields["threshold"] == (1, 2)
    with pytest.raises(TypeError):
        measurement.fields["threshold"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        DepthAboveGround(unit="m")  # type: ignore[arg-type]
    assert Ratio(unit=RatioUnit.PERCENT).unit is RatioUnit.PERCENT


@pytest.mark.parametrize("values", [[math.nan], [math.inf], [-math.inf]])
def test_nonfinite_exposures_are_rejected(values: list[float]) -> None:
    with pytest.raises(ImpactError, match="finite"):
        linear().evaluate(values)


def test_model_requirements_and_typed_context_are_discoverable() -> None:
    reference = Impact.builtin(BuiltinImpact.INUNDATION)
    requirements = reference.requirements
    assert requirements.measurement_type is DepthAboveGround
    assert requirements.accepted_units == (LengthUnit.METRE,)
    assert requirements.context_type is InundationContext
    resolved = reference.resolve(
        requirements.context_type(
            region=FloodRegion.EUROPE, asset_type=AssetType.COMMERCIAL
        )
    )
    assert isinstance(resolved, Impact)
    assert resolved.input == DepthAboveGround(unit=LengthUnit.METRE)
    assert 0 <= cast(float, resolved.evaluate(1.5)) <= 1
    assert ImpactReference.from_json(reference.to_json()).sha256 == reference.sha256
    with pytest.raises(ImpactError, match="InundationContext"):
        reference.resolve()


@pytest.mark.parametrize("model", list(BuiltinImpact))
def test_every_builtin_identity_resolves_with_its_discovered_context(
    model: BuiltinImpact,
) -> None:
    reference = Impact.builtin(model)
    requirements = reference.requirements
    data = {
        "cell": int("852b83b3fffffff", 16),
        "country": "USA" if model is BuiltinImpact.CONSECUTIVE_DRY_DAYS else "CAN",
        "region": FloodRegion.EUROPE,
        "asset_type": AssetType.AGRICULTURAL_BUILDING
        if model is BuiltinImpact.CONSECUTIVE_DRY_DAYS
        else AssetType.RESIDENTIAL,
        "historic_mean": 20.0,
    }
    values = {field.name: data[field.name] for field in requirements.context_fields}
    context = requirements.context_type(**values)
    response = reference.resolve(context)
    assert isinstance(response, Impact)
    assert math.isfinite(cast(float, response.evaluate(3.0)))
    assert Impact.from_json(response.to_json()).evaluate(3.0) == response.evaluate(3.0)


def test_continuous_affine_composition_remains_exact_and_serializable() -> None:
    law = Distribution.histogram([0, 2], [1])
    first = linear(3, intercept=1).pushforward(law)
    second = linear(-2, intercept=4)
    composed = second.pushforward(first)
    assert composed.quantile([0, 0.5, 1]).tolist() == [-10.0, -4.0, 2.0]
    assert second.expectation(first) == pytest.approx(-4.0)
    assert composed.statistics(metrics=[Statistic.MEAN]).values[
        Statistic.MEAN
    ].value == pytest.approx(-4.0)
    assert Distribution.from_json(composed.to_json()).quantile(0.5) == -4.0


def test_parametric_affine_composition_has_no_intermediate_sampling() -> None:
    law = Distribution.parametric(DistributionFamily.GUMBEL_RIGHT, location=0, scale=1)
    first = linear(3, intercept=1).pushforward(law)
    second = linear(-2, intercept=4)
    composed = second.pushforward(first)
    assert second.expectation(first) == pytest.approx(2 - 6 * 0.5772156649015329)
    assert composed.quantile(0.2) == pytest.approx(
        2 - 6 * cast(float, law.quantile(0.8))
    )


def test_finite_hurdle_impact_uses_the_conditioned_tail() -> None:
    source = Distribution.hurdle(
        Distribution.histogram([0, 4], [1]), atom_probability=0.7, atom_location=1
    )
    response = linear(2)
    assert response.expectation(source) == pytest.approx(2.9)
    image = response.pushforward(source)
    assert image.mass(2.0) == pytest.approx(0.7)
    assert image.statistics(metrics=[Statistic.MEAN]).values[
        Statistic.MEAN
    ].value == pytest.approx(2.9)
    reflected = linear(-1).pushforward(source)
    assert reflected.quantile(0.3) == pytest.approx(-1.0)
    assert reflected.cdf(-1.0, inclusive=False) == pytest.approx(0.3)
    assert reflected.cdf(-1.0) == 1.0


def test_exact_hurdle_impact_preserves_a_sub_ulp_atom() -> None:
    source = Distribution.hurdle(
        Distribution.discrete([1, 1e20], [1, 1e-20]), atom_probability=0.2
    )
    identity = linear(1)
    assert identity.expectation(source) == pytest.approx(1.6)
    image = identity.pushforward(source)
    assert image.mass(1e20) == pytest.approx(8e-21, abs=0)
    assert image.exceedance(1.0) == pytest.approx(8e-21, abs=0)
    assert image.quantile(1.0) == 1e20


@pytest.mark.parametrize("hurdle", [False, True])
def test_exact_impact_preserves_a_sub_ulp_continuous_interval(hurdle: bool) -> None:
    base = Distribution.histogram([1, 2, 2e20], [1, 1e-20])
    source = Distribution.hurdle(base, atom_probability=0.2) if hurdle else base
    weight = 0.8 if hurdle else 1
    identity = linear(1)
    assert identity.expectation(source) == pytest.approx(2.5 * weight)
    assert identity.pushforward(source).exceedance(2.0) == pytest.approx(
        1e-20 * weight, abs=0
    )
    assert linear(-1).pushforward(source).quantile(5e-21 * weight) == pytest.approx(
        -1e20
    )


@pytest.mark.parametrize("steepness", [0.22, -0.22])
def test_sigmoid_pushforward_probability_respects_rounded_atom(
    steepness: float,
) -> None:
    source = Distribution.hurdle(
        Distribution.parametric(DistributionFamily.GUMBEL_RIGHT, location=0, scale=1),
        atom_probability=0.2,
    )
    impact = Impact.sigmoid(
        midpoint=-2.33, steepness=steepness, input=Ratio(), output=Ratio()
    )
    atom = cast(float, impact.evaluate(0))
    image = impact.pushforward(source)
    assert image.mass(atom) == pytest.approx(0.2)
    below = 0.0 if steepness > 0 else 0.8
    assert image.cdf(atom, inclusive=False) == pytest.approx(below)
    assert image.cdf(atom) == pytest.approx(below + 0.2)
    assert image.exceedance(atom) == pytest.approx(0.8 - below)
    assert image.exceedance(atom, inclusive=True) == pytest.approx(1 - below)


def test_hurdle_generalized_inverse_excludes_discarded_atoms() -> None:
    base = Distribution.discrete([-1, 2, 3], [0.5, 0.1, 0.4])
    hurdle = Distribution.hurdle(base, atom_probability=0.2)
    assert hurdle.quantile(0.2) == 0.0
    assert hurdle.quantile(math.nextafter(0.2, 1.0)) == 2.0
    assert Distribution.hurdle(base, atom_probability=0).quantile(0.0) == 2.0


@pytest.mark.parametrize("field", ["fields", "schema"])
def test_measurement_extensions_require_mappings(field: str) -> None:
    with pytest.raises(TypeError, match="mapping"):
        Measurement(**{field: ["bad"]})  # type: ignore[arg-type]
    document = json.loads(linear().to_json())
    document["input"][field] = ["bad"]
    with pytest.raises(ImpactError, match="mapping"):
        Impact.from_json(json.dumps(document))


@pytest.mark.parametrize("slope", [2.0, -2.0])
def test_affine_pushforward_propagates_divergent_moments(slope: float) -> None:
    source = Distribution.parametric(
        DistributionFamily.GENERALIZED_PARETO, location=0, scale=1, shape=1
    )
    image = linear(slope, intercept=3).pushforward(source)
    summary = image.statistics(
        metrics=[
            Statistic.MEAN,
            Statistic.VARIANCE,
            Statistic.MINIMUM,
            Statistic.MAXIMUM,
        ]
    )
    expected = (
        StatisticStatus.POSITIVE_INFINITY
        if slope > 0
        else StatisticStatus.NEGATIVE_INFINITY
    )
    assert summary.values[Statistic.MEAN].status is expected
    assert (
        summary.values[Statistic.VARIANCE].status is StatisticStatus.POSITIVE_INFINITY
    )
    infinite_tail = Tail.UPPER if slope > 0 else Tail.LOWER
    cvar = image.statistics(metrics=[Statistic.CVAR], tail=infinite_tail)
    assert cvar.values[Statistic.CVAR].status is expected
    finite_tail = Tail.LOWER if slope > 0 else Tail.UPPER
    cvar = image.statistics(metrics=[Statistic.CVAR], tail=finite_tail)
    assert cvar.values[Statistic.CVAR].status is StatisticStatus.FINITE
    assert image.quantile(1.0 if slope > 0 else 0.0) == math.copysign(math.inf, slope)


def test_affine_pushforward_keeps_finite_mean_and_divergent_variance_distinct() -> None:
    source = Distribution.parametric(
        DistributionFamily.GENERALIZED_PARETO, location=0, scale=1, shape=0.75
    )
    image = linear(-2, intercept=3).pushforward(source)
    summary = image.statistics(metrics=[Statistic.MEAN, Statistic.VARIANCE])
    assert summary.values[Statistic.MEAN].value == pytest.approx(-5)
    assert (
        summary.values[Statistic.VARIANCE].status is StatisticStatus.POSITIVE_INFINITY
    )


@pytest.mark.parametrize("slope", [1.0, -1.0])
def test_affine_clamps_can_complete_a_divergent_tail(slope: float) -> None:
    source = Distribution.parametric(
        DistributionFamily.GENERALIZED_PARETO, location=0, scale=1, shape=1
    )
    response = linear(
        slope, maximum=4 if slope > 0 else None, minimum=-4 if slope < 0 else None
    )
    image = response.pushforward(source)
    summary = image.statistics(metrics=[Statistic.MEAN, Statistic.VARIANCE])
    assert summary.values[Statistic.MEAN].value == pytest.approx(slope * math.log(5))
    assert summary.values[Statistic.VARIANCE].value == pytest.approx(
        2 * (4 - math.log(5)) - math.log(5) ** 2
    )
    assert image.quantile(1 if slope > 0 else 0) == 4 * slope


@pytest.mark.parametrize("slope", [1.0, -1.0])
def test_affine_clamps_on_the_other_end_leave_divergence_visible(slope: float) -> None:
    source = Distribution.parametric(
        DistributionFamily.GENERALIZED_PARETO, location=0, scale=1, shape=1
    )
    response = linear(
        slope, minimum=1 if slope > 0 else None, maximum=-1 if slope < 0 else None
    )
    image = response.pushforward(source)
    summary = image.statistics(metrics=[Statistic.MEAN, Statistic.VARIANCE])
    assert summary.values[Statistic.MEAN].status is (
        StatisticStatus.POSITIVE_INFINITY
        if slope > 0
        else StatisticStatus.NEGATIVE_INFINITY
    )
    assert (
        summary.values[Statistic.VARIANCE].status is StatisticStatus.POSITIVE_INFINITY
    )


def test_zero_slope_is_a_point_law_even_when_source_moments_diverge() -> None:
    source = Distribution.parametric(
        DistributionFamily.GENERALIZED_PARETO, location=0, scale=1, shape=2
    )
    image = linear(0, intercept=7).pushforward(source)
    summary = image.statistics(
        metrics=[Statistic.MEAN, Statistic.VARIANCE, Statistic.CVAR]
    )
    assert summary.values[Statistic.MEAN].value == 7
    assert summary.values[Statistic.VARIANCE].value == 0
    assert summary.values[Statistic.CVAR].value == pytest.approx(7)
    assert image.quantile([0, 1]).tolist() == [7, 7]


def test_affine_variance_scaling_avoids_premature_overflow_and_underflow() -> None:
    source = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=0, scale=1e-100
    )
    image = linear(1e200).pushforward(source)
    variance = image.statistics(metrics=[Statistic.VARIANCE]).values[Statistic.VARIANCE]
    assert variance.status is StatisticStatus.FINITE
    assert variance.value == pytest.approx(math.pi**2 / 6 * 1e200)
    heavy = Distribution.parametric(
        DistributionFamily.GENERALIZED_PARETO, location=0, scale=1, shape=0.75
    )
    image = linear(1e-200).pushforward(heavy)
    assert (
        image.statistics(metrics=[Statistic.VARIANCE]).values[Statistic.VARIANCE].status
        is StatisticStatus.POSITIVE_INFINITY
    )

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
    DistributionKind,
    EmptyContext,
    ExpectationMethod,
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
    Temperature,
    TemperatureUnit,
    WildfireContext,
    _core,
)


def linear(slope: float = 2.0, **kwargs: float | None) -> Impact:
    return Impact.linear(
        slope=slope,
        input=Ratio(),
        output=Ratio(),
        **kwargs,
    )


def test_linear_does_not_hide_a_zero_clamp() -> None:
    response = linear()
    assert response.evaluate(-2.0) == -4.0
    assert response.expectation(Distribution.discrete([-2, 1], [0.25, 0.75])) == 0.5
    assert not callable(response)


def test_discrete_impact_preserves_quantiles_at_cumulative_mass_boundaries() -> None:
    source = Distribution.discrete([0, 1, 2, 3], [0.05, 0.1, 0.15, 0.7])
    image = linear(1).pushforward(source)
    native = _core.NativeNumericalDocument(image.to_json())
    cutoff = 0.3
    above_cutoff = math.nextafter(cutoff, 1.0)

    assert image.cdf(2) == cutoff
    for law in (source, image, native):
        assert law.quantile(cutoff) == 2
        assert law.quantile(above_cutoff) == 3


def test_discrete_impact_keeps_small_masses_when_merging_atoms() -> None:
    small = [1e-17] * 10
    source = Distribution.discrete(list(range(12)), [0.5, *small, 0.5 - 1e-16])
    image = linear(1, minimum=10).pushforward(source)
    merged_mass = math.fsum([0.5, *small])

    assert image.mass(10) == merged_mass
    assert image.quantile(merged_mass) == 10
    assert image.quantile(math.nextafter(merged_mass, 1.0)) == 11


def test_linear_overflow_is_checked_after_intercept_and_clamps() -> None:
    assert linear(2, intercept=-1e308).evaluate(1e308) == 1e308
    assert linear(2, maximum=100).evaluate(1e308) == 100
    assert linear(2, minimum=-100).evaluate(-1e308) == -100
    with pytest.raises(ImpactError, match="non_finite_result"):
        linear(2, maximum=100).evaluate(-1e308)


@pytest.mark.parametrize(
    "source",
    [
        Distribution.point_mass(5),
        Distribution.discrete([1, 5], [0.25, 0.75]),
        Distribution.histogram([0, 5, 10], [0.3, 0.7]),
    ],
)
def test_chained_impacts_validate_the_previous_output_measurement(
    source: Distribution,
) -> None:
    depth = DepthAboveGround(unit=LengthUnit.METRE)
    temperature = Temperature(unit=TemperatureUnit.CELSIUS)
    first = Impact.linear(slope=1, input=depth, output=temperature)
    inconsistent = Impact.linear(slope=2, input=depth, output=Ratio())
    consistent = Impact.linear(slope=2, input=temperature, output=Ratio())
    image = first.pushforward(source)
    restored = Distribution.from_json(image.to_json())
    for law in (image, restored, Distribution.hurdle(restored, atom_probability=0.1)):
        assert law.kind in (DistributionKind.TRANSFORMED, DistributionKind.HURDLE)
        for operation in (inconsistent.pushforward, inconsistent.expectation):
            with pytest.raises(ImpactError, match="source impact output") as failure:
                operation(law)
            assert failure.value.code == "measurement_mismatch"
        assert consistent.pushforward(law).quantile(0.5) == pytest.approx(
            2 * float(law.quantile(0.5))
        )
    assert json.loads(restored.to_json())["impact"]["output"] == json.loads(
        temperature.to_json()
    )


@pytest.mark.parametrize("rank", [1e-16, 1e-18])
def test_reflected_unbounded_law_keeps_small_interior_ranks(rank: float) -> None:
    source = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=0, scale=1
    )
    image = linear(-1).pushforward(source)
    expected = math.log(-math.log1p(-rank))
    assert image.quantile(rank) == pytest.approx(expected, rel=1e-14)
    assert image.cdf(expected) == pytest.approx(rank, rel=1e-14, abs=0)


def test_reflected_hurdle_skips_zero_mass_gap_at_exact_rank() -> None:
    base = Distribution.parametric(
        DistributionFamily.WEIBULL_MINIMUM, location=10, scale=1, shape=2
    )
    image = linear(-1).pushforward(Distribution.hurdle(base, atom_probability=0.25))
    assert image.quantile(0.75) == -10
    assert image._inverse_survival(0.25) == -10


def test_affine_composition_preserves_wide_finite_histogram() -> None:
    response = linear(1)
    source = Distribution.histogram([-1e308, 1e308], [1])
    image = response.pushforward(source)
    composed = response.pushforward(image)
    assert composed.quantile(0.5) == 0
    assert response.expectation(image) == 0
    assert Distribution.from_json(composed.to_json()).quantile(0.5) == 0


@pytest.mark.parametrize(
    "points",
    [
        [(-1e308, -1), (1e308, 1)],
        [(-1, -1e308), (1, 1e308)],
        [(-1, 1e308), (1, -1e308)],
    ],
)
def test_tabulated_midpoint_handles_wide_finite_intervals(
    points: list[tuple[float, float]],
) -> None:
    response = Impact.tabulated(points, input=Ratio(), output=Ratio())
    assert response.evaluate(0) == 0
    assert (
        response.pushforward(
            Distribution.histogram([points[0][0], points[1][0]], [1])
        ).quantile(0.5)
        == 0
    )


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


@pytest.mark.parametrize("value", [False, np.bool_(True), "1"])
def test_binned_rejects_coerced_edges_and_values(value: object) -> None:
    with pytest.raises(TypeError, match="bin edges.*real numbers"):
        Impact.binned([value, 2], [1], input=Ratio(), output=Ratio())  # type: ignore[list-item]
    with pytest.raises(TypeError, match="bin values.*real numbers"):
        Impact.binned([0, 2], [value], input=Ratio(), output=Ratio())  # type: ignore[list-item]


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


@pytest.mark.parametrize(
    ("direction", "upper_cvar", "lower_cvar"),
    [(1, 2.0, 1 / 18), (-1, -1 / 18, -2.0)],
)
def test_piecewise_linear_segment_moments_cover_both_tail_directions(
    direction: int, upper_cvar: float, lower_cvar: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Distribution.hurdle(
        Distribution.histogram([0, 4], [1]), atom_probability=0.25, atom_location=0
    )
    response = Impact.tabulated(
        [(0, 0), (2, direction), (4, 3 * direction)], input=Ratio(), output=Ratio()
    )
    image = response.pushforward(source)

    def unexpected_source_rebuild(*args: object) -> None:
        pytest.fail("the native pushforward already contains the source")

    monkeypatch.setattr(Distribution, "from_json", unexpected_source_rebuild)
    upper = image.statistics(
        metrics=[Statistic.MEAN, Statistic.VARIANCE, Statistic.CVAR], level=0.625
    )
    mean = direction * 15 / 16
    assert upper.values[Statistic.MEAN].value == pytest.approx(mean)
    assert upper.values[Statistic.VARIANCE].value == pytest.approx(7 / 4 - mean**2)
    assert upper.values[Statistic.CVAR].value == pytest.approx(upper_cvar)
    lower = image.statistics(metrics=[Statistic.CVAR], level=0.625, tail=Tail.LOWER)
    assert lower.values[Statistic.CVAR].value == pytest.approx(lower_cvar)


def test_partial_evidence_cannot_be_used_as_a_law() -> None:
    evidence = QuantileCurve([0.9, 0.99], [1, 3])
    with pytest.raises(TypeError, match="Distribution"):
        linear().expectation(evidence)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Distribution"):
        linear().pushforward(evidence)  # type: ignore[arg-type]
    completed = evidence.complete(TailCompletion(lower=0, upper=4))
    assert math.isfinite(linear().expectation(completed))


def test_expectation_uses_numerical_integration_when_exact_is_unavailable() -> None:
    response = Impact.sigmoid(
        midpoint=1, steepness=2, input=Ratio(), output=Ratio()
    )
    law = Distribution.parametric(DistributionFamily.GUMBEL_RIGHT, location=1, scale=0.5)
    image_mean = response.pushforward(law).statistics(metrics=[Statistic.MEAN])
    assert response.expectation(law) == pytest.approx(0.5963473623297835)
    assert response.expectation(law, method=ExpectationMethod.AUTO) == pytest.approx(
        image_mean.values[Statistic.MEAN].value
    )
    with pytest.raises(ImpactError) as error:
        response.expectation(law, method=ExpectationMethod.EXACT)
    assert error.value.code == "exact_method_unavailable"
    with pytest.raises(TypeError, match="ExpectationMethod"):
        response.expectation(law, method="auto")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "response",
    [
        linear(1, minimum=0, maximum=1),
        Impact.builtin(BuiltinImpact.HOT_DAYS).resolve(),
        Impact.builtin(BuiltinImpact.FIRE_WEATHER_INDEX).resolve(
            WildfireContext(wildland_urban_interface=True)
        ),
        Impact.builtin(BuiltinImpact.FIRE_WEATHER_INDEX).resolve(
            WildfireContext(wildland_urban_interface=False)
        ),
    ],
)
def test_clamped_and_builtin_expectations_on_hurdle_laws(response: Impact) -> None:
    base = Distribution.parametric(DistributionFamily.GUMBEL_RIGHT, location=3, scale=0.5)
    law = Distribution.hurdle(base, atom_probability=0.25, atom_location=0)
    image = response.pushforward(law).statistics(metrics=[Statistic.MEAN])
    result = response.expectation(law)
    assert math.isfinite(result)
    assert result == pytest.approx(image.values[Statistic.MEAN].value)


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
        "country": "USA"
        if model
        in (
            BuiltinImpact.CONSECUTIVE_DRY_DAYS,
            BuiltinImpact.COOLING_DEGREE_DAYS_CROP_YIELD,
        )
        else "CAN",
        "region": FloodRegion.EUROPE,
        "asset_type": AssetType.RESIDENTIAL,
        "historic_mean": 20.0,
        "wildland_urban_interface": True,
    }
    values = {field.name: data[field.name] for field in requirements.context_fields}
    context = requirements.context_type(**values)
    response = reference.resolve(context)
    assert isinstance(response, Impact)
    assert math.isfinite(cast(float, response.evaluate(3.0)))
    assert Impact.from_json(response.to_json()).evaluate(3.0) == response.evaluate(3.0)


def test_fire_weather_requires_the_interface_and_is_zero_outside_it() -> None:
    reference = Impact.builtin(BuiltinImpact.FIRE_WEATHER_INDEX)
    assert reference.requirements.context_type is WildfireContext
    with pytest.raises(ImpactError) as missing:
        reference.resolve()
    assert missing.value.code == "missing_context"
    with pytest.raises(ImpactError) as empty:
        reference.resolve(EmptyContext())
    assert empty.value.code == "invalid_context"
    inside = reference.resolve(WildfireContext(wildland_urban_interface=True))
    outside = reference.resolve(WildfireContext(wildland_urban_interface=False))
    values = [0.0, 20.0, 60.0]
    assert np.all(np.asarray(inside.evaluate(values))[1:] > 0)
    assert np.asarray(outside.evaluate(values)).tolist() == [0.0, 0.0, 0.0]
    law = Distribution.parametric(DistributionFamily.GUMBEL_RIGHT, location=30, scale=5)
    assert outside.expectation(law) == 0.0
    assert outside.pushforward(law).quantile([0.5, 0.99]).tolist() == [0.0, 0.0]
    assert outside.input == inside.input and outside.output == inside.output
    loaded = Impact.from_json(outside.to_json())
    assert loaded.to_json() == outside.to_json()
    with pytest.raises(TypeError, match="bool"):
        WildfireContext(wildland_urban_interface=1)  # type: ignore[arg-type]


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
    document["input"] = {
        "kind": "measurement",
        "quantity": "exposure",
        "unit": "1",
        field: ["bad"],
    }
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
@pytest.mark.parametrize(
    "family", [DistributionFamily.GUMBEL_RIGHT, DistributionFamily.GUMBEL_LEFT]
)
def test_affine_clamped_statistics_keep_sub_epsilon_endpoints(
    slope: float, family: DistributionFamily
) -> None:
    base = Distribution.parametric(family, location=0, scale=1)
    side = slope if family is DistributionFamily.GUMBEL_RIGHT else -slope
    response = linear(
        slope, maximum=40 if side > 0 else None, minimum=-40 if side < 0 else None
    )
    image = response.pushforward(base)
    assert 0 < float(image.mass(side * 40)) < math.ulp(1.0)
    metrics = [Statistic.MEAN, Statistic.VARIANCE, Statistic.CVAR]
    tail = Tail.UPPER if side > 0 else Tail.LOWER
    expected = linear(slope).pushforward(base).statistics(metrics=metrics, tail=tail)
    statistics = image.statistics(metrics=metrics, tail=tail)
    for metric in metrics:
        assert statistics.values[metric].status is StatisticStatus.FINITE
        assert statistics.values[metric].value == pytest.approx(
            expected.values[metric].value, abs=1e-9
        )


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


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (Distribution.histogram([-1e200, 1e200], [1]), 1e200 / 3),
        (
            Distribution.parametric(
                DistributionFamily.GUMBEL_RIGHT, location=0, scale=1e200
            ),
            math.pi**2 * 1e200 / 6,
        ),
    ],
)
@pytest.mark.parametrize("slope", [-1e-100, 1e-100])
def test_affine_downscaling_keeps_finite_variance(
    source: Distribution, expected: float, slope: float
) -> None:
    assert (
        source.statistics(metrics=[Statistic.VARIANCE])
        .values[Statistic.VARIANCE]
        .status
        is StatisticStatus.POSITIVE_INFINITY
    )
    image = linear(slope).pushforward(source)
    summary = image.statistics(
        metrics=[Statistic.VARIANCE, Statistic.STANDARD_DEVIATION]
    )
    assert summary.values[Statistic.VARIANCE].value == pytest.approx(expected)
    assert summary.values[Statistic.STANDARD_DEVIATION].value == pytest.approx(
        math.sqrt(expected)
    )


@pytest.mark.parametrize(
    ("source", "unclamped_variance", "clamped_variance"),
    [
        (Distribution.empirical([-1e200, 1e200]), 1.0, 0.25),
        (Distribution.histogram([-1e200, 1e200], [1]), 1 / 3, 5 / 48),
        (
            Distribution.parametric(
                DistributionFamily.GENERALIZED_PARETO,
                location=-1e200,
                scale=2e200,
                shape=-1,
            ),
            1 / 3,
            5 / 48,
        ),
    ],
)
@pytest.mark.parametrize("slope", [-1e-100, 1e-100])
def test_clamped_affine_applies_slope_before_squaring(
    source: Distribution,
    unclamped_variance: float,
    clamped_variance: float,
    slope: float,
) -> None:
    for maximum, expected in [(2e100, unclamped_variance), (0.0, clamped_variance)]:
        image = linear(slope, maximum=maximum).pushforward(source)
        values = image.statistics(
            metrics=[Statistic.VARIANCE, Statistic.STANDARD_DEVIATION]
        ).values
        assert values[Statistic.VARIANCE].value == pytest.approx(expected * 1e200)
        assert values[Statistic.STANDARD_DEVIATION].value == pytest.approx(
            math.sqrt(expected) * 1e100
        )


def test_clamped_continuous_image_keeps_finite_standard_deviation() -> None:
    source = Distribution.histogram([-1e200, 1e200], [1])
    image = linear(1, maximum=0).pushforward(source)
    values = image.statistics(
        metrics=[Statistic.VARIANCE, Statistic.STANDARD_DEVIATION]
    ).values
    assert values[Statistic.VARIANCE].status is StatisticStatus.POSITIVE_INFINITY
    assert values[Statistic.STANDARD_DEVIATION].value == pytest.approx(
        math.sqrt(5 / 48) * 1e200
    )


@pytest.mark.parametrize("orientation", [-1.0, 1.0])
@pytest.mark.parametrize("clamped", [False, True])
def test_affine_fitted_law_is_rescaled_before_querying_overflowing_ranks(
    orientation: float, clamped: bool
) -> None:
    source = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=1e308, scale=1e308
    )
    normalized = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=1, scale=1
    )
    bounds = (
        ({"maximum": 10.0} if orientation > 0 else {"minimum": -10.0})
        if clamped
        else {}
    )
    image = linear(orientation * 1e-308, **bounds).pushforward(source)
    exact = linear(orientation, **bounds).pushforward(normalized)
    ranks = [0.05, 0.5, 0.95]
    assert image.quantile(ranks) == pytest.approx(exact.quantile(ranks))
    restored = Distribution.from_json(image.to_json())
    assert restored.quantile(ranks) == pytest.approx(exact.quantile(ranks))
    metrics = [Statistic.MEAN, Statistic.STANDARD_DEVIATION, Statistic.CVAR]
    for tail in (Tail.LOWER, Tail.UPPER):
        actual = image.statistics(metrics=metrics, tail=tail).values
        expected = exact.statistics(metrics=metrics, tail=tail).values
        for metric in metrics:
            assert actual[metric].value == pytest.approx(
                expected[metric].value, rel=1e-8
            )


def test_affine_gumbel_mean_can_be_finite_when_its_scale_overflows() -> None:
    source = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=0, scale=1e308
    )
    response = linear(2)
    expected = (2 * 0.577_215_664_901_532_9) * 1e308
    assert response.expectation(source) == pytest.approx(expected, rel=1e-14)
    value = (
        response.pushforward(source)
        .statistics(metrics=[Statistic.MEAN])
        .values[Statistic.MEAN]
    )
    assert value.value == pytest.approx(expected, rel=1e-14)


def test_downscaled_affine_expectation_matches_the_pushforward_mean() -> None:
    source = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=1.5e308, scale=1.5e308
    )
    response = linear(1e-308)
    expected = 1.5 * (1 + 0.577_215_664_901_532_9)
    assert response.expectation(source) == pytest.approx(expected, rel=1e-14)
    value = (
        response.pushforward(source)
        .statistics(metrics=[Statistic.MEAN])
        .values[Statistic.MEAN]
    )
    assert value.value == pytest.approx(expected, rel=1e-14)

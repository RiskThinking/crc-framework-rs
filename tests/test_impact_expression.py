"""Public expression API against the shared numerical fixtures."""

from __future__ import annotations

import json
import pickle

import jsonschema
import numpy as np
import pytest

from crc_framework import (
    DiscreteDistribution,
    EmpiricalDistribution,
    FittedDistribution,
    GaussKronrodOptions,
    HistogramExposure,
    ImpactError,
    ImpactExpression,
    LinearImpact,
    PiecewiseLinearImpact,
    PointMassDistribution,
    SigmoidImpact,
    TabulatedDistribution,
    TransformContext,
    impacts,
)
from crc_framework.contracts import load_impact_contract_file, load_impact_schema
from tools.impact_contracts import reference

CALIBRATION = {
    "exposure_variable": "value",
    "exposure_unit": "1",
    "impact_metric": "response",
    "impact_unit": "1",
}


def _close(actual: float, expected: float, tolerance: dict[str, float]) -> bool:
    return abs(actual - expected) <= (
        tolerance["absolute"] + tolerance["relative"] * abs(expected)
    )


def _source(spec: dict):
    kind = spec["kind"]
    if kind == "histogram":
        return HistogramExposure(spec["bin_edges"], spec["weights"])
    if kind == "discrete":
        return DiscreteDistribution(spec["support"], spec["probabilities"])
    parameters = spec["parameters"]
    return _curve(parameters)


def _curve(parameters: dict):
    kind = parameters["curve_kind"]
    if kind == "point_mass":
        return PointMassDistribution(parameters["curve_location"])
    if kind == "fitted":
        return FittedDistribution(
            parameters["curve_type"],
            location=parameters["curve_location"],
            scale=parameters["curve_scale"],
            shape=parameters.get("curve_shape"),
        )
    if kind == "tabulated":
        return TabulatedDistribution(
            parameters["curve_probabilities"], parameters["curve_values"]
        )
    raise AssertionError(kind)


def _method(spec):
    if spec == "auto" or spec == "exact":
        return spec
    return GaussKronrodOptions(
        absolute_tolerance=spec["absolute_tolerance"],
        relative_tolerance=spec["relative_tolerance"],
        max_evaluations=spec["max_evaluations"],
    )


def test_point_fixtures_match_the_reference_and_public_api() -> None:
    document = load_impact_contract_file("fixtures/point-evaluations.json")
    for fixture in document["fixtures"]:
        expression = ImpactExpression.from_config(fixture["definition"])
        assert expression.definition_hash == fixture["definition_hash"]
        assert expression.to_json() == reference.canonical_bytes(
            reference.normalize_definition(fixture["definition"])
        ).decode()
        for case in fixture["cases"]:
            if "x_special" in case:
                exposure = {"NaN": np.nan, "Infinity": np.inf, "-Infinity": -np.inf}[
                    case["x_special"]
                ]
            else:
                exposure = case["x"]
            if "error" in case:
                with pytest.raises(ImpactError) as raised:
                    expression.evaluate(exposure)
                assert raised.value.code == case["error"]
                continue
            actual = expression.evaluate(exposure)
            python = reference.evaluate(fixture["definition"], exposure)
            assert _close(actual, case["expected"], fixture["tolerance"])
            assert _close(python, actual, fixture["tolerance"])


def test_operation_fixtures_match_the_public_api() -> None:
    operations = load_impact_contract_file("fixtures/operations.json")
    for case in operations["expected_value"]:
        expression = ImpactExpression.from_config(case["definition"])
        if "error" in case:
            with pytest.raises(ImpactError) as raised:
                expression.expected_value(_source(case["source"]), method=_method(case["method"]))
            assert raised.value.code == case["error"]
            continue
        actual = expression.expected_value(
            _source(case["source"]), method=_method(case["method"])
        )
        assert _close(actual, case["expected"], case["tolerance"])
        recorded = expression.expected_value_with_record(
            _source(case["source"]), method=_method(case["method"])
        )
        assert recorded.record.to_dict()["operation"]["settings"]["method"]["kind"] in {
            "exact",
            "gauss_kronrod",
        }
    for case in operations["transform_distribution"]:
        expression = ImpactExpression.from_config(case["definition"])
        if "error" in case:
            with pytest.raises(ImpactError) as raised:
                expression.transform_distribution(_source(case["source"]))
            assert raised.value.code == case["error"]
            continue
        distribution = expression.transform_distribution(_source(case["source"]))
        quantiles = distribution.quantiles(case["probabilities"])
        for actual, expected in zip(quantiles, case["quantiles"]):
            assert _close(float(actual), expected, case["tolerance"])
        if "atoms" in case:
            atoms = distribution.atoms()
            assert len(atoms) == len(case["atoms"])
            for atom, expected in zip(atoms, case["atoms"]):
                assert _close(atom.value, expected["value"], case["tolerance"])
                assert _close(atom.probability, expected["probability"], case["tolerance"])


def test_shapes_records_pickle_and_immutability() -> None:
    expression = ImpactExpression.linear(
        2.0, intercept=-1.0, calibration=CALIBRATION, clamp={"minimum": 0.0}
    )
    assert expression.evaluate(1.5) == 2.0
    values = np.array([[0.0, 1.0], [4.0, -3.0]])
    evaluated = expression.evaluate(values)
    assert isinstance(evaluated, np.ndarray)
    assert evaluated.shape == values.shape
    assert not evaluated.flags.writeable
    np.testing.assert_allclose(evaluated, [[0.0, 1.0], [7.0, 0.0]])
    empty = expression.evaluate(np.zeros((0, 2)))
    assert empty.shape == (0, 2)
    assert not empty.flags.writeable

    config = expression.to_config()
    with pytest.raises(TypeError):
        config["model"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        config["model"]["slope"] = 0.0  # type: ignore[index]

    restored = pickle.loads(pickle.dumps(expression))
    assert restored.to_json() == expression.to_json()
    assert restored.evaluate([0.5, 2.0]).tolist() == [0.0, 3.0]

    recorded = expression.evaluate_with_record(values)
    document = recorded.record.to_dict()
    jsonschema.validate(json.loads(recorded.record.to_json()), load_impact_schema("impact_evaluation_record"))
    assert document["replay"]["status"] == "missing_source_snapshot"
    assert document["source"]["shape"] == (2, 2)
    assert json.loads(recorded.record.to_json())["source"]["shape"] == [2, 2]
    assert document["diagnostics"]["exactness"] == "direct"
    assert not recorded.value.flags.writeable

    with pytest.raises(TypeError, match="context"):
        expression.evaluate(1.0, context=TransformContext())  # type: ignore[call-arg]


def test_duplicate_keys_and_domain_errors_expose_codes() -> None:
    with pytest.raises(ImpactError) as raised:
        ImpactExpression.from_json('{"schema_version":"1.0","schema_version":"1.0"}')
    assert raised.value.code == "duplicate_key"
    expression = ImpactExpression.tabulated(
        [(0.0, 0.0), (1.0, 1.0)],
        calibration=CALIBRATION,
        outside_domain="error",
    )
    with pytest.raises(ImpactError) as raised:
        expression.evaluate(2.0)
    assert raised.value.code == "out_of_domain"


def test_decreasing_call_reflects_each_probability() -> None:
    expression = ImpactExpression.linear(-1.0, intercept=10.0, calibration=CALIBRATION, clamp=None)
    # from_config omits clamp when None, which is an absent clamp. linear() passes clamp=None
    # so no clamp key. slope is negative and minimum is absent. Good.
    exposure = TabulatedDistribution([0.0, 1.0], [0.0, 10.0])
    quantiles = expression(exposure, probabilities=[0.1, 0.5, 0.8]).quantiles([0.1, 0.5, 0.8])
    np.testing.assert_allclose(quantiles, [1.0, 5.0, 8.0])
    reflected_wrongly = expression.evaluate(exposure.quantiles([0.1, 0.5, 0.8]))[::-1]
    assert not np.allclose(quantiles, reflected_wrongly)


def test_legacy_point_math_uses_the_native_kernel() -> None:
    linear = LinearImpact(slope=0.5, intercept=0.1, maximum=1.0)
    np.testing.assert_allclose(
        linear.evaluate(np.array([[0.0, 0.5], [1.0, 3.0]])),
        [[0.1, 0.35], [0.6, 1.0]],
    )
    sigmoid = SigmoidImpact(midpoint=1.0, steepness=2.0, minimum=0.1, maximum=0.9, zero_below=0.5)
    exposure = np.array([0.0, 0.5, 1.0, 2.0])
    unit = 1.0 / (1.0 + np.exp(-2.0 * (exposure - 1.0)))
    expected = np.where(exposure < 0.5, 0.0, 0.1 + unit * 0.8)
    np.testing.assert_allclose(sigmoid.evaluate(exposure), expected, rtol=1e-12, atol=1e-12)
    piecewise = PiecewiseLinearImpact([0.0, 1.0, 2.0], [0.0, 0.25, 1.0])
    np.testing.assert_allclose(
        piecewise.evaluate(np.array([-1.0, 0.5, 3.0])),
        np.interp([-1.0, 0.5, 3.0], [0.0, 1.0, 2.0], [0.0, 0.25, 1.0]),
    )


def test_registry_expression_matches_the_legacy_point_path() -> None:
    context = TransformContext(continent="Europe", building_type="Commercial buildings")
    legacy = impacts.for_factor("inundation", context=context)
    expression = ImpactExpression.for_factor("inundation", context=context)
    exposure = np.array([0.0, 0.5, 1.0, 2.0])
    np.testing.assert_allclose(
        expression.evaluate(exposure), legacy.evaluate(exposure, context=context)
    )
    restored = pickle.loads(pickle.dumps(expression))
    assert restored.definition_hash == expression.definition_hash
    np.testing.assert_allclose(restored.evaluate(exposure), expression.evaluate(exposure))


def test_histogram_expectation_records_normalization() -> None:
    expression = ImpactExpression.binned(
        [0.0, 0.5, 2.0],
        [0.0, 1.0],
        calibration=CALIBRATION,
    )
    histogram = HistogramExposure([0.0, 2.0], [1.0])
    recorded = expression.expected_value_with_record(histogram)
    source = recorded.record.to_dict()["source"]
    assert source["kind"] == "histogram"
    assert source["original_mass_sum"] == pytest.approx(histogram.original_mass_sum)
    assert recorded.value == pytest.approx(0.75)
    jsonschema.validate(
        json.loads(recorded.record.to_json()),
        load_impact_schema("impact_evaluation_record"),
    )


def test_partial_curve_is_not_silently_completed() -> None:
    expression = ImpactExpression.linear(1.0, calibration=CALIBRATION)
    partial = TabulatedDistribution([0.1, 0.9], [1.0, 2.0])
    with pytest.raises(ImpactError) as raised:
        expression.expected_value(partial)
    assert raised.value.code == "missing_probability_support"
    with pytest.raises(ImpactError) as raised:
        expression.transform_distribution(partial)
    assert raised.value.code == "missing_probability_support"
    table = expression(partial, probabilities=[0.1, 0.5, 0.9])
    np.testing.assert_allclose(table.quantiles([0.1, 0.5, 0.9]), partial.quantiles([0.1, 0.5, 0.9]))


def test_empirical_and_point_mass_expectations_are_exact() -> None:
    expression = ImpactExpression.linear(2.0, calibration=CALIBRATION)
    # Empirical quantiles are piecewise linear through the order statistics,
    # so the expectation is the integral of that quantile, not the sample mean.
    samples = EmpiricalDistribution([1.0, 1.0, 3.0, 5.0])
    assert expression.expected_value(PointMassDistribution(4.0)) == 8.0
    assert expression.expected_value(samples) == pytest.approx(14.0 / 3.0)
    discrete = DiscreteDistribution([0.0, 1.0, 3.0], [0.2, 0.5, 0.3])
    assert expression.expected_value(discrete) == pytest.approx(2.8)

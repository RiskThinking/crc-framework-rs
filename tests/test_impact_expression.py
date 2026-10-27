"""Public expression API against the shared numerical fixtures."""

from __future__ import annotations

import hashlib
import importlib
import json
import pickle
from collections.abc import Mapping
from typing import Any

import jsonschema
import numpy as np
import pytest

import crc_framework
import crc_framework.transforms as transform_api
from crc_framework import _core
from crc_framework import (
    DiscreteDistribution,
    EmpiricalDistribution,
    FittedDistribution,
    GaussKronrodOptions,
    HistogramExposure,
    HurdleDistribution,
    ImpactError,
    ImpactExpression,
    LinearImpact,
    PiecewiseLinearImpact,
    PointMassDistribution,
    RegistryImpactReference,
    SigmoidImpact,
    TabulatedDistribution,
    TransformContext,
    impact_registry_manifest,
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

    # The record hashes the evaluated exposures, not the clamped responses.
    # [0, 1, 4, -3] -> [0, 1, 7, 0]: inputs and outputs differ here, so a hash
    # of the outputs would not match a hash of the inputs.
    expected_digest = _core.impact_array_digest([2, 2], values.reshape(-1).tolist())
    assert document["source"]["content_digest"]["sha256"] == expected_digest
    wrong_digest = _core.impact_array_digest([2, 2], evaluated.reshape(-1).tolist())
    assert document["source"]["content_digest"]["sha256"] != wrong_digest


def test_record_keeps_zero_components_that_actually_ran() -> None:
    # A constant-zero sigmoid response has no exact-expectation path (sigmoid
    # is not piecewise-linear), so it always runs Gauss-Kronrod; the integral
    # is exactly zero, but it still ran and must stay reported.
    expression = ImpactExpression.sigmoid(
        midpoint=0.0, steepness=1.0, offset=0.0, amplitude=0.0, calibration=CALIBRATION
    )
    histogram = HistogramExposure([0.0, 1.0], [1.0])
    recorded = expression.expected_value_with_record(
        histogram, method=GaussKronrodOptions()
    )
    document = recorded.record.to_dict()
    jsonschema.validate(
        json.loads(recorded.record.to_json()),
        load_impact_schema("impact_evaluation_record"),
    )
    assert recorded.value == pytest.approx(0.0)
    kinds = {component["kind"] for component in document["diagnostics"]["components"]}
    assert kinds == {"numerical"}
    assert document["diagnostics"]["exactness"] == "numerical"

    # A hurdle atom sitting exactly where the response is zero makes the
    # peeled exact component evaluate to zero; it still ran (the continuous
    # tail still needs quadrature) and must stay reported too.
    base = FittedDistribution("gumbel_r", location=0.5, scale=1.2)
    hurdle = HurdleDistribution(base, atom_probability=0.4, atom_location=0.0)
    unclamped = ImpactExpression.linear(1.0, calibration=CALIBRATION)
    recorded = unclamped.expected_value_with_record(hurdle, method=GaussKronrodOptions())
    document = recorded.record.to_dict()
    jsonschema.validate(
        json.loads(recorded.record.to_json()),
        load_impact_schema("impact_evaluation_record"),
    )
    components = {
        component["kind"]: component["value"]
        for component in document["diagnostics"]["components"]
    }
    assert components.keys() == {"exact", "numerical"}
    assert components["exact"] == pytest.approx(0.0)
    assert document["diagnostics"]["exactness"] == "mixed"


def test_untracked_sources_never_claim_complete_replay() -> None:
    # Discrete and empirical sources have no snapshot member in their record
    # schema, so a passed-in snapshot cannot land there; replay must say so.
    expression = ImpactExpression.linear(1.0, calibration=CALIBRATION)
    snapshot = {
        "uri": "s3://bucket/key",
        "revision": None,
        "content_sha256": "0" * 64,
        "retention": "retained",
    }
    for source in (
        DiscreteDistribution([0.0, 1.0], [0.5, 0.5]),
        EmpiricalDistribution([0.0, 1.0, 2.0]),
    ):
        recorded = expression.expected_value_with_record(source, snapshot=snapshot)
        document = recorded.record.to_dict()
        assert document["source"]["kind"] == "untracked"
        assert "snapshot" not in document["source"]
        assert document["replay"]["status"] == "missing_source_snapshot"
        assert tuple(document["replay"]["missing_requirements"]) == ("source_snapshot",)


def test_call_raises_impact_error_with_code_and_path() -> None:
    expression = ImpactExpression.tabulated(
        [(0.0, 0.0), (5.0, 1.0), (10.0, 0.0)],
        calibration=CALIBRATION,
    )
    exposure = TabulatedDistribution([0.0, 1.0], [0.0, 10.0])
    with pytest.raises(ImpactError) as raised:
        expression(exposure)
    assert raised.value.code == "monotonicity_required"
    assert raised.value.path == ""

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


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def test_to_config_round_trips_through_from_config() -> None:
    expression = ImpactExpression.linear(
        2.0,
        intercept=-1.0,
        calibration=CALIBRATION,
        clamp={"minimum": 0.0, "maximum": None},
        provenance={
            "origin": {"kind": "derived", "description": "scaled"},
            "sources": [{"id": "paper", "kind": "publication", "uri": "doi:10.1/example"}],
            "derived_from": ["sha256:" + "ab" * 32],
        },
    )
    restored = ImpactExpression.from_config(expression.to_config())
    assert restored.to_json() == expression.to_json()
    assert restored.definition_hash == expression.definition_hash

    tabulated = ImpactExpression.tabulated(
        [(0.0, 0.0), (1.0, 2.0)],
        calibration=CALIBRATION,
    )
    assert ImpactExpression.from_config(tabulated.to_config()).to_json() == tabulated.to_json()
    mapped = ImpactExpression.tabulated(
        [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
        calibration=CALIBRATION,
    )
    assert mapped.evaluate(0.5) == 0.5


def test_unknown_point_and_context_keys_are_rejected() -> None:
    with pytest.raises(ImpactError) as raised:
        ImpactExpression.tabulated(
            [{"x": 0.0, "y": 0.0, "z": 1.0, "label": "a"}, {"x": 1.0, "y": 1.0}],
            calibration=CALIBRATION,
        )
    assert raised.value.code == "unknown_field"
    assert raised.value.path == "/model/points/0/label"

    with pytest.raises(ImpactError) as raised:
        ImpactExpression.for_factor("fwi", context={"region": "north", "countyr": "CAN"})
    assert raised.value.code == "unknown_field"
    assert raised.value.path == "/context/countyr"

    with pytest.raises(ImpactError) as raised:
        ImpactExpression.for_factor("fwi", context_defaults={"countyr": "CAN"})
    assert raised.value.code == "unknown_field"
    assert raised.value.path == "/context_defaults/countyr"

    reference = RegistryImpactReference.for_factor("fwi")
    with pytest.raises(ImpactError) as raised:
        reference.resolve({"countyr": "CAN"})
    assert raised.value.code == "unknown_field"
    assert raised.value.path == "/context/countyr"


def test_registry_reference_keeps_its_identity() -> None:
    defaults = {"continent": "Europe", "building_type": "Commercial buildings"}
    reference = RegistryImpactReference.for_factor("inundation", context_defaults=defaults)
    assert reference.reference_hash.startswith("sha256:")
    document = json.loads(reference.to_json())
    assert document["context_defaults"]["continent"] == "Europe"
    assert document["context_defaults"]["country"] is None
    assert document["factor"] == "inundation"

    resolved = reference.resolve()
    direct = ImpactExpression.for_factor("inundation", context=defaults)
    via_defaults = ImpactExpression.for_factor("inundation", context_defaults=defaults)
    assert resolved.definition_hash == direct.definition_hash == via_defaults.definition_hash

    partial = reference.resolve({"continent": "Asia"})
    kept = ImpactExpression.for_factor(
        "inundation",
        context={"continent": "Asia"},
        context_defaults=defaults,
    )
    assert partial.definition_hash == kept.definition_hash
    assert partial.definition_hash != resolved.definition_hash

    cell = 0xABC
    pinned = RegistryImpactReference.for_factor(
        "frost_days",
        context_defaults=TransformContext(cell=cell),
    )
    assert json.loads(pinned.to_json())["context_defaults"]["cell"] == f"{cell:015x}"

    restored = pickle.loads(pickle.dumps(reference))
    assert restored.to_json() == reference.to_json()
    assert restored.reference_hash == reference.reference_hash
    assert restored.resolve().definition_hash == resolved.definition_hash

    # The package attribute crc_framework.impacts is the legacy registry singleton,
    # so the impacts module is loaded by its full name.
    impact_module = importlib.import_module("crc_framework.impacts")
    assert impact_module.RegistryImpactReference is RegistryImpactReference
    assert transform_api.RegistryImpactReference is RegistryImpactReference
    assert crc_framework.RegistryImpactReference is RegistryImpactReference


def test_registry_manifest_digest_matches_identity() -> None:
    raw = _core.impact_registry_manifest_json()
    registry_id, revision, resolver, digest = _core.impact_registry_identity()
    assert hashlib.sha256(raw.encode()).hexdigest() == digest

    manifest = impact_registry_manifest()
    assert _plain(manifest) == json.loads(raw)
    assert manifest["registry_id"] == registry_id
    assert manifest["registry_revision"] == revision
    assert manifest["resolver_semantics_version"] == resolver
    factors = {item["factor"]: item for item in manifest["factors"]}
    assert factors["rx1day"]["calibration"]["exposure_unit"] == "mm/day"
    agricultural = factors["cooling_degree_days"]["branch_calibrations"]["agricultural"]
    assert agricultural["impact_metric"] == "crop_yield_change"
    with pytest.raises(TypeError):
        manifest["registry_id"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        factors["rx1day"]["calibration"]["exposure_unit"] = "cm"  # type: ignore[index]

    impact_module = importlib.import_module("crc_framework.impacts")
    assert impact_module.impact_registry_manifest is impact_registry_manifest
    assert transform_api.impact_registry_manifest is impact_registry_manifest
    assert crc_framework.impact_registry_manifest is impact_registry_manifest


def test_hazard_exposure_catalog_uses_factor_calibrations() -> None:
    catalog = load_impact_contract_file("catalogs/hazard-exposures.json")
    factors = {item["factor"]: item for item in impact_registry_manifest()["factors"]}
    seen: set[str] = set()
    expected = {
        "rx1day": "rx1day",
        "fd": "frost_days",
        "cdd": "maximum_consecutive_dry_days",
    }
    found = {}
    for entry in catalog["entries"]:
        factor = factors[entry["factor"]]
        calibration = factor["calibration"]
        for branch in factor.get("branch_calibrations", {}).values():
            assert branch["exposure_variable"] == calibration["exposure_variable"]
            assert branch["exposure_unit"] == calibration["exposure_unit"]
        for name in (entry["id"], *entry["aliases"]):
            assert name not in seen
            seen.add(name)
        found[entry["id"]] = entry["factor"]
        assert calibration["exposure_variable"]
        assert calibration["exposure_unit"]
    assert found == expected
    assert factors["rx1day"]["calibration"] == {
        "exposure_variable": "rx1day",
        "exposure_unit": "mm/day",
        "impact_metric": "damage_ratio",
        "impact_unit": "1",
    }


def test_empirical_and_point_mass_expectations_are_exact() -> None:
    expression = ImpactExpression.linear(2.0, calibration=CALIBRATION)
    # Empirical quantiles are piecewise linear through the order statistics,
    # so the expectation is the integral of that quantile, not the sample mean.
    samples = EmpiricalDistribution([1.0, 1.0, 3.0, 5.0])
    assert expression.expected_value(PointMassDistribution(4.0)) == 8.0
    assert expression.expected_value(samples) == pytest.approx(14.0 / 3.0)
    discrete = DiscreteDistribution([0.0, 1.0, 3.0], [0.2, 0.5, 0.3])
    assert expression.expected_value(discrete) == pytest.approx(2.8)

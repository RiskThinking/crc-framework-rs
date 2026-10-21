"""Conformance gates for the published impact contract artifact."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import pytest

from crc_framework.contracts import (
    IMPACT_CONTRACT_VERSION,
    impact_contract_manifest,
    impact_contract_names,
    load_impact_contract_file,
    load_impact_fixtures,
    load_impact_schema,
    read_impact_contract_file,
    verify_impact_contracts,
)

TOOLS = Path(__file__).resolve().parent.parent / "tools" / "impact_contracts"
sys.path.insert(0, str(TOOLS))

import reference  # noqa: E402

ANNOTATIONS = ("description", "x-crc-canonical", "x-crc-native")


def test_manifest_digests_match_files() -> None:
    verify_impact_contracts()
    manifest = impact_contract_manifest()
    assert manifest["artifact_version"] == IMPACT_CONTRACT_VERSION
    assert set(impact_contract_names()) == {
        "impact_definition",
        "registry_impact_reference",
        "impact_evaluation_record",
        "spatial_combination_policy",
        "spatial_impact_evaluation_record",
    }


def test_artifact_is_generated_from_the_reference() -> None:
    pytest.importorskip("jsonschema")
    import generate

    built = generate.build()
    listed = {entry["path"] for entry in impact_contract_manifest()["files"]} | {"manifest.json"}
    assert set(built) == listed
    for path, content in built.items():
        assert read_impact_contract_file(path) == content, f"{path} is stale; run tools/impact_contracts/generate.py"


@pytest.mark.parametrize("contract", ["impact_definition", "registry_impact_reference", "impact_evaluation_record", "spatial_combination_policy", "spatial_impact_evaluation_record"])
def test_schema_agrees_with_fixtures(contract: str) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = load_impact_schema(contract)
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    fixtures = load_impact_fixtures(contract)
    for case in fixtures["accepted"]:
        assert validator.is_valid(json.loads(case["input_json"])), case["name"]
        assert validator.is_valid(json.loads(case["canonical_json"])), case["name"]
    for case in fixtures["rejected"]:
        if case["schema_valid"] is not None:
            assert validator.is_valid(json.loads(case["input_json"])) is case["schema_valid"], case["name"]


@pytest.mark.parametrize("contract", ["impact_definition", "registry_impact_reference", "impact_evaluation_record", "spatial_combination_policy", "spatial_impact_evaluation_record"])
def test_reference_reproduces_identities_and_errors(contract: str) -> None:
    normalize = reference.NORMALIZERS[contract]
    fixtures = load_impact_fixtures(contract)
    for case in fixtures["accepted"]:
        normalized = normalize(reference.parse_json(case["input_json"]))
        assert reference.canonical_bytes(normalized).decode("utf-8") == case["canonical_json"]
        assert reference.content_hash(normalized) == case["content_hash"]
        reparsed = normalize(reference.parse_json(case["canonical_json"]))
        assert reference.canonical_bytes(reparsed).decode("utf-8") == case["canonical_json"], "canonical form is a fixed point"
    for case in fixtures["rejected"]:
        with pytest.raises(reference.ContractError) as raised:
            normalize(reference.parse_json(case["input_json"]))
        assert (raised.value.code, raised.value.path) == (case["error"]["code"], case["error"]["path"])


def test_canonical_parser_rejection_vectors_are_enforced() -> None:
    vectors = load_impact_contract_file("fixtures/canonical-json.json")["rejected"]
    for case in vectors:
        with pytest.raises(reference.ContractError) as raised:
            reference.parse_json(case["input"])
        assert raised.value.code == case["error"], case["name"]


def test_semantics_inventory_is_complete_and_references_executable_fixtures() -> None:
    inventory = load_impact_contract_file("catalogs/semantics-constants.json")
    assert inventory["semantics_version"] == "crc-impact/1"
    constants = inventory["constants"]
    assert len({item["id"] for item in constants}) == len(constants)
    required = {
        "spei-ln-199", "spei-drought-multiplier", "economic-normalizer",
        "crop-reference-year", "sigmoid-threshold-strict", "cooling-threshold-inclusive",
        "binned-right-edge", "histogram-mass-tolerance", "h3-wire-length",
        "override-catalog",
    }
    by_id = {item["id"]: item for item in constants}
    assert required <= set(by_id)
    for item in constants:
        assert item["owner"], item["id"]
        assert item["fixtures"], item["id"]
    assert by_id["h3-wire-length"]["value"] == 15


def test_native_capability_metadata_is_explicit() -> None:
    capabilities = load_impact_contract_file("catalogs/native-capabilities.json")
    assert capabilities == {
        "supported": ["canonical_json"],
        "required_fixture_families": {"canonical_json": ["numbers", "strings"]},
        "native_parser_rejections": ["leading-bom", "nan", "infinity", "negative-infinity", "binary64-overflow", "unpaired-surrogate"],
        "future_families": ["strict_duplicate_key_parser", "point_evaluations", "monotonicity", "operations", "registry"],
    }


def test_error_catalog_documents_every_code() -> None:
    codes = load_impact_contract_file("catalogs/error-codes.json")["codes"]
    used: Set[str] = set()
    for contract in impact_contract_names():
        used |= {case["error"]["code"] for case in load_impact_fixtures(contract)["rejected"]}
    for name in ("point-evaluations", "operations"):
        used |= set(_error_codes(load_impact_contract_file(f"fixtures/{name}.json")))
    assert used <= set(codes)


def _error_codes(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        if isinstance(value.get("error"), str):
            yield value["error"]
        for item in value.values():
            yield from _error_codes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _error_codes(item)


# Documentation and coverage gates -----------------------------------------


Label = Tuple[str, str]


def _resolve(schema: Dict[str, Any], node: Dict[str, Any], owner: str) -> Tuple[Dict[str, Any], str]:
    while "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        node, owner = schema["$defs"][name], name
    return node, owner


def _branches(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(node.get("oneOf", [])) + list(node.get("anyOf", []))


def _schema_properties(schema: Dict[str, Any]) -> Tuple[Dict[Label, Dict[str, Any]], Set[Label]]:
    """All object properties reachable from the root, and the optional ones."""
    properties: Dict[Label, Dict[str, Any]] = {}
    optional: Set[Label] = set()
    seen: Set[int] = set()

    def visit(node: Dict[str, Any], owner: str) -> None:
        node, owner = _resolve(schema, node, owner)
        if id(node) in seen:
            return
        seen.add(id(node))
        for branch in _branches(node):
            visit(branch, owner)
        if "items" in node:
            visit(node["items"], owner)
        if isinstance(node.get("additionalProperties"), dict):
            visit(node["additionalProperties"], owner)
        for name, child in node.get("properties", {}).items():
            label = (owner, name)
            properties[label] = child
            if name not in node.get("required", []):
                optional.add(label)
            visit(child, owner)

    visit(schema, "root")
    return properties, optional


def _matches(schema: Dict[str, Any], branch: Dict[str, Any], value: Any) -> bool:
    node, _ = _resolve(schema, branch, "")
    if value is None:
        return node.get("type") == "null"
    tag = node.get("properties", {}).get("kind", {}).get("const")
    if tag is not None:
        return isinstance(value, dict) and value.get("kind") == tag
    if node.get("type") == "object" or "properties" in node:
        return isinstance(value, dict)
    if node.get("type") == "array":
        return isinstance(value, list)
    return not isinstance(value, (dict, list))


def _walk(schema: Dict[str, Any], node: Dict[str, Any], owner: str, value: Any, present: Set[Label], absent: Set[Label]) -> None:
    node, owner = _resolve(schema, node, owner)
    branches = _branches(node)
    if branches:
        for branch in branches:
            if _matches(schema, branch, value):
                _walk(schema, branch, owner, value, present, absent)
                return
        return
    if isinstance(value, list) and "items" in node:
        for item in value:
            _walk(schema, node["items"], owner, item, present, absent)
    if isinstance(value, dict) and "properties" in node:
        for name, child in node["properties"].items():
            if name in value:
                present.add((owner, name))
                _walk(schema, child, owner, value[name], present, absent)
            else:
                absent.add((owner, name))


def test_every_property_is_documented() -> None:
    for contract in impact_contract_names():
        schema = load_impact_schema(contract)
        properties, optional = _schema_properties(schema)
        for label, node in properties.items():
            missing = [key for key in ANNOTATIONS if not node.get(key)]
            assert not missing, f"{contract} {label} lacks {missing}"
            if label in optional:
                assert "default" in node or "x-crc-absent" in node, f"{contract} {label} has no default"


def test_every_property_has_a_conformance_case() -> None:
    present: Set[Label] = set()
    absent: Set[Label] = set()
    required: Set[Label] = set()
    defaulted: Set[Label] = set()
    for contract in impact_contract_names():
        schema = load_impact_schema(contract)
        properties, optional = _schema_properties(schema)
        required |= set(properties)
        defaulted |= {label for label in optional if "default" in properties[label]}
        for case in load_impact_fixtures(contract)["accepted"]:
            _walk(schema, schema, "root", json.loads(case["input_json"]), present, absent)
    root_labels = {label for label in required if label[0] == "root"}
    shared = required - root_labels
    assert not sorted(shared - present), "properties never supplied explicitly"
    assert not sorted(defaulted - absent - root_labels), "defaults never exercised by omission"
    for contract in impact_contract_names():
        schema = load_impact_schema(contract)
        properties, optional = _schema_properties(schema)
        contract_present: Set[Label] = set()
        contract_absent: Set[Label] = set()
        for case in load_impact_fixtures(contract)["accepted"]:
            _walk(schema, schema, "root", json.loads(case["input_json"]), contract_present, contract_absent)
        roots = {label for label in properties if label[0] == "root"}
        assert not sorted(roots - contract_present), f"{contract}: root properties never supplied"
        root_defaults = {label for label in roots & optional if "default" in properties[label]}
        assert not sorted(root_defaults - contract_absent), f"{contract}: root defaults never omitted"


# Numerical fixtures --------------------------------------------------------


def _close(actual: float, expected: float, tolerance: Dict[str, float]) -> bool:
    return abs(actual - expected) <= tolerance["absolute"] + tolerance["relative"] * abs(expected)


def _exposure(case: Dict[str, Any]) -> float:
    special = case.get("x_special")
    return float(case["x"]) if special is None else float(special.replace("Infinity", "inf"))


def test_point_fixtures_reproduce_with_reference_kernels() -> None:
    fixtures = load_impact_contract_file("fixtures/point-evaluations.json")["fixtures"]
    kinds = {fixture["definition"]["model"]["kind"] for fixture in fixtures}
    assert kinds == set(reference.MODEL_KINDS)
    for fixture in fixtures:
        definition = reference.normalize_definition(fixture["definition"])
        assert reference.content_hash(definition) == fixture["definition_hash"]
        for case in fixture["cases"]:
            x = _exposure(case)
            if "error" in case:
                with pytest.raises(reference.ContractError) as raised:
                    reference.evaluate(definition, x)
                assert raised.value.code == case["error"]
            else:
                assert _close(reference.evaluate(definition, x), case["expected"], fixture["tolerance"]), (fixture["name"], x)


def test_point_fixtures_pin_boundary_conventions() -> None:
    fixtures = {item["name"]: item for item in load_impact_contract_file("fixtures/point-evaluations.json")["fixtures"]}

    def value(name: str, x: float) -> Optional[float]:
        for case in fixtures[name]["cases"]:
            if case.get("x") == x:
                return case.get("expected")
        raise KeyError(x)

    assert value("binned-flood", 0.1) == 0.05
    assert value("binned-flood", 4.0) == 0.9
    assert value("tabulated-step-left", 1.0) == 0.5
    assert value("tabulated-step-right", 0.5) == 1.0
    assert value("tabulated-step-right", 0.25) == 1.0
    assert value("sigmoid-origin-threshold", 0.5) != 0.0
    assert value("sigmoid-origin-threshold", math.nextafter(0.5, -math.inf)) == 0.0
    assert value("sigmoid-origin-negative-exposure", 0.0) == 0.1
    assert value("sigmoid-constant-extreme", 1e308) == 0.5
    assert value("sigmoid-zero-amplitude-extreme", 1e308) == 0.3
    assert value("snapshot-cooling", 1.0) == 0.0
    overflow = [case for case in fixtures["linear-overflow"]["cases"] if case["x"] == 10.0]
    assert overflow == [{"x": 10.0, "error": "non_finite_result"}]


def test_monotonicity_fixtures_are_true_and_acceptable() -> None:
    for fixture in load_impact_contract_file("fixtures/monotonicity.json")["fixtures"]:
        definition = reference.normalize_definition(fixture["definition"])
        assert reference.monotonicity(definition) == fixture["reference"]
        assert fixture["reference"] in fixture["acceptable"]
        assert "unknown" in fixture["acceptable"]
        if fixture["reference"] in ("non_decreasing", "non_increasing", "constant"):
            grid = [-50.0 + 0.25 * step for step in range(401)]
            values = [reference.evaluate(definition, x) for x in grid if _in_domain(definition, x)]
            steps = [right - left for left, right in zip(values, values[1:])]
            if fixture["reference"] != "non_increasing":
                assert all(step >= -1e-15 for step in steps), fixture["name"]
            if fixture["reference"] != "non_decreasing":
                assert all(step <= 1e-15 for step in steps), fixture["name"]


def test_crop_monotonicity_does_not_lose_underflowing_coefficient_sign() -> None:
    fixtures = load_impact_contract_file("fixtures/monotonicity.json")["fixtures"]
    fixture = next(item for item in fixtures if item["name"] == "crop-underflowing-derivative-product")
    definition = reference.normalize_definition(fixture["definition"])
    assert reference.monotonicity(definition) == "non_increasing"
    assert reference.evaluate(definition, 0.0) == 0.0
    assert reference.evaluate(definition, 7e202) < -1e100


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_mapping_normalization_rejects_non_finite_parameters(value: float) -> None:
    fixture = load_impact_fixtures("impact_definition")["accepted"][0]
    definition = json.loads(fixture["canonical_json"])
    definition["model"] = {"kind": "linear", "slope": value}
    with pytest.raises(reference.ContractError) as raised:
        reference.normalize_definition(definition)
    assert (raised.value.code, raised.value.path) == ("non_finite_number", "/model/slope")


def _in_domain(definition: Dict[str, Any], x: float) -> bool:
    model = definition["model"]
    if model.get("outside_domain") != "error":
        return True
    xs = [point["x"] for point in model["points"]] if model["kind"] == "tabulated" else model["bin_edges"]
    return xs[0] <= x <= xs[-1]


def test_operation_fixtures_pin_plan_acceptance_values() -> None:
    operations = load_impact_contract_file("fixtures/operations.json")
    expected = {case["name"]: case for case in operations["expected_value"]}
    assert expected["flood-aligned-histogram"]["expected"] == 0.1275
    assert expected["uniform-bin-threshold"]["expected"] == 0.75
    assert expected["linear-clamp-inside-bin"]["expected"] == 9.0 / 16.0
    assert expected["tabulated-clamp-inside-bin"]["expected"] == 7.0 / 16.0
    transforms = {case["name"]: case for case in operations["transform_distribution"]}
    assert transforms["decreasing-uniform"]["quantiles"] == [1.0, 5.0, 8.0]
    assert {atom["value"] for atom in transforms["binary-threshold"]["atoms"]} == {0.0, 1.0}
    spatial = {case["name"]: case["results"] for case in operations["spatial_combination"]}
    assert spatial["square"]["exposure"]["values"] == [0.25, 0.25]
    assert spatial["square"]["impact"]["values"] == [0.5, 0.5]
    assert spatial["decreasing-square"]["exposure"]["values"] == [0.75, 0.75]
    assert spatial["non-monotone"]["exposure"]["values"] == [1.0, 1.0]
    assert spatial["non-monotone"]["impact"]["values"] == [0.0, 0.0]
    hashes = {case["definition_hash"] for case in operations["spatial_combination"]}
    assert len(hashes) == 3
    for results in spatial.values():
        assert results["exposure"]["policy_hash"] != results["impact"]["policy_hash"]
    for case in operations["expected_value"]:
        if "expected" in case and case["source"]["kind"] == "histogram":
            assert _histogram_expectation(case) == pytest.approx(case["expected"], abs=1e-12)


def _histogram_expectation(case: Dict[str, Any]) -> float:
    """Integrate affine/constant pieces, splitting at knots and final clamps."""
    definition = reference.normalize_definition(case["definition"])
    edges, weights = case["source"]["bin_edges"], case["source"]["weights"]
    mass = reference.histogram_mass(weights)
    model = definition["model"]
    assert model["kind"] in ("linear", "tabulated", "binned")
    breaks = sorted(set(edges) | set(model.get("bin_edges", [])) | {p["x"] for p in model.get("points", [])})
    total = 0.0
    for (left, right), weight in zip(zip(edges, edges[1:]), mass["normalized_weights"]):
        pieces = [left] + [b for b in breaks if left < b < right] + [right]
        for a, b in zip(pieces, pieces[1:]):
            cuts = [a, b]
            affine = model["kind"] == "linear" or (
                model["kind"] == "tabulated" and model["interpolation"] == "linear"
            )
            if affine and definition["clamp"] is not None:
                ya, yb = reference.raw_response(model, a), reference.raw_response(model, b)
                if ya != yb:
                    for limit in definition["clamp"].values():
                        if limit is not None and min(ya, yb) < limit < max(ya, yb):
                            cuts.append(a + (b - a) * (limit - ya) / (yb - ya))
            cuts.sort()
            for start, end in zip(cuts, cuts[1:]):
                midpoint = 0.5 * (start + end)
                total += weight * (end - start) / (right - left) * reference.evaluate(definition, midpoint)
    return total

"""Independent native-kernel scientific fixtures and generated artifact integrity."""

import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path

import pytest

from tools.impact_contracts import generate, reference

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "python" / "crc_framework" / "contracts" / "impact"


def _fixture(path):
    return json.loads((ARTIFACT / path).read_text())


def _close(actual, expected, tolerance):
    return abs(actual - expected) <= tolerance["absolute"] + tolerance[
        "relative"
    ] * abs(expected)


def _exposure(case) -> float:
    special = case.get("x_special")
    return (
        float(case["x"])
        if special is None
        else float(special.replace("Infinity", "inf"))
    )


def _in_domain(definition, x):
    model = definition["model"]
    if model.get("outside_domain") != "error":
        return True
    xs = (
        [p["x"] for p in model["points"]]
        if model["kind"] == "tabulated"
        else model["bin_edges"]
    )
    return xs[0] <= x <= xs[-1]


def test_generated_kernel_resources_are_current():
    for path, content in generate.build().items():
        assert (ARTIFACT / path).read_bytes() == content


def test_kernel_resources_match_their_embedded_digests():
    manifest = _fixture("manifest.json")
    for entry in manifest["files"]:
        content = (ARTIFACT / entry["path"]).read_bytes()
        assert len(content) == entry["bytes"]
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]


def test_point_fixtures_reproduce_with_reference_kernels() -> None:
    fixtures = _fixture("fixtures/point-evaluations.json")["fixtures"]
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
                assert _close(
                    reference.evaluate(definition, x),
                    case["expected"],
                    fixture["tolerance"],
                ), (fixture["name"], x)


def test_point_fixtures_pin_boundary_conventions() -> None:
    fixtures = {
        item["name"]: item
        for item in _fixture("fixtures/point-evaluations.json")["fixtures"]
    }

    def value(name: str, x: float) -> float | None:
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
    overflow = [
        case for case in fixtures["linear-overflow"]["cases"] if case["x"] == 10.0
    ]
    assert overflow == [{"x": 10.0, "error": "non_finite_result"}]


def test_monotonicity_fixtures_are_true_and_acceptable() -> None:
    for fixture in _fixture("fixtures/monotonicity.json")["fixtures"]:
        definition = reference.normalize_definition(fixture["definition"])
        assert reference.monotonicity(definition) == fixture["reference"]
        assert fixture["reference"] in fixture["acceptable"]
        assert "unknown" in fixture["acceptable"]
        if fixture["reference"] == "constant":
            assert set(fixture["acceptable"]) == {
                "constant",
                "non_decreasing",
                "non_increasing",
                "unknown",
            }, fixture["name"]
        if fixture["reference"] in ("non_decreasing", "non_increasing", "constant"):
            grid = [-50.0 + 0.25 * step for step in range(401)]
            values = [
                reference.evaluate(definition, x)
                for x in grid
                if _in_domain(definition, x)
            ]
            steps = [right - left for left, right in pairwise(values)]
            if fixture["reference"] != "non_increasing":
                assert all(step >= -1e-15 for step in steps), fixture["name"]
            if fixture["reference"] != "non_decreasing":
                assert all(step <= 1e-15 for step in steps), fixture["name"]


def test_crop_monotonicity_does_not_lose_underflowing_coefficient_sign() -> None:
    fixtures = _fixture("fixtures/monotonicity.json")["fixtures"]
    fixture = next(
        item
        for item in fixtures
        if item["name"] == "crop-underflowing-derivative-product"
    )
    definition = reference.normalize_definition(fixture["definition"])
    assert reference.monotonicity(definition) == "non_increasing"
    assert reference.evaluate(definition, 0.0) == 0.0
    assert reference.evaluate(definition, 7e202) < -1e100

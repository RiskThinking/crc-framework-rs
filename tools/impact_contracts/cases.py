"""Conformance case declarations for the impact contract artifact.

Each accepted case is normalized by the independent reference and its
canonical bytes and hash are recorded. Each rejected case declares exactly one
defect together with the expected error code and JSON Pointer path; the
generator fails if the reference disagrees with a declaration.
"""

from __future__ import annotations

import copy
import json
import math
import struct
from typing import Any, Callable, Dict, List, Optional

import reference as ref

Normalizer = Callable[[Any], Dict[str, Any]]


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def f32(value: float) -> float:
    """Widen a binary32 coefficient exactly as the native tables do."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def accept(normalizer: Normalizer, name: str, description: str, text: str) -> Dict[str, Any]:
    normalized = normalizer(ref.parse_json(text))
    return {
        "name": name,
        "description": description,
        "input_json": text,
        "canonical_json": ref.canonical_bytes(normalized).decode("utf-8"),
        "content_hash": ref.content_hash(normalized),
    }


def reject(
    normalizer: Normalizer, name: str, description: str, text: str, code: str, path: str
) -> Dict[str, Any]:
    try:
        normalizer(ref.parse_json(text))
    except ref.ContractError as error:
        if (error.code, error.path) != (code, path):
            raise AssertionError(
                f"{name}: reference reported {error.code} at {error.path!r}, declared {code} at {path!r}"
            ) from error
    else:
        raise AssertionError(f"{name}: reference accepted a rejected case")
    return {"name": name, "description": description, "input_json": text, "error": {"code": code, "path": path}}


# Impact definitions --------------------------------------------------------

FLOOD = {"exposure_variable": "flood_depth", "exposure_unit": "m", "impact_metric": "damage_ratio", "impact_unit": "1"}
GENERIC = {"exposure_variable": "value", "exposure_unit": "1", "impact_metric": "response", "impact_unit": "1"}
USER = {"origin": {"kind": "user_defined"}}
DIGEST_A = "a" * 64
DIGEST_B = "0123456789abcdef" * 4
HASH_A = "sha256:" + "1" * 64
HASH_B = "sha256:" + "2" * 64
MANIFEST = "3" * 64
H3_CELL = "8a2a1072b59ffff"


def definition(model: Dict[str, Any], *, calibration: Optional[Dict[str, Any]] = None, provenance: Optional[Dict[str, Any]] = None, **extra: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema_version": "1.0",
        "semantics_version": "crc-impact/1",
        "calibration": calibration or GENERIC,
        "model": model,
        "provenance": provenance or USER,
    }
    result.update(extra)
    return result


PLAN_EXAMPLE = {
    "schema_version": "1.0",
    "semantics_version": "crc-impact/1",
    "calibration": FLOOD,
    "model": {
        "kind": "tabulated",
        "interpolation": "linear",
        "outside_domain": "clamp",
        "points": [{"x": 0.0, "y": 0.0}, {"x": 0.3, "y": 0.08}, {"x": 1.0, "y": 0.45}, {"x": 2.0, "y": 1.0}],
    },
    "clamp": {"minimum": 0.0, "maximum": 1.0},
    "provenance": {"origin": {"kind": "user_defined"}, "sources": [], "derived_from": []},
}

# Independently written expected bytes for the plan example; the generator
# asserts that the reference encoder reproduces them exactly.
PLAN_EXAMPLE_CANONICAL = (
    '{"calibration":{"exposure_unit":"m","exposure_variable":"flood_depth","impact_metric":"damage_ratio",'
    '"impact_unit":"1"},"clamp":{"maximum":1.0,"minimum":0.0},"model":{"interpolation":"linear","kind":"tabulated",'
    '"outside_domain":"clamp","points":[{"x":0.0,"y":0.0},{"x":0.3,"y":0.08},{"x":1.0,"y":0.45},{"x":2.0,"y":1.0}]},'
    '"provenance":{"derived_from":[],"origin":{"kind":"user_defined"},"sources":[]},"schema_version":"1.0",'
    '"semantics_version":"crc-impact/1"}'
)

CYCLONE_ORIGIN = {
    "kind": "registry",
    "registry_id": "crc-climate-impacts",
    "registry_revision": "2026.10.0",
    "registry_manifest_sha256": MANIFEST,
    "resolver_semantics_version": "crc-impact-registry/1",
    "factor": "cyclone",
    "context": {"country": "PHL"},
    "overrides": {"maximum_damage": 0.8},
    "fallback_policy": "error",
    "lookups": [
        {"table_id": "wind.cyclone_params", "table_sha256": DIGEST_B, "requested_key": "PHL", "resolved_key": "PHL", "status": "exact"}
    ],
}

CYCLONE_SNAPSHOT = definition(
    {
        "kind": "sigmoid",
        "midpoint": f32(74.7),
        "steepness": f32(0.11457369750485913),
        "offset": 0.0,
        "amplitude": 0.8,
        "origin_adjustment": "subtract_at_zero",
        "zero_below": f32(28.5),
    },
    calibration={"exposure_variable": "wind_speed", "exposure_unit": "m/s", "impact_metric": "damage_ratio", "impact_unit": "1"},
    provenance={"origin": CYCLONE_ORIGIN},
)

SPEI_SNAPSHOT = definition(
    {"kind": "spei", "water_consumption_asset": 1250000.5, "original_water_unit_cost": 2.9809},
    calibration={"exposure_variable": "spei", "exposure_unit": "1", "impact_metric": "normalized_water_cost", "impact_unit": "crc-legacy-normalized-cost"},
)

COOLING_SNAPSHOT = definition(
    {"kind": "cooling", "cooling_cost": 52000000.0, "historic_mean": 410.5},
    calibration={"exposure_variable": "cooling_degree_days", "exposure_unit": "K.d", "impact_metric": "normalized_cooling_cost", "impact_unit": "crc-legacy-normalized-cost"},
)

CROP_SNAPSHOT = definition(
    {
        "kind": "crop",
        "weights": {"maize": 0.5, "rice": 0.3, "wheat": 0.25},
        "country": {"maize": 1.2, "rice": 0.8, "wheat": 1.1},
        "yearly": {"maize": 0.0004, "rice": 0.0002, "wheat": 0.0003},
        "index": {"maize": -0.002, "rice": -0.001, "wheat": -0.0015},
        "historic_mean": 150.0,
    },
    calibration={"exposure_variable": "cooling_degree_days", "exposure_unit": "K.d", "impact_metric": "crop_yield_change", "impact_unit": "1"},
)


def definition_cases() -> Dict[str, List[Dict[str, Any]]]:
    n = ref.normalize_definition
    accepted = []

    def ok(name: str, description: str, value: Any) -> None:
        accepted.append(accept(n, name, description, value if isinstance(value, str) else dumps(value)))

    ok("plan-example-tabulated", "Section 2.1 example with every default already materialized.", PLAN_EXAMPLE)
    ok("linear-minimal", "Required fields only: intercept, clamp and provenance arrays are materialized.", definition({"kind": "linear", "slope": 0.5}))
    ok(
        "linear-integer-spelling",
        "Integer spellings and integer negative zero convert to binary64; a one-sided clamp materializes its null member.",
        '{"schema_version":"1.0","semantics_version":"crc-impact/1","calibration":' + dumps(GENERIC)
        + ',"model":{"kind":"linear","slope":2,"intercept":-0},"clamp":{"minimum":0},"provenance":{"origin":{"kind":"user_defined"}}}',
    )
    ok(
        "linear-float-spelling",
        "Same function as linear-integer-spelling with float, exponent and negative-zero spellings; identical bytes and hash.",
        '{"provenance":{"origin":{"kind":"user_defined"},"sources":[],"derived_from":[]},"clamp":{"maximum":null,"minimum":0.0E0},'
        '"model":{"intercept":-0.0,"slope":20e-1,"kind":"linear"},"calibration":' + dumps(GENERIC)
        + ',"semantics_version":"crc-impact/1","schema_version":"1.0"}',
    )
    ok("linear-negative-slope-upper-clamp", "Negative slope with only an upper clamp.", definition({"kind": "linear", "slope": -1.5, "intercept": 3.0}, clamp={"maximum": 1.0}))
    ok("linear-equal-clamp", "Equal clamp limits give a constant response.", definition({"kind": "linear", "slope": 1.0}, clamp={"minimum": 0.5, "maximum": 0.5}))
    ok("linear-zero-slope", "Zero slope is a valid constant model.", definition({"kind": "linear", "slope": 0.0, "intercept": 0.25}))
    ok("clamp-both-null", "A clamp with both members null normalizes to null.", definition({"kind": "linear", "slope": 1.0}, clamp={"minimum": None, "maximum": None}))
    ok("clamp-empty-object", "An empty clamp object normalizes to null.", definition({"kind": "linear", "slope": 1.0}, clamp={}))
    ok("clamp-explicit-null", "Explicit null clamp.", definition({"kind": "linear", "slope": 1.0}, clamp=None))
    ok("sigmoid-minimal", "Sigmoid defaults: offset 0, amplitude 1, no origin adjustment, no threshold.", definition({"kind": "sigmoid", "midpoint": 35.65, "steepness": 0.2051873853077659}))
    ok(
        "sigmoid-full",
        "Every sigmoid member, origin subtraction, threshold and final clamp.",
        definition(
            {"kind": "sigmoid", "midpoint": 2.0, "steepness": -1.25, "offset": 0.1, "amplitude": 0.9, "origin_adjustment": "subtract_at_zero", "zero_below": 0.5},
            clamp={"minimum": 0.0, "maximum": 1.0},
        ),
    )
    ok("sigmoid-constant", "Zero amplitude and zero steepness are valid constant responses.", definition({"kind": "sigmoid", "midpoint": 0.0, "steepness": 0.0, "amplitude": 0.0, "offset": 0.3}))
    ok(
        "tabulated-step-left-error-domain",
        "Step-left interpolation, outside-domain error and decreasing response values.",
        definition({"kind": "tabulated", "points": [{"x": -1.0, "y": 1.0}, {"x": 0.0, "y": 0.5}, {"x": 2.5, "y": 0.0}], "interpolation": "step_left", "outside_domain": "error"}),
    )
    ok(
        "tabulated-step-right-non-monotone",
        "Step-right interpolation with a non-monotone response.",
        definition({"kind": "tabulated", "points": [{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 1.0}, {"x": 1.0, "y": 0.0}], "interpolation": "step_right"}),
    )
    ok(
        "binned-flood",
        "Binned response used by the aligned histogram expectation fixture.",
        definition({"kind": "binned", "bin_edges": [0.0, 0.1, 0.5, 1.5, 4.0], "values": [0.0, 0.05, 0.35, 0.9]}, calibration=FLOOD),
    )
    ok("binned-single-interval-error", "One interval with outside-domain error.", definition({"kind": "binned", "bin_edges": [0.0, 1.0], "values": [-2.5], "outside_domain": "error"}))
    ok("snapshot-spei", "SPEI registry snapshot model.", SPEI_SNAPSHOT)
    ok("snapshot-cooling", "Cooling registry snapshot model.", COOLING_SNAPSHOT)
    ok("snapshot-crop", "Crop registry snapshot model; weights are not renormalized.", CROP_SNAPSHOT)
    ok("snapshot-cyclone-registry-origin", "Registry origin trace with exact f32-widened coefficients.", CYCLONE_SNAPSHOT)
    ok(
        "origin-registry-legacy-fallback",
        "Registry origin with composite lookup keys, a recorded fallback, a null override and full context.",
        definition(
            {"kind": "sigmoid", "midpoint": 682.5, "steepness": 0.007, "origin_adjustment": "subtract_at_zero"},
            calibration={"exposure_variable": "rx1day", "exposure_unit": "mm", "impact_metric": "damage_ratio", "impact_unit": "1"},
            provenance={
                "origin": {
                    "kind": "registry",
                    "registry_id": "crc-climate-impacts",
                    "registry_revision": "2026.10.0",
                    "registry_manifest_sha256": MANIFEST,
                    "resolver_semantics_version": "crc-impact-registry/1",
                    "factor": "rx1day",
                    "context": {"cell": H3_CELL, "country": None, "continent": None, "building_type": None, "historic_mean": None},
                    "overrides": {"zero_below": None, "steepness": 0.007},
                    "fallback_policy": "legacy_defaults",
                    "lookups": [
                        {
                            "table_id": "precipitation.rx1day_threshold",
                            "table_sha256": DIGEST_A,
                            "requested_key": {"h3_resolution": "4", "cell": "0842a107fffffff"},
                            "resolved_key": "global_default",
                            "status": "fallback",
                        }
                    ],
                }
            },
        ),
    )
    ok(
        "origin-published",
        "Published origin with a complete source record.",
        definition(
            {"kind": "linear", "slope": 0.000222739726},
            provenance={
                "origin": {"kind": "published", "source_id": "huizinga-2017"},
                "sources": [{"id": "huizinga-2017", "kind": "publication", "uri": "https://doi.org/10.2760/16510", "revision": "v1", "content_sha256": DIGEST_A}],
            },
        ),
    )
    ok(
        "origin-imported",
        "Imported origin; an omitted build digest materializes as null and source defaults expand.",
        definition(
            {"kind": "linear", "slope": 1.0},
            provenance={
                "origin": {"kind": "imported", "source_id": "table", "importer_name": "crc-curve-import", "importer_version": "0.3.1"},
                "sources": [{"id": "table", "kind": "dataset", "uri": "s3://curves/table.csv"}, {"id": "code", "kind": "code", "uri": "git+https://example.org/importer"}],
            },
        ),
    )
    ok(
        "origin-imported-build",
        "Imported origin with an importer build digest.",
        definition(
            {"kind": "binned", "bin_edges": [0.0, 1.0, 2.0], "values": [0.1, 0.2]},
            provenance={
                "origin": {"kind": "imported", "source_id": "table", "importer_name": "crc-curve-import", "importer_version": "0.3.1", "importer_build_sha256": DIGEST_B},
                "sources": [{"id": "table", "kind": "dataset", "uri": "s3://curves/table.csv", "revision": "2024-01", "content_sha256": DIGEST_A}],
            },
        ),
    )
    ok(
        "origin-derived-unicode",
        "Derived origin: parents are sorted; Unicode, quotes, backslashes and control characters use the canonical escapes.",
        '{"schema_version":"1.0","semantics_version":"crc-impact/1","calibration":{"exposure_variable":"profondeur_d’eau",'
        '"exposure_unit":"m","impact_metric":"coût_relatif","impact_unit":"€/m²"},"model":{"kind":"linear","slope":1.0},'
        '"provenance":{"origin":{"kind":"derived","description":"Rescaled \\"Huizinga\\" \\\\ curve\\n\\ttab \\u0001 del:\\u007f ls:\\u2028 '
        'emoji:\\ud83d\\ude00 é"},"derived_from":["' + HASH_B + '","' + HASH_A + '"],"sources":[{"id":"parent","kind":"definition","uri":"crc:definition/' + HASH_A + '"}]}}',
    )
    ok(
        "number-boundaries",
        "Exponent placement, subnormal, large integer literal and shortest round-trip boundaries.",
        '{"schema_version":"1.0","semantics_version":"crc-impact/1","calibration":' + dumps(GENERIC) + ',"model":{"kind":"tabulated","points":['
        '{"x":-1.7976931348623157e308,"y":5e-324},{"x":-1e-7,"y":0.00001},{"x":0,"y":0.30000000000000004},{"x":1e15,"y":9007199254740993},'
        '{"x":1e16,"y":1.5e300},{"x":123456789012345678901,"y":2.2250738585072014e-308}]},"provenance":{"origin":{"kind":"user_defined"}}}',
    )

    rejected = []

    def bad(name: str, description: str, value: Any, code: str, path: str) -> None:
        rejected.append(reject(n, name, description, value if isinstance(value, str) else dumps(value), code, path))

    base = definition({"kind": "linear", "slope": 1.0})

    def mutate(change: Callable[[Dict[str, Any]], None], source: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        value = copy.deepcopy(source or base)
        change(value)
        return value

    linear_text = dumps(base)
    bad("invalid-json-syntax", "Truncated JSON text.", linear_text[:-1], "invalid_json", "")
    bad("invalid-json-trailing-comma", "Trailing comma.", linear_text[:-1] + ",}", "invalid_json", "")
    bad("invalid-json-bom", "Leading byte order mark.", "\ufeff" + linear_text, "invalid_json", "")
    bad("invalid-json-lone-surrogate", "Unpaired surrogate escape in a string.", linear_text.replace('"value"', '"\\ud800"', 1), "invalid_json", "/calibration/exposure_variable")
    bad("duplicate-key-root", "Duplicate top-level key.", '{"schema_version":"1.0",' + linear_text[1:], "duplicate_key", "/schema_version")
    bad("duplicate-key-model", "Duplicate model key.", linear_text.replace('"slope":1.0', '"slope":1.0,"slope":2.0'), "duplicate_key", "/model/slope")
    bad("non-finite-nan", "NaN token.", linear_text.replace('"slope":1.0', '"slope":NaN'), "non_finite_number", "/model/slope")
    bad("non-finite-infinity", "Infinity token.", linear_text.replace('"slope":1.0', '"slope":Infinity'), "non_finite_number", "/model/slope")
    bad("non-finite-negative-infinity", "Negative infinity token.", linear_text.replace('"slope":1.0', '"slope":-Infinity'), "non_finite_number", "/model/slope")
    bad("overflow-float-literal", "Float literal beyond binary64.", linear_text.replace('"slope":1.0', '"slope":1e400'), "number_overflow", "/model/slope")
    bad("overflow-integer-literal", "Integer literal beyond binary64.", linear_text.replace('"slope":1.0', '"slope":1' + "0" * 400), "number_overflow", "/model/slope")
    bad("boolean-number", "Boolean in a numeric field.", mutate(lambda d: d["model"].update(slope=True)), "invalid_type", "/model/slope")
    bad("string-number", "Numeric string in a numeric field.", mutate(lambda d: d["model"].update(slope="1.0")), "invalid_type", "/model/slope")
    bad("string-clamp", "Numeric string clamp member.", mutate(lambda d: d.update(clamp={"minimum": "0"})), "invalid_type", "/clamp/minimum")
    bad("calibration-type", "Non-string calibration member.", mutate(lambda d: d["calibration"].update(exposure_unit=1)), "invalid_type", "/calibration/exposure_unit")
    bad("schema-version-number", "Schema version spelled as a number.", mutate(lambda d: d.update(schema_version=1.0)), "invalid_type", "/schema_version")
    bad("schema-version-unsupported", "Unsupported schema version.", mutate(lambda d: d.update(schema_version="2.0")), "unsupported_schema_version", "/schema_version")
    bad("schema-version-minor", "Unknown minor schema version.", mutate(lambda d: d.update(schema_version="1.1")), "unsupported_schema_version", "/schema_version")
    bad("semantics-version-unsupported", "Unsupported numerical semantics.", mutate(lambda d: d.update(semantics_version="crc-impact/2")), "unsupported_semantics_version", "/semantics_version")
    bad("missing-schema-version", "Versions are never inferred.", mutate(lambda d: d.pop("schema_version")), "missing_field", "/schema_version")
    bad("missing-semantics-version", "Versions are never inferred.", mutate(lambda d: d.pop("semantics_version")), "missing_field", "/semantics_version")
    bad("missing-calibration", "Calibration is required.", mutate(lambda d: d.pop("calibration")), "missing_field", "/calibration")
    bad("missing-calibration-member", "All four calibration members are required.", mutate(lambda d: d["calibration"].pop("impact_unit")), "missing_field", "/calibration/impact_unit")
    bad("missing-provenance", "Provenance is required.", mutate(lambda d: d.pop("provenance")), "missing_field", "/provenance")
    bad("missing-origin", "Origin is required.", mutate(lambda d: d.update(provenance={"sources": []})), "missing_field", "/provenance/origin")
    bad("missing-model-kind", "Model kind is required.", mutate(lambda d: d["model"].pop("kind")), "missing_field", "/model/kind")
    bad("missing-slope", "Linear slope is required.", mutate(lambda d: d["model"].pop("slope")), "missing_field", "/model/slope")
    bad("unknown-root-field", "Unknown top-level field.", mutate(lambda d: d.update(output={"bounds": [0, 1]})), "unknown_field", "/output")
    bad("unknown-label-field", "Human labels live outside the definition.", mutate(lambda d: d.update(name="flood curve")), "unknown_field", "/name")
    bad("unknown-calibration-field", "Unknown calibration field.", mutate(lambda d: d["calibration"].update(currency="USD")), "unknown_field", "/calibration/currency")
    bad("unknown-model-field", "Legacy linear minimum is not a model field.", mutate(lambda d: d["model"].update(minimum=0.0)), "unknown_field", "/model/minimum")
    bad(
        "unknown-sigmoid-legacy-boolean",
        "The ambiguous adjust_at_zero boolean is not accepted.",
        definition({"kind": "sigmoid", "midpoint": 1.0, "steepness": 1.0, "adjust_at_zero": True}),
        "unknown_field",
        "/model/adjust_at_zero",
    )
    bad("unknown-clamp-field", "Unknown clamp field.", mutate(lambda d: d.update(clamp={"minimum": 0.0, "inclusive": True})), "unknown_field", "/clamp/inclusive")
    bad("unknown-provenance-field", "Unknown provenance field.", mutate(lambda d: d["provenance"].update(created_at="2026-10-21")), "unknown_field", "/provenance/created_at")
    bad("unknown-origin-field", "Unknown origin field.", mutate(lambda d: d["provenance"].update(origin={"kind": "user_defined", "author": "x"})), "unknown_field", "/provenance/origin/author")
    bad(
        "unknown-source-field",
        "Unknown source field.",
        mutate(lambda d: d["provenance"].update(sources=[{"id": "s", "kind": "code", "uri": "u", "license": "MIT"}])),
        "unknown_field",
        "/provenance/sources/0/license",
    )
    bad(
        "unknown-point-field",
        "Unknown knot field.",
        definition({"kind": "tabulated", "points": [{"x": 0.0, "y": 0.0, "w": 1.0}, {"x": 1.0, "y": 1.0}]}),
        "unknown_field",
        "/model/points/0/w",
    )
    bad("unsupported-model-kind", "Unknown model kind.", mutate(lambda d: d.update(model={"kind": "cubic", "slope": 1.0})), "unsupported_kind", "/model/kind")
    bad("registry-model-kind", "Registry references are not model kinds.", mutate(lambda d: d.update(model={"kind": "registry", "factor": "hot_days"})), "unsupported_kind", "/model/kind")
    bad("unsupported-origin-kind", "Unknown origin kind.", mutate(lambda d: d["provenance"].update(origin={"kind": "latest"})), "unsupported_kind", "/provenance/origin/kind")
    bad(
        "unsupported-interpolation",
        "Unknown interpolation.",
        definition({"kind": "tabulated", "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}], "interpolation": "cubic"}),
        "unsupported_value",
        "/model/interpolation",
    )
    bad(
        "unsupported-outside-domain",
        "Extrapolation is not supported in v1.",
        definition({"kind": "binned", "bin_edges": [0.0, 1.0], "values": [1.0], "outside_domain": "extrapolate"}),
        "unsupported_value",
        "/model/outside_domain",
    )
    bad(
        "unsupported-origin-adjustment",
        "Unknown origin adjustment.",
        definition({"kind": "sigmoid", "midpoint": 1.0, "steepness": 1.0, "origin_adjustment": "normalize"}),
        "unsupported_value",
        "/model/origin_adjustment",
    )
    bad(
        "unsupported-source-kind",
        "Unknown source kind.",
        mutate(lambda d: d["provenance"].update(sources=[{"id": "s", "kind": "blog", "uri": "u"}])),
        "unsupported_value",
        "/provenance/sources/0/kind",
    )
    bad("calibration-empty", "Empty calibration member.", mutate(lambda d: d["calibration"].update(exposure_unit="")), "invalid_value", "/calibration/exposure_unit")
    bad("calibration-whitespace", "Surrounding whitespace in a calibration member.", mutate(lambda d: d["calibration"].update(exposure_unit=" m")), "invalid_value", "/calibration/exposure_unit")
    bad("clamp-order", "Clamp minimum above maximum.", mutate(lambda d: d.update(clamp={"minimum": 1.0, "maximum": 0.0})), "invalid_value", "/clamp")
    bad("sigmoid-negative-amplitude", "Amplitude must be non-negative.", definition({"kind": "sigmoid", "midpoint": 1.0, "steepness": 1.0, "amplitude": -0.5}), "invalid_value", "/model/amplitude")
    bad("tabulated-one-point", "Tables need two knots.", definition({"kind": "tabulated", "points": [{"x": 0.0, "y": 0.0}]}), "invalid_value", "/model/points")
    bad(
        "tabulated-equal-knots",
        "Knots must strictly increase; they are never sorted.",
        definition({"kind": "tabulated", "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 2.0}]}),
        "invalid_value",
        "/model/points/2/x",
    )
    bad(
        "tabulated-unsorted-knots",
        "Unsorted knots are rejected, not repaired.",
        definition({"kind": "tabulated", "points": [{"x": 1.0, "y": 0.0}, {"x": 0.0, "y": 1.0}]}),
        "invalid_value",
        "/model/points/1/x",
    )
    bad("binned-decreasing-edges", "Edges must strictly increase.", definition({"kind": "binned", "bin_edges": [0.0, 2.0, 1.0], "values": [0.0, 1.0]}), "invalid_value", "/model/bin_edges/2")
    bad("binned-value-count", "One value per interval.", definition({"kind": "binned", "bin_edges": [0.0, 1.0, 2.0], "values": [0.0]}), "invalid_value", "/model/values")
    bad("spei-negative-cost", "Costs are non-negative.", definition({"kind": "spei", "water_consumption_asset": 1.0, "original_water_unit_cost": -1.0}), "invalid_value", "/model/original_water_unit_cost")
    bad("cooling-zero-mean", "Cooling historic mean is positive.", definition({"kind": "cooling", "cooling_cost": 1.0, "historic_mean": 0.0}), "invalid_value", "/model/historic_mean")
    bad("crop-zero-weights", "Crop weights need a positive total.", mutate(lambda d: d["model"].update(weights={"maize": 0.0, "rice": 0.0, "wheat": 0.0}), CROP_SNAPSHOT), "invalid_value", "/model/weights")
    bad("crop-negative-weight", "Crop weights are non-negative.", mutate(lambda d: d["model"]["weights"].update(rice=-0.1), CROP_SNAPSHOT), "invalid_value", "/model/weights/rice")
    bad("crop-missing-crop", "Crop vectors contain exactly maize, rice and wheat.", mutate(lambda d: d["model"]["index"].pop("wheat"), CROP_SNAPSHOT), "missing_field", "/model/index/wheat")
    bad("crop-extra-crop", "Crop vectors contain exactly maize, rice and wheat.", mutate(lambda d: d["model"]["index"].update(soy=0.1), CROP_SNAPSHOT), "unknown_field", "/model/index/soy")
    bad("derived-from-uppercase", "Parent hashes use lowercase hex.", mutate(lambda d: d["provenance"].update(derived_from=["sha256:" + "A" * 64])), "invalid_value", "/provenance/derived_from/0")
    bad("derived-from-bare", "Parent hashes carry the sha256: prefix.", mutate(lambda d: d["provenance"].update(derived_from=["1" * 64])), "invalid_value", "/provenance/derived_from/0")
    bad("derived-from-duplicate", "Parents are duplicate-free.", mutate(lambda d: d["provenance"].update(derived_from=[HASH_A, HASH_A])), "duplicate_identifier", "/provenance/derived_from/1")
    bad(
        "source-digest-length",
        "Source digests are 64 hex characters.",
        mutate(lambda d: d["provenance"].update(sources=[{"id": "s", "kind": "code", "uri": "u", "content_sha256": "abc"}])),
        "invalid_value",
        "/provenance/sources/0/content_sha256",
    )
    bad(
        "source-duplicate-id",
        "Source ids are unique.",
        mutate(lambda d: d["provenance"].update(sources=[{"id": "s", "kind": "code", "uri": "u"}, {"id": "s", "kind": "dataset", "uri": "v"}])),
        "duplicate_identifier",
        "/provenance/sources/1/id",
    )
    bad("published-missing-source", "Published source_id must resolve.", mutate(lambda d: d["provenance"].update(origin={"kind": "published", "source_id": "nope"})), "invalid_reference", "/provenance/origin/source_id")
    bad("derived-without-parents", "Derived origins need a parent.", mutate(lambda d: d["provenance"].update(origin={"kind": "derived", "description": "edited"})), "invalid_reference", "/provenance/derived_from")

    def registry(change: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
        value = copy.deepcopy(CYCLONE_SNAPSHOT)
        change(value["provenance"]["origin"])
        return value

    bad("registry-unknown-override", "Unknown override key.", registry(lambda o: o.update(overrides={"maximum": 1.0})), "unknown_override", "/provenance/origin/overrides/maximum")
    bad("registry-inapplicable-override", "Override key not accepted for the factor.", registry(lambda o: o.update(overrides={"slope": 1.0})), "inapplicable_override", "/provenance/origin/overrides/slope")
    bad("registry-null-override", "Null only clears thresholds or linear clamps.", registry(lambda o: o.update(overrides={"midpoint": None})), "invalid_value", "/provenance/origin/overrides/midpoint")
    bad("registry-unknown-factor", "Unknown registry factor.", registry(lambda o: o.update(factor="drought")), "unsupported_value", "/provenance/origin/factor")
    bad("registry-resolver-version", "Unsupported resolver semantics.", registry(lambda o: o.update(resolver_semantics_version="crc-impact-registry/2")), "unsupported_semantics_version", "/provenance/origin/resolver_semantics_version")
    bad("registry-cell-uppercase", "Persisted H3 cells use 15 lowercase hex characters.", registry(lambda o: o.update(context={"cell": H3_CELL.upper()})), "invalid_value", "/provenance/origin/context/cell")
    bad("registry-cell-padded", "Persisted H3 cells never carry a leading zero.", registry(lambda o: o.update(context={"cell": "0" + H3_CELL})), "invalid_value", "/provenance/origin/context/cell")
    bad("registry-cell-short", "Persisted H3 cells have exactly 15 characters.", registry(lambda o: o.update(context={"cell": H3_CELL[1:]})), "invalid_value", "/provenance/origin/context/cell")
    bad("registry-cell-non-hex", "Persisted H3 cells are hexadecimal.", registry(lambda o: o.update(context={"cell": H3_CELL[:-1] + "g"})), "invalid_value", "/provenance/origin/context/cell")
    bad("registry-fallback-under-error", "Fallback lookups require legacy_defaults.", registry(lambda o: o["lookups"][0].update(status="fallback")), "invalid_value", "/provenance/origin/lookups/0/status")
    bad("registry-missing-lookups", "The resolution trace is complete.", registry(lambda o: o.pop("lookups")), "missing_field", "/provenance/origin/lookups")
    bad("registry-unknown-context-field", "Unknown context field.", registry(lambda o: o.update(context={"region": "x"})), "unknown_field", "/provenance/origin/context/region")
    return {"accepted": accepted, "rejected": rejected}


# Registry references -------------------------------------------------------

REFERENCE = {
    "schema_version": "1.0",
    "registry_id": "crc-climate-impacts",
    "registry_revision": "2026.10.0",
    "registry_manifest_sha256": MANIFEST,
    "resolver_semantics_version": "crc-impact-registry/1",
    "factor": "cyclone",
}


def reference_cases() -> Dict[str, List[Dict[str, Any]]]:
    n = ref.normalize_reference

    def with_(**changes: Any) -> str:
        value = copy.deepcopy(REFERENCE)
        value.update(changes)
        return dumps(value)

    accepted = [
        accept(n, "minimal", "Required members only; overrides, context defaults and fallback policy materialize.", dumps(REFERENCE)),
        accept(n, "override-omitted", "Omitted zero_below uses the pinned registry default.", with_(overrides={"maximum_damage": 0.8})),
        accept(n, "override-null", "Explicit null clears zero_below; distinct identity from override-omitted.", with_(overrides={"maximum_damage": 0.8, "zero_below": None})),
        accept(n, "sigmoid-overrides", "Every sigmoid override key.", with_(factor="fwi", overrides={"midpoint": 30, "steepness": 0.25, "minimum_damage": 0.05, "maximum_damage": 0.95, "zero_below": 5.0})),
        accept(n, "spei-override", "SPEI water unit cost override.", with_(factor="spei", overrides={"water_unit_cost": 3.1}, context_defaults={"country": "CAN", "building_type": "Commercial buildings"})),
        accept(
            n,
            "context-defaults-legacy",
            "Context defaults and explicit legacy fallbacks.",
            with_(factor="cflood", overrides={"maximum_damage": 2}, context_defaults={"continent": "Europe", "building_type": "Residential buildings"}, fallback_policy="legacy_defaults"),
        ),
        accept(n, "hot-days-null-clamp", "Hot-days clamps are nullable overrides.", with_(factor="hot_days", overrides={"minimum_damage": None, "maximum_damage": 0.25, "slope": 0.0002, "intercept": -0.01})),
        accept(n, "cooling-branch-override", "Factor-level acceptance; branch applicability is checked after context resolution.", with_(factor="cooling_degree_days", overrides={"cooling_cost": 1.5e7})),
        accept(n, "no-override-factor", "maximum_consecutive_dry_days accepts no overrides.", with_(factor="maximum_consecutive_dry_days", context_defaults={"cell": H3_CELL, "historic_mean": 12.5})),
    ]
    rejected = [
        reject(n, "missing-manifest", "References pin the manifest digest.", dumps({k: v for k, v in REFERENCE.items() if k != "registry_manifest_sha256"}), "missing_field", "/registry_manifest_sha256"),
        reject(n, "missing-revision", "No persisted reference means latest.", dumps({k: v for k, v in REFERENCE.items() if k != "registry_revision"}), "missing_field", "/registry_revision"),
        reject(n, "unknown-field", "Unknown reference field.", with_(latest=True), "unknown_field", "/latest"),
        reject(n, "unknown-override", "Typo override key.", with_(overrides={"midpiont": 1.0}), "unknown_override", "/overrides/midpiont"),
        reject(n, "inapplicable-override", "Override key not accepted for the factor.", with_(factor="spei", overrides={"cooling_cost": 1.0}), "inapplicable_override", "/overrides/cooling_cost"),
        reject(n, "no-override-factor", "maximum_consecutive_dry_days accepts no overrides.", with_(factor="maximum_consecutive_dry_days", overrides={"maximum_damage": 1.0}), "inapplicable_override", "/overrides/maximum_damage"),
        reject(n, "null-override", "Null is not accepted for midpoint.", with_(overrides={"midpoint": None}), "invalid_value", "/overrides/midpoint"),
        reject(n, "string-override", "Override values are numbers.", with_(overrides={"midpoint": "70"}), "invalid_type", "/overrides/midpoint"),
        reject(n, "schema-version", "Unsupported reference schema.", with_(schema_version="2.0"), "unsupported_schema_version", "/schema_version"),
        reject(n, "resolver-version", "Unsupported resolver semantics.", with_(resolver_semantics_version="latest"), "unsupported_semantics_version", "/resolver_semantics_version"),
        reject(n, "fallback-policy", "Unknown fallback policy.", with_(fallback_policy="nearest"), "unsupported_value", "/fallback_policy"),
        reject(n, "unknown-factor", "Unknown factor.", with_(factor="heatwave"), "unsupported_value", "/factor"),
        reject(n, "context-null-defaults", "Context defaults are an object, not null.", with_(context_defaults=None), "invalid_type", "/context_defaults"),
        reject(n, "context-empty-country", "Textual context members are non-empty.", with_(context_defaults={"country": ""}), "invalid_value", "/context_defaults/country"),
        reject(n, "manifest-digest-prefixed", "Manifest digests are bare hex.", with_(registry_manifest_sha256="sha256:" + MANIFEST), "invalid_value", "/registry_manifest_sha256"),
    ]
    return {"accepted": accepted, "rejected": rejected}


# Evaluation records --------------------------------------------------------

ENGINE = {
    "framework_version": "0.5.0",
    "source_revision": "1288d99",
    "build_id": None,
    "target": "x86_64-unknown-linux-gnu",
    "numerical_methods": {"point": "crc-impact/1"},
}
REPLAY_COMPLETE = {"status": "complete", "missing_requirements": []}
SNAPSHOT = {"uri": "file:///data/exposure.parquet", "content_sha256": DIGEST_A, "retention": "retained"}
POINT = {"kind": "point", "settings": {"alignment": "event_aligned", "shape": "preserve"}}
DIRECT = {"exactness": "direct"}


def _definition_hash(value: Dict[str, Any]) -> str:
    return ref.content_hash(ref.normalize_definition(value))


def record(**changes: Any) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "schema_version": "1.0",
        "function": {"kind": "definition", "definition_hash": _definition_hash(PLAN_EXAMPLE)},
        "operation": POINT,
        "engine": ENGINE,
        "source": {
            "kind": "point_array",
            "shape": [3, 2],
            "content_digest": {"encoding": "crc-array-f64le/1", "sha256": DIGEST_B},
            "snapshot": SNAPSHOT,
            "selection": {"dataset": "jrc-flood", "columns": ["depth_rp100"], "rows": "asset_id in portfolio", "scenario": "ssp245", "horizon": "2050"},
        },
        "preprocessing": [],
        "resolution": None,
        "diagnostics": DIRECT,
        "replay": REPLAY_COMPLETE,
    }
    value.update(copy.deepcopy(changes))
    return value


REFERENCE_CONTEXT = {"cell": None, "country": "PHL", "continent": None, "building_type": None, "historic_mean": None}


def record_cases() -> Dict[str, List[Dict[str, Any]]]:
    n = ref.normalize_evaluation_record
    gk = {"kind": "gauss_kronrod", "version": "crc-gk15/1", "absolute_tolerance": 1e-10, "relative_tolerance": 1e-8, "max_evaluations": 100000}
    reference_hash = ref.content_hash(ref.normalize_reference(REFERENCE))

    def curve_record(**parameters: Any) -> Dict[str, Any]:
        return record(source={"kind": "canonical_distribution", "parameters": {
            "curve_kind": "fitted", "curve_type": "gumbel_r", "curve_location": 0.0,
            "curve_scale": 1.0, **parameters,
        }, "snapshot": SNAPSHOT})

    accepted = [
        accept(n, "point-complete", "Point evaluation with a retained array snapshot.", dumps(record())),
        accept(
            n,
            "point-missing-snapshot",
            "In-memory array with a digest but no retained snapshot; preprocessing converted millimetres to metres.",
            dumps(
                record(
                    source={"kind": "point_array", "shape": [], "content_digest": {"encoding": "crc-array-f64le/1", "sha256": DIGEST_B}},
                    preprocessing=[
                        {
                            "operation": "unit_conversion",
                            "description": "mm to m",
                            "parameters": {"from_unit": "mm", "to_unit": "m", "factor": 0.001, "exact": True, "note": None},
                            "input_digest": {"encoding": "crc-array-f64le/1", "sha256": DIGEST_A},
                            "output_digest": {"encoding": "crc-array-f64le/1", "sha256": DIGEST_B},
                        }
                    ],
                    replay={"status": "missing_source_snapshot", "missing_requirements": ["source_snapshot"]},
                )
            ),
        ),
        accept(
            n,
            "expected-histogram-exact",
            "Exact histogram expectation with recorded mass normalization and components.",
            dumps(
                record(
                    function={"kind": "definition", "definition_hash": _definition_hash(definition({"kind": "binned", "bin_edges": [0.0, 0.1, 0.5, 1.5, 4.0], "values": [0.0, 0.05, 0.35, 0.9]}, calibration=FLOOD))},
                    operation={"kind": "expected_value", "settings": {"requested_method": "auto", "method": {"kind": "exact", "version": "crc-exact-piecewise/1"}}},
                    source={"kind": "histogram", "interval_count": 4, "within_bin": "uniform", "original_mass_sum": 1.0000000000002, "normalization_factor": 1 / 1.0000000000002, "snapshot": SNAPSHOT},
                    diagnostics={"exactness": "exact", "estimated_error": 0.0, "evaluations": 0, "components": [{"kind": "exact", "value": 0.1275}]},
                )
            ),
        ),
        accept(
            n,
            "expected-curve-quadrature",
            "Auto request resolved to Gauss-Kronrod with every effective default.",
            dumps(
                record(
                    operation={"kind": "expected_value", "settings": {"requested_method": "auto", "method": gk}},
                    source={"kind": "canonical_distribution", "parameters": {"curve_kind": "fitted", "curve_type": "gumbel_r", "curve_location": 1.0, "curve_scale": 0.8}, "snapshot": SNAPSHOT},
                    engine={**ENGINE, "numerical_methods": {"quadrature": "crc-gk15/1", "point": "crc-impact/1"}},
                    diagnostics={"exactness": "numerical", "estimated_error": 3.2e-12, "evaluations": 465, "components": [{"kind": "numerical", "value": 2.92354, "estimated_error": 3.2e-12, "evaluations": 465}]},
                )
            ),
        ),
        accept(
            n,
            "transform-registry-resolution",
            "Distribution transform of a registry-resolved definition with its effective context identity.",
            dumps(
                record(
                    function={"kind": "definition", "definition_hash": _definition_hash(CYCLONE_SNAPSHOT)},
                    operation={"kind": "transform_distribution", "settings": {"algorithm": "crc-pushforward/1"}},
                    source={"kind": "canonical_distribution", "parameters": {"curve_kind": "tabulated", "curve_type": "linear_probability", "curve_probabilities": [0.0, 1.0], "curve_values": [0.0, 10.0]}, "snapshot": SNAPSHOT},
                    resolution={"reference_hash": reference_hash, "context": {"country": "PHL"}, "context_hash": ref.content_hash(REFERENCE_CONTEXT)},
                    diagnostics={"exactness": "exact"},
                )
            ),
        ),
        accept(
            n,
            "tabulate-approximation",
            "Explicit approximate tabulation with its full probability grid.",
            dumps(
                record(
                    operation={"kind": "tabulate", "settings": {"probabilities": [0.0, 0.5, 0.9, 0.99, 1.0], "interpolation": "linear_probability", "approximate": True}},
                    source={"kind": "canonical_distribution", "parameters": {"curve_kind": "point_mass", "curve_type": "point_mass", "curve_location": 0.5, "curve_scale": 0, "curve_atom_probability": 1, "curve_atom_location": 0.5}, "snapshot": {**SNAPSHOT, "retention": "external", "revision": "v3"}},
                    diagnostics={"exactness": "exact"},
                )
            ),
        ),
        accept(
            n,
            "custom-runtime-only",
            "Custom callable with untracked source and its available runtime identity.",
            dumps(
                record(
                    function={"kind": "custom", "module": "portfolio.impacts", "qualified_name": "DepthDamage.evaluate", "type": "DepthDamage"},
                    source={"kind": "untracked", "description": "caller-held numpy array"},
                    replay={"status": "runtime_only", "missing_requirements": ["source_snapshot", "function_definition"]},
                )
            ),
        ),
        accept(
            n,
            "custom-anonymous-minimal",
            "Custom identity unavailable; optional engine, source and preprocessing members materialize as null or empty.",
            dumps(
                record(
                    function={"kind": "custom"},
                    engine={"framework_version": "0.5.0", "target": "aarch64-apple-darwin", "numerical_methods": {}},
                    source={"kind": "untracked"},
                    preprocessing=[{"operation": "fill_missing"}],
                    replay={"status": "runtime_only", "missing_requirements": ["function_definition", "source_snapshot", "engine_build"]},
                )
            ),
        ),
        accept(
            n,
            "point-minimal-source",
            "Point array source with only a retained snapshot and an empty selection.",
            dumps(record(source={"kind": "point_array", "shape": [4], "snapshot": SNAPSHOT, "selection": {}})),
        ),
        accept(
            n,
            "canonical-distribution-identified",
            "Canonical distribution with shape, digest and selection but no retained snapshot.",
            dumps(
                record(
                    operation={"kind": "expected_value", "settings": {"requested_method": "gauss_kronrod", "method": gk}},
                    source={
                        "kind": "canonical_distribution",
                        "parameters": {"curve_kind": "hurdle", "curve_type": "genextreme", "curve_shape": -0.1, "curve_location": 1.0, "curve_scale": 0.5, "curve_atom_probability": 0.2, "curve_atom_location": 0.0},
                        "content_digest": {"encoding": "crc-curve-parameters/1", "sha256": DIGEST_B},
                        "selection": {"dataset": "crc-open/heat", "columns": ["curve_*"], "scenario": "ssp585", "horizon": "2080"},
                    },
                    diagnostics={"exactness": "mixed", "estimated_error": 1e-11, "evaluations": 225, "components": [{"kind": "exact", "value": 0.0}, {"kind": "numerical", "value": 0.61, "estimated_error": 1e-11, "evaluations": 225}]},
                    replay={"status": "missing_source_snapshot", "missing_requirements": ["source_snapshot"]},
                )
            ),
        ),
        accept(
            n,
            "histogram-identified",
            "Histogram with digest and selection but no retained snapshot.",
            dumps(
                record(
                    operation={"kind": "expected_value", "settings": {"requested_method": "exact", "method": {"kind": "exact", "version": "crc-exact-piecewise/1"}}},
                    source={"kind": "histogram", "interval_count": 2, "within_bin": "uniform", "original_mass_sum": 1, "normalization_factor": 1, "content_digest": {"encoding": "crc-histogram/1", "sha256": DIGEST_A}, "selection": {"rows": "region = 'EU'"}},
                    diagnostics={"exactness": "exact"},
                    replay={"status": "missing_source_snapshot", "missing_requirements": ["source_snapshot"]},
                )
            ),
        ),
    ]
    accepted.append(accept(n, "u64-max-shape", "Integer metadata preserves the largest native u64 exactly.", dumps(record(source={"kind": "point_array", "shape": [ref.U64_MAX], "snapshot": SNAPSHOT}))))
    accepted.append(accept(n, "gumbel-unused-shape", "Canonical Gumbel normalization removes the unused shape.", dumps(curve_record(curve_shape=7.0))))
    for family in ref.FITTED_CURVE_TYPES:
        shape = None if family in ("gumbel_r", "gumbel_l") else 0.2
        accepted.append(accept(n, "curve-family-" + family, "A reconstructible canonical fitted source.", dumps(curve_record(curve_type=family, curve_shape=shape))))
    rejected = [
        reject(n, "shape-u64-overflow", "Integer metadata must fit the declared Rust u64.", dumps(record(source={"kind": "point_array", "shape": [ref.U64_MAX + 1], "snapshot": SNAPSHOT})), "invalid_value", "/source/shape/0"),
        reject(n, "evaluations-u64-overflow", "Integration budgets must fit the declared Rust u64.", dumps(record(operation={"kind": "expected_value", "settings": {"requested_method": "auto", "method": {**gk, "max_evaluations": ref.U64_MAX + 1}}})), "invalid_value", "/operation/settings/method/max_evaluations"),
        reject(n, "curve-negative-scale", "Source curves require a positive fitted scale.", dumps(curve_record(curve_scale=-1.0)), "invalid_value", "/source/parameters/curve_scale"),
        reject(n, "curve-missing-location", "A fitted source requires its location.", dumps(curve_record(curve_location=None)), "invalid_value", "/source/parameters/curve_location"),
        reject(n, "curve-unknown-kind", "Arbitrary curve kinds cannot claim complete replay.", dumps(curve_record(curve_kind="invented")), "unsupported_value", "/source/parameters/curve_kind"),
        reject(n, "curve-unknown-family", "A fitted source must use a supported family.", dumps(curve_record(curve_type="invented")), "unsupported_value", "/source/parameters/curve_type"),
        reject(n, "curve-required-shape", "Non-Gumbel fitted families require shape.", dumps(curve_record(curve_type="genextreme")), "invalid_value", "/source/parameters/curve_shape"),
        reject(n, "curve-fitted-atom", "A fitted source cannot smuggle hurdle parameters.", dumps(curve_record(curve_atom_probability=0.2)), "invalid_value", "/source/parameters/curve_atom_probability"),
        reject(n, "curve-hurdle-probability", "Hurdle atom mass must be strictly between zero and one.", dumps(curve_record(curve_kind="hurdle", curve_atom_probability=1.0, curve_atom_location=0.0)), "invalid_value", "/source/parameters/curve_atom_probability"),
        reject(n, "curve-point-mass-mismatch", "A point mass has one consistent atom and curve location.", dumps(curve_record(curve_kind="point_mass", curve_type="point_mass", curve_scale=0.0, curve_atom_probability=1.0, curve_atom_location=1.0)), "invalid_value", "/source/parameters/curve_atom_location"),
        reject(n, "curve-tabulated-length", "Quantile knot arrays have equal lengths.", dumps(curve_record(curve_kind="tabulated", curve_type="linear_probability", curve_location=None, curve_scale=None, curve_probabilities=[0.0, 1.0], curve_values=[0.0])), "invalid_value", "/source/parameters/curve_values"),
        reject(n, "curve-tabulated-order", "Quantile probabilities strictly increase.", dumps(curve_record(curve_kind="tabulated", curve_type="linear_probability", curve_location=None, curve_scale=None, curve_probabilities=[0.5, 0.4], curve_values=[0.0, 1.0])), "invalid_value", "/source/parameters/curve_probabilities/1"),
        reject(n, "curve-tabulated-values", "Quantile values cannot decrease.", dumps(curve_record(curve_kind="tabulated", curve_type="linear_probability", curve_location=None, curve_scale=None, curve_probabilities=[0.0, 1.0], curve_values=[1.0, 0.0])), "invalid_value", "/source/parameters/curve_values/1"),
        reject(n, "histogram-normalization-factor", "Normalization diagnostics must equal the reciprocal of original mass.", dumps(record(source={"kind": "histogram", "interval_count": 1, "within_bin": "uniform", "original_mass_sum": 1.0, "normalization_factor": 2.0, "snapshot": SNAPSHOT})), "invalid_value", "/source/normalization_factor"),
        reject(n, "custom-complete", "Custom functions cannot claim complete replay.", dumps(record(function={"kind": "custom"})), "invalid_value", "/replay/status"),
        reject(n, "definition-runtime-only", "Definition functions are not runtime-only.", dumps(record(replay={"status": "runtime_only", "missing_requirements": ["source_snapshot"]})), "invalid_value", "/replay/status"),
        reject(n, "complete-with-missing", "Complete replay lists no missing requirements.", dumps(record(replay={"status": "complete", "missing_requirements": ["engine_build"]})), "invalid_value", "/replay/missing_requirements"),
        reject(n, "missing-snapshot-unlisted", "missing_source_snapshot lists source_snapshot.", dumps(record(replay={"status": "missing_source_snapshot", "missing_requirements": ["engine_build"]})), "invalid_value", "/replay/missing_requirements"),
        reject(n, "complete-without-snapshot", "Hashing alone does not retain data.", dumps(record(source={"kind": "point_array", "shape": [2]})), "invalid_value", "/replay/status"),
        reject(n, "duplicate-requirement", "Requirements are duplicate-free.", dumps(record(replay={"status": "missing_source_snapshot", "missing_requirements": ["source_snapshot", "source_snapshot"]})), "duplicate_identifier", "/replay/missing_requirements/1"),
        reject(n, "unknown-requirement", "Requirements come from the closed list.", dumps(record(replay={"status": "missing_source_snapshot", "missing_requirements": ["source_snapshot", "luck"]})), "unsupported_value", "/replay/missing_requirements/1"),
        reject(n, "method-contradiction", "An explicit request equals the resolved method.", dumps(record(operation={"kind": "expected_value", "settings": {"requested_method": "exact", "method": gk}})), "invalid_value", "/operation/settings/method"),
        reject(n, "unresolved-auto", "Records store the resolved method, not only auto.", dumps(record(operation={"kind": "expected_value", "settings": {"requested_method": "auto", "method": {"kind": "auto"}}})), "unsupported_kind", "/operation/settings/method/kind"),
        reject(n, "fractional-evaluations", "Integer fields reject fractional spellings.", dumps(record(operation={"kind": "expected_value", "settings": {"requested_method": "auto", "method": {**gk, "max_evaluations": 1.5}}})), "invalid_type", "/operation/settings/method/max_evaluations"),
        reject(n, "integral-float-evaluations", "Integer fields reject float spellings.", dumps(record(operation={"kind": "expected_value", "settings": {"requested_method": "auto", "method": {**gk, "max_evaluations": 100000.0}}})), "invalid_type", "/operation/settings/method/max_evaluations"),
        reject(n, "zero-evaluations", "Maximum evaluations is positive.", dumps(record(operation={"kind": "expected_value", "settings": {"requested_method": "auto", "method": {**gk, "max_evaluations": 0}}})), "invalid_value", "/operation/settings/method/max_evaluations"),
        reject(n, "tabulate-not-approximate", "Tabulation is always approximate.", dumps(record(operation={"kind": "tabulate", "settings": {"probabilities": [0.5], "interpolation": "linear_probability", "approximate": False}})), "invalid_value", "/operation/settings/approximate"),
        reject(n, "tabulate-unsorted", "Tabulation grids strictly increase.", dumps(record(operation={"kind": "tabulate", "settings": {"probabilities": [0.5, 0.1], "interpolation": "linear_probability", "approximate": True}})), "invalid_value", "/operation/settings/probabilities/1"),
        reject(n, "tabulate-no-interpolation", "Tabulation requires an interpolation choice.", dumps(record(operation={"kind": "tabulate", "settings": {"probabilities": [0.5], "approximate": True}})), "missing_field", "/operation/settings/interpolation"),
        reject(n, "histogram-mass", "Histogram mass within 1e-12 of one.", dumps(record(source={"kind": "histogram", "interval_count": 1, "within_bin": "uniform", "original_mass_sum": 1.1, "normalization_factor": 1 / 1.1, "snapshot": SNAPSHOT})), "invalid_value", "/source/original_mass_sum"),
        reject(n, "context-hash-mismatch", "Context identity matches the canonical context.", dumps(record(resolution={"reference_hash": reference_hash, "context": {"country": "PHL"}, "context_hash": HASH_A})), "invalid_value", "/resolution/context_hash"),
        reject(n, "unknown-operation", "Aggregation is not an operation kind.", dumps(record(operation={"kind": "aggregate", "settings": {}})), "unsupported_kind", "/operation/kind"),
        reject(n, "unknown-source", "Unknown source kind.", dumps(record(source={"kind": "zarr"})), "unsupported_kind", "/source/kind"),
        reject(n, "negative-shape", "Shape dimensions are non-negative integers.", dumps(record(source={"kind": "point_array", "shape": [-1], "snapshot": SNAPSHOT})), "invalid_value", "/source/shape/0"),
        reject(n, "output-columns", "Records never prescribe output storage.", dumps(record(output_column="damage")), "unknown_field", "/output_column"),
        reject(n, "missing-engine", "Engine identity is required.", dumps({k: v for k, v in record().items() if k != "engine"}), "missing_field", "/engine"),
        reject(n, "schema-version", "Unsupported record schema.", dumps(record(schema_version="0.9")), "unsupported_schema_version", "/schema_version"),
    ]
    return {"accepted": accepted, "rejected": rejected}


# Spatial policies and records ----------------------------------------------

POLICY = {
    "schema_version": "1.0",
    "semantics_version": "crc-spatial-combination/1",
    "stage": "impact",
    "reducer": "weighted_mean",
    "weights": {"kind": "intersection_area"},
    "coverage": "require_full",
}


def policy_cases() -> Dict[str, List[Dict[str, Any]]]:
    n = ref.normalize_spatial_policy

    def with_(**changes: Any) -> str:
        value = copy.deepcopy(POLICY)
        value.update(changes)
        return dumps({k: v for k, v in value.items() if v is not None})

    accepted = [
        accept(n, "plan-example", "Section 8.3 example.", dumps(POLICY)),
        accept(n, "exposure-stage", "Before-impact combination; differs from plan-example only by stage.", with_(stage="exposure")),
        accept(n, "provided-defaults", "Provided weights; reducer and coverage defaults materialize.", with_(reducer=None, coverage=None, weights={"kind": "provided", "meaning": "asset_value"})),
        accept(n, "covered-only", "Explicit covered-only normalization.", with_(coverage="covered_only")),
    ]
    rejected = [
        reject(n, "missing-stage", "No implicit stage for multiple contributors.", with_(stage=None), "missing_field", "/stage"),
        reject(n, "auto-stage", "No implicit stage.", with_(stage="auto"), "unsupported_value", "/stage"),
        reject(n, "sum-reducer", "Only weighted_mean in v1.", with_(reducer="sum"), "unsupported_value", "/reducer"),
        reject(n, "quantile-pairing", "Quantile pairing is deferred.", with_(reducer="quantile_mean"), "unsupported_value", "/reducer"),
        reject(n, "missing-weights", "Weight basis is required.", with_(weights=None), "missing_field", "/weights"),
        reject(n, "provided-without-meaning", "Provided weights declare their meaning.", with_(weights={"kind": "provided"}), "missing_field", "/weights/meaning"),
        reject(n, "unknown-weight-kind", "Unknown weight basis.", with_(weights={"kind": "population"}), "unsupported_kind", "/weights/kind"),
        reject(n, "coverage", "Unknown coverage treatment.", with_(coverage="zero_fill"), "unsupported_value", "/coverage"),
        reject(n, "distribution-aggregation", "Source-distribution aggregation is not part of the policy.", with_(aggregate_distributions=True), "unknown_field", "/aggregate_distributions"),
        reject(n, "semantics-version", "Unsupported spatial semantics.", with_(semantics_version="crc-spatial-combination/2"), "unsupported_semantics_version", "/semantics_version"),
    ]
    return {"accepted": accepted, "rejected": rejected}


def spatial_record(policy: Dict[str, Any], **changes: Any) -> Dict[str, Any]:
    normalized = ref.normalize_spatial_policy(policy)
    snapshot = {"uri": "file:///runs/r1/contributors.json", "revision": None, "content_sha256": DIGEST_A, "retention": "retained"}
    exposure = normalized["stage"] == "exposure"
    value: Dict[str, Any] = {
        "schema_version": "1.0",
        "policy": policy,
        "policy_hash": ref.content_hash(normalized),
        "target": {"id": "asset-17", "kind": "geometry", "geometry_sha256": DIGEST_B},
        "contributors": snapshot,
        "contributor_count": 2,
        "geometry": {"engine": "geos", "engine_version": "3.12.1", "crs": "EPSG:6933", "area_method": "planar", "target_area": 12500.0, "covered_fraction": 1.0, "overlap_fraction": 0.0, "tolerance": 1e-9},
        "weights": {"raw": {**snapshot, "uri": "file:///runs/r1/weights-raw.json"}, "normalized": {**snapshot, "uri": "file:///runs/r1/weights.json"}},
        "combined_exposure": {"encoding": "crc-array-f64le/1", "sha256": DIGEST_B} if exposure else None,
        "children": [{"record_hash": HASH_A}] if exposure else [{"contributor_id": "cell-a", "record_hash": HASH_A}, {"contributor_id": "cell-b", "record_hash": HASH_B, "uri": "file:///runs/r1/b.json"}],
        "replay": REPLAY_COMPLETE,
    }
    value.update(copy.deepcopy(changes))
    return value


def spatial_record_cases() -> Dict[str, List[Dict[str, Any]]]:
    n = ref.normalize_spatial_record
    exposure_policy = {**POLICY, "stage": "exposure"}
    provided = {**POLICY, "weights": {"kind": "provided", "meaning": "asset_value"}, "coverage": "covered_only"}
    impact = spatial_record(POLICY)
    accepted = [
        accept(n, "impact-stage", "After-impact combination with one child record per contributor.", dumps(impact)),
        accept(
            n,
            "exposure-stage",
            "Before-impact combination with one target child and combined exposure identity; policy defaults materialize.",
            dumps(spatial_record({k: v for k, v in exposure_policy.items() if k not in ("reducer", "coverage")})),
        ),
        accept(
            n,
            "provided-covered-only",
            "Provided weights with a unit and explicit partial coverage on an H3 target.",
            dumps(
                spatial_record(
                    provided,
                    target={"id": "cell", "kind": "h3_cell", "cell": H3_CELL},
                    weights={**impact["weights"], "unit": "USD"},
                    geometry={**impact["geometry"], "covered_fraction": 0.82},
                    replay={"status": "missing_source_snapshot", "missing_requirements": ["source_snapshot"]},
                )
            ),
        ),
    ]
    rejected = [
        reject(n, "policy-hash-mismatch", "Policy identity matches the canonical policy.", dumps(spatial_record(POLICY, policy_hash=HASH_B)), "invalid_value", "/policy_hash"),
        reject(n, "function-hash", "No single function hash for an application.", dumps(spatial_record(POLICY, function_hash=HASH_A)), "unknown_field", "/function_hash"),
        reject(n, "exposure-two-children", "Exposure stage evaluates one target function.", dumps(spatial_record(exposure_policy, children=[{"record_hash": HASH_A}, {"record_hash": HASH_B}])), "invalid_value", "/children"),
        reject(n, "exposure-without-combined", "Exposure stage retains the combined exposure identity.", dumps(spatial_record(exposure_policy, combined_exposure=None)), "invalid_value", "/combined_exposure"),
        reject(n, "impact-missing-child", "Impact stage retains every child record.", dumps(spatial_record(POLICY, children=[{"contributor_id": "cell-a", "record_hash": HASH_A}])), "invalid_value", "/children"),
        reject(n, "impact-duplicate-contributor", "Duplicate contributor ids.", dumps(spatial_record(POLICY, children=[{"contributor_id": "a", "record_hash": HASH_A}, {"contributor_id": "a", "record_hash": HASH_B}])), "duplicate_identifier", "/children/1/contributor_id"),
        reject(n, "overlap", "Positive-area overlap above tolerance.", dumps(spatial_record(POLICY, geometry={**impact["geometry"], "overlap_fraction": 1e-6})), "invalid_value", "/geometry/overlap_fraction"),
        reject(n, "require-full-uncovered", "require_full rejects uncovered area above tolerance.", dumps(spatial_record(POLICY, geometry={**impact["geometry"], "covered_fraction": 0.9})), "invalid_value", "/geometry/covered_fraction"),
        reject(n, "provided-without-unit", "Provided weights require a common unit.", dumps(spatial_record(provided)), "invalid_value", "/weights/unit"),
        reject(n, "h3-target-without-cell", "H3 targets carry the cell.", dumps(spatial_record(POLICY, target={"id": "t", "kind": "h3_cell"})), "invalid_value", "/target/cell"),
        reject(n, "h3-target-padded", "H3 targets use 15-character wire strings.", dumps(spatial_record(POLICY, target={"id": "t", "kind": "h3_cell", "cell": "0" + H3_CELL})), "invalid_value", "/target/cell"),
        reject(n, "h3-target-short", "H3 targets use exactly 15 hexadecimal characters.", dumps(spatial_record(POLICY, target={"id": "t", "kind": "h3_cell", "cell": H3_CELL[1:]})), "invalid_value", "/target/cell"),
        reject(n, "h3-target-non-hex", "H3 targets are hexadecimal.", dumps(spatial_record(POLICY, target={"id": "t", "kind": "h3_cell", "cell": H3_CELL[:-1] + "g"})), "invalid_value", "/target/cell"),
        reject(n, "zero-area-target", "Empty or zero-area targets are rejected.", dumps(spatial_record(POLICY, geometry={**impact["geometry"], "target_area": 0.0})), "invalid_value", "/geometry/target_area"),
        reject(n, "relaxed-tolerance", "Recorded diagnostics cannot relax the versioned spatial policy.", dumps(spatial_record(POLICY, geometry={**impact["geometry"], "tolerance": 1.0, "covered_fraction": 0.0, "overlap_fraction": 1.0})), "invalid_value", "/geometry/tolerance"),
        reject(n, "changed-tolerance", "The spatial semantics fixes the effective tolerance.", dumps(spatial_record(POLICY, geometry={**impact["geometry"], "tolerance": 1e-10})), "invalid_value", "/geometry/tolerance"),
        reject(n, "count-u64-overflow", "Contributor counts must fit native u64 even for an exposure-stage record.", dumps(spatial_record(exposure_policy, contributor_count=ref.U64_MAX + 1)), "invalid_value", "/contributor_count"),
    ]
    return {"accepted": accepted, "rejected": rejected}


# Executable impact-semantics inventory -------------------------------------

def semantics_constants() -> List[Dict[str, Any]]:
    """Every normative numeric constant, version, and boundary rule in crc-impact/1.

    `fixtures` names executable artifact vectors.  The generator verifies these
    references, preventing a documented semantic from becoming untested.
    """
    return [
        {"id": "impact-semantics-version", "value": "crc-impact/1", "owner": "ImpactDefinition.semantics_version", "fixtures": ["definition:plan-example-tabulated"]},
        {"id": "canonical-profile", "value": "crc-canonical-json/1", "owner": "canonical_json", "fixtures": ["number:1e+16"]},
        {"id": "spei-ln-199", "formula": "ln(199)", "owner": "SpeiImpact", "fixtures": ["point:snapshot-spei"]},
        {"id": "spei-drought-multiplier", "value": 10, "owner": "SpeiImpact", "fixtures": ["point:snapshot-spei"]},
        {"id": "economic-normalizer", "value": 10000000, "owner": "SpeiImpact,CoolingImpact", "fixtures": ["point:snapshot-spei", "point:snapshot-cooling"]},
        {"id": "crop-reference-year", "value": 2010, "owner": "CropImpact", "fixtures": ["point:snapshot-crop"]},
        {"id": "sigmoid-threshold-strict", "rule": "x < zero_below", "owner": "SigmoidImpact", "fixtures": ["point:sigmoid-origin-threshold"]},
        {"id": "cooling-threshold-inclusive", "rule": "x <= 1", "owner": "CoolingImpact", "fixtures": ["point:snapshot-cooling"]},
        {"id": "binned-right-edge", "rule": "interior edge selects right interval", "owner": "BinnedImpact", "fixtures": ["point:binned-flood"]},
        {"id": "histogram-mass-tolerance", "value": 1e-12, "owner": "histogram_mass", "fixtures": ["operation:histogram-normalization", "operation:histogram-mass-mismatch"]},
        {"id": "quadrature-absolute-tolerance", "value": 1e-10, "owner": "expectation", "fixtures": ["operation:linear-gumbel-mean"]},
        {"id": "quadrature-relative-tolerance", "value": 1e-8, "owner": "expectation", "fixtures": ["operation:linear-gumbel-mean"]},
        {"id": "resolver-semantics-version", "value": "crc-impact-registry/1", "owner": "RegistryImpactReference", "fixtures": ["registry-reference:minimal"]},
        {"id": "spatial-semantics-version", "value": "crc-spatial-combination/1", "owner": "SpatialCombinationPolicy", "fixtures": ["spatial-policy:plan-example"]},
        {"id": "h3-wire-length", "value": 15, "owner": "ImpactContext.cell", "fixtures": ["definition:origin-registry-legacy-fallback", "definition:registry-cell-padded", "spatial-record:h3-target-padded"]},
        {"id": "spatial-geometry-tolerance", "value": ref.SPATIAL_GEOMETRY_TOLERANCE, "owner": "SpatialImpactEvaluationRecord.geometry", "fixtures": ["spatial-record:overlap", "spatial-record:relaxed-tolerance"]},
        {"id": "native-u64-maximum", "value": ref.U64_MAX, "owner": "integer metadata", "fixtures": ["evaluation-record:u64-max-shape", "evaluation-record:shape-u64-overflow"]},
        {"id": "override-catalog", "rule": "factor override keys and nullability", "owner": "ImpactRegistry", "fixtures": ["definition:registry-null-override", "registry-reference:no-override-factor"]},
    ]


def native_capabilities() -> Dict[str, Any]:
    return {
        "supported": ["canonical_json"],
        "required_fixture_families": {"canonical_json": ["numbers", "strings"]},
        "native_parser_rejections": ["leading-bom", "nan", "infinity", "negative-infinity", "binary64-overflow", "unpaired-surrogate"],
        "future_families": ["strict_duplicate_key_parser", "point_evaluations", "monotonicity", "operations", "registry"],
    }


# Canonical numbers ---------------------------------------------------------

NUMBER_INPUTS = [
    "0", "-0", "0.0", "-0.0", "1", "1.0", "1E0", "10e-1", "-1", "0.1", "0.30000000000000004", "1e-5", "0.00001",
    "1e-6", "1e-7", "1.5e-7", "123.456", "1e15", "1e16", "1e+16", "1e+16 ", "1.5e16", "12345678901234567", "9007199254740993",
    "123456789012345678901", "1e21", "1e22", "1e23", "1.7976931348623157e308", "2.2250738585072014e-308",
    "2.225073858507201e-308", "5e-324", "4.9e-324", "1e-400", "0.1e1", "100", "1e2", "33.333333333333336",
    "-2.5e-10", "6.02214076e23", "299792458",
]


def number_cases() -> List[Dict[str, Any]]:
    cases = []
    for text in NUMBER_INPUTS:
        value = ref._number(json.loads(text), "")
        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        cases.append({"name": text, "input": text, "bits": f"{bits:016x}", "canonical": ref.format_binary64(value)})
    return cases


PARSER_REJECTIONS = [
    ("leading-bom", "\ufeff0", "invalid_json"),
    ("nan", "NaN", "non_finite_number"),
    ("infinity", "Infinity", "non_finite_number"),
    ("negative-infinity", "-Infinity", "non_finite_number"),
    ("binary64-overflow", "1e400", "number_overflow"),
    ("unpaired-surrogate", '"\\ud800"', "invalid_json"),
    ("duplicate-key", '{"a": 1, "a": 2}', "duplicate_key"),
]


def parser_rejection_cases() -> List[Dict[str, str]]:
    result = []
    for name, text, code in PARSER_REJECTIONS:
        try:
            ref.parse_json(text)
        except ref.ContractError as error:
            if error.code != code:
                raise AssertionError(f"{name}: expected {code}, got {error.code}") from error
        else:
            raise AssertionError(f"{name}: strict parser accepted invalid JSON")
        result.append({"name": name, "input": text, "error": code})
    return result


STRING_CASES = [
    ("plain", "flood_depth"),
    ("quote-backslash", 'a"b\\c'),
    ("short-escapes", "\b\f\n\r\t"),
    ("control", "\u0000\u0001\u001f"),
    ("delete-unescaped", "\u007f"),
    ("non-ascii-raw", "é€²"),
    ("line-separator-raw", "\u2028\u2029"),
    ("astral-raw", "😀"),
    ("slash-unescaped", "a/b"),
]


def string_cases() -> List[Dict[str, Any]]:
    return [
        {"name": name, "value": value, "canonical": ref.canonical_bytes(value).decode("utf-8"), "utf8_hex": ref.canonical_bytes(value).hex()}
        for name, value in STRING_CASES
    ]


# Point evaluation and monotonicity ------------------------------------------

POINT_TOLERANCE = {"absolute": 1e-12, "relative": 1e-12}


def _after(value: float) -> float:
    return math.nextafter(value, math.inf)


def _before(value: float) -> float:
    return math.nextafter(value, -math.inf)


def point_cases() -> List[Dict[str, Any]]:
    plan = PLAN_EXAMPLE
    sets: List[tuple] = [
        ("tabulated-linear", plan, [-1.0, 0.0, _after(0.0), 0.15, 0.3, _before(0.3), _after(0.3), 1.0, 1.5, 2.0, _after(2.0), 3.0], None),
        ("tabulated-step-left", definition({"kind": "tabulated", "points": [{"x": 0.0, "y": 1.0}, {"x": 1.0, "y": 0.5}, {"x": 2.0, "y": 0.0}], "interpolation": "step_left", "outside_domain": "error"}), [0.0, 0.5, _before(1.0), 1.0, _after(1.0), 2.0, _before(0.0), _after(2.0)], None),
        ("tabulated-step-right", definition({"kind": "tabulated", "points": [{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 1.0}, {"x": 1.0, "y": 0.0}], "interpolation": "step_right"}), [-5.0, 0.0, 0.25, _before(0.5), 0.5, 0.75, 1.0, 7.0], None),
        ("binned-flood", definition({"kind": "binned", "bin_edges": [0.0, 0.1, 0.5, 1.5, 4.0], "values": [0.0, 0.05, 0.35, 0.9]}, calibration=FLOOD), [-1.0, 0.0, _before(0.1), 0.1, 0.3, 0.5, 1.5, _before(4.0), 4.0, 9.0], None),
        ("binned-single-error", definition({"kind": "binned", "bin_edges": [0.0, 1.0], "values": [-2.5], "outside_domain": "error"}), [0.0, 0.5, 1.0, _before(0.0), _after(1.0)], None),
        ("linear-clamped", definition({"kind": "linear", "slope": 0.0813 / 365.0}, clamp={"minimum": 0.0}), [-10.0, 0.0, 30.0, 365.0], None),
        ("linear-negative-slope", definition({"kind": "linear", "slope": -1.5, "intercept": 3.0}, clamp={"maximum": 1.0}), [0.0, 4.0 / 3.0, 2.0, 10.0], None),
        ("linear-equal-clamp", definition({"kind": "linear", "slope": 1.0}, clamp={"minimum": 0.5, "maximum": 0.5}), [-1e300, 0.0, 1e300], None),
        ("linear-overflow", definition({"kind": "linear", "slope": 1e308}, clamp={"minimum": 0.0, "maximum": 1.0}), [0.5, 10.0], "A final clamp must not conceal an overflow in the raw response."),
        ("sigmoid-plain", definition({"kind": "sigmoid", "midpoint": 35.65, "steepness": 0.2051873853077659}), [-1e6, 0.0, 35.65, 50.0, 1e6], None),
        ("sigmoid-negative-steepness", definition({"kind": "sigmoid", "midpoint": 2.0, "steepness": -3.0, "offset": 0.1, "amplitude": 0.5}), [-100.0, 0.0, 2.0, 5.0], None),
        ("sigmoid-zero-steepness", definition({"kind": "sigmoid", "midpoint": 4.0, "steepness": 0.0, "amplitude": 0.6, "offset": -0.1}), [-3.0, 4.0, 1e9], None),
        ("sigmoid-constant-extreme", definition({"kind": "sigmoid", "midpoint": -1e308, "steepness": 0.0}), [-1e308, 1e308], "Zero steepness remains constant when x - midpoint would overflow."),
        ("sigmoid-zero-amplitude-extreme", definition({"kind": "sigmoid", "midpoint": -1e308, "steepness": 0.0, "amplitude": 0.0, "offset": 0.3}), [-1e308, 1e308], "Zero amplitude returns offset without evaluating an indeterminate argument."),
        ("sigmoid-origin-threshold", definition({"kind": "sigmoid", "midpoint": 2.0, "steepness": 1.25, "offset": 0.1, "amplitude": 0.9, "origin_adjustment": "subtract_at_zero", "zero_below": 0.5}), [-2.0, 0.0, _before(0.5), 0.5, 2.0, 40.0], None),
        ("sigmoid-origin-negative-exposure", definition({"kind": "sigmoid", "midpoint": 2.0, "steepness": 1.25, "offset": 0.1, "amplitude": 0.9, "origin_adjustment": "subtract_at_zero"}), [-5.0, -1.0, 0.0, 1.0], "Origin subtraction returns offset at zero and can go below offset for negative exposures."),
        ("sigmoid-clamp-order", definition({"kind": "sigmoid", "midpoint": 2.0, "steepness": 1.25, "offset": -0.2, "amplitude": 1.5, "zero_below": 1.0}, clamp={"minimum": 0.0, "maximum": 1.0}), [0.0, 1.0, 2.0, 10.0], "Threshold first, then a single final clamp."),
        ("snapshot-cyclone-f32", CYCLONE_SNAPSHOT, [0.0, 28.0, f32(28.5), 74.7, 120.0], None),
        ("snapshot-spei", SPEI_SNAPSHOT, [-3.0, -1.0, _before(0.0), 0.0, 2.0], "The SPEI drought branch applies strictly below zero."),
        ("snapshot-cooling", COOLING_SNAPSHOT, [0.0, 1.0, _after(1.0), 410.5, 900.0], None),
        ("snapshot-crop", CROP_SNAPSHOT, [0.0, 150.0, 300.0, 1000.0], None),
        ("snapshot-crop-overflow", definition({**CROP_SNAPSHOT["model"], "index": {"maize": 1.0, "rice": 0.0, "wheat": 0.0}}, calibration=CROP_SNAPSHOT["calibration"]), [0.0, 800.0], "exp overflow in the future term is an error."),
    ]
    results = []
    for name, value, xs, note in sets:
        normalized = ref.normalize_definition(value)
        cases = []
        for x in xs:
            try:
                cases.append({"x": x, "expected": ref.evaluate(normalized, x)})
            except ref.ContractError as error:
                cases.append({"x": x, "error": error.code})
        for special in ("NaN", "Infinity", "-Infinity"):
            if name in ("tabulated-linear", "sigmoid-plain"):
                cases.append({"x_special": special, "error": "non_finite_exposure"})
        entry = {"name": name, "definition": normalized, "definition_hash": ref.content_hash(normalized), "tolerance": POINT_TOLERANCE, "cases": cases}
        if note:
            entry["note"] = note
        results.append(entry)
    return results


def monotonicity_cases() -> List[Dict[str, Any]]:
    sets = [
        ("linear-increasing", definition({"kind": "linear", "slope": 2.0}), ["non_decreasing", "unknown"]),
        ("linear-decreasing-clamped", definition({"kind": "linear", "slope": -2.0}, clamp={"minimum": 0.0}), ["non_increasing", "unknown"]),
        ("linear-zero", definition({"kind": "linear", "slope": 0.0}), ["constant", "non_decreasing", "non_increasing", "unknown"]),
        ("linear-equal-clamp", definition({"kind": "linear", "slope": 3.0}, clamp={"minimum": 1.0, "maximum": 1.0}), ["constant", "non_decreasing", "unknown"]),
        ("sigmoid-increasing", definition({"kind": "sigmoid", "midpoint": 1.0, "steepness": 2.0}), ["non_decreasing", "unknown"]),
        ("sigmoid-decreasing", definition({"kind": "sigmoid", "midpoint": 1.0, "steepness": -2.0}), ["non_increasing", "unknown"]),
        ("sigmoid-zero-amplitude", definition({"kind": "sigmoid", "midpoint": 1.0, "steepness": 2.0, "amplitude": 0.0}), ["constant", "non_decreasing", "non_increasing", "unknown"]),
        ("sigmoid-threshold-rising-jump", definition({"kind": "sigmoid", "midpoint": 1.0, "steepness": 2.0, "zero_below": 0.5}), ["non_decreasing", "unknown"]),
        ("sigmoid-threshold-breaks-decreasing", definition({"kind": "sigmoid", "midpoint": 1.0, "steepness": -2.0, "zero_below": 0.5}), ["non_monotone", "unknown"]),
        ("sigmoid-threshold-negative-jump", definition({"kind": "sigmoid", "midpoint": 1.0, "steepness": 2.0, "offset": -1.0, "amplitude": 0.5, "zero_below": 0.5}), ["non_monotone", "unknown"]),
        ("sigmoid-threshold-hidden-by-clamp", definition({"kind": "sigmoid", "midpoint": 1.0, "steepness": 2.0, "offset": -1.0, "amplitude": 2.0, "zero_below": 0.5}, clamp={"minimum": 0.0}), ["non_decreasing", "unknown"]),
        ("sigmoid-origin-subtraction-threshold", definition({"kind": "sigmoid", "midpoint": 2.0, "steepness": 1.0, "origin_adjustment": "subtract_at_zero", "zero_below": 1.0}), ["non_decreasing", "unknown"]),
        ("tabulated-decreasing", definition({"kind": "tabulated", "points": [{"x": 0.0, "y": 1.0}, {"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 0.0}]}), ["non_increasing", "unknown"]),
        ("tabulated-non-monotone", definition({"kind": "tabulated", "points": [{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 1.0}, {"x": 1.0, "y": 0.0}]}), ["non_monotone", "unknown"]),
        ("tabulated-non-monotone-clamped-flat", definition({"kind": "tabulated", "points": [{"x": 0.0, "y": 2.0}, {"x": 0.5, "y": 3.0}, {"x": 1.0, "y": 2.5}]}, clamp={"maximum": 1.0}), ["constant", "non_decreasing", "non_increasing", "unknown"]),
        ("binned-step-up", definition({"kind": "binned", "bin_edges": [0.0, 5.0, 10.0], "values": [0.0, 1.0]}), ["non_decreasing", "unknown"]),
        ("binned-constant", definition({"kind": "binned", "bin_edges": [0.0, 5.0, 10.0], "values": [0.4, 0.4]}), ["constant", "non_decreasing", "non_increasing", "unknown"]),
        ("spei", SPEI_SNAPSHOT, ["non_increasing", "unknown"]),
        ("cooling", COOLING_SNAPSHOT, ["non_decreasing", "unknown"]),
        ("crop-same-sign", CROP_SNAPSHOT, ["non_decreasing", "unknown"]),
        (
            "crop-underflowing-derivative-product",
            definition({"kind": "crop", "weights": {"maize": 1e-200, "rice": 0.0, "wheat": 0.0}, "country": {"maize": 0.0, "rice": 0.0, "wheat": 0.0}, "yearly": {"maize": 0.0, "rice": 0.0, "wheat": 0.0}, "index": {"maize": 1e-200, "rice": 0.0, "wheat": 0.0}, "historic_mean": 0.0}),
            ["non_increasing", "unknown"],
        ),
        (
            "crop-mixed-sign-equal-clamp",
            definition({**CROP_SNAPSHOT["model"], "index": {"maize": -0.002, "rice": 0.004, "wheat": 0.0}}, clamp={"minimum": 1.0, "maximum": 1.0}),
            ["constant", "non_decreasing", "non_increasing", "unknown"],
        ),
        (
            "crop-mixed-sign",
            definition({**CROP_SNAPSHOT["model"], "index": {"maize": -0.002, "rice": 0.004, "wheat": 0.0}}, calibration=CROP_SNAPSHOT["calibration"]),
            ["non_monotone", "unknown"],
        ),
    ]
    results = []
    for name, value, acceptable in sets:
        normalized = ref.normalize_definition(value)
        expected = ref.monotonicity(normalized)
        if expected not in acceptable:
            raise AssertionError(f"{name}: reference monotonicity {expected} not in {acceptable}")
        results.append({"name": name, "definition": normalized, "definition_hash": ref.content_hash(normalized), "reference": expected, "acceptable": acceptable})
    return results


# Operation reference values ---------------------------------------------------

EULER_GAMMA = 0.5772156649015329
QUADRATURE_TOLERANCE = {"absolute": 1e-10, "relative": 1e-8}
EXACT_TOLERANCE = {"absolute": 1e-12, "relative": 1e-12}
UNIFORM_0_10 = {"kind": "canonical_distribution", "parameters": {"curve_kind": "tabulated", "curve_type": "linear_probability", "curve_probabilities": [0.0, 1.0], "curve_values": [0.0, 10.0]}}


def _hist(edges: List[float], weights: List[float]) -> Dict[str, Any]:
    return {"kind": "histogram", "bin_edges": edges, "weights": weights, "within_bin": "uniform"}


def _point_mass(location: float) -> Dict[str, Any]:
    return {"kind": "canonical_distribution", "parameters": {"curve_kind": "point_mass", "curve_type": "point_mass", "curve_location": location, "curve_scale": 0.0, "curve_atom_probability": 1.0, "curve_atom_location": location}}


def _fitted(family: str, shape: Optional[float], location: float, scale: float) -> Dict[str, Any]:
    return {"kind": "canonical_distribution", "parameters": {"curve_kind": "fitted", "curve_type": family, "curve_shape": shape, "curve_location": location, "curve_scale": scale}}


def _with_definition(entry: Dict[str, Any], value: Dict[str, Any]) -> Dict[str, Any]:
    normalized = ref.normalize_definition(value)
    return {"definition": normalized, "definition_hash": ref.content_hash(normalized), **entry}


def operation_cases() -> Dict[str, Any]:
    flood = definition({"kind": "binned", "bin_edges": [0.0, 0.1, 0.5, 1.5, 4.0], "values": [0.0, 0.05, 0.35, 0.9]}, calibration=FLOOD)
    step = definition({"kind": "binned", "bin_edges": [0.0, 0.5, 2.0], "values": [0.0, 1.0]})
    near_one = ref.histogram_mass([0.5, 0.5 + 4e-13])
    gumbel_linear = definition({"kind": "linear", "slope": 2.0})
    genpareto_heavy = _fitted("genpareto", 1.0, 0.0, 1.0)
    expectations = [
        _with_definition({"name": "flood-aligned-histogram", "source": _hist([0.0, 0.1, 0.5, 1.5, 4.0], [0.5, 0.25, 0.2, 0.05]), "method": "auto", "resolved_method": "exact", "expected": 0.1275, "tolerance": EXACT_TOLERANCE, "derivation": "0.5*0 + 0.25*0.05 + 0.2*0.35 + 0.05*0.9"}, flood),
        _with_definition({"name": "uniform-bin-threshold", "source": _hist([0.0, 2.0], [1.0]), "method": "exact", "resolved_method": "exact", "expected": 0.75, "tolerance": EXACT_TOLERANCE, "derivation": "uniform [0,2]; response 0 below 0.5 and 1 above: (2 - 0.5) / 2"}, step),
        _with_definition({"name": "misaligned-bins", "source": _hist([0.0, 1.0, 2.0], [0.5, 0.5]), "method": "auto", "resolved_method": "exact", "expected": 0.75, "tolerance": EXACT_TOLERANCE, "derivation": "split at 0.5: 0.5*(0.5*0 + 0.5*1) + 0.5*1"}, step),
        _with_definition({"name": "linear-histogram", "source": _hist([0.0, 1.0, 3.0], [0.5, 0.5]), "method": "auto", "resolved_method": "exact", "expected": 3.5, "tolerance": EXACT_TOLERANCE, "derivation": "2*E[X] + 1 with E[X] = 0.5*0.5 + 0.5*2"}, definition({"kind": "linear", "slope": 2.0, "intercept": 1.0})),
        _with_definition({"name": "knot-inside-bin", "source": _hist([0.0, 2.0], [1.0]), "method": "auto", "resolved_method": "exact", "expected": 0.75, "tolerance": EXACT_TOLERANCE, "derivation": "(integral of min(x,1) on [0,2]) / 2 = (0.5 + 1) / 2"}, definition({"kind": "tabulated", "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}]})),
        _with_definition({"name": "linear-clamp-inside-bin", "source": _hist([0.0, 2.0], [1.0]), "method": "exact", "resolved_method": "exact", "expected": 0.5625, "tolerance": EXACT_TOLERANCE, "derivation": "integral of max(x-0.5,0) on [0,2], divided by 2: 1.5**2 / 4 = 9/16"}, definition({"kind": "linear", "slope": 1.0, "intercept": -0.5}, clamp={"minimum": 0.0})),
        _with_definition({"name": "tabulated-clamp-inside-bin", "source": _hist([0.0, 2.0], [1.0]), "method": "exact", "resolved_method": "exact", "expected": 0.4375, "tolerance": EXACT_TOLERANCE, "derivation": "integral of min(x,0.5) on [0,2], divided by 2: (0.125+0.75)/2 = 7/16"}, definition({"kind": "tabulated", "points": [{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": 2.0}]}, clamp={"maximum": 0.5})),
        _with_definition({"name": "atom-on-bin-edge", "source": _point_mass(0.5), "method": "auto", "resolved_method": "exact", "expected": 1.0, "tolerance": EXACT_TOLERANCE, "derivation": "interior edges select the interval to their right"}, step),
        _with_definition({"name": "atom-on-threshold", "source": _point_mass(0.5), "method": "auto", "resolved_method": "exact", "expected": 0.5, "tolerance": EXACT_TOLERANCE, "derivation": "threshold equality evaluates the sigmoid: L(0) = 0.5"}, definition({"kind": "sigmoid", "midpoint": 0.5, "steepness": 10.0, "zero_below": 0.5})),
        _with_definition({"name": "discrete-weighted-sum", "source": {"kind": "discrete", "support": [0.0, 1.0, 3.0], "probabilities": [0.2, 0.5, 0.3]}, "method": "auto", "resolved_method": "exact", "expected": 2.8, "tolerance": EXACT_TOLERANCE, "derivation": "2 * (0.5*1 + 0.3*3)"}, gumbel_linear),
        _with_definition(
            {
                "name": "histogram-normalization",
                "source": _hist([0.0, 1.0, 2.0], [0.5, 0.5 + 4e-13]),
                "method": "exact",
                "resolved_method": "exact",
                "expected": near_one["normalized_weights"][1],
                "tolerance": EXACT_TOLERANCE,
                "source_diagnostics": {"original_mass_sum": near_one["original_mass_sum"], "normalization_factor": near_one["normalization_factor"]},
                "derivation": "Neumaier sum in bin order; weights divided by the sum once",
            },
            definition({"kind": "binned", "bin_edges": [0.0, 1.0, 2.0], "values": [0.0, 1.0]}),
        ),
        _with_definition({"name": "linear-gumbel-mean", "source": _fitted("gumbel_r", None, 1.0, 0.8), "method": "auto", "resolved_method": "any", "expected": 2.0 * (1.0 + 0.8 * EULER_GAMMA), "tolerance": QUADRATURE_TOLERANCE, "derivation": "2 * (location + scale * Euler-Mascheroni); unclamped linear with finite first moment"}, gumbel_linear),
        _with_definition({"name": "clamped-heavy-tail", "source": genpareto_heavy, "method": "auto", "resolved_method": "any", "expected": math.log(2.0), "tolerance": QUADRATURE_TOLERANCE, "derivation": "E[min(X,1)] = integral_0^1 1/(1+x) dx for genpareto c=1"}, definition({"kind": "linear", "slope": 1.0}, clamp={"maximum": 1.0})),
        _with_definition({"name": "heavy-tail-divergent", "source": genpareto_heavy, "method": "auto", "error": "not_integrable", "derivation": "genpareto c=1 has no finite mean; quadrature convergence is not proof"}, gumbel_linear),
        _with_definition({"name": "incomplete-support", "source": {"kind": "canonical_distribution", "parameters": {"curve_kind": "tabulated", "curve_type": "linear_probability", "curve_probabilities": [0.1, 0.9], "curve_values": [0.0, 1.0]}}, "method": "auto", "error": "missing_probability_support", "derivation": "tails are never silently omitted or renormalized"}, step),
        _with_definition({"name": "histogram-mass-mismatch", "source": _hist([0.0, 1.0, 2.0], [0.5, 0.6]), "method": "auto", "error": "invalid_histogram", "derivation": "total mass 1.1 is outside 1e-12 of one"}, step),
        _with_definition({"name": "histogram-negative-weight", "source": _hist([0.0, 1.0, 2.0], [1.5, -0.5]), "method": "auto", "error": "invalid_histogram", "derivation": "weights are non-negative"}, step),
        _with_definition({"name": "exact-unavailable", "source": _fitted("skewnorm", 0.0, 0.0, 1.0), "method": "exact", "error": "exact_method_unavailable", "derivation": "sigmoid of a normal has no implemented exact path"}, definition({"kind": "sigmoid", "midpoint": 0.0, "steepness": 1.0})),
        _with_definition({"name": "quadrature-budget", "source": _fitted("skewnorm", 0.0, 0.0, 1.0), "method": {"kind": "gauss_kronrod", "absolute_tolerance": 1e-300, "relative_tolerance": 1e-300, "max_evaluations": 15}, "error": "convergence_failed", "derivation": "tolerance unreachable within one 15-point rule"}, definition({"kind": "sigmoid", "midpoint": 0.0, "steepness": 1.0})),
    ]
    transforms = [
        _with_definition({"name": "decreasing-uniform", "source": UNIFORM_0_10, "probabilities": [0.1, 0.5, 0.8], "quantiles": [1.0, 5.0, 8.0], "tolerance": EXACT_TOLERANCE, "derivation": "Q_Y(p) = 10 - Q_X(1 - p) = 10p"}, definition({"kind": "linear", "slope": -1.0, "intercept": 10.0})),
        _with_definition({"name": "binary-threshold", "source": UNIFORM_0_10, "probabilities": [0.25, 0.5, 0.75], "quantiles": [0.0, 0.0, 1.0], "atoms": [{"value": 0.0, "probability": 0.5}, {"value": 1.0, "probability": 0.5}], "tolerance": EXACT_TOLERANCE, "derivation": "only responses 0 and 1; generalized inverse at the tie returns 0"}, definition({"kind": "binned", "bin_edges": [0.0, 5.0, 10.0], "values": [0.0, 1.0]})),
        _with_definition({"name": "decreasing-step", "source": UNIFORM_0_10, "probabilities": [0.25, 0.5, 0.75], "quantiles": [0.0, 0.0, 1.0], "atoms": [{"value": 0.0, "probability": 0.5}, {"value": 1.0, "probability": 0.5}], "tolerance": EXACT_TOLERANCE, "derivation": "decreasing step: P(Y = 0) = P(X >= 5)"}, definition({"kind": "binned", "bin_edges": [0.0, 5.0, 10.0], "values": [1.0, 0.0]})),
        _with_definition({"name": "flat-segment", "source": UNIFORM_0_10, "probabilities": [0.1, 0.2, 0.5, 0.95], "quantiles": [0.5, 1.0, 1.0, 1.5], "atoms": [{"value": 1.0, "probability": 0.6}], "tolerance": EXACT_TOLERANCE, "derivation": "flat between x = 2 and 8 creates an atom of mass 0.6 at 1"}, definition({"kind": "tabulated", "points": [{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": 1.0}, {"x": 8.0, "y": 1.0}, {"x": 10.0, "y": 2.0}]})),
        _with_definition({"name": "atom-on-edge", "source": _point_mass(5.0), "probabilities": [0.0, 0.5, 1.0], "quantiles": [1.0, 1.0, 1.0], "atoms": [{"value": 1.0, "probability": 1.0}], "tolerance": EXACT_TOLERANCE, "derivation": "an atom exactly on an interior edge maps to the right interval"}, definition({"kind": "binned", "bin_edges": [0.0, 5.0, 10.0], "values": [0.0, 1.0]})),
        _with_definition({"name": "non-monotone", "source": UNIFORM_0_10, "error": "monotonicity_required", "derivation": "pushforward requires a certified monotone function"}, definition({"kind": "tabulated", "points": [{"x": 0.0, "y": 0.0}, {"x": 5.0, "y": 1.0}, {"x": 10.0, "y": 0.0}]})),
        _with_definition({"name": "threshold-breaks-monotonicity", "source": UNIFORM_0_10, "error": "monotonicity_required", "derivation": "a decreasing sigmoid with a zero threshold is non-monotone"}, definition({"kind": "sigmoid", "midpoint": 1.0, "steepness": -2.0, "zero_below": 0.5})),
        _with_definition({"name": "partial-support", "source": {"kind": "canonical_distribution", "parameters": {"curve_kind": "tabulated", "curve_type": "linear_probability", "curve_probabilities": [0.1, 0.9], "curve_values": [0.0, 1.0]}}, "error": "missing_probability_support", "derivation": "partial quantile curves are not full distributions"}, definition({"kind": "linear", "slope": 1.0})),
    ]
    spatial = []
    curves = [
        ("square", [(0.0, 0.0), (0.5, 0.25), (1.0, 1.0)], 0.25, 0.5),
        ("decreasing-square", [(0.0, 1.0), (0.5, 0.75), (1.0, 0.0)], 0.75, 0.5),
        ("non-monotone", [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)], 1.0, 0.0),
    ]
    policies = {stage: ref.normalize_spatial_policy({**POLICY, "stage": stage}) for stage in ("exposure", "impact")}
    for name, knots, exposure_stage, impact_stage in curves:
        value = definition({"kind": "tabulated", "points": [{"x": x, "y": y} for x, y in knots]})
        normalized = ref.normalize_definition(value)
        combined = 0.5 * 0.0 + 0.5 * 1.0
        assert ref.evaluate(normalized, combined) == exposure_stage
        assert 0.5 * ref.evaluate(normalized, 0.0) + 0.5 * ref.evaluate(normalized, 1.0) == impact_stage
        spatial.append(
            {
                "name": name,
                "definition": normalized,
                "definition_hash": ref.content_hash(normalized),
                "contributors": [{"id": "a", "weight": 1.0, "exposure": [0.0, 1.0]}, {"id": "b", "weight": 1.0, "exposure": [1.0, 0.0]}],
                "results": {
                    stage: {
                        "policy_hash": ref.content_hash(policies[stage]),
                        "values": [
                            ref.evaluate(normalized, 0.5 * a + 0.5 * b) if stage == "exposure" else 0.5 * ref.evaluate(normalized, a) + 0.5 * ref.evaluate(normalized, b)
                            for a, b in zip([0.0, 1.0], [1.0, 0.0])
                        ],
                    }
                    for stage in ("exposure", "impact")
                },
                "tolerance": EXACT_TOLERANCE,
                "derivation": "equal weights; event arrays are combined elementwise and never reordered",
            }
        )
    return {
        "description": "Conformance vectors for impact operations. Source encodings here are fixture inputs, not persisted contracts. resolved_method 'any' accepts exact or numerical paths within tolerance.",
        "policies": policies,
        "expected_value": expectations,
        "transform_distribution": transforms,
        "spatial_combination": spatial,
    }

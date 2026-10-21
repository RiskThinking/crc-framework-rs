"""Generate the versioned impact contract artifact.

Run from the repository root:
    python tools/impact_contracts/generate.py          # rewrite the artifact
    python tools/impact_contracts/generate.py --check  # fail if it is stale

Requires ``jsonschema`` to record whether each case is schema-valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cases  # noqa: E402
import reference as ref  # noqa: E402
import schemas  # noqa: E402

ARTIFACT = "crc-impact-contracts"
ARTIFACT_VERSION = "1.0.0"
OUTPUT = HERE.parent.parent / "python" / "crc_framework" / "contracts" / "impact"

FIXTURE_CONTRACTS = {
    "fixtures/definitions.json": ("impact_definition", "definition_hash", cases.definition_cases),
    "fixtures/registry-references.json": ("registry_impact_reference", "reference_hash", cases.reference_cases),
    "fixtures/evaluation-records.json": ("impact_evaluation_record", "record_hash", cases.record_cases),
    "fixtures/spatial-policies.json": ("spatial_combination_policy", "policy_hash", cases.policy_cases),
    "fixtures/spatial-records.json": ("spatial_impact_evaluation_record", "record_hash", cases.spatial_record_cases),
}

ERROR_CODES = {
    "invalid_json": ("parse", "Malformed JSON text, a byte order mark, or an unpaired surrogate."),
    "duplicate_key": ("parse", "An object repeats a key; reported at the repeated key."),
    "non_finite_number": ("parse", "NaN or Infinity tokens."),
    "number_overflow": ("parse", "A numeric literal overflows binary64."),
    "unknown_field": ("validate", "A field not defined by the schema at that level."),
    "missing_field": ("validate", "A required field is absent."),
    "invalid_type": ("validate", "A value has the wrong JSON type, including booleans or strings in numeric fields and fractional integers."),
    "unsupported_schema_version": ("validate", "The schema_version is not readable by this implementation."),
    "unsupported_semantics_version": ("validate", "A numerical or resolver semantics version is not supported."),
    "unsupported_kind": ("validate", "A tagged-union kind is not supported."),
    "unsupported_value": ("validate", "An enumerated value is not supported."),
    "invalid_value": ("validate", "A value violates a documented constraint."),
    "invalid_reference": ("validate", "A reference does not resolve within the document."),
    "duplicate_identifier": ("validate", "An identifier or set member is repeated."),
    "unknown_override": ("validate", "An override key is not in the override catalog."),
    "inapplicable_override": ("resolve", "An override key is not accepted for the factor or resolved branch."),
    "missing_context": ("resolve", "Required resolution context is missing after merging."),
    "registry_unavailable": ("resolve", "The requested registry revision is not installed."),
    "registry_manifest_mismatch": ("resolve", "The installed manifest digest differs from the pinned digest."),
    "fallback_required": ("resolve", "A table key is missing and the fallback policy is error."),
    "non_finite_exposure": ("evaluate", "An exposure value is NaN or infinite."),
    "non_finite_result": ("evaluate", "The raw or final response is not finite."),
    "out_of_domain": ("evaluate", "Exposure outside a table domain with outside_domain error."),
    "invalid_histogram": ("operation", "Histogram edges or weights violate the histogram contract."),
    "missing_probability_support": ("operation", "The source does not define the full probability interval."),
    "not_integrable": ("operation", "Integrability of the response cannot be established."),
    "convergence_failed": ("operation", "Numerical integration did not reach its tolerance."),
    "exact_method_unavailable": ("operation", "An explicitly requested exact method is not implemented for the case."),
    "monotonicity_required": ("operation", "The operation requires a certified monotone function."),
    "ambiguous_contributors": ("spatial", "Multiple contributors without a spatial combination policy."),
    "overlapping_contributors": ("spatial", "Positive-area overlap or competing coverage."),
    "incomplete_coverage": ("spatial", "Uncovered target area under require_full."),
    "invalid_weights": ("spatial", "Weights are missing, negative, non-finite or sum to zero."),
    "incompatible_contributors": ("spatial", "Contributors disagree on quantity, unit, scenario, horizon or metric."),
    "unaligned_contributors": ("spatial", "Contributors lack a justified common event identity."),
    "unsupported_combination": ("spatial", "Source-distribution or quantile combination is not supported."),
}

UNIT_ALIASES = {
    "description": "Documented v1 alias normalization applied before exact canonical unit equality. Physical conversions are never implied.",
    "aliases": {"metre": "m", "metres": "m", "meter": "m", "meters": "m", "day": "d", "days": "d", "fraction": "1", "ratio": "1", "dimensionless": "1"},
}

OVERRIDE_MEANINGS = {
    "sigmoid": "minimum_damage and maximum_damage set curve levels (offset and offset + amplitude), not a final clamp.",
    "hot_days": "minimum_damage and maximum_damage set the final clamp; null clears a limit.",
    "flood": "maximum_damage scales table values; it is not a clamp.",
}


def _schema_validator(contract: str) -> Any:
    try:
        import jsonschema
    except ImportError as error:
        raise SystemExit("generate.py requires jsonschema") from error
    schema = schemas.all_schemas()[contract]
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _schema_valid(validator: Any, text: str) -> Any:
    try:
        ref.parse_json(text)
    except ref.ContractError:
        return None
    return validator.is_valid(json.loads(text))


def _override_catalog() -> Dict[str, Any]:
    family = {
        **{name: "sigmoid" for name in ("daily_freezethaw_cycles", "frost_days", "rx1day", "cyclone", "wind_max_daily_max", "fwi")},
        "hot_days": "hot_days",
        **{name: "flood" for name in ("cflood", "rflood", "inundation")},
    }
    factors: Dict[str, Any] = {}
    for factor, entries in ref.OVERRIDE_CATALOG.items():
        item: Dict[str, Any] = {"overrides": {key: {"nullable": nullable} for key, nullable in entries.items()}}
        if factor in family:
            item["meaning"] = OVERRIDE_MEANINGS[family[factor]]
        if factor == "cooling_degree_days":
            item["branches"] = {"non_agricultural": ["cooling_cost"], "agricultural": []}
        if factor == "maximum_consecutive_dry_days":
            item["branches"] = {"agricultural": []}
        factors[factor] = item
    return {"resolver_semantics_version": ref.RESOLVER_SEMANTICS_VERSION, "factors": factors}


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def build() -> Dict[str, bytes]:
    files: Dict[str, bytes] = {}
    for contract, (name, _) in schemas.SCHEMAS.items():
        document = schemas.all_schemas()[contract]
        _schema_validator(contract)
        files[f"schemas/{name}.schema.json"] = _json(document)
    for path, (contract, identity, build_cases) in FIXTURE_CONTRACTS.items():
        validator = _schema_validator(contract)
        built = build_cases()
        for group in ("accepted", "rejected"):
            for case in built[group]:
                case["schema_valid"] = _schema_valid(validator, case["input_json"])
                if group == "accepted" and case["schema_valid"] is not True:
                    raise AssertionError(f"{path}:{case['name']} is accepted but schema-invalid")
        files[path] = _json(
            {
                "contract": contract,
                "schema": f"schemas/{schemas.SCHEMAS[contract][0]}.schema.json",
                "identity": identity,
                "description": "Accepted cases record exact canonical UTF-8 text and content_hash = 'sha256:' + "
                "hex(sha256(canonical bytes)). Rejected cases contain exactly one defect and record its error code and "
                "JSON Pointer path. schema_valid records whether the JSON Schema alone accepts the input (null when the "
                "text is not strict JSON); semantic rules beyond JSON Schema are listed in x-crc-rules.",
                **built,
            }
        )
    definitions = json.loads(files["fixtures/definitions.json"])
    plan = next(case for case in definitions["accepted"] if case["name"] == "plan-example-tabulated")
    if plan["canonical_json"] != cases.PLAN_EXAMPLE_CANONICAL:
        raise AssertionError("reference encoder disagrees with the hand-written plan example bytes")
    integer = next(case for case in definitions["accepted"] if case["name"] == "linear-integer-spelling")
    floats = next(case for case in definitions["accepted"] if case["name"] == "linear-float-spelling")
    if integer["content_hash"] != floats["content_hash"]:
        raise AssertionError("numeric spellings of one function must share identity")
    files["fixtures/canonical-json.json"] = _json(
        {
            "profile": ref.CANONICAL_PROFILE,
            "description": "Number vectors: input JSON number text, its binary64 bits (big-endian hex) and the canonical text. "
            "String vectors: the canonical JSON string literal and its UTF-8 bytes.",
            "numbers": cases.number_cases(),
            "strings": cases.string_cases(),
            "rejected": cases.parser_rejection_cases(),
        }
    )
    files["fixtures/point-evaluations.json"] = _json(
        {
            "description": "Point evaluations of normalized definitions. A case has expected or error; x_special is a non-finite "
            "exposure that JSON cannot spell. Compare with |actual - expected| <= absolute + relative * |expected|.",
            "fixtures": cases.point_cases(),
        }
    )
    files["fixtures/monotonicity.json"] = _json(
        {
            "description": "Certified monotonicity of the full function including thresholds and clamp. reference is the most "
            "specific class proven by the reference; an implementation must return a member of acceptable.",
            "fixtures": cases.monotonicity_cases(),
        }
    )
    files["fixtures/operations.json"] = _json(cases.operation_cases())
    fixture_ids = set()
    for _, (contract, _, build_cases) in FIXTURE_CONTRACTS.items():
        prefix = {
            "impact_definition": "definition",
            "registry_impact_reference": "registry-reference",
            "impact_evaluation_record": "evaluation-record",
            "spatial_combination_policy": "spatial-policy",
            "spatial_impact_evaluation_record": "spatial-record",
        }[contract]
        for group in ("accepted", "rejected"):
            fixture_ids |= {f"{prefix}:{case['name']}" for case in build_cases()[group]}
    fixture_ids |= {f"number:{case['name']}" for case in cases.number_cases()}
    fixture_ids |= {f"point:{case['name']}" for case in cases.point_cases()}
    operations = cases.operation_cases()
    fixture_ids |= {f"operation:{case['name']}" for group in ("expected_value", "transform_distribution", "spatial_combination") for case in operations[group]}
    semantics = cases.semantics_constants()
    declared = [entry["id"] for entry in semantics]
    if len(declared) != len(set(declared)):
        raise AssertionError("semantics constants have duplicate ids")
    for entry in semantics:
        if not entry.get("owner") or not entry.get("fixtures"):
            raise AssertionError(f"semantics constant {entry['id']} lacks owner or fixtures")
        missing = set(entry["fixtures"]) - fixture_ids
        if missing:
            raise AssertionError(f"semantics constant {entry['id']} references missing fixtures: {sorted(missing)}")
    files["catalogs/semantics-constants.json"] = _json({"semantics_version": ref.IMPACT_SEMANTICS_VERSION, "constants": semantics})
    files["catalogs/native-capabilities.json"] = _json(cases.native_capabilities())
    files["catalogs/registry-overrides.json"] = _json(_override_catalog())
    files["catalogs/error-codes.json"] = _json(
        {"codes": {code: {"stage": stage, "meaning": meaning} for code, (stage, meaning) in ERROR_CODES.items()}}
    )
    files["catalogs/unit-aliases.json"] = _json(UNIT_ALIASES)
    used = {
        case["error"]["code"]
        for path in FIXTURE_CONTRACTS
        for case in json.loads(files[path])["rejected"]
    }
    unknown = used - set(ERROR_CODES)
    if unknown:
        raise AssertionError(f"fixtures use undocumented error codes: {sorted(unknown)}")
    manifest = {
        "artifact": ARTIFACT,
        "artifact_version": ARTIFACT_VERSION,
        "canonical_profile": ref.CANONICAL_PROFILE,
        "contracts": {
            contract: {"schema": f"schemas/{name}.schema.json", "fixtures": next(p for p, v in FIXTURE_CONTRACTS.items() if v[0] == contract)}
            for contract, (name, _) in schemas.SCHEMAS.items()
        },
        "versions": {
            "impact_definition_schema": ref.DEFINITION_SCHEMA_VERSION,
            "impact_semantics": ref.IMPACT_SEMANTICS_VERSION,
            "registry_reference_schema": ref.REFERENCE_SCHEMA_VERSION,
            "resolver_semantics": ref.RESOLVER_SEMANTICS_VERSION,
            "evaluation_record_schema": ref.RECORD_SCHEMA_VERSION,
            "spatial_policy_schema": ref.POLICY_SCHEMA_VERSION,
            "spatial_semantics": ref.SPATIAL_SEMANTICS_VERSION,
            "spatial_record_schema": ref.SPATIAL_RECORD_SCHEMA_VERSION,
        },
        "files": [
            {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            for path, content in sorted(files.items())
        ],
    }
    files["manifest.json"] = _json(manifest)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the committed artifact differs")
    args = parser.parse_args()
    files = build()
    stale = []
    for path, content in files.items():
        target = OUTPUT / path
        if args.check:
            if not target.exists() or target.read_bytes() != content:
                stale.append(path)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    existing = {p.relative_to(OUTPUT).as_posix() for p in OUTPUT.rglob("*") if p.is_file() and p.suffix == ".json"} if OUTPUT.exists() else set()
    extra = sorted(existing - set(files))
    if args.check and (stale or extra):
        print("stale impact contract files:", *stale, *extra, sep="\n  ", file=sys.stderr)
        return 1
    if not args.check:
        for path in extra:
            (OUTPUT / path).unlink()
        print(f"wrote {len(files)} files to {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

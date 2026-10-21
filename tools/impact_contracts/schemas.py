"""JSON Schema (draft 2020-12) sources for the impact contract artifact.

Every property carries its meaning (``description``), canonical form
(``x-crc-canonical``) and native mapping (``x-crc-native``). Optional
properties also declare a ``default``. Cross-field rules that JSON Schema
cannot express are listed in each object's ``x-crc-rules`` annotation and are
exercised by the semantic conformance fixtures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from reference import (
    FALLBACK_POLICIES,
    OVERRIDE_CATALOG,
    OVERRIDE_KEYS,
    REGISTRY_FACTORS,
    REPLAY_REQUIREMENTS,
    SPATIAL_GEOMETRY_TOLERANCE,
    SOURCE_KINDS,
    TABULATION_INTERPOLATIONS,
    U64_MAX,
)

DIALECT = "https://json-schema.org/draft/2020-12/schema"
ID_PREFIX = "https://schemas.riskthinking.ai/crc/impact/1.0.0/"

_NO_DEFAULT = object()

NUMBER = "binary64 shortest round-trip number; negative zero emitted as 0.0"
INTEGER = "JSON integer without fraction or exponent"
STRING = "string preserved exactly; JSON escapes per the canonical profile"
ALWAYS = "always present in canonical form"


def field(
    schema: Dict[str, Any],
    meaning: str,
    canonical: str,
    native: str,
    default: Any = _NO_DEFAULT,
) -> Dict[str, Any]:
    result = dict(schema)
    result["description"] = meaning
    result["x-crc-canonical"] = canonical
    result["x-crc-native"] = native
    if default is not _NO_DEFAULT:
        result["default"] = default
    return result


def obj(
    description: str,
    properties: Dict[str, Dict[str, Any]],
    required: Sequence[str],
    rules: Sequence[str] = (),
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "type": "object",
        "description": description,
        "additionalProperties": False,
        "properties": properties,
        "required": list(required),
    }
    if rules:
        result["x-crc-rules"] = list(rules)
    return result


def ref(name: str) -> Dict[str, Any]:
    return {"$ref": f"#/$defs/{name}"}


def nullable(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def const(value: str) -> Dict[str, Any]:
    return {"type": "string", "const": value}


def enum(values: Sequence[str]) -> Dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def kind(value: str, native: str) -> Dict[str, Any]:
    return field(const(value), f"Variant tag {value!r}.", STRING, native)


COMMON_DEFS: Dict[str, Dict[str, Any]] = {
    "identifier": {
        "type": "string",
        "minLength": 1,
        "pattern": "^\\S(?:[\\s\\S]*\\S)?$",
        "description": "Non-empty, case-sensitive string without surrounding whitespace.",
    },
    "text": {"type": "string", "minLength": 1, "description": "Non-empty free text."},
    "number": {
        "type": "number",
        "description": "JSON number converted once to finite IEEE binary64; booleans and strings are rejected.",
    },
    "non_negative_number": {"type": "number", "minimum": 0},
    "positive_number": {"type": "number", "exclusiveMinimum": 0},
    "probability": {"type": "number", "minimum": 0, "maximum": 1},
    "count": {"type": "integer", "minimum": 0, "maximum": U64_MAX},
    "positive_count": {"type": "integer", "minimum": 1, "maximum": U64_MAX},
    "sha256": {
        "type": "string",
        "pattern": "^[0-9a-f]{64}$",
        "description": "Bare SHA-256 digest as 64 lowercase hexadecimal characters.",
    },
    "content_hash": {
        "type": "string",
        "pattern": "^sha256:[0-9a-f]{64}$",
        "description": "Prefixed content identity: sha256 of the canonical JSON bytes.",
    },
    "h3_cell": {
        "type": "string",
        "pattern": "^[0-9a-f]{15}$",
        "description": "H3 cell index as 15 lowercase hexadecimal characters.",
    },
}


def _context() -> Dict[str, Any]:
    native = "ImpactContext::{}"
    return obj(
        "Registry resolution context. Every member is nullable and defaults to null; "
        "only context used by the resolved branch carries values in a frozen origin.",
        {
            "cell": field(
                nullable(ref("h3_cell")),
                "H3 cell of the matched hazard row.",
                f"{STRING}; {ALWAYS}",
                "ImpactContext::cell: Option<u64> (validated H3 index)",
                None,
            ),
            "country": field(
                nullable(ref("identifier")),
                "Country key, normalized only by the pinned registry manifest.",
                f"{STRING}; {ALWAYS}",
                native.format("country: Option<String>"),
                None,
            ),
            "continent": field(
                nullable(ref("identifier")),
                "Continent key used by flood tables.",
                f"{STRING}; {ALWAYS}",
                native.format("continent: Option<String>"),
                None,
            ),
            "building_type": field(
                nullable(ref("identifier")),
                "Building type key used by flood, SPEI and baseline-relative tables.",
                f"{STRING}; {ALWAYS}",
                native.format("building_type: Option<String>"),
                None,
            ),
            "historic_mean": field(
                nullable(ref("number")),
                "Historical baseline exposure for baseline-relative factors.",
                f"{NUMBER}; {ALWAYS}",
                native.format("historic_mean: Option<f64>"),
                None,
            ),
        },
        [],
    )


def _overrides() -> Dict[str, Any]:
    meanings = {
        "midpoint": "Sigmoid midpoint for sigmoid registry factors.",
        "steepness": "Sigmoid steepness for sigmoid registry factors.",
        "minimum_damage": "Sigmoid lower curve level (maps to offset), or the hot-days lower clamp; null clears the hot-days clamp.",
        "maximum_damage": "Sigmoid upper curve level (offset + amplitude), hot-days upper clamp (nullable), or flood table scale factor.",
        "zero_below": "Strict sigmoid threshold below which the response is zero; null clears it.",
        "slope": "Hot-days linear slope.",
        "intercept": "Hot-days linear intercept.",
        "water_unit_cost": "SPEI original water unit cost.",
        "cooling_cost": "Cooling cost for non-agricultural cooling-degree-days branches.",
    }
    nullable_keys = {key for entries in OVERRIDE_CATALOG.values() for key, allowed in entries.items() if allowed}
    properties = {}
    for key in OVERRIDE_KEYS:
        schema = nullable(ref("number")) if key in nullable_keys else ref("number")
        properties[key] = field(
            schema,
            meanings[key],
            f"{NUMBER}; present only when supplied",
            f"RegistryOverrides::{key}: Override<f64>",
        )
        properties[key]["x-crc-absent"] = "Absence is meaningful: the pinned registry default applies."
    return obj(
        "Typed registry overrides. Absent keys use the pinned registry default and are never materialized.",
        properties,
        [],
        [
            "Keys must be accepted for the selected factor (see catalogs/registry-overrides.json); otherwise inapplicable_override.",
            "Explicit null is accepted only where the catalog marks the key nullable for the factor; otherwise invalid_value.",
            "After context resolution, keys must also be accepted for the resolved branch.",
        ],
    )


def _lookup() -> Dict[str, Any]:
    key = {
        "oneOf": [
            ref("identifier"),
            {
                "type": "object",
                "minProperties": 1,
                "propertyNames": ref("identifier"),
                "additionalProperties": ref("identifier"),
            },
        ]
    }
    return obj(
        "One registry table lookup performed during resolution.",
        {
            "table_id": field(ref("identifier"), "Manifest table identifier.", STRING, "RegistryLookup::table_id: String"),
            "table_sha256": field(ref("sha256"), "Digest of the table content in the pinned manifest.", STRING, "RegistryLookup::table_sha256: [u8; 32]"),
            "requested_key": field(key, "Key requested after manifest normalization; composite keys use named string fields.", "string, or object with sorted keys", "RegistryLookup::requested_key: LookupKey"),
            "resolved_key": field(key, "Key actually used; differs from requested_key only for fallbacks.", "string, or object with sorted keys", "RegistryLookup::resolved_key: LookupKey"),
            "status": field(enum(("exact", "fallback")), "Whether the requested key was found or a documented fallback applied.", STRING, "RegistryLookup::status: LookupStatus"),
        },
        ["table_id", "table_sha256", "requested_key", "resolved_key", "status"],
    )


def _registry_identity(owner: str) -> Dict[str, Dict[str, Any]]:
    return {
        "registry_id": field(ref("identifier"), "Registry identity.", STRING, f"{owner}::registry_id: String"),
        "registry_revision": field(ref("identifier"), "Pinned registry revision; never 'latest' by omission.", STRING, f"{owner}::registry_revision: String"),
        "registry_manifest_sha256": field(ref("sha256"), "Digest of the pinned registry manifest; resolution rejects a different manifest.", STRING, f"{owner}::registry_manifest_sha256: [u8; 32]"),
        "resolver_semantics_version": field(const("crc-impact-registry/1"), "Resolver semantics: override catalog, context precedence, fallbacks and key normalization.", STRING, f"{owner}::resolver_semantics_version: ResolverSemantics"),
        "factor": field(enum(REGISTRY_FACTORS), "Registry factor.", STRING, f"{owner}::factor: RegistryFactor"),
    }


def _document(name: str, title: str, description: str, root: Dict[str, Any], defs: Dict[str, Any]) -> Dict[str, Any]:
    document: Dict[str, Any] = {
        "$schema": DIALECT,
        "$id": f"{ID_PREFIX}{name}.schema.json",
        "title": title,
    }
    document.update(root)
    document["description"] = description
    document["$defs"] = {**COMMON_DEFS, **defs}
    return document


# Impact definition ---------------------------------------------------------


def _crop_vector(minimum: Optional[float]) -> Dict[str, Any]:
    number = ref("non_negative_number") if minimum == 0 else ref("number")
    return obj(
        "Per-crop coefficient vector with exactly maize, rice and wheat.",
        {
            crop: field(number, f"{crop.capitalize()} coefficient.", NUMBER, f"CropValues::{crop}: f64")
            for crop in ("maize", "rice", "wheat")
        },
        ["maize", "rice", "wheat"],
    )


def _models() -> Dict[str, Dict[str, Any]]:
    domain = field(
        enum(("clamp", "error")),
        "Below or above the table domain: clamp returns the nearest endpoint response; error rejects the exposure.",
        f"{STRING}; {ALWAYS}",
        "OutsideDomain",
        "clamp",
    )
    return {
        "model_linear": obj(
            "raw(x) = slope * x + intercept.",
            {
                "kind": kind("linear", "ModelSpec::Linear"),
                "slope": field(ref("number"), "Slope; positive, zero or negative.", NUMBER, "LinearKernel::slope: f64"),
                "intercept": field(ref("number"), "Intercept.", f"{NUMBER}; {ALWAYS}", "LinearKernel::intercept: f64", 0.0),
            },
            ["kind", "slope"],
        ),
        "model_sigmoid": obj(
            "raw(x) = offset + amplitude * (L(steepness * (x - midpoint)) - [L(-steepness * midpoint) if subtract_at_zero]); "
            "raw(x) = 0 when zero_below is present and x < zero_below.",
            {
                "kind": kind("sigmoid", "ModelSpec::Sigmoid"),
                "midpoint": field(ref("number"), "Logistic midpoint.", NUMBER, "SigmoidKernel::midpoint: f64"),
                "steepness": field(ref("number"), "Logistic steepness; zero gives a constant response.", NUMBER, "SigmoidKernel::steepness: f64"),
                "offset": field(ref("number"), "Lower curve level (legacy minimum).", f"{NUMBER}; {ALWAYS}", "SigmoidKernel::offset: f64", 0.0),
                "amplitude": field(ref("non_negative_number"), "Curve height (legacy maximum minus minimum).", f"{NUMBER}; {ALWAYS}", "SigmoidKernel::amplitude: f64", 1.0),
                "origin_adjustment": field(
                    enum(("none", "subtract_at_zero")),
                    "subtract_at_zero subtracts the unit logistic at x = 0 without renormalizing the upper asymptote.",
                    f"{STRING}; {ALWAYS}",
                    "SigmoidKernel::origin_adjustment: OriginAdjustment",
                    "none",
                ),
                "zero_below": field(
                    nullable(ref("number")),
                    "Strict threshold: x < zero_below gives raw response 0; equality evaluates the sigmoid.",
                    f"{NUMBER} or null; {ALWAYS}",
                    "SigmoidKernel::zero_below: Option<f64>",
                    None,
                ),
            },
            ["kind", "midpoint", "steepness"],
        ),
        "model_tabulated": obj(
            "Response table with strictly increasing x knots. Between knots: linear interpolates, step_left returns y_i, step_right returns y_(i+1); every mode returns y_i at x_i.",
            {
                "kind": kind("tabulated", "ModelSpec::Tabulated"),
                "points": field(
                    {"type": "array", "minItems": 2, "items": ref("point")},
                    "Ordered knots; order is preserved and never sorted.",
                    "array order preserved",
                    "TabulatedKernel::points: Arc<[Point]>",
                ),
                "interpolation": field(
                    enum(("linear", "step_left", "step_right")),
                    "Interpolation between adjacent knots.",
                    f"{STRING}; {ALWAYS}",
                    "TabulatedKernel::interpolation: TableInterpolation",
                    "linear",
                ),
                "outside_domain": domain,
            },
            ["kind", "points"],
            ["Knot x values strictly increase."],
        ),
        "point": obj(
            "One tabulated knot.",
            {
                "x": field(ref("number"), "Exposure in the calibrated exposure unit.", NUMBER, "Point::x: f64"),
                "y": field(ref("number"), "Raw response at x.", NUMBER, "Point::y: f64"),
            },
            ["x", "y"],
        ),
        "model_binned": obj(
            "Interval lookup over [edge_i, edge_(i+1)); the final interval includes the final edge and interior edges select the interval to their right.",
            {
                "kind": kind("binned", "ModelSpec::Binned"),
                "bin_edges": field(
                    {"type": "array", "minItems": 2, "items": ref("number")},
                    "Strictly increasing interval edges.",
                    "array order preserved",
                    "BinnedKernel::edges: Arc<[f64]>",
                ),
                "values": field(
                    {"type": "array", "minItems": 1, "items": ref("number")},
                    "One finite response per interval.",
                    "array order preserved",
                    "BinnedKernel::values: Arc<[f64]>",
                ),
                "outside_domain": domain,
            },
            ["kind", "bin_edges", "values"],
            ["bin_edges strictly increase.", "len(values) == len(bin_edges) - 1."],
        ),
        "model_spei": obj(
            "Registry snapshot: water-cost response with the fixed crc-impact/1 SPEI constants.",
            {
                "kind": kind("spei", "ModelSpec::Spei"),
                "water_consumption_asset": field(ref("non_negative_number"), "Asset water consumption.", NUMBER, "SpeiKernel::water_consumption_asset: f64"),
                "original_water_unit_cost": field(ref("non_negative_number"), "Water unit cost before drought scaling.", NUMBER, "SpeiKernel::original_water_unit_cost: f64"),
            },
            ["kind", "water_consumption_asset", "original_water_unit_cost"],
        ),
        "model_cooling": obj(
            "Registry snapshot: raw = 0 for x <= 1, else cooling_cost * x / (historic_mean * 10_000_000).",
            {
                "kind": kind("cooling", "ModelSpec::Cooling"),
                "cooling_cost": field(ref("non_negative_number"), "Cooling cost coefficient.", NUMBER, "CoolingKernel::cooling_cost: f64"),
                "historic_mean": field(ref("positive_number"), "Historical baseline exposure.", NUMBER, "CoolingKernel::historic_mean: f64"),
            },
            ["kind", "cooling_cost", "historic_mean"],
        ),
        "model_crop": obj(
            "Registry snapshot: weighted historical/future crop response with reference year 2010.",
            {
                "kind": kind("crop", "ModelSpec::Crop"),
                "weights": field(ref("crop_weights"), "Non-negative crop weights with positive total; never renormalized.", "object with sorted keys", "CropKernel::weights: CropValues"),
                "country": field(ref("crop_vector"), "Country coefficients.", "object with sorted keys", "CropKernel::country: CropValues"),
                "yearly": field(ref("crop_vector"), "Yearly trend coefficients.", "object with sorted keys", "CropKernel::yearly: CropValues"),
                "index": field(ref("crop_vector"), "Exposure index coefficients.", "object with sorted keys", "CropKernel::index: CropValues"),
                "historic_mean": field(ref("number"), "Historical baseline exposure.", NUMBER, "CropKernel::historic_mean: f64"),
            },
            ["kind", "weights", "country", "yearly", "index", "historic_mean"],
            ["The sum of crop weights is positive."],
        ),
        "crop_vector": _crop_vector(None),
        "crop_weights": _crop_vector(0),
    }


def _origins() -> Dict[str, Dict[str, Any]]:
    source_id = field(ref("identifier"), "Identifier of a source record in this definition.", STRING, "Origin::source_id: String")
    return {
        "origin_user_defined": obj(
            "Caller-authored coefficients.",
            {"kind": kind("user_defined", "Origin::UserDefined")},
            ["kind"],
        ),
        "origin_published": obj(
            "Coefficients taken from a cited source.",
            {"kind": kind("published", "Origin::Published"), "source_id": source_id},
            ["kind", "source_id"],
            ["source_id names a record in provenance.sources."],
        ),
        "origin_imported": obj(
            "Coefficients produced by an importer from a cited source.",
            {
                "kind": kind("imported", "Origin::Imported"),
                "source_id": source_id,
                "importer_name": field(ref("identifier"), "Importer name.", STRING, "Origin::Imported::importer_name: String"),
                "importer_version": field(ref("identifier"), "Importer version.", STRING, "Origin::Imported::importer_version: String"),
                "importer_build_sha256": field(
                    nullable(ref("sha256")),
                    "Importer build digest when known.",
                    f"{STRING}; {ALWAYS}",
                    "Origin::Imported::importer_build_sha256: Option<[u8; 32]>",
                    None,
                ),
            },
            ["kind", "source_id", "importer_name", "importer_version"],
            ["source_id names a record in provenance.sources."],
        ),
        "origin_derived": obj(
            "Coefficients derived from parent definitions.",
            {
                "kind": kind("derived", "Origin::Derived"),
                "description": field(ref("text"), "Description of the derivation.", STRING, "Origin::Derived::description: String"),
            },
            ["kind", "description"],
            ["provenance.derived_from contains at least one parent."],
        ),
        "origin_registry": obj(
            "Complete registry resolution trace. Contains only context used to construct the function, never evaluated exposures.",
            {
                "kind": kind("registry", "Origin::Registry"),
                **_registry_identity("RegistryOrigin"),
                "context": field(ref("context"), "Effective context used by the resolved branch; unused members are null.", "object with all members present", "RegistryOrigin::context: ImpactContext"),
                "overrides": field(ref("overrides"), "Supplied override keys after type normalization; resolved defaults are not listed.", "object with sorted keys; absent keys omitted", "RegistryOrigin::overrides: RegistryOverrides"),
                "fallback_policy": field(enum(FALLBACK_POLICIES), "Fallback policy used during resolution.", STRING, "RegistryOrigin::fallback_policy: FallbackPolicy"),
                "lookups": field({"type": "array", "items": ref("lookup")}, "Every table lookup in resolution order.", "array order preserved", "RegistryOrigin::lookups: Vec<RegistryLookup>"),
            },
            ["kind", "registry_id", "registry_revision", "registry_manifest_sha256", "resolver_semantics_version", "factor", "context", "overrides", "fallback_policy", "lookups"],
            ["A fallback lookup requires fallback_policy legacy_defaults."],
        ),
    }


def definition_schema() -> Dict[str, Any]:
    models = _models()
    defs: Dict[str, Any] = {
        **models,
        **_origins(),
        "context": _context(),
        "overrides": _overrides(),
        "lookup": _lookup(),
        "calibration": obj(
            "Units and meaning of the function's argument and response. Never selects a kernel or implies a clamp.",
            {
                "exposure_variable": field(ref("identifier"), "Exposure variable, preferably a canonical hazard or quantity identifier.", STRING, "Calibration::exposure_variable: String"),
                "exposure_unit": field(ref("identifier"), "Declared exposure unit token; raw arrays are interpreted in this unit.", STRING, "Calibration::exposure_unit: String"),
                "impact_metric": field(ref("identifier"), "Response metric.", STRING, "Calibration::impact_metric: String"),
                "impact_unit": field(ref("identifier"), "Declared response unit token.", STRING, "Calibration::impact_unit: String"),
            },
            ["exposure_variable", "exposure_unit", "impact_metric", "impact_unit"],
        ),
        "clamp": obj(
            "Final numerical limits applied once after the raw model and its thresholds.",
            {
                "minimum": field(nullable(ref("number")), "Lower limit.", f"{NUMBER} or null; {ALWAYS} when the clamp is not null", "Clamp::minimum: Option<f64>", None),
                "maximum": field(nullable(ref("number")), "Upper limit.", f"{NUMBER} or null; {ALWAYS} when the clamp is not null", "Clamp::maximum: Option<f64>", None),
            },
            [],
            ["minimum <= maximum when both are present.", "A clamp with both members null normalizes to null."],
        ),
        "source": obj(
            "Citation or evidence record.",
            {
                "id": field(ref("identifier"), "Identifier unique within the definition.", STRING, "SourceRecord::id: String"),
                "kind": field(enum(SOURCE_KINDS), "Source category.", STRING, "SourceRecord::kind: SourceKind"),
                "uri": field(ref("identifier"), "Citation URI; a URI alone is not immutable evidence.", STRING, "SourceRecord::uri: String"),
                "revision": field(nullable(ref("identifier")), "Source revision when known.", f"{STRING}; {ALWAYS}", "SourceRecord::revision: Option<String>", None),
                "content_sha256": field(nullable(ref("sha256")), "Digest of the cited content when retained.", f"{STRING}; {ALWAYS}", "SourceRecord::content_sha256: Option<[u8; 32]>", None),
            },
            ["id", "kind", "uri"],
        ),
        "provenance": obj(
            "Definition origin, evidence and parents. Audit-only: never changes evaluation, always changes identity.",
            {
                "origin": field(
                    {"oneOf": [ref(name) for name in ("origin_user_defined", "origin_published", "origin_imported", "origin_derived", "origin_registry")]},
                    "Tagged origin of the coefficients.",
                    "tagged object with all variant members present",
                    "Provenance::origin: Origin",
                ),
                "sources": field({"type": "array", "items": ref("source")}, "Citation and evidence records.", f"array order preserved; {ALWAYS}", "Provenance::sources: Arc<[SourceRecord]>", []),
                "derived_from": field(
                    {"type": "array", "uniqueItems": True, "items": ref("content_hash")},
                    "Parent definition hashes; audit references that evaluation never retrieves.",
                    f"sorted ascending, duplicate-free; {ALWAYS}",
                    "Provenance::derived_from: BTreeSet<DefinitionHash>",
                    [],
                ),
            },
            ["origin"],
            ["Source ids are unique.", "derived_from contains no duplicates."],
        ),
    }
    model_variants = [ref(name) for name in models if name.startswith("model_")]
    root = obj(
        "",
        {
            "schema_version": field(const("1.0"), "Wire format and validation rules.", STRING, "ImpactDefinition::schema_version"),
            "semantics_version": field(const("crc-impact/1"), "Equations, fixed constants, boundary rules and normalization.", STRING, "ImpactDefinition::semantics_version"),
            "calibration": field(ref("calibration"), "Declared units and meaning.", "object with sorted keys", "ImpactDefinition::calibration: Calibration"),
            "model": field({"oneOf": model_variants}, "Tagged numerical model containing every coefficient needed for evaluation.", "tagged object with all defaults materialized", "ImpactDefinition::model: ModelSpec"),
            "clamp": field(
                nullable(ref("clamp")),
                "Optional final clamp; omitted or null means no clamp.",
                f"null or object with both members; {ALWAYS}",
                "ImpactDefinition::clamp: Option<Clamp>",
                None,
            ),
            "provenance": field(ref("provenance"), "Origin, sources and parent definitions.", "object with sorted keys", "ImpactDefinition::provenance: Provenance"),
        },
        ["schema_version", "semantics_version", "calibration", "model", "provenance"],
    )
    return _document(
        "impact-definition-1.0",
        "CRC impact definition 1.0",
        "Immutable, self-contained impact function: model, calibration, final clamp and provenance. "
        "Contains no exposure data, aggregation, output or spatial settings. definition_hash is sha256 of the "
        "crc-canonical-json/1 encoding of the normalized definition and is never embedded in it.",
        root,
        defs,
    )


def reference_schema() -> Dict[str, Any]:
    root = obj(
        "",
        {
            "schema_version": field(const("1.0"), "Reference wire format.", STRING, "RegistryImpactReference::schema_version"),
            **_registry_identity("RegistryImpactReference"),
            "overrides": field(ref("overrides"), "Typed overrides; absent keys use pinned registry defaults.", f"object with sorted keys; absent keys omitted; {ALWAYS}", "RegistryImpactReference::overrides: RegistryOverrides", {}),
            "context_defaults": field(ref("context"), "Lowest-precedence context values.", f"object with all members present; {ALWAYS}", "RegistryImpactReference::context_defaults: ImpactContext", {}),
            "fallback_policy": field(enum(FALLBACK_POLICIES), "error rejects missing table keys; legacy_defaults permits and records documented fallbacks.", f"{STRING}; {ALWAYS}", "RegistryImpactReference::fallback_policy: FallbackPolicy", "error"),
        },
        ["schema_version", "registry_id", "registry_revision", "registry_manifest_sha256", "resolver_semantics_version", "factor"],
    )
    return _document(
        "registry-impact-reference-1.0",
        "CRC registry impact reference 1.0",
        "Pinned recipe that resolves to a self-contained impact definition. reference_hash uses the definition "
        "canonicalization and hashing rules and is never embedded in the payload.",
        root,
        {"context": _context(), "overrides": _overrides()},
    )


# Evaluation records --------------------------------------------------------


def _record_defs() -> Dict[str, Any]:
    digest = obj(
        "Domain-separated content digest of a canonical snapshot encoding.",
        {
            "encoding": field(ref("identifier"), "Versioned encoding prefix included in the hash preimage, for example crc-array-f64le/1 or raw-bytes.", STRING, "ContentDigest::encoding: String"),
            "sha256": field(ref("sha256"), "Digest of the encoded bytes.", STRING, "ContentDigest::sha256: [u8; 32]"),
        },
        ["encoding", "sha256"],
    )
    snapshot = obj(
        "Immutable snapshot reference. Hashing alone does not retain data.",
        {
            "uri": field(ref("identifier"), "Location of the snapshot.", STRING, "SnapshotRef::uri: String"),
            "revision": field(nullable(ref("identifier")), "Snapshot revision when versioned.", f"{STRING}; {ALWAYS}", "SnapshotRef::revision: Option<String>", None),
            "content_sha256": field(ref("sha256"), "Digest of the snapshot bytes.", STRING, "SnapshotRef::content_sha256: [u8; 32]"),
            "retention": field(enum(("retained", "external")), "retained: written by the workflow; external: the caller must preserve it.", STRING, "SnapshotRef::retention: Retention"),
        },
        ["uri", "content_sha256", "retention"],
    )
    selection = obj(
        "Applied row, column and scenario selection.",
        {
            "dataset": field(nullable(ref("identifier")), "Dataset identity.", f"{STRING}; {ALWAYS}", "Selection::dataset: Option<String>", None),
            "columns": field({"type": "array", "items": ref("identifier")}, "Selected columns.", f"array order preserved; {ALWAYS}", "Selection::columns: Vec<String>", []),
            "rows": field(nullable(ref("text")), "Row selection description.", f"{STRING}; {ALWAYS}", "Selection::rows: Option<String>", None),
            "scenario": field(nullable(ref("identifier")), "Scenario.", f"{STRING}; {ALWAYS}", "Selection::scenario: Option<String>", None),
            "horizon": field(nullable(ref("identifier")), "Horizon.", f"{STRING}; {ALWAYS}", "Selection::horizon: Option<String>", None),
        },
        [],
    )
    tracked = {
        "content_digest": field(nullable(ref("digest")), "Digest of the evaluated data in its canonical snapshot encoding.", f"{ALWAYS}", "SourceIdentity::content_digest: Option<ContentDigest>", None),
        "snapshot": field(nullable(ref("snapshot")), "Retained or external snapshot; required for complete replay.", f"{ALWAYS}", "SourceIdentity::snapshot: Option<SnapshotRef>", None),
        "selection": field(nullable(ref("selection")), "Applied selection.", f"{ALWAYS}", "SourceIdentity::selection: Option<Selection>", None),
    }
    curve_scalar = lambda name: field(nullable(ref("number")), f"Canonical curve {name}.", f"{NUMBER} or null; {ALWAYS}", f"CurveParameters::curve_{name}: Option<f64>", None)  # noqa: E731
    curve_array = lambda name: field(nullable({"type": "array", "items": ref("number")}), f"Canonical curve {name}.", f"array order preserved or null; {ALWAYS}", f"CurveParameters::curve_{name}: Option<Vec<f64>>", None)  # noqa: E731
    method_version = field(ref("identifier"), "Versioned numerical method identity.", STRING, "IntegrationMethod::version: String")
    return {
        "digest": digest,
        "snapshot": snapshot,
        "selection": selection,
        "context": _context(),
        "function_definition": obj(
            "Function identified by an impact definition hash.",
            {
                "kind": kind("definition", "FunctionIdentity::Definition"),
                "definition_hash": field(ref("content_hash"), "Definition identity.", STRING, "FunctionIdentity::Definition::definition_hash: DefinitionHash"),
            },
            ["kind", "definition_hash"],
        ),
        "function_custom": obj(
            "Runtime-only custom function; never claims serializable reproducibility.",
            {
                "kind": kind("custom", "FunctionIdentity::Custom"),
                "module": field(nullable(ref("identifier")), "Python module when available.", f"{STRING}; {ALWAYS}", "FunctionIdentity::Custom::module: Option<String>", None),
                "qualified_name": field(nullable(ref("identifier")), "Qualified name when available.", f"{STRING}; {ALWAYS}", "FunctionIdentity::Custom::qualified_name: Option<String>", None),
                "type": field(nullable(ref("identifier")), "Runtime type name when available.", f"{STRING}; {ALWAYS}", "FunctionIdentity::Custom::type_name: Option<String>", None),
            },
            ["kind"],
        ),
        "settings_point": obj(
            "Point evaluation settings.",
            {
                "alignment": field(const("event_aligned"), "Exposures are event-aligned and never sorted.", STRING, "PointSettings::alignment"),
                "shape": field(const("preserve"), "Scalar/array form, shape and order are preserved.", STRING, "PointSettings::shape"),
            },
            ["alignment", "shape"],
        ),
        "method_exact": obj(
            "Exact integration.",
            {"kind": kind("exact", "IntegrationMethod::Exact"), "version": method_version},
            ["kind", "version"],
        ),
        "method_gauss_kronrod": obj(
            "Adaptive Gauss-Kronrod 15/7 quadrature on the probability interval.",
            {
                "kind": kind("gauss_kronrod", "IntegrationMethod::GaussKronrod"),
                "version": method_version,
                "absolute_tolerance": field(ref("positive_number"), "Absolute error tolerance.", NUMBER, "GaussKronrodOptions::absolute_tolerance: f64"),
                "relative_tolerance": field(ref("positive_number"), "Relative error tolerance.", NUMBER, "GaussKronrodOptions::relative_tolerance: f64"),
                "max_evaluations": field(ref("positive_count"), "Maximum integrand evaluations.", INTEGER, "GaussKronrodOptions::max_evaluations: u64"),
            },
            ["kind", "version", "absolute_tolerance", "relative_tolerance", "max_evaluations"],
        ),
        "settings_expected_value": obj(
            "Expectation settings: E[f(X)], never f(E[X]).",
            {
                "requested_method": field(enum(("auto", "exact", "gauss_kronrod")), "Method requested by the caller.", STRING, "ExpectationSettings::requested: MethodRequest"),
                "method": field({"oneOf": [ref("method_exact"), ref("method_gauss_kronrod")]}, "Resolved method with every effective default.", "tagged object", "ExpectationSettings::method: IntegrationMethod"),
            },
            ["requested_method", "method"],
            ["A non-auto request equals the resolved method kind."],
        ),
        "settings_transform_distribution": obj(
            "Distribution pushforward settings.",
            {"algorithm": field(ref("identifier"), "Exact pushforward algorithm identity, for example crc-pushforward/1.", STRING, "TransformSettings::algorithm: String")},
            ["algorithm"],
        ),
        "settings_tabulate": obj(
            "Explicit approximate tabulation of a distribution.",
            {
                "probabilities": field({"type": "array", "minItems": 1, "items": ref("probability")}, "Full probability grid.", "array order preserved", "TabulationSettings::probabilities: Vec<f64>"),
                "interpolation": field(enum(TABULATION_INTERPOLATIONS), "Required interpolation choice.", STRING, "TabulationSettings::interpolation: Interpolation"),
                "approximate": field({"const": True}, "The artifact is an approximation.", "true", "TabulationSettings::approximate"),
            },
            ["probabilities", "interpolation", "approximate"],
            ["probabilities strictly increase."],
        ),
        **{
            f"operation_{name}": obj(
                f"{name} operation.",
                {
                    "kind": kind(name, f"Operation::{name}"),
                    "settings": field(ref(f"settings_{name}"), "Typed operation settings.", "object with sorted keys", f"Operation::{name}::settings"),
                },
                ["kind", "settings"],
            )
            for name in ("point", "expected_value", "transform_distribution", "tabulate")
        },
        "engine": obj(
            "Engine identity.",
            {
                "framework_version": field(ref("identifier"), "Framework package version.", STRING, "EngineIdentity::framework_version: String"),
                "source_revision": field(nullable(ref("identifier")), "Source revision when known.", f"{STRING}; {ALWAYS}", "EngineIdentity::source_revision: Option<String>", None),
                "build_id": field(nullable(ref("identifier")), "Build identity when available.", f"{STRING}; {ALWAYS}", "EngineIdentity::build_id: Option<String>", None),
                "target": field(ref("identifier"), "Target platform triple.", STRING, "EngineIdentity::target: String"),
                "numerical_methods": field(
                    {"type": "object", "propertyNames": ref("identifier"), "additionalProperties": ref("identifier")},
                    "Numerical method name to version.",
                    "object with sorted keys",
                    "EngineIdentity::numerical_methods: BTreeMap<String, String>",
                ),
            },
            ["framework_version", "target", "numerical_methods"],
        ),
        "curve_parameters": obj(
            "Canonical CRC curve snapshot in the existing CurveParameters form.",
            {
                "curve_kind": field(enum(("fitted", "hurdle", "point_mass", "tabulated")), "Canonical curve kind; no-data rows are not exposure distributions.", STRING, "CurveParameters::curve_kind: String"),
                "curve_type": field(ref("identifier"), "Canonical curve type.", STRING, "CurveParameters::curve_type: String"),
                **{f"curve_{name}": curve_scalar(name) for name in ("shape", "location", "scale", "atom_probability", "atom_location")},
                **{f"curve_{name}": curve_array(name) for name in ("probabilities", "values")},
            },
            ["curve_kind", "curve_type"],
            [
                "Curve parameters must reconstruct a supported canonical CRC distribution.",
                "Fitted/hurdle sources require a supported family, location, positive scale, and the family's required shape; Gumbel shape normalizes to null.",
                "Fitted sources contain no atom parameters; hurdle sources require an atom location and probability strictly between zero and one.",
                "Point masses require no shape, zero scale, probability one, and matching curve and atom locations.",
                "Tabulated sources require linear_probability, no scalar parameters, equal knot arrays of length at least two, increasing probabilities in [0,1], and non-decreasing values.",
                "The native reconstructor additionally checks that a hurdle's conditional tail is numerically viable.",
            ],
        ),
        "source_point_array": obj(
            "Event-aligned exposure array.",
            {
                "kind": kind("point_array", "SourceIdentity::PointArray"),
                "shape": field({"type": "array", "items": ref("count")}, "Array shape; empty for scalars.", "array of integers", "SourceIdentity::PointArray::shape: Vec<u64>"),
                **tracked,
            },
            ["kind", "shape"],
        ),
        "source_canonical_distribution": obj(
            "Canonical CRC distribution.",
            {
                "kind": kind("canonical_distribution", "SourceIdentity::CanonicalDistribution"),
                "parameters": field(ref("curve_parameters"), "Curve snapshot.", "object with all members present", "SourceIdentity::CanonicalDistribution::parameters: CurveParameters"),
                **tracked,
            },
            ["kind", "parameters"],
        ),
        "source_histogram": obj(
            "Caller-supplied uniform-within-bin histogram; edges and weights live in the snapshot.",
            {
                "kind": kind("histogram", "SourceIdentity::Histogram"),
                "interval_count": field(ref("positive_count"), "Number of intervals.", INTEGER, "SourceIdentity::Histogram::interval_count: u64"),
                "within_bin": field(const("uniform"), "Within-bin interpretation.", STRING, "WithinBin::Uniform"),
                "original_mass_sum": field(ref("positive_number"), "Neumaier sum of weights in input-bin order before normalization.", NUMBER, "HistogramMass::original_sum: f64"),
                "normalization_factor": field(ref("positive_number"), "1 / original_mass_sum, reported for audit; weights are divided by original_mass_sum once.", NUMBER, "HistogramMass::normalization_factor: f64"),
                **tracked,
            },
            ["kind", "interval_count", "within_bin", "original_mass_sum", "normalization_factor"],
            ["|original_mass_sum - 1| <= 1e-12.", "normalization_factor equals binary64 1 / original_mass_sum."],
        ),
        "source_untracked": obj(
            "External data with no retained identity.",
            {
                "kind": kind("untracked", "SourceIdentity::Untracked"),
                "description": field(nullable(ref("text")), "Caller description.", f"{STRING}; {ALWAYS}", "SourceIdentity::Untracked::description: Option<String>", None),
            },
            ["kind"],
        ),
        "preprocessing_step": obj(
            "One explicit caller conversion or normalization, in application order.",
            {
                "operation": field(ref("identifier"), "Operation identity, for example unit_conversion.", STRING, "PreprocessingStep::operation: String"),
                "description": field(nullable(ref("text")), "Description.", f"{STRING}; {ALWAYS}", "PreprocessingStep::description: Option<String>", None),
                "parameters": field(
                    {"type": "object", "propertyNames": ref("identifier"), "additionalProperties": {"type": ["string", "number", "boolean", "null"]}},
                    "Scalar operation parameters.",
                    f"object with sorted keys; {ALWAYS}",
                    "PreprocessingStep::parameters: BTreeMap<String, Scalar>",
                    {},
                ),
                "input_digest": field(nullable(ref("digest")), "Input data identity.", ALWAYS, "PreprocessingStep::input_digest: Option<ContentDigest>", None),
                "output_digest": field(nullable(ref("digest")), "Resulting data identity.", ALWAYS, "PreprocessingStep::output_digest: Option<ContentDigest>", None),
            },
            ["operation"],
        ),
        "resolution": obj(
            "Registry resolution performed for this application.",
            {
                "reference_hash": field(ref("content_hash"), "Registry reference identity.", STRING, "Resolution::reference_hash: ReferenceHash"),
                "context": field(ref("context"), "Effective merged context.", "object with all members present", "Resolution::context: ImpactContext"),
                "context_hash": field(ref("content_hash"), "Identity of the canonical effective context.", STRING, "Resolution::context_hash: ContentHash"),
            },
            ["reference_hash", "context", "context_hash"],
            ["context_hash equals the content hash of the canonical context."],
        ),
        "component": obj(
            "Exact or numerical subcomponent of a result.",
            {
                "kind": field(enum(("exact", "numerical")), "Component method class.", STRING, "Component::kind"),
                "value": field(ref("number"), "Component value.", NUMBER, "Component::value: f64"),
                "estimated_error": field(nullable(ref("non_negative_number")), "Estimated numerical error.", f"{NUMBER} or null; {ALWAYS}", "Component::estimated_error: Option<f64>", None),
                "evaluations": field(nullable(ref("count")), "Integrand evaluations.", f"{INTEGER} or null; {ALWAYS}", "Component::evaluations: Option<u64>", None),
            },
            ["kind", "value"],
        ),
        "diagnostics": obj(
            "Method diagnostics; no invented statistical confidence.",
            {
                "exactness": field(enum(("direct", "exact", "numerical", "mixed")), "direct point evaluation, exact integration, numerical integration, or a mix.", STRING, "Diagnostics::exactness"),
                "estimated_error": field(nullable(ref("non_negative_number")), "Estimated numerical error.", f"{NUMBER} or null; {ALWAYS}", "Diagnostics::estimated_error: Option<f64>", None),
                "evaluations": field(nullable(ref("count")), "Total integrand evaluations.", f"{INTEGER} or null; {ALWAYS}", "Diagnostics::evaluations: Option<u64>", None),
                "components": field({"type": "array", "items": ref("component")}, "Exact and numerical subcomponents.", f"array order preserved; {ALWAYS}", "Diagnostics::components: Vec<Component>", []),
            },
            ["exactness"],
        ),
        "replay": obj(
            "Replay completeness.",
            {
                "status": field(enum(("complete", "missing_source_snapshot", "runtime_only")), "Replay status.", STRING, "Replay::status"),
                "missing_requirements": field(
                    {"type": "array", "uniqueItems": True, "items": enum(REPLAY_REQUIREMENTS)},
                    "Requirements missing for complete replay.",
                    "sorted ascending, duplicate-free",
                    "Replay::missing_requirements: BTreeSet<ReplayRequirement>",
                ),
            },
            ["status", "missing_requirements"],
            [
                "complete if and only if missing_requirements is empty.",
                "missing_source_snapshot lists source_snapshot.",
            ],
        ),
    }


def record_schema() -> Dict[str, Any]:
    defs = _record_defs()
    root = obj(
        "",
        {
            "schema_version": field(const("1.0"), "Evaluation-record format.", STRING, "ImpactEvaluationRecord::schema_version"),
            "function": field({"oneOf": [ref("function_definition"), ref("function_custom")]}, "Function identity.", "tagged object", "ImpactEvaluationRecord::function: FunctionIdentity"),
            "operation": field(
                {"oneOf": [ref(f"operation_{name}") for name in ("point", "expected_value", "transform_distribution", "tabulate")]},
                "Operation and its typed settings.",
                "tagged object",
                "ImpactEvaluationRecord::operation: Operation",
            ),
            "engine": field(ref("engine"), "Engine identity.", "object with all members present", "ImpactEvaluationRecord::engine: EngineIdentity"),
            "source": field(
                {"oneOf": [ref(f"source_{name}") for name in ("point_array", "canonical_distribution", "histogram", "untracked")]},
                "Evaluated-input description.",
                "tagged object with all members present",
                "ImpactEvaluationRecord::source: SourceIdentity",
            ),
            "preprocessing": field({"type": "array", "items": ref("preprocessing_step")}, "Ordered explicit preprocessing.", "array order preserved", "ImpactEvaluationRecord::preprocessing: Vec<PreprocessingStep>"),
            "resolution": field(nullable(ref("resolution")), "Registry resolution for this application, or null.", "object or null", "ImpactEvaluationRecord::resolution: Option<Resolution>"),
            "diagnostics": field(ref("diagnostics"), "Method diagnostics.", "object with all members present", "ImpactEvaluationRecord::diagnostics: Diagnostics"),
            "replay": field(ref("replay"), "Replay completeness.", "object", "ImpactEvaluationRecord::replay: Replay"),
        },
        ["schema_version", "function", "operation", "engine", "source", "preprocessing", "resolution", "diagnostics", "replay"],
        [
            "A custom function has replay status runtime_only; a definition function never does.",
            "Complete replay requires a source snapshot.",
        ],
    )
    return _document(
        "impact-evaluation-record-1.0",
        "CRC impact evaluation record 1.0",
        "Independently versioned record of one impact operation: function identity, data identity, operation "
        "settings, engine identity and replay completeness. record_hash uses the canonical profile.",
        root,
        defs,
    )


# Spatial combination -------------------------------------------------------


def _policy_root() -> Dict[str, Any]:
    return obj(
        "",
        {
            "schema_version": field(const("1.0"), "Policy wire format.", STRING, "SpatialCombinationPolicy::schema_version"),
            "semantics_version": field(const("crc-spatial-combination/1"), "Reducer and coverage semantics.", STRING, "SpatialCombinationPolicy::semantics_version"),
            "stage": field(enum(("exposure", "impact")), "exposure: f(sum w_i x_i); impact: sum w_i f_i(x_i). No implicit choice.", STRING, "SpatialCombinationPolicy::stage: Stage"),
            "reducer": field(const("weighted_mean"), "Reducer; only weighted_mean in v1.", f"{STRING}; {ALWAYS}", "SpatialCombinationPolicy::reducer: Reducer", "weighted_mean"),
            "weights": field({"oneOf": [ref("weights_intersection_area"), ref("weights_provided")]}, "Weight basis; actual geometries and values are application data.", "tagged object", "SpatialCombinationPolicy::weights: WeightBasis"),
            "coverage": field(enum(("require_full", "covered_only")), "require_full rejects uncovered area above 1e-9 of target area; covered_only normalizes over covered contributors and reports the uncovered fraction.", f"{STRING}; {ALWAYS}", "SpatialCombinationPolicy::coverage: Coverage", "require_full"),
        },
        ["schema_version", "semantics_version", "stage", "weights"],
    )


def _policy_defs() -> Dict[str, Any]:
    return {
        "weights_intersection_area": obj(
            "Weights are contributor intersection areas with the target.",
            {"kind": kind("intersection_area", "WeightBasis::IntersectionArea")},
            ["kind"],
        ),
        "weights_provided": obj(
            "Caller-provided weights with a retained contributor-to-weight mapping.",
            {
                "kind": kind("provided", "WeightBasis::Provided"),
                "meaning": field(ref("text"), "Declared meaning of the weights, for example asset_value.", STRING, "WeightBasis::Provided::meaning: String"),
            },
            ["kind", "meaning"],
        ),
    }


def policy_schema() -> Dict[str, Any]:
    return _document(
        "spatial-combination-policy-1.0",
        "CRC spatial combination policy 1.0",
        "Application-layer weighted-mean combination of aligned point exposures. Never part of an impact "
        "definition or its hash; policy_hash uses the canonical profile.",
        _policy_root(),
        _policy_defs(),
    )


def spatial_record_schema() -> Dict[str, Any]:
    record_defs = _record_defs()
    defs = {
        **_policy_defs(),
        "policy": {**_policy_root(), "description": "Canonical spatial combination policy."},
        "digest": record_defs["digest"],
        "snapshot": record_defs["snapshot"],
        "replay": record_defs["replay"],
        "target": obj(
            "Target identity.",
            {
                "id": field(ref("identifier"), "Stable target identifier.", STRING, "SpatialTarget::id: String"),
                "kind": field(enum(("h3_cell", "geometry")), "Target kind.", STRING, "SpatialTarget::kind"),
                "cell": field(nullable(ref("h3_cell")), "H3 target cell.", f"{STRING}; {ALWAYS}", "SpatialTarget::cell: Option<u64>", None),
                "geometry_sha256": field(nullable(ref("sha256")), "Digest of the target geometry snapshot.", f"{STRING}; {ALWAYS}", "SpatialTarget::geometry_sha256: Option<[u8; 32]>", None),
            },
            ["id", "kind"],
            ["h3_cell targets have a cell; geometry targets have a geometry digest."],
        ),
        "geometry": obj(
            "Geometry engine identity and measured coverage diagnostics.",
            {
                "engine": field(ref("identifier"), "Geometry engine.", STRING, "GeometryDiagnostics::engine: String"),
                "engine_version": field(ref("identifier"), "Geometry engine version.", STRING, "GeometryDiagnostics::engine_version: String"),
                "crs": field(ref("identifier"), "Coordinate reference system.", STRING, "GeometryDiagnostics::crs: String"),
                "area_method": field(ref("identifier"), "Area measurement method.", STRING, "GeometryDiagnostics::area_method: String"),
                "target_area": field(ref("positive_number"), "Measured target area.", NUMBER, "GeometryDiagnostics::target_area: f64"),
                "covered_fraction": field(ref("probability"), "Union of contributor intersections over target area.", NUMBER, "GeometryDiagnostics::covered_fraction: f64"),
                "overlap_fraction": field(ref("probability"), "Positive-area overlap over target area.", NUMBER, "GeometryDiagnostics::overlap_fraction: f64"),
                "tolerance": field({"type": "number", "const": SPATIAL_GEOMETRY_TOLERANCE}, "Fixed relative tolerance for crc-spatial-combination/1; a record cannot relax coverage or overlap policy.", NUMBER, "GeometryDiagnostics::tolerance: f64"),
            },
            ["engine", "engine_version", "crs", "area_method", "target_area", "covered_fraction", "overlap_fraction", "tolerance"],
            ["overlap_fraction <= tolerance.", "require_full: 1 - covered_fraction <= tolerance."],
        ),
        "weights": obj(
            "Raw and normalized weight snapshots in stable contributor-ID order.",
            {
                "raw": field(ref("snapshot"), "Raw weight snapshot.", "object", "SpatialWeights::raw: SnapshotRef"),
                "normalized": field(ref("snapshot"), "Normalized weight snapshot.", "object", "SpatialWeights::normalized: SnapshotRef"),
                "unit": field(nullable(ref("identifier")), "Common weight unit; required for provided weights.", f"{STRING}; {ALWAYS}", "SpatialWeights::unit: Option<String>", None),
            },
            ["raw", "normalized"],
        ),
        "child": obj(
            "Child point-evaluation record reference.",
            {
                "contributor_id": field(nullable(ref("identifier")), "Contributor for impact-stage children; null for the exposure-stage target child.", f"{STRING}; {ALWAYS}", "SpatialChild::contributor_id: Option<String>", None),
                "record_hash": field(ref("content_hash"), "Child evaluation record identity.", STRING, "SpatialChild::record_hash: ContentHash"),
                "uri": field(nullable(ref("identifier")), "Child record location.", f"{STRING}; {ALWAYS}", "SpatialChild::uri: Option<String>", None),
            },
            ["record_hash"],
        ),
    }
    root = obj(
        "",
        {
            "schema_version": field(const("1.0"), "Spatial record format.", STRING, "SpatialImpactEvaluationRecord::schema_version"),
            "policy": field(ref("policy"), "Canonical policy.", "object with all members present", "SpatialImpactEvaluationRecord::policy"),
            "policy_hash": field(ref("content_hash"), "Policy identity.", STRING, "SpatialImpactEvaluationRecord::policy_hash"),
            "target": field(ref("target"), "Target identity.", "object with all members present", "SpatialImpactEvaluationRecord::target"),
            "contributors": field(ref("snapshot"), "Contributor snapshot mapping stable IDs to source, selection, alignment and function identities.", "object", "SpatialImpactEvaluationRecord::contributors: SnapshotRef"),
            "contributor_count": field(ref("positive_count"), "Number of contributors.", INTEGER, "SpatialImpactEvaluationRecord::contributor_count: u64"),
            "geometry": field(ref("geometry"), "Geometry and coverage diagnostics.", "object", "SpatialImpactEvaluationRecord::geometry"),
            "weights": field(ref("weights"), "Weight snapshots.", "object with all members present", "SpatialImpactEvaluationRecord::weights"),
            "combined_exposure": field(nullable(ref("digest")), "Combined exposure identity for the exposure stage; null for the impact stage.", "object or null", "SpatialImpactEvaluationRecord::combined_exposure: Option<ContentDigest>"),
            "children": field({"type": "array", "minItems": 1, "items": ref("child")}, "Child evaluation records.", "array order preserved", "SpatialImpactEvaluationRecord::children: Vec<SpatialChild>"),
            "replay": field(ref("replay"), "Replay completeness.", "object", "SpatialImpactEvaluationRecord::replay: Replay"),
        },
        ["schema_version", "policy", "policy_hash", "target", "contributors", "contributor_count", "geometry", "weights", "combined_exposure", "children", "replay"],
        [
            "policy_hash equals the content hash of the canonical policy.",
            "exposure stage: combined_exposure is present and children is one target child with null contributor_id.",
            "impact stage: combined_exposure is null and children has one record per contributor with unique ids.",
            "Provided weights require a weight unit.",
        ],
    )
    return _document(
        "spatial-impact-evaluation-record-1.0",
        "CRC spatial impact evaluation record 1.0",
        "Application record for a spatial combination. Never assigns a single function hash to an application "
        "that evaluated multiple definitions; function identities live in child records and the contributor snapshot.",
        root,
        defs,
    )


SCHEMAS = {
    "impact_definition": ("impact-definition-1.0", definition_schema),
    "registry_impact_reference": ("registry-impact-reference-1.0", reference_schema),
    "impact_evaluation_record": ("impact-evaluation-record-1.0", record_schema),
    "spatial_combination_policy": ("spatial-combination-policy-1.0", policy_schema),
    "spatial_impact_evaluation_record": ("spatial-impact-evaluation-record-1.0", spatial_record_schema),
}


def all_schemas() -> Dict[str, Dict[str, Any]]:
    return {contract: build() for contract, (_, build) in SCHEMAS.items()}


def schema_files() -> List[str]:
    return [f"schemas/{name}.schema.json" for name, _ in SCHEMAS.values()]

"""Independent reference semantics for the CRC impact contracts.

This module is the oracle used to generate the published conformance
fixtures. It deliberately shares no code with the Rust core or the public
Python API, so canonical bytes, hashes and reference values are not produced
by the implementation under test. It favours clarity over speed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

DEFINITION_SCHEMA_VERSION = "1.0"
IMPACT_SEMANTICS_VERSION = "crc-impact/1"
REFERENCE_SCHEMA_VERSION = "1.0"
RESOLVER_SEMANTICS_VERSION = "crc-impact-registry/1"
RECORD_SCHEMA_VERSION = "1.0"
POLICY_SCHEMA_VERSION = "1.0"
SPATIAL_SEMANTICS_VERSION = "crc-spatial-combination/1"
SPATIAL_RECORD_SCHEMA_VERSION = "1.0"
CANONICAL_PROFILE = "crc-canonical-json/1"
HISTOGRAM_MASS_TOLERANCE = 1e-12
SPATIAL_GEOMETRY_TOLERANCE = 1e-9
U64_MAX = (1 << 64) - 1
FITTED_CURVE_TYPES = (
    "genextreme", "weibull_min", "weibull_max", "skewnorm", "gumbel_r", "gumbel_l", "genpareto"
)

MODEL_KINDS = ("linear", "sigmoid", "tabulated", "binned", "spei", "cooling", "crop")
ORIGIN_KINDS = ("user_defined", "published", "imported", "derived", "registry")
SOURCE_KINDS = ("publication", "dataset", "code", "definition")
CROPS = ("maize", "rice", "wheat")
CONTEXT_FIELDS = ("cell", "country", "continent", "building_type", "historic_mean")
FALLBACK_POLICIES = ("error", "legacy_defaults")
MONOTONICITY_CLASSES = (
    "non_decreasing",
    "non_increasing",
    "constant",
    "non_monotone",
    "unknown",
)
REPLAY_REQUIREMENTS = (
    "engine_build",
    "function_definition",
    "preprocessing_snapshot",
    "registry_context_snapshot",
    "source_snapshot",
)
TABULATION_INTERPOLATIONS = ("linear_probability", "log_return_period")

_SIGMOID_OVERRIDES = {
    "midpoint": False,
    "steepness": False,
    "minimum_damage": False,
    "maximum_damage": False,
    "zero_below": True,
}
# Factor -> override key -> whether explicit null is accepted.
OVERRIDE_CATALOG: Dict[str, Dict[str, bool]] = {
    "daily_freezethaw_cycles": dict(_SIGMOID_OVERRIDES),
    "frost_days": dict(_SIGMOID_OVERRIDES),
    "rx1day": dict(_SIGMOID_OVERRIDES),
    "cyclone": dict(_SIGMOID_OVERRIDES),
    "wind_max_daily_max": dict(_SIGMOID_OVERRIDES),
    "fwi": dict(_SIGMOID_OVERRIDES),
    "hot_days": {
        "slope": False,
        "intercept": False,
        "minimum_damage": True,
        "maximum_damage": True,
    },
    "cflood": {"maximum_damage": False},
    "rflood": {"maximum_damage": False},
    "inundation": {"maximum_damage": False},
    "spei": {"water_unit_cost": False},
    "cooling_degree_days": {"cooling_cost": False},
    "maximum_consecutive_dry_days": {},
}
OVERRIDE_KEYS = tuple(
    sorted({key for entries in OVERRIDE_CATALOG.values() for key in entries})
)
REGISTRY_FACTORS = tuple(OVERRIDE_CATALOG)

_IDENTIFIER = re.compile(r"\S(?:[\s\S]*\S)?")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_CONTENT_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_H3_CELL = re.compile(r"[0-9a-f]{15}")


class ContractError(ValueError):
    """A contract violation identified by a stable code and JSON Pointer path."""

    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(f"{code} at {path or '/'}: {message}")
        self.code = code
        self.path = path
        self.message = message


def _fail(code: str, path: str, message: str) -> Any:
    raise ContractError(code, path, message)


def pointer(path: str, key: Any) -> str:
    token = str(key).replace("~", "~0").replace("/", "~1")
    return f"{path}/{token}"


# Strict JSON parsing ------------------------------------------------------


class _NonFinite:
    def __init__(self, token: str) -> None:
        self.token = token


class _Overflow:
    def __init__(self, token: str) -> None:
        self.token = token


class _Pairs(dict):  # type: ignore[type-arg]
    duplicates: List[str]


def _pairs_hook(pairs: Sequence[Any]) -> _Pairs:
    result = _Pairs()
    result.duplicates = []
    for key, value in pairs:
        if key in result:
            result.duplicates.append(key)
        else:
            result[key] = value
    return result


def _parse_float(token: str) -> Any:
    value = float(token)
    return _Overflow(token) if math.isinf(value) else value


def _check_text(value: str, path: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        _fail("invalid_json", path, "strings must not contain unpaired surrogates")


def _clean(value: Any, path: str) -> Any:
    if isinstance(value, _NonFinite):
        return _fail("non_finite_number", path, f"{value.token} is not a JSON number")
    if isinstance(value, _Overflow):
        return _fail("number_overflow", path, f"{value.token} overflows binary64")
    if isinstance(value, str):
        _check_text(value, path)
        return value
    if isinstance(value, list):
        return [_clean(item, pointer(path, index)) for index, item in enumerate(value)]
    if isinstance(value, dict):
        duplicates = getattr(value, "duplicates", [])
        if duplicates:
            _fail("duplicate_key", pointer(path, duplicates[0]), "duplicate object key")
        result = {}
        for key, item in value.items():
            _check_text(key, pointer(path, key))
            result[key] = _clean(item, pointer(path, key))
        return result
    return value


def parse_json(text: str) -> Any:
    """Parse JSON text, rejecting duplicate keys, non-finite tokens and overflow."""
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_hook,
            parse_constant=_NonFinite,
            parse_float=_parse_float,
        )
    except ValueError as error:
        return _fail("invalid_json", "", str(error))
    return _clean(value, "")


# Field validation ---------------------------------------------------------


def _utf8(key: str) -> bytes:
    return key.encode("utf-8")


def _fields(
    value: Any,
    path: str,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return _fail("invalid_type", path, "expected an object")
    allowed = set(required) | set(optional)
    for key in sorted(value, key=_utf8):
        if key not in allowed:
            _fail("unknown_field", pointer(path, key), "unknown field")
    for key in required:
        if key not in value:
            _fail("missing_field", pointer(path, key), "required field is missing")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: Optional[float] = None,
    positive: bool = False,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _fail("invalid_type", path, "expected a JSON number")
    if isinstance(value, int):
        try:
            result = float(value)
        except OverflowError:
            return _fail("number_overflow", path, "integer overflows binary64")
    else:
        result = value
    if not math.isfinite(result):
        _fail("non_finite_number", path, "numbers must be finite binary64 values")
    if result == 0.0:
        result = 0.0
    if minimum is not None and result < minimum:
        _fail("invalid_value", path, f"must be at least {minimum}")
    if maximum is not None and result > maximum:
        _fail("invalid_value", path, f"must be at most {maximum}")
    if positive and not result > 0.0:
        _fail("invalid_value", path, "must be positive")
    return result


def _optional_number(value: Any, path: str, **limits: Any) -> Optional[float]:
    return None if value is None else _number(value, path, **limits)


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return _fail("invalid_type", path, "expected a JSON integer")
    if value < minimum:
        _fail("invalid_value", path, f"must be at least {minimum}")
    if value > U64_MAX:
        _fail("invalid_value", path, "must fit an unsigned 64-bit integer")
    return value


def _optional_integer(value: Any, path: str, *, minimum: int = 0) -> Optional[int]:
    return None if value is None else _integer(value, path, minimum=minimum)


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        return _fail("invalid_type", path, "expected a string")
    return value


def identifier(value: Any, path: str) -> str:
    text = _string(value, path)
    if not _IDENTIFIER.fullmatch(text):
        _fail("invalid_value", path, "must be non-empty without surrounding whitespace")
    return text


def _optional_identifier(value: Any, path: str) -> Optional[str]:
    return None if value is None else identifier(value, path)


def _text(value: Any, path: str) -> str:
    text = _string(value, path)
    if not text:
        _fail("invalid_value", path, "must be non-empty")
    return text


def _optional_text(value: Any, path: str) -> Optional[str]:
    return None if value is None else _text(value, path)


def _choice(value: Any, path: str, options: Sequence[str]) -> str:
    text = _string(value, path)
    if text not in options:
        _fail("unsupported_value", path, f"expected one of {', '.join(options)}")
    return text


def _constant(value: Any, path: str, expected: Any) -> Any:
    if value != expected or type(value) is not type(expected):
        _fail("invalid_value", path, f"must be {json.dumps(expected)}")
    return value


def _sha256(value: Any, path: str) -> str:
    text = _string(value, path)
    if not _HEX64.fullmatch(text):
        _fail("invalid_value", path, "must be 64 lowercase hexadecimal characters")
    return text


def _optional_sha256(value: Any, path: str) -> Optional[str]:
    return None if value is None else _sha256(value, path)


def _content_hash(value: Any, path: str) -> str:
    text = _string(value, path)
    if not _CONTENT_HASH.fullmatch(text):
        _fail("invalid_value", path, "must be sha256: followed by 64 lowercase hex")
    return text


def _array(value: Any, path: str, *, min_items: int = 0) -> List[Any]:
    if not isinstance(value, list):
        return _fail("invalid_type", path, "expected an array")
    if len(value) < min_items:
        _fail("invalid_value", path, f"requires at least {min_items} items")
    return value


def _kind(value: Any, path: str, options: Sequence[str]) -> str:
    if not isinstance(value, dict):
        return _fail("invalid_type", path, "expected an object")
    if "kind" not in value:
        _fail("missing_field", pointer(path, "kind"), "required field is missing")
    kind = _string(value["kind"], pointer(path, "kind"))
    if kind not in options:
        _fail("unsupported_kind", pointer(path, "kind"), f"unsupported kind {kind!r}")
    return kind


def _version(value: Any, path: str, field: str, expected: str, code: str) -> None:
    if not isinstance(value, dict):
        _fail("invalid_type", path, "expected an object")
    if field in value:
        found = _string(value[field], pointer(path, field))
        if found != expected:
            _fail(code, pointer(path, field), f"unsupported version {found!r}")


# Impact definitions -------------------------------------------------------


def _calibration(value: Any, path: str) -> Dict[str, Any]:
    names = ("exposure_variable", "exposure_unit", "impact_metric", "impact_unit")
    fields = _fields(value, path, names)
    return {name: identifier(fields[name], pointer(path, name)) for name in names}


def _increasing(values: Sequence[float], path: str, suffix: Optional[str]) -> None:
    for index in range(1, len(values)):
        if not values[index] > values[index - 1]:
            location = pointer(path, index)
            if suffix is not None:
                location = pointer(location, suffix)
            _fail("invalid_value", location, "values must strictly increase")


def _crop_vector(value: Any, path: str, *, minimum: Optional[float] = None) -> Dict[str, float]:
    fields = _fields(value, path, CROPS)
    return {
        crop: _number(fields[crop], pointer(path, crop), minimum=minimum) for crop in CROPS
    }


def _model(value: Any, path: str) -> Dict[str, Any]:
    kind = _kind(value, path, MODEL_KINDS)

    def at(name: str) -> str:
        return pointer(path, name)

    if kind == "linear":
        fields = _fields(value, path, ("kind", "slope"), ("intercept",))
        return {
            "kind": kind,
            "slope": _number(fields["slope"], at("slope")),
            "intercept": _number(fields.get("intercept", 0.0), at("intercept")),
        }
    if kind == "sigmoid":
        fields = _fields(
            value,
            path,
            ("kind", "midpoint", "steepness"),
            ("offset", "amplitude", "origin_adjustment", "zero_below"),
        )
        return {
            "kind": kind,
            "midpoint": _number(fields["midpoint"], at("midpoint")),
            "steepness": _number(fields["steepness"], at("steepness")),
            "offset": _number(fields.get("offset", 0.0), at("offset")),
            "amplitude": _number(fields.get("amplitude", 1.0), at("amplitude"), minimum=0.0),
            "origin_adjustment": _choice(
                fields.get("origin_adjustment", "none"),
                at("origin_adjustment"),
                ("none", "subtract_at_zero"),
            ),
            "zero_below": _optional_number(fields.get("zero_below"), at("zero_below")),
        }
    if kind == "tabulated":
        fields = _fields(value, path, ("kind", "points"), ("interpolation", "outside_domain"))
        raw = _array(fields["points"], at("points"), min_items=2)
        points = []
        for index, item in enumerate(raw):
            location = pointer(at("points"), index)
            point = _fields(item, location, ("x", "y"))
            points.append(
                {
                    "x": _number(point["x"], pointer(location, "x")),
                    "y": _number(point["y"], pointer(location, "y")),
                }
            )
        _increasing([point["x"] for point in points], at("points"), "x")
        return {
            "kind": kind,
            "points": points,
            "interpolation": _choice(
                fields.get("interpolation", "linear"),
                at("interpolation"),
                ("linear", "step_left", "step_right"),
            ),
            "outside_domain": _choice(
                fields.get("outside_domain", "clamp"), at("outside_domain"), ("clamp", "error")
            ),
        }
    if kind == "binned":
        fields = _fields(value, path, ("kind", "bin_edges", "values"), ("outside_domain",))
        edges = [
            _number(item, pointer(at("bin_edges"), index))
            for index, item in enumerate(_array(fields["bin_edges"], at("bin_edges"), min_items=2))
        ]
        _increasing(edges, at("bin_edges"), None)
        values = [
            _number(item, pointer(at("values"), index))
            for index, item in enumerate(_array(fields["values"], at("values"), min_items=1))
        ]
        if len(values) != len(edges) - 1:
            _fail("invalid_value", at("values"), "requires exactly one value per interval")
        return {
            "kind": kind,
            "bin_edges": edges,
            "values": values,
            "outside_domain": _choice(
                fields.get("outside_domain", "clamp"), at("outside_domain"), ("clamp", "error")
            ),
        }
    if kind == "spei":
        names = ("water_consumption_asset", "original_water_unit_cost")
        fields = _fields(value, path, ("kind", *names))
        result: Dict[str, Any] = {"kind": kind}
        for name in names:
            result[name] = _number(fields[name], at(name), minimum=0.0)
        return result
    if kind == "cooling":
        fields = _fields(value, path, ("kind", "cooling_cost", "historic_mean"))
        return {
            "kind": kind,
            "cooling_cost": _number(fields["cooling_cost"], at("cooling_cost"), minimum=0.0),
            "historic_mean": _number(fields["historic_mean"], at("historic_mean"), positive=True),
        }
    fields = _fields(value, path, ("kind", "weights", "country", "yearly", "index", "historic_mean"))
    weights = _crop_vector(fields["weights"], at("weights"), minimum=0.0)
    if not sum(weights.values()) > 0.0:
        _fail("invalid_value", at("weights"), "crop weights require a positive total")
    return {
        "kind": kind,
        "weights": weights,
        "country": _crop_vector(fields["country"], at("country")),
        "yearly": _crop_vector(fields["yearly"], at("yearly")),
        "index": _crop_vector(fields["index"], at("index")),
        "historic_mean": _number(fields["historic_mean"], at("historic_mean")),
    }


def _clamp(value: Any, path: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    fields = _fields(value, path, (), ("minimum", "maximum"))
    minimum = _optional_number(fields.get("minimum"), pointer(path, "minimum"))
    maximum = _optional_number(fields.get("maximum"), pointer(path, "maximum"))
    if minimum is None and maximum is None:
        return None
    if minimum is not None and maximum is not None and minimum > maximum:
        _fail("invalid_value", path, "clamp minimum must not exceed maximum")
    return {"minimum": minimum, "maximum": maximum}


def normalize_context(value: Any, path: str) -> Dict[str, Any]:
    fields = _fields(value, path, (), CONTEXT_FIELDS)
    cell = fields.get("cell")
    if cell is not None:
        cell = _string(cell, pointer(path, "cell"))
        if not _H3_CELL.fullmatch(cell):
            _fail("invalid_value", pointer(path, "cell"), "must be 15 lowercase hex characters")
    return {
        "cell": cell,
        "country": _optional_identifier(fields.get("country"), pointer(path, "country")),
        "continent": _optional_identifier(fields.get("continent"), pointer(path, "continent")),
        "building_type": _optional_identifier(
            fields.get("building_type"), pointer(path, "building_type")
        ),
        "historic_mean": _optional_number(
            fields.get("historic_mean"), pointer(path, "historic_mean")
        ),
    }


def _factor(value: Any, path: str) -> str:
    return _choice(identifier(value, path), path, REGISTRY_FACTORS)


def _overrides(value: Any, path: str, factor: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return _fail("invalid_type", path, "expected an object")
    accepted = OVERRIDE_CATALOG[factor]
    result: Dict[str, Any] = {}
    for key in sorted(value, key=_utf8):
        location = pointer(path, key)
        if key not in OVERRIDE_KEYS:
            _fail("unknown_override", location, "unknown override key")
        if key not in accepted:
            _fail("inapplicable_override", location, f"not accepted for factor {factor}")
        item = value[key]
        if item is None:
            if not accepted[key]:
                _fail("invalid_value", location, "null is accepted only to clear a threshold or clamp")
            result[key] = None
        else:
            result[key] = _number(item, location)
    return result


def _lookup_key(value: Any, path: str) -> Any:
    if isinstance(value, str):
        return identifier(value, path)
    if not isinstance(value, dict):
        return _fail("invalid_type", path, "expected a string or an object of strings")
    if not value:
        _fail("invalid_value", path, "composite keys require at least one field")
    result = {}
    for name in sorted(value, key=_utf8):
        identifier(name, pointer(path, name))
        result[name] = identifier(value[name], pointer(path, name))
    return result


def _lookup(value: Any, path: str) -> Dict[str, Any]:
    names = ("table_id", "table_sha256", "requested_key", "resolved_key", "status")
    fields = _fields(value, path, names)
    return {
        "table_id": identifier(fields["table_id"], pointer(path, "table_id")),
        "table_sha256": _sha256(fields["table_sha256"], pointer(path, "table_sha256")),
        "requested_key": _lookup_key(fields["requested_key"], pointer(path, "requested_key")),
        "resolved_key": _lookup_key(fields["resolved_key"], pointer(path, "resolved_key")),
        "status": _choice(fields["status"], pointer(path, "status"), ("exact", "fallback")),
    }


def _registry_identity(fields: Mapping[str, Any], path: str) -> Dict[str, Any]:
    resolver = pointer(path, "resolver_semantics_version")
    if _string(fields["resolver_semantics_version"], resolver) != RESOLVER_SEMANTICS_VERSION:
        _fail("unsupported_semantics_version", resolver, "unsupported resolver semantics")
    return {
        "registry_id": identifier(fields["registry_id"], pointer(path, "registry_id")),
        "registry_revision": identifier(
            fields["registry_revision"], pointer(path, "registry_revision")
        ),
        "registry_manifest_sha256": _sha256(
            fields["registry_manifest_sha256"], pointer(path, "registry_manifest_sha256")
        ),
        "resolver_semantics_version": RESOLVER_SEMANTICS_VERSION,
    }


def _origin(
    value: Any, path: str, source_ids: Sequence[str], parents: Sequence[str], provenance: str
) -> Dict[str, Any]:
    kind = _kind(value, path, ORIGIN_KINDS)

    def source_reference(fields: Mapping[str, Any]) -> str:
        location = pointer(path, "source_id")
        source_id = identifier(fields["source_id"], location)
        if source_id not in source_ids:
            _fail("invalid_reference", location, "source_id does not name a source record")
        return source_id

    if kind == "user_defined":
        _fields(value, path, ("kind",))
        return {"kind": kind}
    if kind == "published":
        fields = _fields(value, path, ("kind", "source_id"))
        return {"kind": kind, "source_id": source_reference(fields)}
    if kind == "imported":
        fields = _fields(
            value,
            path,
            ("kind", "source_id", "importer_name", "importer_version"),
            ("importer_build_sha256",),
        )
        return {
            "kind": kind,
            "source_id": source_reference(fields),
            "importer_name": identifier(fields["importer_name"], pointer(path, "importer_name")),
            "importer_version": identifier(
                fields["importer_version"], pointer(path, "importer_version")
            ),
            "importer_build_sha256": _optional_sha256(
                fields.get("importer_build_sha256"), pointer(path, "importer_build_sha256")
            ),
        }
    if kind == "derived":
        fields = _fields(value, path, ("kind", "description"))
        description = _text(fields["description"], pointer(path, "description"))
        if not parents:
            _fail(
                "invalid_reference",
                pointer(provenance, "derived_from"),
                "a derived origin requires at least one parent definition",
            )
        return {"kind": kind, "description": description}
    names = (
        "kind",
        "registry_id",
        "registry_revision",
        "registry_manifest_sha256",
        "resolver_semantics_version",
        "factor",
        "context",
        "overrides",
        "fallback_policy",
        "lookups",
    )
    fields = _fields(value, path, names)
    result: Dict[str, Any] = {"kind": kind, **_registry_identity(fields, path)}
    factor = _factor(fields["factor"], pointer(path, "factor"))
    policy = _choice(fields["fallback_policy"], pointer(path, "fallback_policy"), FALLBACK_POLICIES)
    lookups = [
        _lookup(item, pointer(pointer(path, "lookups"), index))
        for index, item in enumerate(_array(fields["lookups"], pointer(path, "lookups")))
    ]
    if policy == "error":
        for index, lookup in enumerate(lookups):
            if lookup["status"] == "fallback":
                _fail(
                    "invalid_value",
                    pointer(pointer(pointer(path, "lookups"), index), "status"),
                    "fallback lookups require fallback_policy legacy_defaults",
                )
    result.update(
        factor=factor,
        context=normalize_context(fields["context"], pointer(path, "context")),
        overrides=_overrides(fields["overrides"], pointer(path, "overrides"), factor),
        fallback_policy=policy,
        lookups=lookups,
    )
    return result


def _source(value: Any, path: str) -> Dict[str, Any]:
    fields = _fields(value, path, ("id", "kind", "uri"), ("revision", "content_sha256"))
    return {
        "id": identifier(fields["id"], pointer(path, "id")),
        "kind": _choice(fields["kind"], pointer(path, "kind"), SOURCE_KINDS),
        "uri": identifier(fields["uri"], pointer(path, "uri")),
        "revision": _optional_identifier(fields.get("revision"), pointer(path, "revision")),
        "content_sha256": _optional_sha256(
            fields.get("content_sha256"), pointer(path, "content_sha256")
        ),
    }


def _provenance(value: Any, path: str) -> Dict[str, Any]:
    fields = _fields(value, path, ("origin",), ("sources", "derived_from"))
    sources_path = pointer(path, "sources")
    sources = []
    seen: set = set()
    for index, item in enumerate(_array(fields.get("sources", []), sources_path)):
        source = _source(item, pointer(sources_path, index))
        if source["id"] in seen:
            _fail("duplicate_identifier", pointer(pointer(sources_path, index), "id"), "duplicate source id")
        seen.add(source["id"])
        sources.append(source)
    parents_path = pointer(path, "derived_from")
    parents: List[str] = []
    for index, item in enumerate(_array(fields.get("derived_from", []), parents_path)):
        parent = _content_hash(item, pointer(parents_path, index))
        if parent in parents:
            _fail("duplicate_identifier", pointer(parents_path, index), "duplicate parent hash")
        parents.append(parent)
    origin = _origin(
        fields["origin"], pointer(path, "origin"), [source["id"] for source in sources], parents, path
    )
    return {"origin": origin, "sources": sources, "derived_from": sorted(parents)}


def normalize_definition(value: Any, path: str = "") -> Dict[str, Any]:
    """Validate an impact definition and return its fully materialized form."""
    _version(value, path, "schema_version", DEFINITION_SCHEMA_VERSION, "unsupported_schema_version")
    _version(
        value, path, "semantics_version", IMPACT_SEMANTICS_VERSION, "unsupported_semantics_version"
    )
    fields = _fields(
        value,
        path,
        ("schema_version", "semantics_version", "calibration", "model", "provenance"),
        ("clamp",),
    )
    return {
        "schema_version": DEFINITION_SCHEMA_VERSION,
        "semantics_version": IMPACT_SEMANTICS_VERSION,
        "calibration": _calibration(fields["calibration"], pointer(path, "calibration")),
        "model": _model(fields["model"], pointer(path, "model")),
        "clamp": _clamp(fields.get("clamp"), pointer(path, "clamp")),
        "provenance": _provenance(fields["provenance"], pointer(path, "provenance")),
    }


# Registry references ------------------------------------------------------


def normalize_reference(value: Any, path: str = "") -> Dict[str, Any]:
    """Validate a registry reference; absent override keys stay absent."""
    _version(value, path, "schema_version", REFERENCE_SCHEMA_VERSION, "unsupported_schema_version")
    fields = _fields(
        value,
        path,
        (
            "schema_version",
            "registry_id",
            "registry_revision",
            "registry_manifest_sha256",
            "resolver_semantics_version",
            "factor",
        ),
        ("overrides", "context_defaults", "fallback_policy"),
    )
    result: Dict[str, Any] = {"schema_version": REFERENCE_SCHEMA_VERSION}
    result.update(_registry_identity(fields, path))
    factor = _factor(fields["factor"], pointer(path, "factor"))
    result.update(
        factor=factor,
        overrides=_overrides(fields.get("overrides", {}), pointer(path, "overrides"), factor),
        context_defaults=normalize_context(
            fields.get("context_defaults", {}), pointer(path, "context_defaults")
        ),
        fallback_policy=_choice(
            fields.get("fallback_policy", "error"), pointer(path, "fallback_policy"), FALLBACK_POLICIES
        ),
    )
    return result


# Evaluation records -------------------------------------------------------


def _digest(value: Any, path: str) -> Dict[str, Any]:
    fields = _fields(value, path, ("encoding", "sha256"))
    return {
        "encoding": identifier(fields["encoding"], pointer(path, "encoding")),
        "sha256": _sha256(fields["sha256"], pointer(path, "sha256")),
    }


def _optional_digest(value: Any, path: str) -> Optional[Dict[str, Any]]:
    return None if value is None else _digest(value, path)


def _snapshot(value: Any, path: str) -> Dict[str, Any]:
    fields = _fields(value, path, ("uri", "content_sha256", "retention"), ("revision",))
    return {
        "uri": identifier(fields["uri"], pointer(path, "uri")),
        "revision": _optional_identifier(fields.get("revision"), pointer(path, "revision")),
        "content_sha256": _sha256(fields["content_sha256"], pointer(path, "content_sha256")),
        "retention": _choice(fields["retention"], pointer(path, "retention"), ("retained", "external")),
    }


def _optional_snapshot(value: Any, path: str) -> Optional[Dict[str, Any]]:
    return None if value is None else _snapshot(value, path)


def _selection(value: Any, path: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    names = ("dataset", "columns", "rows", "scenario", "horizon")
    fields = _fields(value, path, (), names)
    columns_path = pointer(path, "columns")
    return {
        "dataset": _optional_identifier(fields.get("dataset"), pointer(path, "dataset")),
        "columns": [
            identifier(item, pointer(columns_path, index))
            for index, item in enumerate(_array(fields.get("columns", []), columns_path))
        ],
        "rows": _optional_text(fields.get("rows"), pointer(path, "rows")),
        "scenario": _optional_identifier(fields.get("scenario"), pointer(path, "scenario")),
        "horizon": _optional_identifier(fields.get("horizon"), pointer(path, "horizon")),
    }


def _number_array(value: Any, path: str) -> Optional[List[float]]:
    if value is None:
        return None
    return [_number(item, pointer(path, index)) for index, item in enumerate(_array(value, path))]


def _curve_parameters(value: Any, path: str) -> Dict[str, Any]:
    scalars = (
        "curve_shape",
        "curve_location",
        "curve_scale",
        "curve_atom_probability",
        "curve_atom_location",
    )
    arrays = ("curve_probabilities", "curve_values")
    fields = _fields(value, path, ("curve_kind", "curve_type"), (*scalars, *arrays))
    result: Dict[str, Any] = {
        "curve_kind": identifier(fields["curve_kind"], pointer(path, "curve_kind")),
        "curve_type": identifier(fields["curve_type"], pointer(path, "curve_type")),
    }
    for name in scalars:
        result[name] = _optional_number(fields.get(name), pointer(path, name))
    for name in arrays:
        result[name] = _number_array(fields.get(name), pointer(path, name))
    _validate_curve_parameters(result, path)
    return result


def _validate_curve_parameters(curve: Dict[str, Any], path: str) -> None:
    """Check reconstructible CRC source curves without importing the native core."""
    kind = _choice(curve["curve_kind"], pointer(path, "curve_kind"), ("fitted", "hurdle", "point_mass", "tabulated"))
    family = curve["curve_type"]
    # Existing canonical CRC normalization ignores Gumbel's unused shape.
    if family in ("gumbel_r", "gumbel_l"):
        curve["curve_shape"] = None

    def invalid(name: str, message: str) -> None:
        _fail("invalid_value", pointer(path, name), message)

    scalars = ("curve_shape", "curve_location", "curve_scale", "curve_atom_probability", "curve_atom_location")
    arrays = ("curve_probabilities", "curve_values")
    if kind == "tabulated":
        _choice(family, pointer(path, "curve_type"), ("linear_probability",))
        for name in scalars:
            if curve[name] is not None:
                invalid(name, "tabulated curves must not contain scalar parameters")
        probabilities, values = curve["curve_probabilities"], curve["curve_values"]
        if probabilities is None or len(probabilities) < 2:
            invalid("curve_probabilities", "tabulated curves require at least two probability knots")
        if values is None or len(values) != len(probabilities):
            invalid("curve_values", "tabulated value and probability arrays must have equal length")
        for index, probability in enumerate(probabilities):
            if not 0.0 <= probability <= 1.0:
                _fail("invalid_value", pointer(pointer(path, "curve_probabilities"), index), "probabilities must be in [0, 1]")
        _increasing(probabilities, pointer(path, "curve_probabilities"), None)
        for index in range(1, len(values)):
            if values[index] < values[index - 1]:
                _fail("invalid_value", pointer(pointer(path, "curve_values"), index), "quantile values must not decrease")
        return
    for name in arrays:
        if curve[name] is not None:
            invalid(name, "non-tabulated curves must not contain knots")
    if kind == "point_mass":
        _choice(family, pointer(path, "curve_type"), ("point_mass",))
        if curve["curve_shape"] is not None:
            invalid("curve_shape", "point masses have no shape")
        if curve["curve_location"] is None:
            invalid("curve_location", "point masses require a location")
        if curve["curve_scale"] != 0.0:
            invalid("curve_scale", "point masses require zero scale")
        if curve["curve_atom_probability"] != 1.0:
            invalid("curve_atom_probability", "point masses require probability one")
        if curve["curve_atom_location"] != curve["curve_location"]:
            invalid("curve_atom_location", "point-mass atom and curve location must agree")
        return
    _choice(family, pointer(path, "curve_type"), FITTED_CURVE_TYPES)
    if curve["curve_location"] is None:
        invalid("curve_location", "fitted and hurdle curves require a location")
    if curve["curve_scale"] is None or curve["curve_scale"] <= 0.0:
        invalid("curve_scale", "fitted and hurdle curves require positive scale")
    if family not in ("gumbel_r", "gumbel_l") and curve["curve_shape"] is None:
        invalid("curve_shape", "this fitted family requires a shape")
    if kind == "fitted":
        for name in ("curve_atom_probability", "curve_atom_location"):
            if curve[name] is not None:
                invalid(name, "fitted curves must not contain hurdle atom parameters")
    else:
        probability = curve["curve_atom_probability"]
        if probability is None or not 0.0 < probability < 1.0:
            invalid("curve_atom_probability", "hurdle atom probability must be strictly between zero and one")
        if curve["curve_atom_location"] is None:
            invalid("curve_atom_location", "hurdle curves require an atom location")


def _record_source(value: Any, path: str) -> Dict[str, Any]:
    kind = _kind(value, path, ("point_array", "canonical_distribution", "histogram", "untracked"))
    tracked = ("content_digest", "snapshot", "selection")

    def common(fields: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "content_digest": _optional_digest(fields.get("content_digest"), pointer(path, "content_digest")),
            "snapshot": _optional_snapshot(fields.get("snapshot"), pointer(path, "snapshot")),
            "selection": _selection(fields.get("selection"), pointer(path, "selection")),
        }

    if kind == "untracked":
        fields = _fields(value, path, ("kind",), ("description",))
        return {
            "kind": kind,
            "description": _optional_text(fields.get("description"), pointer(path, "description")),
        }
    if kind == "point_array":
        fields = _fields(value, path, ("kind", "shape"), tracked)
        shape_path = pointer(path, "shape")
        shape = [
            _integer(item, pointer(shape_path, index))
            for index, item in enumerate(_array(fields["shape"], shape_path))
        ]
        return {"kind": kind, "shape": shape, **common(fields)}
    if kind == "canonical_distribution":
        fields = _fields(value, path, ("kind", "parameters"), tracked)
        return {
            "kind": kind,
            "parameters": _curve_parameters(fields["parameters"], pointer(path, "parameters")),
            **common(fields),
        }
    fields = _fields(
        value,
        path,
        ("kind", "interval_count", "within_bin", "original_mass_sum", "normalization_factor"),
        tracked,
    )
    mass = _number(fields["original_mass_sum"], pointer(path, "original_mass_sum"), positive=True)
    if abs(mass - 1.0) > HISTOGRAM_MASS_TOLERANCE:
        _fail("invalid_value", pointer(path, "original_mass_sum"), "histogram mass must be one within 1e-12")
    factor = _number(fields["normalization_factor"], pointer(path, "normalization_factor"), positive=True)
    if factor != 1.0 / mass:
        _fail("invalid_value", pointer(path, "normalization_factor"), "must equal 1 / original_mass_sum")
    return {
        "kind": kind,
        "interval_count": _integer(fields["interval_count"], pointer(path, "interval_count"), minimum=1),
        "within_bin": _choice(fields["within_bin"], pointer(path, "within_bin"), ("uniform",)),
        "original_mass_sum": mass,
        "normalization_factor": factor,
        **common(fields),
    }


def _integration_method(value: Any, path: str) -> Dict[str, Any]:
    kind = _kind(value, path, ("exact", "gauss_kronrod"))
    if kind == "exact":
        fields = _fields(value, path, ("kind", "version"))
        return {"kind": kind, "version": identifier(fields["version"], pointer(path, "version"))}
    names = ("kind", "version", "absolute_tolerance", "relative_tolerance", "max_evaluations")
    fields = _fields(value, path, names)
    return {
        "kind": kind,
        "version": identifier(fields["version"], pointer(path, "version")),
        "absolute_tolerance": _number(
            fields["absolute_tolerance"], pointer(path, "absolute_tolerance"), positive=True
        ),
        "relative_tolerance": _number(
            fields["relative_tolerance"], pointer(path, "relative_tolerance"), positive=True
        ),
        "max_evaluations": _integer(
            fields["max_evaluations"], pointer(path, "max_evaluations"), minimum=1
        ),
    }


def _operation(value: Any, path: str) -> Dict[str, Any]:
    kind = _kind(value, path, ("point", "expected_value", "transform_distribution", "tabulate"))
    fields = _fields(value, path, ("kind", "settings"))
    settings_path = pointer(path, "settings")
    raw = fields["settings"]
    if kind == "point":
        settings = _fields(raw, settings_path, ("alignment", "shape"))
        result = {
            "alignment": _choice(settings["alignment"], pointer(settings_path, "alignment"), ("event_aligned",)),
            "shape": _choice(settings["shape"], pointer(settings_path, "shape"), ("preserve",)),
        }
    elif kind == "expected_value":
        settings = _fields(raw, settings_path, ("requested_method", "method"))
        result = {
            "requested_method": _choice(
                settings["requested_method"],
                pointer(settings_path, "requested_method"),
                ("auto", "exact", "gauss_kronrod"),
            ),
            "method": _integration_method(settings["method"], pointer(settings_path, "method")),
        }
        if result["requested_method"] not in ("auto", result["method"]["kind"]):
            _fail("invalid_value", pointer(settings_path, "method"), "resolved method contradicts the request")
    elif kind == "transform_distribution":
        settings = _fields(raw, settings_path, ("algorithm",))
        result = {"algorithm": identifier(settings["algorithm"], pointer(settings_path, "algorithm"))}
    else:
        settings = _fields(raw, settings_path, ("probabilities", "interpolation", "approximate"))
        grid_path = pointer(settings_path, "probabilities")
        grid = [
            _number(item, pointer(grid_path, index), minimum=0.0, maximum=1.0)
            for index, item in enumerate(_array(settings["probabilities"], grid_path, min_items=1))
        ]
        _increasing(grid, grid_path, None)
        result = {
            "probabilities": grid,
            "interpolation": _choice(
                settings["interpolation"], pointer(settings_path, "interpolation"), TABULATION_INTERPOLATIONS
            ),
            "approximate": _constant(settings["approximate"], pointer(settings_path, "approximate"), True),
        }
    return {"kind": kind, "settings": result}


def _function(value: Any, path: str) -> Dict[str, Any]:
    kind = _kind(value, path, ("definition", "custom"))
    if kind == "definition":
        fields = _fields(value, path, ("kind", "definition_hash"))
        return {
            "kind": kind,
            "definition_hash": _content_hash(fields["definition_hash"], pointer(path, "definition_hash")),
        }
    fields = _fields(value, path, ("kind",), ("module", "qualified_name", "type"))
    return {
        "kind": kind,
        **{
            name: _optional_identifier(fields.get(name), pointer(path, name))
            for name in ("module", "qualified_name", "type")
        },
    }


def _engine(value: Any, path: str) -> Dict[str, Any]:
    fields = _fields(
        value,
        path,
        ("framework_version", "target", "numerical_methods"),
        ("source_revision", "build_id"),
    )
    methods_path = pointer(path, "numerical_methods")
    methods = fields["numerical_methods"]
    if not isinstance(methods, dict):
        _fail("invalid_type", methods_path, "expected an object")
    return {
        "framework_version": identifier(fields["framework_version"], pointer(path, "framework_version")),
        "source_revision": _optional_identifier(fields.get("source_revision"), pointer(path, "source_revision")),
        "build_id": _optional_identifier(fields.get("build_id"), pointer(path, "build_id")),
        "target": identifier(fields["target"], pointer(path, "target")),
        "numerical_methods": {
            identifier(name, pointer(methods_path, name)): identifier(
                methods[name], pointer(methods_path, name)
            )
            for name in sorted(methods, key=_utf8)
        },
    }


def _parameter_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    return _number(value, path)


def _preprocessing(value: Any, path: str) -> List[Dict[str, Any]]:
    steps = []
    for index, item in enumerate(_array(value, path)):
        location = pointer(path, index)
        fields = _fields(
            item,
            location,
            ("operation",),
            ("description", "parameters", "input_digest", "output_digest"),
        )
        parameters = fields.get("parameters", {})
        parameters_path = pointer(location, "parameters")
        if not isinstance(parameters, dict):
            _fail("invalid_type", parameters_path, "expected an object")
        steps.append(
            {
                "operation": identifier(fields["operation"], pointer(location, "operation")),
                "description": _optional_text(fields.get("description"), pointer(location, "description")),
                "parameters": {
                    identifier(name, pointer(parameters_path, name)): _parameter_value(
                        parameters[name], pointer(parameters_path, name)
                    )
                    for name in sorted(parameters, key=_utf8)
                },
                "input_digest": _optional_digest(fields.get("input_digest"), pointer(location, "input_digest")),
                "output_digest": _optional_digest(fields.get("output_digest"), pointer(location, "output_digest")),
            }
        )
    return steps


def _resolution(value: Any, path: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    fields = _fields(value, path, ("reference_hash", "context", "context_hash"))
    context = normalize_context(fields["context"], pointer(path, "context"))
    context_hash = _content_hash(fields["context_hash"], pointer(path, "context_hash"))
    if context_hash != content_hash(context):
        _fail("invalid_value", pointer(path, "context_hash"), "does not match the canonical context")
    return {
        "reference_hash": _content_hash(fields["reference_hash"], pointer(path, "reference_hash")),
        "context": context,
        "context_hash": context_hash,
    }


def _diagnostics(value: Any, path: str) -> Dict[str, Any]:
    fields = _fields(value, path, ("exactness",), ("estimated_error", "evaluations", "components"))
    components_path = pointer(path, "components")
    components = []
    for index, item in enumerate(_array(fields.get("components", []), components_path)):
        location = pointer(components_path, index)
        component = _fields(item, location, ("kind", "value"), ("estimated_error", "evaluations"))
        components.append(
            {
                "kind": _choice(component["kind"], pointer(location, "kind"), ("exact", "numerical")),
                "value": _number(component["value"], pointer(location, "value")),
                "estimated_error": _optional_number(
                    component.get("estimated_error"), pointer(location, "estimated_error"), minimum=0.0
                ),
                "evaluations": _optional_integer(component.get("evaluations"), pointer(location, "evaluations")),
            }
        )
    return {
        "exactness": _choice(
            fields["exactness"], pointer(path, "exactness"), ("direct", "exact", "numerical", "mixed")
        ),
        "estimated_error": _optional_number(fields.get("estimated_error"), pointer(path, "estimated_error"), minimum=0.0),
        "evaluations": _optional_integer(fields.get("evaluations"), pointer(path, "evaluations")),
        "components": components,
    }


def _replay(value: Any, path: str) -> Dict[str, Any]:
    fields = _fields(value, path, ("status", "missing_requirements"))
    status = _choice(
        fields["status"], pointer(path, "status"), ("complete", "missing_source_snapshot", "runtime_only")
    )
    missing_path = pointer(path, "missing_requirements")
    missing: List[str] = []
    for index, item in enumerate(_array(fields["missing_requirements"], missing_path)):
        requirement = _choice(item, pointer(missing_path, index), REPLAY_REQUIREMENTS)
        if requirement in missing:
            _fail("duplicate_identifier", pointer(missing_path, index), "duplicate requirement")
        missing.append(requirement)
    if (status == "complete") != (not missing):
        _fail("invalid_value", missing_path, "complete replay requires, and only allows, no missing requirements")
    if status == "missing_source_snapshot" and "source_snapshot" not in missing:
        _fail("invalid_value", missing_path, "missing_source_snapshot must list source_snapshot")
    return {"status": status, "missing_requirements": sorted(missing)}


def normalize_evaluation_record(value: Any, path: str = "") -> Dict[str, Any]:
    _version(value, path, "schema_version", RECORD_SCHEMA_VERSION, "unsupported_schema_version")
    names = ("schema_version", "function", "operation", "engine", "source", "preprocessing", "resolution", "diagnostics", "replay")
    fields = _fields(value, path, names)
    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "function": _function(fields["function"], pointer(path, "function")),
        "operation": _operation(fields["operation"], pointer(path, "operation")),
        "engine": _engine(fields["engine"], pointer(path, "engine")),
        "source": _record_source(fields["source"], pointer(path, "source")),
        "preprocessing": _preprocessing(fields["preprocessing"], pointer(path, "preprocessing")),
        "resolution": _resolution(fields["resolution"], pointer(path, "resolution")),
        "diagnostics": _diagnostics(fields["diagnostics"], pointer(path, "diagnostics")),
        "replay": _replay(fields["replay"], pointer(path, "replay")),
    }
    status_path = pointer(pointer(path, "replay"), "status")
    status = record["replay"]["status"]
    if record["function"]["kind"] == "custom" and status != "runtime_only":
        _fail("invalid_value", status_path, "custom functions are replayable only at runtime")
    if record["function"]["kind"] == "definition" and status == "runtime_only":
        _fail("invalid_value", status_path, "definition functions are not runtime-only")
    if status == "complete" and record["source"].get("snapshot") is None:
        _fail("invalid_value", status_path, "complete replay requires a source snapshot")
    return record


# Spatial combination ------------------------------------------------------


def normalize_spatial_policy(value: Any, path: str = "") -> Dict[str, Any]:
    _version(value, path, "schema_version", POLICY_SCHEMA_VERSION, "unsupported_schema_version")
    _version(value, path, "semantics_version", SPATIAL_SEMANTICS_VERSION, "unsupported_semantics_version")
    fields = _fields(
        value,
        path,
        ("schema_version", "semantics_version", "stage", "weights"),
        ("reducer", "coverage"),
    )
    weights_path = pointer(path, "weights")
    kind = _kind(fields["weights"], weights_path, ("intersection_area", "provided"))
    if kind == "intersection_area":
        _fields(fields["weights"], weights_path, ("kind",))
        weights: Dict[str, Any] = {"kind": kind}
    else:
        raw = _fields(fields["weights"], weights_path, ("kind", "meaning"))
        weights = {"kind": kind, "meaning": _text(raw["meaning"], pointer(weights_path, "meaning"))}
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "semantics_version": SPATIAL_SEMANTICS_VERSION,
        "stage": _choice(fields["stage"], pointer(path, "stage"), ("exposure", "impact")),
        "reducer": _choice(fields.get("reducer", "weighted_mean"), pointer(path, "reducer"), ("weighted_mean",)),
        "weights": weights,
        "coverage": _choice(
            fields.get("coverage", "require_full"), pointer(path, "coverage"), ("require_full", "covered_only")
        ),
    }


def _target(value: Any, path: str) -> Dict[str, Any]:
    fields = _fields(value, path, ("id", "kind"), ("cell", "geometry_sha256"))
    kind = _choice(fields["kind"], pointer(path, "kind"), ("h3_cell", "geometry"))
    cell = fields.get("cell")
    if cell is not None:
        cell = _string(cell, pointer(path, "cell"))
        if not _H3_CELL.fullmatch(cell):
            _fail("invalid_value", pointer(path, "cell"), "must be 15 lowercase hex characters")
    geometry = _optional_sha256(fields.get("geometry_sha256"), pointer(path, "geometry_sha256"))
    if kind == "h3_cell" and cell is None:
        _fail("invalid_value", pointer(path, "cell"), "H3 targets require a cell")
    if kind == "geometry" and geometry is None:
        _fail("invalid_value", pointer(path, "geometry_sha256"), "geometry targets require a geometry digest")
    return {"id": identifier(fields["id"], pointer(path, "id")), "kind": kind, "cell": cell, "geometry_sha256": geometry}


def _geometry(value: Any, path: str) -> Dict[str, Any]:
    names = ("engine", "engine_version", "crs", "area_method", "target_area", "covered_fraction", "overlap_fraction", "tolerance")
    fields = _fields(value, path, names)
    result: Dict[str, Any] = {
        name: identifier(fields[name], pointer(path, name)) for name in names[:4]
    }
    result["target_area"] = _number(fields["target_area"], pointer(path, "target_area"), positive=True)
    result["covered_fraction"] = _number(
        fields["covered_fraction"], pointer(path, "covered_fraction"), minimum=0.0, maximum=1.0
    )
    result["overlap_fraction"] = _number(
        fields["overlap_fraction"], pointer(path, "overlap_fraction"), minimum=0.0, maximum=1.0
    )
    result["tolerance"] = _number(fields["tolerance"], pointer(path, "tolerance"), minimum=0.0)
    if result["tolerance"] != SPATIAL_GEOMETRY_TOLERANCE:
        _fail("invalid_value", pointer(path, "tolerance"), "crc-spatial-combination/1 requires tolerance 1e-9")
    if result["overlap_fraction"] > result["tolerance"]:
        _fail("invalid_value", pointer(path, "overlap_fraction"), "positive-area overlap exceeds tolerance")
    return result


def normalize_spatial_record(value: Any, path: str = "") -> Dict[str, Any]:
    _version(value, path, "schema_version", SPATIAL_RECORD_SCHEMA_VERSION, "unsupported_schema_version")
    names = (
        "schema_version", "policy", "policy_hash", "target", "contributors", "contributor_count",
        "geometry", "weights", "combined_exposure", "children", "replay",
    )
    fields = _fields(value, path, names)
    policy = normalize_spatial_policy(fields["policy"], pointer(path, "policy"))
    policy_hash = _content_hash(fields["policy_hash"], pointer(path, "policy_hash"))
    if policy_hash != content_hash(policy):
        _fail("invalid_value", pointer(path, "policy_hash"), "does not match the canonical policy")
    count = _integer(fields["contributor_count"], pointer(path, "contributor_count"), minimum=1)
    geometry = _geometry(fields["geometry"], pointer(path, "geometry"))
    if policy["coverage"] == "require_full" and 1.0 - geometry["covered_fraction"] > geometry["tolerance"]:
        _fail("invalid_value", pointer(pointer(path, "geometry"), "covered_fraction"), "require_full needs full coverage")
    weights_path = pointer(path, "weights")
    raw_weights = _fields(fields["weights"], weights_path, ("raw", "normalized"), ("unit",))
    weights = {
        "raw": _snapshot(raw_weights["raw"], pointer(weights_path, "raw")),
        "normalized": _snapshot(raw_weights["normalized"], pointer(weights_path, "normalized")),
        "unit": _optional_identifier(raw_weights.get("unit"), pointer(weights_path, "unit")),
    }
    if policy["weights"]["kind"] == "provided" and weights["unit"] is None:
        _fail("invalid_value", pointer(weights_path, "unit"), "provided weights require a common unit")
    combined = _optional_digest(fields["combined_exposure"], pointer(path, "combined_exposure"))
    children_path = pointer(path, "children")
    children = []
    for index, item in enumerate(_array(fields["children"], children_path, min_items=1)):
        location = pointer(children_path, index)
        child = _fields(item, location, ("record_hash",), ("contributor_id", "uri"))
        children.append(
            {
                "contributor_id": _optional_identifier(child.get("contributor_id"), pointer(location, "contributor_id")),
                "record_hash": _content_hash(child["record_hash"], pointer(location, "record_hash")),
                "uri": _optional_identifier(child.get("uri"), pointer(location, "uri")),
            }
        )
    if policy["stage"] == "exposure":
        if combined is None:
            _fail("invalid_value", pointer(path, "combined_exposure"), "exposure stage requires the combined exposure")
        if len(children) != 1 or children[0]["contributor_id"] is not None:
            _fail("invalid_value", children_path, "exposure stage has one target child record")
    else:
        if combined is not None:
            _fail("invalid_value", pointer(path, "combined_exposure"), "impact stage combines responses, not exposure")
        ids = [child["contributor_id"] for child in children]
        if len(children) != count or None in ids:
            _fail("invalid_value", children_path, "impact stage has one child record per contributor")
        for index, contributor in enumerate(ids):
            if contributor in ids[:index]:
                _fail("duplicate_identifier", pointer(pointer(children_path, index), "contributor_id"), "duplicate contributor")
    return {
        "schema_version": SPATIAL_RECORD_SCHEMA_VERSION,
        "policy": policy,
        "policy_hash": policy_hash,
        "target": _target(fields["target"], pointer(path, "target")),
        "contributors": _snapshot(fields["contributors"], pointer(path, "contributors")),
        "contributor_count": count,
        "geometry": geometry,
        "weights": weights,
        "combined_exposure": combined,
        "children": children,
        "replay": _replay(fields["replay"], pointer(path, "replay")),
    }


# Canonical JSON -----------------------------------------------------------


def format_binary64(value: float) -> str:
    """Shortest round-trip binary64 text using the frozen native profile.

    The digits are Python's shortest round-trip digits. Their placement
    follows the serde_json (zmij) rules: plain notation, with a trailing
    ``.0`` for integral values, when the scientific exponent is in [-5, 15];
    otherwise ``d.ddde+x`` or ``d.ddde-x`` with an explicit sign and no
    padding. Classic ryu omits the ``+``, so serializer upgrades must be
    checked against these vectors.
    """
    if not math.isfinite(value):
        raise ValueError("canonical JSON numbers must be finite")
    if value == 0.0:
        return "0.0"
    sign = "-" if value < 0.0 else ""
    _, digit_tuple, exponent = Decimal(repr(abs(value))).as_tuple()
    digits = list(digit_tuple)
    assert isinstance(exponent, int)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    while len(digits) > 1 and digits[0] == 0:
        digits.pop(0)
    text = "".join(str(digit) for digit in digits)
    length = len(text)
    position = length + exponent
    if 0 <= exponent and position <= 16:
        body = text + "0" * exponent + ".0"
    elif 0 < position <= 16:
        body = text[:position] + "." + text[position:]
    elif -5 < position <= 0:
        body = "0." + "0" * (-position) + text
    else:
        mantissa = text if length == 1 else f"{text[0]}.{text[1:]}"
        exponent_value = position - 1
        body = f"{mantissa}e{'+' if exponent_value >= 0 else '-'}{abs(exponent_value)}"
    return sign + body


_SHORT_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _encode_string(value: str) -> str:
    parts = ['"']
    for character in value:
        if character in _SHORT_ESCAPES:
            parts.append(_SHORT_ESCAPES[character])
        elif ord(character) < 0x20:
            parts.append(f"\\u{ord(character):04x}")
        else:
            parts.append(character)
    parts.append('"')
    return "".join(parts)


def _encode(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_binary64(value)
    if isinstance(value, str):
        return _encode_string(value)
    if isinstance(value, list):
        return "[" + ",".join(_encode(item) for item in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value, key=_utf8)
        return "{" + ",".join(_encode_string(key) + ":" + _encode(value[key]) for key in keys) + "}"
    raise TypeError(f"cannot canonicalize {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Encode an already-normalized contract value with the canonical profile."""
    return _encode(value).encode("utf-8")


def content_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


# Point kernels ------------------------------------------------------------


def _exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def logistic(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + _exp(-value))
    exponential = _exp(value)
    return exponential / (1.0 + exponential)


SPEI_MIDPOINT = -1.0 + math.log(199.0) / -3.0


def _table(model: Mapping[str, Any], value: float) -> float:
    if model["kind"] == "tabulated":
        xs = [point["x"] for point in model["points"]]
        ys = [point["y"] for point in model["points"]]
    else:
        xs = model["bin_edges"]
        ys = model["values"]
    if value < xs[0] or value > xs[-1]:
        if model["outside_domain"] == "error":
            _fail("out_of_domain", "", f"exposure {value!r} is outside [{xs[0]!r}, {xs[-1]!r}]")
        return ys[0] if value < xs[0] else ys[-1]
    if model["kind"] == "binned":
        for index in range(len(ys)):
            if value < xs[index + 1]:
                return ys[index]
        return ys[-1]
    for index, knot in enumerate(xs):
        if value == knot:
            return ys[index]
    upper = next(index for index, knot in enumerate(xs) if knot > value)
    lower = upper - 1
    mode = model["interpolation"]
    if mode == "step_left":
        return ys[lower]
    if mode == "step_right":
        return ys[upper]
    weight = (value - xs[lower]) / (xs[upper] - xs[lower])
    return ys[lower] + weight * (ys[upper] - ys[lower])


def raw_response(model: Mapping[str, Any], value: float) -> float:
    kind = model["kind"]
    if kind == "linear":
        return model["slope"] * value + model["intercept"]
    if kind == "sigmoid":
        threshold = model["zero_below"]
        if threshold is not None and value < threshold:
            return 0.0
        if model["amplitude"] == 0.0:
            return model["offset"]
        unit = 0.5 if model["steepness"] == 0.0 else logistic(model["steepness"] * (value - model["midpoint"]))
        if model["origin_adjustment"] == "subtract_at_zero":
            unit -= logistic(model["steepness"] * (0.0 - model["midpoint"]))
        return model["offset"] + model["amplitude"] * unit
    if kind in ("tabulated", "binned"):
        return _table(model, value)
    if kind == "spei":
        drought = logistic(-3.0 * (value - SPEI_MIDPOINT))
        cost = model["original_water_unit_cost"]
        if value < 0.0:
            cost = cost * (1.0 + 10.0 * drought)
        return cost * model["water_consumption_asset"] / 10_000_000.0
    if kind == "cooling":
        if value <= 1.0:
            return 0.0
        return model["cooling_cost"] * value / (model["historic_mean"] * 10_000_000.0)
    total = 0.0
    for crop in CROPS:
        base = model["country"][crop] + model["yearly"][crop] * 2010.0
        historical = _exp(base + model["index"][crop] * model["historic_mean"])
        future = _exp(base + model["index"][crop] * value)
        term = model["weights"][crop] * -(future - historical) / historical if historical > 0.0 else 0.0
        total = total + term
    return total


def apply_clamp(clamp: Optional[Mapping[str, Any]], value: float) -> float:
    if clamp is not None:
        if clamp["minimum"] is not None:
            value = max(value, clamp["minimum"])
        if clamp["maximum"] is not None:
            value = min(value, clamp["maximum"])
    return 0.0 if value == 0.0 else value


def evaluate(definition: Mapping[str, Any], value: float) -> float:
    """Evaluate a normalized definition at one exposure value."""
    if not math.isfinite(value):
        _fail("non_finite_exposure", "", "exposure values must be finite")
    result = raw_response(definition["model"], value)
    if not math.isfinite(result):
        _fail("non_finite_result", "", "impact evaluation produced a non-finite response")
    return apply_clamp(definition["clamp"], result)


# Monotonicity -------------------------------------------------------------


def _sequence_class(values: Sequence[float]) -> str:
    steps = [right - left for left, right in zip(values, values[1:])]
    if all(step == 0.0 for step in steps):
        return "constant"
    if all(step >= 0.0 for step in steps):
        return "non_decreasing"
    if all(step <= 0.0 for step in steps):
        return "non_increasing"
    return "non_monotone"


def _jump(base: str, before: float, at: float) -> str:
    """Combine a constant prefix ``before`` with a branch starting at ``at``."""
    rising = at > before
    falling = at < before
    if base == "constant":
        return "constant" if at == before else ("non_decreasing" if rising else "non_increasing")
    if base == "non_decreasing":
        return "non_monotone" if falling else "non_decreasing"
    return "non_monotone" if rising else "non_increasing"


def _clamped(direction: str, clamp: Optional[Mapping[str, Any]]) -> str:
    if clamp is not None and clamp["minimum"] is not None and clamp["minimum"] == clamp["maximum"]:
        return "constant"
    return direction


def monotonicity(definition: Mapping[str, Any]) -> str:
    """A true certified ordering of the full function, including its clamp.

    The result is the most specific class this reference can prove; for
    mixed-sign crop responses it conservatively returns ``unknown``.
    """
    model = definition["model"]
    clamp = definition["clamp"]
    if clamp is not None and clamp["minimum"] is not None and clamp["minimum"] == clamp["maximum"]:
        return "constant"
    kind = model["kind"]
    if kind == "linear":
        slope = model["slope"]
        direction = "constant" if slope == 0.0 else ("non_decreasing" if slope > 0.0 else "non_increasing")
    elif kind == "sigmoid":
        if model["amplitude"] == 0.0 or model["steepness"] == 0.0:
            base = "constant"
        else:
            base = "non_decreasing" if model["steepness"] > 0.0 else "non_increasing"
        threshold = model["zero_below"]
        if threshold is None:
            direction = base
        else:
            before = apply_clamp(clamp, 0.0)
            at = apply_clamp(clamp, raw_response(model, threshold))
            direction = _jump(base, before, at)
    elif kind in ("tabulated", "binned"):
        values = [point["y"] for point in model["points"]] if kind == "tabulated" else model["values"]
        direction = _sequence_class([apply_clamp(clamp, value) for value in values])
    elif kind == "spei":
        scale = model["original_water_unit_cost"] * model["water_consumption_asset"]
        direction = "constant" if scale == 0.0 else "non_increasing"
    elif kind == "cooling":
        direction = "constant" if model["cooling_cost"] == 0.0 else "non_decreasing"
    else:
        # Validated weights are non-negative. Multiplication can underflow and
        # erase the sign of a real response derivative, so inspect factors.
        signs = {
            (model["index"][crop] > 0.0) - (model["index"][crop] < 0.0)
            for crop in CROPS if model["weights"][crop] > 0.0
        } - {0}
        if not signs:
            direction = "constant"
        elif signs == {1}:
            direction = "non_increasing"
        elif signs == {-1}:
            direction = "non_decreasing"
        else:
            direction = "unknown"
    if direction in ("non_monotone", "unknown"):
        return direction
    return _clamped(direction, clamp)


# Histogram mass -----------------------------------------------------------


def neumaier_sum(values: Sequence[float]) -> float:
    total = 0.0
    compensation = 0.0
    for value in values:
        candidate = total + value
        if abs(total) >= abs(value):
            compensation += (total - candidate) + value
        else:
            compensation += (value - candidate) + total
        total = candidate
    return total + compensation


def histogram_mass(weights: Sequence[float]) -> Dict[str, Any]:
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        _fail("invalid_histogram", "/weights", "weights must be finite and non-negative")
    total = neumaier_sum(weights)
    if not total > 0.0 or abs(total - 1.0) > HISTOGRAM_MASS_TOLERANCE:
        _fail("invalid_histogram", "/weights", "total mass must be one within 1e-12")
    return {
        "original_mass_sum": total,
        "normalization_factor": 1.0 / total,
        "normalized_weights": [weight / total for weight in weights],
    }


NORMALIZERS: Dict[str, Callable[[Any], Dict[str, Any]]] = {
    "impact_definition": normalize_definition,
    "registry_impact_reference": normalize_reference,
    "impact_evaluation_record": normalize_evaluation_record,
    "spatial_combination_policy": normalize_spatial_policy,
    "spatial_impact_evaluation_record": normalize_spatial_record,
}

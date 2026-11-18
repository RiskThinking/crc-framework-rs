"""Typed scientific measurements, exact model contexts, and schema validation."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache
from importlib.resources import files
from types import MappingProxyType
from typing import ClassVar, Protocol, TypeAlias, cast, runtime_checkable
from urllib.parse import unquote

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

from . import _core
from ._diagnostics import ScientificValue


class LengthUnit(Enum):
    METRE = "m"
    CENTIMETRE = "cm"
    MILLIMETRE = "mm"
    FOOT = "ft"


class TimeUnit(Enum):
    SECOND = "s"
    HOUR = "h"
    DAY = "d"
    MONTH = "month"
    YEAR = "yr"


class RatioUnit(Enum):
    FRACTION = "1"
    PERCENT = "percent"


class TemperatureUnit(Enum):
    CELSIUS = "degC"
    KELVIN = "K"
    FAHRENHEIT = "degF"


class SpeedUnit(Enum):
    METRES_PER_SECOND = "m/s"
    KILOMETRES_PER_HOUR = "km/h"


class PrecipitationRateUnit(Enum):
    MILLIMETRES_PER_DAY = "mm/day"
    MILLIMETRES_PER_HOUR = "mm/h"


class DegreeDayUnit(Enum):
    KELVIN_DAYS = "K.d"
    CELSIUS_DAYS = "degC.d"


Unit: TypeAlias = (
    LengthUnit
    | TimeUnit
    | RatioUnit
    | TemperatureUnit
    | SpeedUnit
    | PrecipitationRateUnit
    | DegreeDayUnit
)
_UNIT_TYPES = (
    LengthUnit,
    TimeUnit,
    RatioUnit,
    TemperatureUnit,
    SpeedUnit,
    PrecipitationRateUnit,
    DegreeDayUnit,
)


class Aggregation(Enum):
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    MEAN = "mean"
    SUM = "sum"
    COUNT = "count"


class Event(Enum):
    FREEZE_THAW_CYCLE = "freeze_thaw_cycle"
    FROST_DAY = "frost_day"
    HOT_DAY = "hot_day"
    DRY_DAY = "dry_day"


class ClimateIndexKind(Enum):
    FIRE_WEATHER_INDEX = "fire_weather_index"
    STANDARDIZED_PRECIPITATION_EVAPOTRANSPIRATION_INDEX = (
        "standardized_precipitation_evapotranspiration_index"
    )


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(
            f"{name} must be a nonempty exact string without boundary whitespace"
        )
    return value


def _finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _freeze(value: ScientificValue) -> ScientificValue:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("scientific JSON object keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(_freeze(v) for v in value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("scientific values must be finite JSON values")


def _plain(value: ScientificValue) -> ScientificValue:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_plain(v) for v in value]
    return value


def _dumps(value: object) -> str:
    return json.dumps(
        _canonical_value(cast(ScientificValue, value)),
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _canonical_value(value: ScientificValue) -> ScientificValue:
    """Canonical JSON-number spelling, with booleans kept distinct from numbers."""
    if isinstance(value, Mapping):
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def _scientific_equal(left: ScientificValue, right: ScientificValue) -> bool:
    # Comparing canonical JSON keeps False different from 0, unlike Python's
    # ordinary container equality, while 3 and 3.0 share scientific meaning.
    return _dumps(left) == _dumps(right)


def _parse(text: str) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate scientific field {key!r}")
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise ValueError(f"nonfinite scientific JSON number {value}")

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise TypeError("scientific record must be an object")
    _freeze(cast(ScientificValue, value))
    return value


def _enum(value: object, enum: type[Enum], name: str) -> None:
    if not isinstance(value, enum):
        raise TypeError(
            f"{name} must be {enum.__name__}; accepted members: "
            + ", ".join(v.name for v in enum)
        )


def _object(
    properties: dict[str, object], required: Sequence[str]
) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _choice(enum: type[Enum]) -> dict[str, object]:
    return {"type": "string", "enum": [v.value for v in enum]}


_IDENTIFIER_SCHEMA: dict[str, object] = {
    "type": "string",
    "minLength": 1,
    "pattern": r"^\S(?:[\s\S]*\S)?$",
}
_NUMBER: dict[str, object] = {"type": "number"}
_DURATION_SCHEMA = _object(
    {"value": {"type": "number", "exclusiveMinimum": 0}, "unit": _choice(TimeUnit)},
    ("value", "unit"),
)
_THRESHOLD_SCHEMA = _object(
    {"value": _NUMBER, "unit": _IDENTIFIER_SCHEMA}, ("value", "unit")
)
_REFERENCE_SCHEMA = _object(
    {"start": {"type": "integer"}, "end": {"type": "integer"}}, ("start", "end")
)


@dataclass(frozen=True)
class Duration:
    """A positive duration; a year does not imply a calendar or fixed day count."""

    value: float
    unit: TimeUnit

    def __init__(self, value: float, *, unit: TimeUnit) -> None:
        _enum(unit, TimeUnit, "unit")
        value = _finite(value, "value")
        if value <= 0:
            raise ValueError("duration value must be positive")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "unit", unit)

    def _document(self) -> dict[str, ScientificValue]:
        return {"value": self.value, "unit": self.unit.value}

    def to_json(self) -> str:
        return _dumps(self._document())


@dataclass(frozen=True)
class Threshold:
    """An explicitly stated threshold in an actual physical unit."""

    value: float
    unit: Unit | str

    def __init__(self, value: float, *, unit: Unit | str) -> None:
        if not isinstance(unit, _UNIT_TYPES):
            _identifier(unit, "unit")
        object.__setattr__(self, "value", _finite(value, "value"))
        object.__setattr__(self, "unit", unit)

    def _document(self) -> dict[str, ScientificValue]:
        return {
            "value": self.value,
            "unit": self.unit.value if isinstance(self.unit, Enum) else self.unit,
        }

    def to_json(self) -> str:
        return _dumps(self._document())


@dataclass(frozen=True)
class ReferencePeriod:
    """Inclusive reference years. An unspecified calendar remains unspecified."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or self.start > self.end
        ):
            raise ValueError(
                "reference start and end must be inclusive ordered integer years"
            )

    def _document(self) -> dict[str, ScientificValue]:
        return {"start": self.start, "end": self.end}

    def to_json(self) -> str:
        return _dumps(self._document())


def _value(value: object) -> ScientificValue:
    if isinstance(value, Enum):
        return cast(ScientificValue, value.value)
    if isinstance(value, (Duration, Threshold, ReferencePeriod)):
        return value._document()
    return cast(ScientificValue, value)


# One authoritative field definition feeds runtime validation, published JSON
# Schema, model requirements and the generated scientific reference.
_MEASUREMENT_FIELDS: dict[str, tuple[str, dict[str, object], tuple[str, ...]]] = {
    "measurement": (
        "Open user-owned scientific meaning.",
        {
            "quantity": _IDENTIFIER_SCHEMA,
            "unit": _IDENTIFIER_SCHEMA,
            "fields": {"type": "object"},
            "schema": {"type": "object"},
        },
        (),
    ),
    "depth_above_ground": (
        "Depth measured above the local ground surface.",
        {"unit": _choice(LengthUnit)},
        ("unit",),
    ),
    "water_surface_elevation": (
        "Water surface elevation in the explicitly identified vertical datum.",
        {"unit": _choice(LengthUnit), "datum": _IDENTIFIER_SCHEMA},
        ("unit", "datum"),
    ),
    "ratio": (
        "Dimensionless ratio in fractions or percent.",
        {"unit": _choice(RatioUnit)},
        ("unit",),
    ),
    "time_duration": (
        "Elapsed time in an explicitly selected time unit.",
        {"unit": _choice(TimeUnit)},
        ("unit",),
    ),
    "temperature": (
        "Temperature in the selected scale.",
        {"unit": _choice(TemperatureUnit)},
        ("unit",),
    ),
    "wind_speed": (
        "Wind speed with optional explicit averaging and reduction facts.",
        {
            "unit": _choice(SpeedUnit),
            "averaging_duration": _DURATION_SCHEMA,
            "aggregation": _choice(Aggregation),
        },
        ("unit",),
    ),
    "precipitation_rate": (
        "Precipitation rate over an explicit accumulation duration and reduction.",
        {
            "unit": _choice(PrecipitationRateUnit),
            "accumulation_duration": _DURATION_SCHEMA,
            "aggregation": _choice(Aggregation),
        },
        ("unit", "accumulation_duration", "aggregation"),
    ),
    "precipitation_amount": (
        "Accumulated precipitation amount over an explicit interval.",
        {"unit": _choice(LengthUnit), "accumulation_duration": _DURATION_SCHEMA},
        ("unit", "accumulation_duration"),
    ),
    "event_count": (
        "Count of explicitly identified events, with optional threshold and counting duration.",
        {
            "event": _choice(Event),
            "aggregation": _choice(Aggregation),
            "threshold": _THRESHOLD_SCHEMA,
            "duration": _DURATION_SCHEMA,
        },
        ("event", "aggregation"),
    ),
    "climate_index": (
        "An identified climate index, with explicit timescale and reference years when known.",
        {
            "index": _choice(ClimateIndexKind),
            "accumulation_duration": _DURATION_SCHEMA,
            "reference_period": _REFERENCE_SCHEMA,
        },
        ("index",),
    ),
    "degree_days": (
        "Integrated temperature departures from a declared threshold over a declared duration.",
        {
            "unit": _choice(DegreeDayUnit),
            "threshold": _THRESHOLD_SCHEMA,
            "duration": _DURATION_SCHEMA,
        },
        ("unit", "threshold", "duration"),
    ),
}


_MEASUREMENT_TYPE_NAMES = {
    "measurement": "Measurement",
    "depth_above_ground": "DepthAboveGround",
    "water_surface_elevation": "WaterSurfaceElevation",
    "ratio": "Ratio",
    "time_duration": "TimeDuration",
    "temperature": "Temperature",
    "wind_speed": "WindSpeed",
    "precipitation_rate": "PrecipitationRate",
    "precipitation_amount": "PrecipitationAmount",
    "event_count": "EventCount",
    "climate_index": "ClimateIndex",
    "degree_days": "DegreeDays",
}


def _measurement_schema(kind: str) -> dict[str, object]:
    description, fields, required = _MEASUREMENT_FIELDS[kind]
    calibration = (
        {"quantity_field": "quantity", "unit_field": "unit"}
        if kind == "measurement"
        else {"quantity": kind, "fixed_unit": "count"}
        if kind == "event_count"
        else {"quantity": kind, "fixed_unit": "1"}
        if kind == "climate_index"
        else {"quantity": kind, "unit_field": "unit"}
    )
    schema = {
        "title": _MEASUREMENT_TYPE_NAMES[kind],
        "description": description,
        "x-crc-calibration": calibration,
        **_object({"kind": {"const": kind}, **fields}, ("kind", *required)),
    }
    if kind == "event_count":
        schema["allOf"] = [
            {
                "if": {
                    "properties": {
                        "event": {
                            "enum": [
                                Event.FROST_DAY.value,
                                Event.HOT_DAY.value,
                                Event.FREEZE_THAW_CYCLE.value,
                            ]
                        }
                    },
                    "required": ["event"],
                },
                "then": {
                    "properties": {
                        "threshold": {"properties": {"unit": _choice(TemperatureUnit)}}
                    }
                },
            }
        ]
    return schema


def _measurement_union_schema() -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://climateriskcommons.org/schemas/measurement-1.0.schema.json",
        "title": "Measurement",
        "oneOf": [_measurement_schema(kind) for kind in _MEASUREMENT_FIELDS],
    }


def _validate_measurement(document: Mapping[str, object]) -> None:
    kind = document.get("kind")
    if not isinstance(kind, str) or kind not in _MEASUREMENT_FIELDS:
        raise ValueError("unknown measurement kind")
    try:
        Draft202012Validator(_measurement_schema(kind)).validate(document)
    except ValidationError as error:
        path = "/".join(str(p) for p in error.absolute_path)
        raise ValueError(f"measurement/{path}: {error.message}") from error


def _local_references(value: object) -> None:
    """Inspect schema positions, leaving const/enum/default/example data opaque."""
    pending = [value]
    visited: set[int] = set()
    while pending:
        node = pending.pop()
        if not isinstance(node, dict) or id(node) in visited:
            continue
        visited.add(id(node))
        if (
            "$schema" in node
            and node["$schema"] != "https://json-schema.org/draft/2020-12/schema"
        ):
            raise ValueError("custom scientific schemas use JSON Schema draft 2020-12")
        for key in ("$ref", "$dynamicRef"):
            if key not in node:
                continue
            reference = node[key]
            if not isinstance(reference, str) or not reference.startswith("#"):
                raise ValueError(
                    "custom scientific schemas use local references; constructors never retrieve external schemas"
                )
            # A local JSON Pointer can target an otherwise opaque data position.
            # Once referenced, that target is being used as a schema as well.
            pointer = unquote(reference[1:])
            target = value
            if pointer.startswith("/"):
                try:
                    for part in pointer[1:].split("/"):
                        part = part.replace("~1", "/").replace("~0", "~")
                        target = (
                            target[int(part)]
                            if isinstance(target, list)
                            else cast(dict[str, object], target)[part]
                        )
                except (KeyError, IndexError, ValueError, TypeError):
                    continue
                pending.append(target)
        pending.extend(DRAFT202012.subresources_of(node))


@cache
def _scientific_validator(name: str) -> Draft202012Validator:
    root = files("crc_framework").joinpath("contracts", "scientific")
    registry = Registry()
    schema = None
    for filename in (
        "measurement-1.0.schema.json",
        "impact-context-1.0.schema.json",
        "impact-reference-2.0.schema.json",
        "impact-2.0.schema.json",
    ):
        document = json.loads(root.joinpath(filename).read_text(encoding="utf-8"))
        registry = registry.with_resource(
            document["$id"], Resource.from_contents(document)
        )
        if filename == name:
            schema = document
    if schema is None:
        raise ValueError("unknown scientific schema")
    return Draft202012Validator(schema, registry=registry)


def _validate_scientific(name: str, document: Mapping[str, object]) -> None:
    try:
        _scientific_validator(name).validate(document)
    except ValidationError as error:
        raise ValueError(
            f"{'/'.join(str(p) for p in error.absolute_path)}: {error.message}"
        ) from error


class Measurement:
    """Open scientific meaning; custom quantities and units require no registration.

    ``schema`` is an optional JSON Schema for the custom ``fields`` object. It is
    enforced and participates in scientific identity. Basic data access requires
    neither custom fields nor an extension schema.
    """

    _quantity: str | None
    _unit: Unit | str | None
    _fields: Mapping[str, ScientificValue]
    _schema: Mapping[str, ScientificValue] | None
    _kind: str
    _parameters: Mapping[str, object]
    __slots__ = ("_fields", "_kind", "_parameters", "_quantity", "_schema", "_unit")

    def __init__(
        self,
        *,
        quantity: str | None = None,
        unit: Unit | str | None = None,
        fields: Mapping[str, ScientificValue] | None = None,
        schema: Mapping[str, ScientificValue] | None = None,
    ) -> None:
        if quantity is not None:
            _identifier(quantity, "quantity")
        if unit is not None and not isinstance(unit, _UNIT_TYPES):
            _identifier(unit, "unit")
        if fields is not None and not isinstance(fields, Mapping):
            raise TypeError("fields must be a mapping")
        if schema is not None and not isinstance(schema, Mapping):
            raise TypeError("schema must be a mapping")
        frozen_fields = cast(
            Mapping[str, ScientificValue], _freeze({} if fields is None else fields)
        )
        frozen_schema = (
            None
            if schema is None
            else cast(Mapping[str, ScientificValue], _freeze(schema))
        )
        if frozen_schema is not None:
            try:
                raw_schema = cast(dict[str, object], _plain(frozen_schema))
                Draft202012Validator.check_schema(raw_schema)
                _local_references(raw_schema)
                Draft202012Validator(raw_schema).validate(_plain(frozen_fields))
            except (SchemaError, ValidationError, Unresolvable) as error:
                raise ValueError(
                    f"custom measurement fields do not satisfy schema: {getattr(error, 'message', str(error))}"
                ) from error
        object.__setattr__(self, "_kind", "measurement")
        object.__setattr__(self, "_quantity", quantity)
        object.__setattr__(self, "_unit", unit)
        object.__setattr__(self, "_fields", frozen_fields)
        object.__setattr__(self, "_schema", frozen_schema)
        object.__setattr__(self, "_parameters", MappingProxyType({}))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("measurements are immutable")

    @property
    def quantity(self) -> str | None:
        return self._quantity

    @property
    def unit(self) -> Unit | str | None:
        return self._unit

    @property
    def fields(self) -> Mapping[str, ScientificValue]:
        return self._fields

    @property
    def schema(self) -> Mapping[str, ScientificValue] | None:
        if self._kind == "measurement":
            return self._schema
        return cast(
            Mapping[str, ScientificValue],
            _freeze(cast(ScientificValue, _measurement_schema(self._kind))),
        )

    def _document(self) -> dict[str, ScientificValue]:
        result: dict[str, ScientificValue] = {"kind": self._kind}
        if self._kind != "measurement":
            result.update(
                {
                    name: _value(value)
                    for name, value in self._parameters.items()
                    if value is not None
                }
            )
            if "unit" in _MEASUREMENT_FIELDS[self._kind][1]:
                result["unit"] = _value(self.unit)
            return result
        if self.quantity is not None:
            result["quantity"] = self.quantity
        if self.unit is not None:
            result["unit"] = _value(self.unit)
        if self.fields:
            result["fields"] = _plain(self.fields)
        if self._schema is not None:
            result["schema"] = _plain(self._schema)
        return result

    def to_json(self) -> str:
        return _dumps(self._document())

    @classmethod
    def from_json(cls, text: str) -> Measurement:
        result = _measurement(_parse(text))
        if cls is not Measurement and not isinstance(result, cls):
            raise ValueError(
                f"expected {cls.__name__}, received {type(result).__name__}"
            )
        return result

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Measurement) and _scientific_equal(
            self._document(), other._document()
        )

    def __repr__(self) -> str:
        parameters = (
            self._parameters
            if self._kind != "measurement"
            else {"quantity": self.quantity, "unit": self.unit}
        )
        return (
            f"{type(self).__name__}("
            + ", ".join(f"{key}={value!r}" for key, value in parameters.items())
            + ")"
        )

    def _initialize(
        self, kind: str, quantity: str, unit: Unit | str, parameters: dict[str, object]
    ) -> None:
        Measurement.__init__(self, quantity=quantity, unit=unit)
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_parameters", MappingProxyType(parameters))
        document = self._document()
        _validate_measurement(document)
        object.__setattr__(
            self,
            "_fields",
            _freeze({k: v for k, v in document.items() if k not in ("kind", "unit")}),
        )


class _BuiltinMeasurement(Measurement):
    @property
    def quantity(self) -> str:
        return cast(str, self._quantity)

    @property
    def unit(self) -> Unit | str:
        return cast(Unit | str, self._unit)


class DepthAboveGround(_BuiltinMeasurement):
    """Depth measured above the local ground surface, never absolute elevation."""

    __slots__ = ()

    def __init__(self, *, unit: LengthUnit) -> None:
        _enum(unit, LengthUnit, "unit")
        self._initialize(
            "depth_above_ground", "depth_above_ground", unit, {"unit": unit}
        )

    @property
    def quantity(self) -> str:
        return "depth_above_ground"

    @property
    def unit(self) -> LengthUnit:
        return cast(LengthUnit, self._unit)


class WaterSurfaceElevation(_BuiltinMeasurement):
    """Water surface elevation relative to an explicitly identified vertical datum."""

    __slots__ = ()

    def __init__(self, *, unit: LengthUnit, datum: str) -> None:
        _enum(unit, LengthUnit, "unit")
        _identifier(datum, "datum")
        self._initialize(
            "water_surface_elevation",
            "water_surface_elevation",
            unit,
            {"unit": unit, "datum": datum},
        )

    @property
    def datum(self) -> str:
        return cast(str, self._parameters["datum"])

    @property
    def unit(self) -> LengthUnit:
        return cast(LengthUnit, self._unit)


class Ratio(_BuiltinMeasurement):
    """A dimensionless ratio in fractions or percent."""

    __slots__ = ()

    def __init__(self, *, unit: RatioUnit = RatioUnit.FRACTION) -> None:
        _enum(unit, RatioUnit, "unit")
        self._initialize("ratio", "ratio", unit, {"unit": unit})

    @property
    def quantity(self) -> str:
        return "ratio"

    @property
    def unit(self) -> RatioUnit:
        return cast(RatioUnit, self._unit)


class TimeDuration(_BuiltinMeasurement):
    """Elapsed time in the explicitly selected time unit."""

    __slots__ = ()

    def __init__(self, *, unit: TimeUnit) -> None:
        _enum(unit, TimeUnit, "unit")
        self._initialize("time_duration", "time_duration", unit, {"unit": unit})

    @property
    def quantity(self) -> str:
        return "time_duration"

    @property
    def unit(self) -> TimeUnit:
        return cast(TimeUnit, self._unit)


class Temperature(_BuiltinMeasurement):
    """Temperature in an explicit scale."""

    __slots__ = ()

    def __init__(self, *, unit: TemperatureUnit) -> None:
        _enum(unit, TemperatureUnit, "unit")
        self._initialize("temperature", "temperature", unit, {"unit": unit})

    @property
    def unit(self) -> TemperatureUnit:
        return cast(TemperatureUnit, self._unit)


class WindSpeed(_BuiltinMeasurement):
    """Wind speed, retaining averaging duration and aggregation only when known."""

    __slots__ = ()

    def __init__(
        self,
        *,
        unit: SpeedUnit,
        averaging_duration: Duration | None = None,
        aggregation: Aggregation | None = None,
    ) -> None:
        _enum(unit, SpeedUnit, "unit")
        if averaging_duration is not None and not isinstance(
            averaging_duration, Duration
        ):
            raise TypeError("averaging_duration must be Duration")
        if aggregation is not None:
            _enum(aggregation, Aggregation, "aggregation")
        self._initialize(
            "wind_speed",
            "wind_speed",
            unit,
            {
                "unit": unit,
                "averaging_duration": averaging_duration,
                "aggregation": aggregation,
            },
        )

    @property
    def unit(self) -> SpeedUnit:
        return cast(SpeedUnit, self._unit)

    @property
    def averaging_duration(self) -> Duration | None:
        return cast(Duration | None, self._parameters["averaging_duration"])

    @property
    def aggregation(self) -> Aggregation | None:
        return cast(Aggregation | None, self._parameters["aggregation"])


class PrecipitationRate(_BuiltinMeasurement):
    """Precipitation rate over a declared accumulation duration and aggregation."""

    __slots__ = ()

    def __init__(
        self,
        *,
        unit: PrecipitationRateUnit,
        accumulation_duration: Duration,
        aggregation: Aggregation,
    ) -> None:
        _enum(unit, PrecipitationRateUnit, "unit")
        _enum(aggregation, Aggregation, "aggregation")
        if not isinstance(accumulation_duration, Duration):
            raise TypeError("accumulation_duration must be Duration")
        self._initialize(
            "precipitation_rate",
            "precipitation_rate",
            unit,
            {
                "unit": unit,
                "accumulation_duration": accumulation_duration,
                "aggregation": aggregation,
            },
        )

    @property
    def unit(self) -> PrecipitationRateUnit:
        return cast(PrecipitationRateUnit, self._unit)

    @property
    def accumulation_duration(self) -> Duration:
        return cast(Duration, self._parameters["accumulation_duration"])

    @property
    def aggregation(self) -> Aggregation:
        return cast(Aggregation, self._parameters["aggregation"])


class PrecipitationAmount(_BuiltinMeasurement):
    """Accumulated precipitation, distinct from inundation depth and a rate."""

    __slots__ = ()

    def __init__(self, *, unit: LengthUnit, accumulation_duration: Duration) -> None:
        _enum(unit, LengthUnit, "unit")
        if not isinstance(accumulation_duration, Duration):
            raise TypeError("accumulation_duration must be Duration")
        self._initialize(
            "precipitation_amount",
            "precipitation_amount",
            unit,
            {"unit": unit, "accumulation_duration": accumulation_duration},
        )

    @property
    def unit(self) -> LengthUnit:
        return cast(LengthUnit, self._unit)

    @property
    def accumulation_duration(self) -> Duration:
        return cast(Duration, self._parameters["accumulation_duration"])


class EventCount(_BuiltinMeasurement):
    """Count of declared events; absent thresholds and durations remain unknown."""

    __slots__ = ()

    def __init__(
        self,
        *,
        event: Event,
        aggregation: Aggregation,
        threshold: Threshold | None = None,
        duration: Duration | None = None,
    ) -> None:
        _enum(event, Event, "event")
        _enum(aggregation, Aggregation, "aggregation")
        if threshold is not None and not isinstance(threshold, Threshold):
            raise TypeError("threshold must be Threshold")
        if (
            threshold is not None
            and event in (Event.FROST_DAY, Event.HOT_DAY, Event.FREEZE_THAW_CYCLE)
            and not isinstance(threshold.unit, TemperatureUnit)
        ):
            raise TypeError("temperature-event thresholds must use TemperatureUnit")
        if duration is not None and not isinstance(duration, Duration):
            raise TypeError("duration must be Duration")
        self._initialize(
            "event_count",
            "event_count",
            "count",
            {
                "event": event,
                "aggregation": aggregation,
                "threshold": threshold,
                "duration": duration,
            },
        )

    @property
    def unit(self) -> str:
        return "count"

    @property
    def event(self) -> Event:
        return cast(Event, self._parameters["event"])

    @property
    def aggregation(self) -> Aggregation:
        return cast(Aggregation, self._parameters["aggregation"])

    @property
    def threshold(self) -> Threshold | None:
        return cast(Threshold | None, self._parameters["threshold"])

    @property
    def duration(self) -> Duration | None:
        return cast(Duration | None, self._parameters["duration"])


class ClimateIndex(_BuiltinMeasurement):
    """A named index with its actual timescale and reference period when known."""

    __slots__ = ()

    def __init__(
        self,
        *,
        index: ClimateIndexKind,
        accumulation_duration: Duration | None = None,
        reference_period: ReferencePeriod | None = None,
    ) -> None:
        _enum(index, ClimateIndexKind, "index")
        if accumulation_duration is not None and not isinstance(
            accumulation_duration, Duration
        ):
            raise TypeError("accumulation_duration must be Duration")
        if reference_period is not None and not isinstance(
            reference_period, ReferencePeriod
        ):
            raise TypeError("reference_period must be ReferencePeriod")
        self._initialize(
            "climate_index",
            "climate_index",
            RatioUnit.FRACTION,
            {
                "index": index,
                "accumulation_duration": accumulation_duration,
                "reference_period": reference_period,
            },
        )

    @property
    def unit(self) -> RatioUnit:
        return cast(RatioUnit, self._unit)

    @property
    def index(self) -> ClimateIndexKind:
        return cast(ClimateIndexKind, self._parameters["index"])

    @property
    def accumulation_duration(self) -> Duration | None:
        return cast(Duration | None, self._parameters["accumulation_duration"])

    @property
    def reference_period(self) -> ReferencePeriod | None:
        return cast(ReferencePeriod | None, self._parameters["reference_period"])


class DegreeDays(_BuiltinMeasurement):
    """Temperature departures accumulated relative to an explicit threshold."""

    __slots__ = ()

    def __init__(
        self, *, unit: DegreeDayUnit, threshold: Threshold, duration: Duration
    ) -> None:
        _enum(unit, DegreeDayUnit, "unit")
        if not isinstance(threshold, Threshold) or not isinstance(
            threshold.unit, TemperatureUnit
        ):
            raise TypeError("threshold must use TemperatureUnit")
        if not isinstance(duration, Duration):
            raise TypeError("duration must be Duration")
        self._initialize(
            "degree_days",
            "degree_days",
            unit,
            {"unit": unit, "threshold": threshold, "duration": duration},
        )

    @property
    def unit(self) -> DegreeDayUnit:
        return cast(DegreeDayUnit, self._unit)

    @property
    def threshold(self) -> Threshold:
        return cast(Threshold, self._parameters["threshold"])

    @property
    def duration(self) -> Duration:
        return cast(Duration, self._parameters["duration"])


def _unit(value: object) -> Unit | str:
    for enum in _UNIT_TYPES:
        try:
            return enum(value)
        except (ValueError, TypeError):
            pass
    return _identifier(cast(str, value), "unit")


def _duration(value: object) -> Duration:
    record = cast(dict[str, object], value)
    return Duration(cast(float, record["value"]), unit=TimeUnit(record["unit"]))


def _threshold(value: object) -> Threshold:
    record = cast(dict[str, object], value)
    return Threshold(cast(float, record["value"]), unit=_unit(record["unit"]))


def _measurement(document: object) -> Measurement:
    if not isinstance(document, dict):
        raise TypeError("measurement must be an object")
    _freeze(cast(ScientificValue, document))
    if document.get("kind") == "measurement":
        for name in ("fields", "schema"):
            if name in document and not isinstance(document[name], Mapping):
                raise TypeError(f"{name} must be a mapping")
    _validate_measurement(document)
    kind = document["kind"]
    if kind == "measurement":
        return Measurement(
            quantity=document.get("quantity"),
            unit=None if "unit" not in document else _unit(document["unit"]),
            fields=document.get("fields"),
            schema=document.get("schema"),
        )
    unit = document.get("unit")
    if kind == "depth_above_ground":
        return DepthAboveGround(unit=LengthUnit(unit))
    if kind == "water_surface_elevation":
        return WaterSurfaceElevation(unit=LengthUnit(unit), datum=document["datum"])
    if kind == "ratio":
        return Ratio(unit=RatioUnit(unit))
    if kind == "time_duration":
        return TimeDuration(unit=TimeUnit(unit))
    if kind == "temperature":
        return Temperature(unit=TemperatureUnit(unit))
    if kind == "wind_speed":
        return WindSpeed(
            unit=SpeedUnit(unit),
            averaging_duration=None
            if "averaging_duration" not in document
            else _duration(document["averaging_duration"]),
            aggregation=None
            if "aggregation" not in document
            else Aggregation(document["aggregation"]),
        )
    if kind == "precipitation_rate":
        return PrecipitationRate(
            unit=PrecipitationRateUnit(unit),
            accumulation_duration=_duration(document["accumulation_duration"]),
            aggregation=Aggregation(document["aggregation"]),
        )
    if kind == "precipitation_amount":
        return PrecipitationAmount(
            unit=LengthUnit(unit),
            accumulation_duration=_duration(document["accumulation_duration"]),
        )
    if kind == "event_count":
        return EventCount(
            event=Event(document["event"]),
            aggregation=Aggregation(document["aggregation"]),
            threshold=None
            if "threshold" not in document
            else _threshold(document["threshold"]),
            duration=None
            if "duration" not in document
            else _duration(document["duration"]),
        )
    if kind == "climate_index":
        reference = document.get("reference_period")
        return ClimateIndex(
            index=ClimateIndexKind(document["index"]),
            accumulation_duration=None
            if "accumulation_duration" not in document
            else _duration(document["accumulation_duration"]),
            reference_period=None
            if reference is None
            else ReferencePeriod(int(reference["start"]), int(reference["end"])),
        )
    if kind == "degree_days":
        return DegreeDays(
            unit=DegreeDayUnit(unit),
            threshold=_threshold(document["threshold"]),
            duration=_duration(document["duration"]),
        )
    raise ValueError("unknown measurement kind")


class AssetType(Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    AGRICULTURAL_BUILDING = "agricultural_building"
    TRANSPORT = "transport"
    ROADS = "roads"
    AGRICULTURAL_LAND = "agricultural_land"


class FloodRegion(Enum):
    AFRICA = "africa"
    ASIA = "asia"
    EUROPE = "europe"
    NORTH_AMERICA = "north_america"
    SOUTH_AMERICA = "south_america"
    OCEANIA = "oceania"
    GLOBAL = "global"


@runtime_checkable
class ContextValue(Protocol):
    @property
    def name(self) -> str: ...


class ImpactContext(ABC):
    _kind: ClassVar[str]

    @property
    @abstractmethod
    def schema(self) -> Mapping[str, ScientificValue]: ...
    @abstractmethod
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, text: str) -> ImpactContext:
        result = _context(_parse(text))
        if cls is not ImpactContext and not isinstance(result, cls):
            raise ValueError(
                f"expected {cls.__name__}, received {type(result).__name__}"
            )
        return result


def _binding_or(value: object, expected: type[object], name: str) -> None:
    if isinstance(value, ContextValue) and not isinstance(value, Enum):
        _identifier(value.name, name + " binding")
        return
    if expected is int:
        if type(value) is not int or not 0 < value < 2**64:
            raise TypeError(
                f"{name} must be a positive H3 integer or a context binding"
            )
    elif expected is float:
        _finite(cast(float, value), name)
    elif expected is str:
        _identifier(cast(str, value), name)
    elif not isinstance(value, expected):
        raise TypeError(f"{name} must be {expected.__name__} or a context binding")


_BINDING_SCHEMA = _object({"column": _IDENTIFIER_SCHEMA}, ("column",))
_CONTEXT_DEFINITIONS: dict[str, dict[str, object]] = {
    "empty_context": {},
    "cell_context": {"cell": {"type": "integer", "minimum": 1, "maximum": 2**64 - 1}},
    "country_context": {"country": _IDENTIFIER_SCHEMA},
    "inundation_context": {
        "region": _choice(FloodRegion),
        "asset_type": {
            "enum": [
                a.value for a in AssetType if a is not AssetType.AGRICULTURAL_BUILDING
            ]
        },
    },
    "water_cost_context": {
        "country": _IDENTIFIER_SCHEMA,
        "asset_type": {
            "enum": [
                a.value
                for a in (
                    AssetType.RESIDENTIAL,
                    AssetType.COMMERCIAL,
                    AssetType.INDUSTRIAL,
                )
            ]
        },
    },
    "cooling_cost_context": {
        "country": _IDENTIFIER_SCHEMA,
        "asset_type": {
            "enum": [
                a.value
                for a in (
                    AssetType.RESIDENTIAL,
                    AssetType.COMMERCIAL,
                    AssetType.INDUSTRIAL,
                    AssetType.AGRICULTURAL_BUILDING,
                )
            ]
        },
        "historic_mean": _NUMBER,
    },
    "crop_context": {"country": _IDENTIFIER_SCHEMA, "historic_mean": _NUMBER},
}


def _context_schema(kind: str) -> dict[str, object]:
    fields = _CONTEXT_DEFINITIONS[kind]
    properties = {
        name: {"oneOf": [definition, _BINDING_SCHEMA]}
        for name, definition in fields.items()
    }
    return _object({"kind": {"const": kind}, **properties}, ("kind", *fields))


def _context_union_schema() -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://climateriskcommons.org/schemas/impact-context-1.0.schema.json",
        "title": "ImpactContext",
        "oneOf": [_context_schema(k) for k in _CONTEXT_DEFINITIONS],
    }


class _ContextRecord(ImpactContext):
    _kind: ClassVar[str]

    def _validate(self) -> None:
        try:
            Draft202012Validator(_context_schema(self._kind)).validate(
                json.loads(self.to_json())
            )
        except ValidationError as error:
            raise ValueError(
                f"context/{'/'.join(str(p) for p in error.absolute_path)}: {error.message}"
            ) from error

    @property
    def schema(self) -> Mapping[str, ScientificValue]:
        return cast(
            Mapping[str, ScientificValue],
            _freeze(cast(ScientificValue, _context_schema(self._kind))),
        )

    def to_json(self) -> str:
        result: dict[str, ScientificValue] = {"kind": self._kind}
        for name in _CONTEXT_DEFINITIONS[self._kind]:
            value = getattr(self, name)
            if isinstance(value, Enum):
                result[name] = value.value
            elif isinstance(value, ContextValue):
                result[name] = {"column": value.name}
            else:
                result[name] = cast(ScientificValue, value)
        return _dumps(result)


@dataclass(frozen=True, kw_only=True)
class EmptyContext(_ContextRecord):
    _kind = "empty_context"


@dataclass(frozen=True, kw_only=True)
class CellContext(_ContextRecord):
    cell: int | ContextValue
    _kind = "cell_context"

    def __post_init__(self) -> None:
        _binding_or(self.cell, int, "cell")
        if type(self.cell) is int:
            _core.lookup_continent(self.cell)
        self._validate()


@dataclass(frozen=True, kw_only=True)
class CountryContext(_ContextRecord):
    country: str | ContextValue
    _kind = "country_context"

    def __post_init__(self) -> None:
        _binding_or(self.country, str, "country")
        self._validate()


@dataclass(frozen=True, kw_only=True)
class InundationContext(_ContextRecord):
    """Exact flood-table region and asset class. Global is an explicit calibration."""

    region: FloodRegion | ContextValue
    asset_type: AssetType | ContextValue
    _kind = "inundation_context"

    def __post_init__(self) -> None:
        _binding_or(self.region, FloodRegion, "region")
        _binding_or(self.asset_type, AssetType, "asset_type")
        self._validate()


@dataclass(frozen=True, kw_only=True)
class WaterCostContext(_ContextRecord):
    country: str | ContextValue
    asset_type: AssetType | ContextValue
    _kind = "water_cost_context"

    def __post_init__(self) -> None:
        _binding_or(self.country, str, "country")
        _binding_or(self.asset_type, AssetType, "asset_type")
        self._validate()


@dataclass(frozen=True, kw_only=True)
class CoolingCostContext(_ContextRecord):
    country: str | ContextValue
    asset_type: AssetType | ContextValue
    historic_mean: float | ContextValue
    _kind = "cooling_cost_context"

    def __post_init__(self) -> None:
        _binding_or(self.country, str, "country")
        _binding_or(self.asset_type, AssetType, "asset_type")
        self._validate()
        _binding_or(self.historic_mean, float, "historic_mean")
        self._validate()


@dataclass(frozen=True, kw_only=True)
class CropContext(_ContextRecord):
    country: str | ContextValue
    historic_mean: float | ContextValue
    _kind = "crop_context"

    def __post_init__(self) -> None:
        _binding_or(self.country, str, "country")
        _binding_or(self.historic_mean, float, "historic_mean")
        self._validate()


_CONTEXT_TYPES: dict[str, type[ImpactContext]] = {
    "empty_context": EmptyContext,
    "cell_context": CellContext,
    "country_context": CountryContext,
    "inundation_context": InundationContext,
    "water_cost_context": WaterCostContext,
    "cooling_cost_context": CoolingCostContext,
    "crop_context": CropContext,
}


def _context(document: object) -> ImpactContext:
    if not isinstance(document, dict):
        raise TypeError("context must be an object")
    _freeze(cast(ScientificValue, document))
    kind = document.get("kind")
    if not isinstance(kind, str) or kind not in _CONTEXT_TYPES:
        raise ValueError("unknown context kind")
    try:
        Draft202012Validator(_context_schema(kind)).validate(document)
    except ValidationError as error:
        raise ValueError(
            f"context/{'/'.join(str(p) for p in error.absolute_path)}: {error.message}"
        ) from error
    values = {k: v for k, v in document.items() if k != "kind"}
    for name, value in tuple(values.items()):
        if isinstance(value, dict):
            values[name] = _ColumnBinding(value["column"])
            continue
        if name == "asset_type":
            values[name] = AssetType(value)
        elif name == "region":
            values[name] = FloodRegion(value)
        elif name == "cell":
            values[name] = int(value)
    return _CONTEXT_TYPES[kind](**values)


@dataclass(frozen=True)
class _ColumnBinding:
    name: str

    def __post_init__(self) -> None:
        _identifier(self.name, "column")

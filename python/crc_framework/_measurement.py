"""Immutable scientific measurements and typed physical units."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from types import MappingProxyType
from typing import Protocol, TypeAlias, cast, runtime_checkable

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


Unit: TypeAlias = (
    LengthUnit
    | TimeUnit
    | RatioUnit
    | TemperatureUnit
    | SpeedUnit
    | PrecipitationRateUnit
)
_UNIT_TYPES = (
    LengthUnit,
    TimeUnit,
    RatioUnit,
    TemperatureUnit,
    SpeedUnit,
    PrecipitationRateUnit,
)


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(
            f"{name} must be a nonempty string without boundary whitespace"
        )
    return value


def _freeze(value: ScientificValue) -> ScientificValue:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {_identifier(k, "field name"): _freeze(v) for k, v in value.items()}
        )
    if isinstance(value, (tuple, list)):
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


class Measurement:
    """Open scientific meaning; custom quantities and units need no registration."""

    _quantity: str | None
    _unit: Unit | str | None
    _fields: Mapping[str, ScientificValue]
    _schema: Mapping[str, ScientificValue] | None
    __slots__ = ("_fields", "_quantity", "_schema", "_unit")

    def __init__(
        self,
        *,
        quantity: str | None = None,
        unit: Unit | str | None = None,
        fields: Mapping[str, ScientificValue] | None = None,
        schema: Mapping[str, ScientificValue] | None = None,
    ) -> None:
        if fields is not None and not isinstance(fields, Mapping):
            raise TypeError("fields must be a mapping")
        if schema is not None and not isinstance(schema, Mapping):
            raise TypeError("schema must be a mapping")
        if quantity is not None:
            _identifier(quantity, "quantity")
        if unit is not None and not isinstance(unit, _UNIT_TYPES):
            _identifier(unit, "unit")
        object.__setattr__(self, "_quantity", quantity)
        object.__setattr__(self, "_unit", unit)
        object.__setattr__(self, "_fields", _freeze({} if fields is None else fields))
        object.__setattr__(self, "_schema", None if schema is None else _freeze(schema))

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
        return self._schema

    def to_json(self) -> str:
        return json.dumps(
            self._document(), sort_keys=True, allow_nan=False, separators=(",", ":")
        )

    def _document(self) -> dict[str, ScientificValue]:
        return {
            "kind": "measurement",
            "quantity": self.quantity,
            "unit": self.unit.value if isinstance(self.unit, Enum) else self.unit,
            "fields": _plain(self.fields),
            "schema": _plain(self.schema),
        }

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Measurement) and self.to_json() == other.to_json()

    def __repr__(self) -> str:
        return f"Measurement(quantity={self.quantity!r}, unit={self.unit!r})"


class DepthAboveGround(Measurement):
    """Depth above the local ground surface, rather than an absolute elevation."""

    __slots__ = ()

    def __init__(self, *, unit: LengthUnit) -> None:
        if not isinstance(unit, LengthUnit):
            raise TypeError("unit must be LengthUnit")
        super().__init__(quantity="depth_above_ground", unit=unit)

    @property
    def quantity(self) -> str:
        return "depth_above_ground"

    @property
    def unit(self) -> LengthUnit:
        return cast(LengthUnit, self._unit)

    def _document(self) -> dict[str, ScientificValue]:
        return {"kind": "depth_above_ground", "unit": self.unit.value}

    def __repr__(self) -> str:
        return f"DepthAboveGround(unit={self.unit!r})"


class Ratio(Measurement):
    """A dimensionless ratio in fractions or percent."""

    __slots__ = ()

    def __init__(self, *, unit: RatioUnit = RatioUnit.FRACTION) -> None:
        if not isinstance(unit, RatioUnit):
            raise TypeError("unit must be RatioUnit")
        super().__init__(quantity="ratio", unit=unit)

    @property
    def quantity(self) -> str:
        return "ratio"

    @property
    def unit(self) -> RatioUnit:
        return cast(RatioUnit, self._unit)

    def _document(self) -> dict[str, ScientificValue]:
        return {"kind": "ratio", "unit": self.unit.value}

    def __repr__(self) -> str:
        return f"Ratio(unit={self.unit!r})"


class TimeDuration(Measurement):
    """An elapsed time duration in the explicitly selected unit."""

    __slots__ = ()

    def __init__(self, *, unit: TimeUnit) -> None:
        if not isinstance(unit, TimeUnit):
            raise TypeError("unit must be TimeUnit")
        super().__init__(quantity="time_duration", unit=unit)

    @property
    def quantity(self) -> str:
        return "time_duration"

    @property
    def unit(self) -> TimeUnit:
        return cast(TimeUnit, self._unit)

    def _document(self) -> dict[str, ScientificValue]:
        return {"kind": "time_duration", "unit": self.unit.value}


def _measurement(document: object) -> Measurement:
    if not isinstance(document, dict):
        raise TypeError("measurement must be an object")
    kind = document.get("kind")
    if kind in ("depth_above_ground", "ratio", "time_duration"):
        if set(document) != {"kind", "unit"}:
            raise ValueError(f"unknown or missing {kind} fields")
        if kind == "depth_above_ground":
            return DepthAboveGround(unit=LengthUnit(document["unit"]))
        if kind == "time_duration":
            return TimeDuration(unit=TimeUnit(document["unit"]))
        return Ratio(unit=RatioUnit(document["unit"]))
    if kind != "measurement" or set(document) != {
        "kind",
        "quantity",
        "unit",
        "fields",
        "schema",
    }:
        raise ValueError("unknown or missing measurement fields")
    unit = document["unit"]
    for cls in _UNIT_TYPES:
        try:
            unit = cls(unit)
            break
        except ValueError:
            pass
    return Measurement(
        quantity=document["quantity"],
        unit=unit,
        fields=document["fields"],
        schema=document["schema"],
    )


class BuildingType(Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    AGRICULTURAL = "agricultural"


@runtime_checkable
class ContextValue(Protocol):
    @property
    def name(self) -> str: ...


@runtime_checkable
class ImpactContext(Protocol):
    @property
    def schema(self) -> Mapping[str, ScientificValue]: ...
    def to_json(self) -> str: ...


class InundationContext:
    """Geography and building use for the published inundation damage table.

    Supply the continent used by the published depth damage table.
    ``ContextValue`` bindings are resolved by data workflows before numerical use.
    """

    _country: str | ContextValue | None
    _continent: str | ContextValue | None
    _building_type: BuildingType | ContextValue | None
    __slots__ = ("_building_type", "_continent", "_country")

    def __init__(
        self,
        *,
        country: str | ContextValue | None = None,
        continent: str | ContextValue | None = None,
        building_type: BuildingType | ContextValue | None = None,
    ) -> None:
        for name, value in (("country", country), ("continent", continent)):
            if value is not None and not (
                isinstance(value, ContextValue) and not isinstance(value, Enum)
            ):
                _identifier(cast(str, value), name)
        if building_type is not None and not (
            isinstance(building_type, BuildingType)
            or (
                isinstance(building_type, ContextValue)
                and not isinstance(building_type, Enum)
            )
        ):
            raise TypeError("building_type must be BuildingType or a context binding")
        object.__setattr__(self, "_country", country)
        object.__setattr__(self, "_continent", continent)
        object.__setattr__(self, "_building_type", building_type)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("impact contexts are immutable")

    @property
    def country(self) -> str | ContextValue | None:
        return self._country

    @property
    def continent(self) -> str | ContextValue | None:
        return self._continent

    @property
    def building_type(self) -> BuildingType | ContextValue | None:
        return self._building_type

    @property
    def schema(self) -> Mapping[str, ScientificValue]:
        return MappingProxyType(
            {
                "country": "str | ContextValue | None",
                "continent": "str | ContextValue | None",
                "building_type": tuple(v.value for v in BuildingType),
            }
        )

    def to_json(self) -> str:
        values: dict[str, ScientificValue] = {}
        for name in ("country", "continent", "building_type"):
            value = getattr(self, name)
            if isinstance(value, Enum):
                values[name] = value.value
            elif isinstance(value, ContextValue):
                values[name] = {"column": value.name}
            elif value is not None:
                values[name] = value
        return json.dumps(values, sort_keys=True, separators=(",", ":"))

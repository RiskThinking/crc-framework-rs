"""Calibrated point impacts and exact distribution transformations."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from functools import cache
from types import MappingProxyType
from typing import cast

import numpy as np

from . import _core
from ._diagnostics import ScientificValue
from ._distribution import (
    ArrayLike,
    Distribution,
    ValueResult,
    Values,
    _real_array,
    _snapshot,
)
from ._measurement import (
    _BINDING_SCHEMA,
    _CONTEXT_DEFINITIONS,
    Aggregation,
    AssetType,
    CellContext,
    ClimateIndex,
    ClimateIndexKind,
    CoolingCostContext,
    CountryContext,
    CropContext,
    DegreeDays,
    DegreeDayUnit,
    DepthAboveGround,
    Duration,
    EmptyContext,
    Event,
    EventCount,
    FloodRegion,
    ImpactContext,
    InundationContext,
    LengthUnit,
    Measurement,
    PrecipitationAmount,
    Ratio,
    RatioUnit,
    SpeedUnit,
    TemperatureUnit,
    Threshold,
    TimeUnit,
    Unit,
    WaterCostContext,
    WildfireContext,
    WindSpeed,
    _canonical_value,
    _context_schema,
    _freeze,
    _measurement,
    _measurement_schema,
    _scientific_equal,
    _validate_scientific,
)


class ExpectationMethod(Enum):
    """Select how the expected response is computed."""

    AUTO = "auto"
    EXACT = "exact"


class ImpactError(Exception):
    """A calibrated response cannot perform the requested operation."""

    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(f"{code} at {path or '/'}: {message}")
        self._code, self._path, self._message = code, path, message

    @property
    def code(self) -> str:
        return self._code

    @property
    def path(self) -> str:
        return self._path

    @property
    def message(self) -> str:
        return self._message


def _error(error: Exception) -> ImpactError:
    if isinstance(error, ImpactError):
        return error
    code = getattr(error, "code", "invalid_value")
    path = getattr(error, "path", "")
    message = str(error)
    prefix = f"{code} at {path or '/'}: "
    return ImpactError(str(code), str(path), message.removeprefix(prefix))


def _dumps(value: object) -> str:
    return json.dumps(
        _canonical_value(cast(ScientificValue, value)),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ImpactError("duplicate_field", "/" + key, "duplicate JSON field")
        value[key] = item
    return value


def _load(text: str) -> dict[str, object]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (ValueError, TypeError) as error:
        raise ImpactError("invalid_json", "", str(error)) from error
    if not isinstance(value, dict):
        raise ImpactError("invalid_json", "", "expected an object")
    return value


def _unit(measurement: Measurement) -> str:
    unit = measurement.unit
    return unit.value if isinstance(unit, Enum) else cast(str, unit)


def _calibration(input: Measurement, output: Measurement) -> dict[str, str]:
    for name, measurement in (("input", input), ("output", output)):
        if not isinstance(measurement, Measurement):
            raise TypeError(f"{name} must be Measurement")
        if measurement.quantity is None or measurement.unit is None:
            raise ImpactError(
                "missing_measurement",
                "/" + name,
                "calibrated impacts require quantity and unit",
            )
    return {
        "exposure_variable": cast(str, input.quantity),
        "exposure_unit": _unit(input),
        "impact_metric": cast(str, output.quantity),
        "impact_unit": _unit(output),
    }


class Impact:
    """A resolved, immutable, context-free calibrated mathematical response.

    Tabulated and binned functions reject values outside their declared domain.
    Expectation and pushforward are exact operations. Unsupported combinations
    raise ``ImpactError`` rather than silently switching to approximation.
    """

    _native: _core.NativeImpactExpression
    _input: Measurement
    _output: Measurement
    _json: str
    _sha256: str
    __slots__ = ("_input", "_json", "_native", "_output", "_sha256")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("impacts are immutable")

    @classmethod
    def _create(
        cls,
        model: Mapping[str, object],
        input: Measurement,
        output: Measurement,
        clamp: Mapping[str, float | None] | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> Impact:
        calibration = _calibration(input, output)
        internal: dict[str, object] = {
            "schema_version": "1.0",
            "semantics_version": "crc-impact/1",
            "calibration": calibration,
            "model": dict(model),
            "provenance": {"origin": {"kind": "user_defined"}},
        }
        if clamp is not None:
            internal["clamp"] = dict(clamp)
        try:
            native = _core.NativeImpactExpression.from_json(_dumps(internal))
        except Exception as error:
            raise _error(error) from error
        return cls._from_native(native, input, output, provenance=provenance)

    @classmethod
    def _from_native(
        cls,
        native: _core.NativeImpactExpression,
        input: Measurement,
        output: Measurement,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> Impact:
        compiled = _load(native.canonical_json())
        model = cast(dict[str, object], compiled["model"])
        model = {
            key: value
            for key, value in model.items()
            if value is not None
            and not (key == "origin_adjustment" and value == "none")
        }
        document: dict[str, object] = {
            "schema_version": "2.0",
            "input": input._document(),
            "output": output._document(),
            "model": model,
        }
        if compiled["clamp"] is not None:
            document["clamp"] = {
                key: value
                for key, value in cast(dict[str, object], compiled["clamp"]).items()
                if value is not None
            }
        if provenance is not None:
            document["provenance"] = dict(provenance)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_native", native)
        object.__setattr__(instance, "_input", input)
        object.__setattr__(instance, "_output", output)
        object.__setattr__(instance, "_json", _dumps(document))
        object.__setattr__(
            instance,
            "_sha256",
            "sha256:" + hashlib.sha256(instance._json.encode()).hexdigest(),
        )
        return instance

    @classmethod
    def builtin(cls, model: BuiltinImpact) -> ImpactReference:
        if not isinstance(model, BuiltinImpact):
            raise TypeError("model must be BuiltinImpact")
        if model in _RETIRED:
            warnings.warn(
                f"{model.name}: {_RETIRED[model]}", DeprecationWarning, stacklevel=2
            )
        return ImpactReference._create(model)

    @classmethod
    def linear(
        cls,
        *,
        slope: float,
        input: Measurement,
        output: Measurement,
        intercept: float = 0.0,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> Impact:
        return cls._create(
            {"kind": "linear", "slope": slope, "intercept": intercept},
            input,
            output,
            {"minimum": minimum, "maximum": maximum},
        )

    @classmethod
    def sigmoid(
        cls,
        *,
        midpoint: float,
        steepness: float,
        input: Measurement,
        output: Measurement,
        offset: float = 0.0,
        amplitude: float = 1.0,
    ) -> Impact:
        return cls._create(
            {
                "kind": "sigmoid",
                "midpoint": midpoint,
                "steepness": steepness,
                "offset": offset,
                "amplitude": amplitude,
            },
            input,
            output,
        )

    @classmethod
    def tabulated(
        cls,
        points: Sequence[tuple[float, float]],
        *,
        input: Measurement,
        output: Measurement,
    ) -> Impact:
        return cls._create(
            {
                "kind": "tabulated",
                "points": [{"x": x, "y": y} for x, y in points],
                "interpolation": "linear",
                "outside_domain": "error",
            },
            input,
            output,
        )

    @classmethod
    def binned(
        cls,
        edges: ArrayLike,
        values: ArrayLike,
        *,
        input: Measurement,
        output: Measurement,
    ) -> Impact:
        edge_array, value_array = (
            _real_array(edges, "bin edges"),
            _real_array(values, "bin values"),
        )
        if edge_array.ndim != 1 or value_array.ndim != 1:
            raise ValueError("binned edges and values must be one-dimensional")
        return cls._create(
            {
                "kind": "binned",
                "bin_edges": edge_array.tolist(),
                "values": value_array.tolist(),
                "outside_domain": "error",
            },
            input,
            output,
        )

    @property
    def input(self) -> Measurement:
        return self._input

    @property
    def output(self) -> Measurement:
        return self._output

    @property
    def sha256(self) -> str:
        return self._sha256

    def evaluate(self, values: Values) -> ValueResult:
        array = _real_array(values, "exposure values")
        if np.any(~np.isfinite(array)):
            raise ImpactError(
                "invalid_value", "/values", "exposure values must be finite"
            )
        try:
            flat = self._native.evaluate_values(array.reshape(-1).tolist())
        except Exception as error:
            raise _error(error) from error
        result = np.asarray(flat, dtype=np.float64).reshape(array.shape)
        if array.ndim == 0:
            return float(result)
        return _snapshot(result)

    def _mean_deviation(
        self, source: Distribution, image: Distribution
    ) -> tuple[float, float] | None:
        """Affine moments inherit the source's explicit finite/infinite outcomes."""
        document = _load(self.to_json())
        model = cast(dict[str, object], document["model"])
        if model["kind"] != "linear":
            return None
        slope = float(cast(float, model["slope"]))
        intercept = float(cast(float, model["intercept"]))
        if slope == 0:
            return float(self.evaluate(0.0)), 0.0
        if document.get("clamp") is None:
            mean, deviation = source._mean_deviation()
            return slope * mean + intercept, abs(slope) * deviation
        clamped_mean = self._integral(source, 0.0, 1.0)
        if clamped_mean is None:
            return None
        if not math.isfinite(clamped_mean):
            return clamped_mean, math.inf if math.isinf(clamped_mean) else math.nan
        return clamped_mean, image._deviation(clamped_mean)

    def _integral(
        self,
        source: Distribution,
        lo: float,
        hi: float,
        *,
        center: float = 0.0,
        square: bool = False,
        survival: bool = False,
        scale: float = 1.0,
    ) -> float | None:
        """Integrate affine ranks in the nearer tail's probability coordinates."""
        document = _load(self.to_json())
        model = cast(dict[str, object], document["model"])
        if model["kind"] != "linear":
            return None
        slope = float(cast(float, model["slope"]))
        intercept = float(cast(float, model["intercept"]))
        clamp = cast(dict[str, float], document.get("clamp", {}))
        minimum, maximum = clamp.get("minimum"), clamp.get("maximum")

        def constant(value: float, width: float) -> float:
            weighted_scale = scale * (math.sqrt(width) if square else width)
            difference = value - center
            scaled = (
                difference * weighted_scale
                if math.isfinite(difference)
                else value * weighted_scale - center * weighted_scale
            )
            return scaled * scaled if square else scaled

        if lo == hi:
            return 0.0
        if slope == 0 or minimum == maximum and minimum is not None:
            return constant(float(self.evaluate(0.0)), hi - lo)

        source_scale = abs(slope) * scale
        source_center = (center - intercept) / slope if square else 0.0
        if (
            source_scale == 0
            or not math.isfinite(source_scale)
            or not math.isfinite(source_center)
            or any(
                bound is not None and not math.isfinite((bound - intercept) / slope)
                for bound in (minimum, maximum)
            )
        ):
            return None

        def add(values: Sequence[float]) -> float:
            try:
                return math.fsum(values)
            except ValueError:
                return math.nan

        def part(left: float, right: float, from_upper: bool) -> float:
            if right <= left:
                return 0.0
            source_upper = from_upper if slope > 0 else not from_upper

            def cut(value: float) -> float:
                threshold = (value - intercept) / slope
                if source_upper:
                    return float(source.exceedance(threshold, inclusive=not from_upper))
                return float(source.cdf(threshold, inclusive=not from_upper))

            # The first and last ranks are assigned to the corresponding
            # clamp. Rank cuts are evaluated directly as CDF or survival.
            first, last = (maximum, minimum) if from_upper else (minimum, maximum)
            contributions: list[float] = []
            if first is not None:
                boundary = cut(first)
                if boundary > left:
                    contributions.append(constant(first, min(right, boundary) - left))
                left = max(left, boundary)
            if last is not None:
                boundary = cut(last)
                if boundary < right and right > left:
                    contributions.append(constant(last, right - max(left, boundary)))
                right = min(right, boundary)
            if right > left:
                contribution = source._integral(
                    left,
                    right,
                    center=source_center,
                    square=square,
                    survival=source_upper,
                    scale=source_scale,
                )
                if not square:
                    contribution = math.copysign(1.0, slope) * contribution + (
                        intercept - center
                    ) * scale * (right - left)
                contributions.append(contribution)
            return add(contributions)

        # Map only ranks at or above one half into their complements.
        # This also keeps a tiny endpoint atom representable when the
        # complementary ordinary probability rounds to one.
        return add(
            (
                part(lo, min(hi, 0.5), survival),
                part(1 - hi, 1 - max(lo, 0.5), not survival),
            )
        )

    def _operation(
        self, distribution: Distribution
    ) -> tuple[_core.NativeImpactExpression, _core.NativeImpactSource]:
        output = distribution._output_measurement()
        if output is not None and output != self.input:
            raise ImpactError(
                "measurement_mismatch",
                "/input",
                f"source impact output {output!r} does not match {self.input!r}",
            )
        try:
            return self._native, distribution._source()
        except TypeError:
            if distribution._data["kind"] != "transformed":
                raise
        document = _load(self.to_json())
        model = cast(dict[str, float | str], document["model"])
        if model["kind"] != "linear" or document.get("clamp") is not None:
            raise TypeError(
                "this composition has no supported exact source representation"
            )
        previous = Impact.from_json(_dumps(distribution._data["impact"]))
        previous_document = _load(previous.to_json())
        previous_model = cast(dict[str, float | str], previous_document["model"])
        if previous_model["kind"] != "linear":
            raise TypeError(
                "this composition has no supported exact source representation"
            )
        slope, intercept = float(model["slope"]), float(model["intercept"])
        previous_clamp = cast(
            dict[str, float | None] | None, previous_document.get("clamp")
        )
        minimum = maximum = None
        if previous_clamp is not None:
            lo, hi = previous_clamp.get("minimum"), previous_clamp.get("maximum")
            minimum = None if lo is None else slope * lo + intercept
            maximum = None if hi is None else slope * hi + intercept
            if slope < 0:
                minimum, maximum = maximum, minimum
        combined = Impact.linear(
            slope=slope * float(previous_model["slope"]),
            intercept=slope * float(previous_model["intercept"]) + intercept,
            minimum=minimum,
            maximum=maximum,
            input=previous.input,
            output=self.output,
        )
        original = Distribution.from_json(_dumps(distribution._data["source"]))
        return combined._operation(original)

    def expectation(
        self,
        distribution: Distribution,
        *,
        method: ExpectationMethod = ExpectationMethod.AUTO,
    ) -> float:
        """Return the expected response, using quadrature when needed by default."""
        if not isinstance(distribution, Distribution):
            raise TypeError("distribution must be Distribution")
        if not isinstance(method, ExpectationMethod):
            raise TypeError("method must be ExpectationMethod")
        try:
            native, source = self._operation(distribution)
        except (TypeError, ValueError) as error:
            raise ImpactError("exact_method_unavailable", "", str(error)) from error
        try:
            return native.expectation(source, method is ExpectationMethod.EXACT)
        except Exception as error:
            raise _error(error) from error

    def pushforward(self, distribution: Distribution) -> Distribution:
        if not isinstance(distribution, Distribution):
            raise TypeError("distribution must be Distribution")
        try:
            expression, source = self._operation(distribution)
            native = expression.pushforward(source)
        except TypeError as error:
            raise ImpactError("exact_method_unavailable", "", str(error)) from error
        except Exception as error:
            raise _error(error) from error
        return Distribution._from_transformed(
            distribution, _load(self.to_json()), native
        )

    @classmethod
    def from_json(cls, text: str) -> Impact:
        document = _load(text)
        if document.get("schema_version") != "2.0":
            raise ImpactError(
                "unsupported_schema_version",
                "/schema_version",
                "expected impact schema 2.0",
            )
        try:
            _measurement(document.get("input"))
            _measurement(document.get("output"))
            _validate_scientific("impact-2.0.schema.json", document)
        except (TypeError, ValueError) as error:
            raise ImpactError("invalid_fields", "", str(error)) from error
        if "provenance" in document:
            proof = cast(dict[str, object], document["provenance"])
            reference = ImpactReference.from_json(_dumps(proof["model"]))
            context = ImpactContext.from_json(_dumps(proof["context"]))
            resolved = reference.resolve(context)
            if not _scientific_equal(
                cast(ScientificValue, document),
                cast(ScientificValue, _load(resolved.to_json())),
            ):
                raise ImpactError(
                    "provenance_mismatch",
                    "/provenance",
                    "resolved model, calibration or context contradicts its pinned provenance",
                )
            return resolved
        required = {"schema_version", "input", "output", "model"}
        allowed = required | {"clamp", "provenance"}
        if document.get("schema_version") != "2.0":
            raise ImpactError(
                "unsupported_schema_version",
                "/schema_version",
                "expected impact schema 2.0",
            )
        if not required <= set(document) <= allowed:
            raise ImpactError("invalid_fields", "", "unknown or missing impact fields")
        try:
            return cls._create(
                cast(Mapping[str, object], document["model"]),
                _measurement(document["input"]),
                _measurement(document["output"]),
                cast(Mapping[str, float | None] | None, document.get("clamp")),
                cast(Mapping[str, object] | None, document.get("provenance")),
            )
        except ImpactError:
            raise
        except (ValueError, TypeError, KeyError) as error:
            raise ImpactError("invalid_value", "", str(error)) from error

    def to_json(self) -> str:
        return self._json


class BuiltinImpact(Enum):
    """Published RiskThinking models.

    ``HOT_DAYS`` is retired from the RiskThinking methodology: selecting it warns
    with ``DeprecationWarning``, it never binds data, and 2.0 removes it.
    """

    FREEZE_THAW_CYCLES = "crc:freeze_thaw_cycles"
    FROST_DAYS = "crc:frost_days"
    MAXIMUM_DAILY_PRECIPITATION = "crc:maximum_daily_precipitation"
    CYCLONE = "crc:cyclone"
    MAXIMUM_DAILY_WIND_SPEED = "crc:maximum_daily_wind_speed"
    FIRE_WEATHER_INDEX = "crc:fire_weather_index"
    HOT_DAYS = "crc:hot_days"
    COASTAL_FLOOD = "crc:coastal_flood"
    RIVER_FLOOD = "crc:river_flood"
    INUNDATION = "crc:inundation"
    STANDARDIZED_PRECIPITATION_EVAPOTRANSPIRATION_INDEX = (
        "crc:standardized_precipitation_evapotranspiration_index"
    )
    COOLING_DEGREE_DAYS = "crc:cooling_degree_days"
    COOLING_DEGREE_DAYS_CROP_YIELD = "crc:cooling_degree_days_crop_yield"
    CONSECUTIVE_DRY_DAYS = "crc:consecutive_dry_days"


# Models removed from the RiskThinking methodology. They stay readable so saved
# documents remain valid, never bind to data, and are removed in 2.0.
_RETIRED = {
    BuiltinImpact.HOT_DAYS: (
        "the hot-days productivity-loss model is retired from the RiskThinking "
        "methodology and will be removed in crc-framework 2.0"
    ),
}

_FACTORS = {
    BuiltinImpact.FREEZE_THAW_CYCLES: "daily_freezethaw_cycles",
    BuiltinImpact.FROST_DAYS: "frost_days",
    BuiltinImpact.MAXIMUM_DAILY_PRECIPITATION: "rx1day",
    BuiltinImpact.CYCLONE: "cyclone",
    BuiltinImpact.MAXIMUM_DAILY_WIND_SPEED: "wind_max_daily_max",
    BuiltinImpact.FIRE_WEATHER_INDEX: "fwi",
    BuiltinImpact.HOT_DAYS: "hot_days",
    BuiltinImpact.COASTAL_FLOOD: "cflood",
    BuiltinImpact.RIVER_FLOOD: "rflood",
    BuiltinImpact.INUNDATION: "inundation",
    BuiltinImpact.STANDARDIZED_PRECIPITATION_EVAPOTRANSPIRATION_INDEX: "spei",
    BuiltinImpact.COOLING_DEGREE_DAYS: "cooling_degree_days",
    BuiltinImpact.COOLING_DEGREE_DAYS_CROP_YIELD: "cooling_degree_days",
    BuiltinImpact.CONSECUTIVE_DRY_DAYS: "maximum_consecutive_dry_days",
}
_FIXED_ASSET_TYPES = {
    BuiltinImpact.COOLING_DEGREE_DAYS_CROP_YIELD: AssetType.AGRICULTURAL_BUILDING,
    BuiltinImpact.CONSECUTIVE_DRY_DAYS: AssetType.AGRICULTURAL_BUILDING,
}
_ASSET_NAMES = {
    AssetType.RESIDENTIAL: "Residential buildings",
    AssetType.COMMERCIAL: "Commercial buildings",
    AssetType.INDUSTRIAL: "Industrial buildings",
    AssetType.AGRICULTURAL_BUILDING: "Agricultural buildings",
    AssetType.TRANSPORT: "Transport",
    AssetType.ROADS: "Infrastructure - roads",
    AssetType.AGRICULTURAL_LAND: "Agriculture",
}
_REGION_NAMES = {
    FloodRegion.AFRICA: "Africa",
    FloodRegion.ASIA: "Asia",
    FloodRegion.EUROPE: "Europe",
    FloodRegion.NORTH_AMERICA: "North America",
    FloodRegion.SOUTH_AMERICA: "South America",
    FloodRegion.OCEANIA: "Oceania",
    FloodRegion.GLOBAL: "Global",
}


@dataclass(frozen=True)
class ContextField:
    """A named context requirement and its accepted constant values."""

    name: str
    annotation: str
    required: bool
    choices: tuple[str, ...]
    choices_by_asset_type: Mapping[AssetType, tuple[str, ...]] = dataclass_field(
        default_factory=dict, kw_only=True, hash=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "choices_by_asset_type",
            MappingProxyType(dict(self.choices_by_asset_type)),
        )


@dataclass(frozen=True)
class ImpactRequirements:
    """Published model science. Numerical resolution does not assert data binding.

    ``unestablished_facts`` explicitly identifies calibration facts absent from
    published source tables. Supplying a plausible value cannot fill those gaps.
    """

    input: Measurement
    measurement_type: type[Measurement]
    measurement_schema: Mapping[str, ScientificValue]
    accepted_units: tuple[Unit | str, ...]
    context_type: type[ImpactContext]
    context_fields: tuple[ContextField, ...]
    unestablished_facts: tuple[str, ...]
    output: Measurement
    _retired: str | None = dataclass_field(default=None, repr=False, compare=False)

    def bind(self, measurement: Measurement) -> None:
        if self._retired is not None:
            raise ImpactError("model_retired", "/model", self._retired)
        if not isinstance(measurement, self.measurement_type):
            raise ImpactError(
                "measurement_type_mismatch",
                "/measurement",
                f"requires {self.measurement_type.__name__}; inspect model.requirements.measurement_schema for its typed constructor fields",
            )
        unit = measurement.unit
        if unit not in self.accepted_units:
            expected = ", ".join(repr(v) for v in self.accepted_units)
            raise ImpactError(
                "unit_mismatch",
                "/measurement/unit",
                f"current unit {unit!r}; accepted units: {expected}. Convert numerical values with convert_units(variable=..., unit=...) before binding; relabelling does not convert data",
            )
        if self.unestablished_facts:
            raise ImpactError(
                "calibration_unestablished",
                "/measurement",
                "published calibration has not established: "
                + ", ".join(self.unestablished_facts)
                + "; data binding is unavailable until those model-owned facts are published",
            )
        if measurement != self.input:
            raise ImpactError(
                "measurement_mismatch",
                "/measurement",
                f"requires {self.input!r}; datum, threshold, duration, aggregation and reference facts cannot be repaired by changing units",
            )


@dataclass(frozen=True)
class _ModelScience:
    input: Measurement
    measurement_type: type[Measurement]
    context_type: type[ImpactContext]
    unestablished: tuple[str, ...] = ()


_FREEZING = Threshold(0, unit=TemperatureUnit.CELSIUS)
_YEAR = Duration(1, unit=TimeUnit.YEAR)


def _model_science(model: BuiltinImpact) -> _ModelScience:
    """Each model's input states the facts its calibration assumed.

    Binding is exact equality, so data must declare the same event threshold,
    counting or accumulation duration and wind averaging. Per-cell thresholds
    for frost days, freeze-thaw cycles and rx1day are return levels of the
    1985–2024 historic baseline; they are model data, not input facts.
    """
    if model in (
        BuiltinImpact.INUNDATION,
        BuiltinImpact.COASTAL_FLOOD,
        BuiltinImpact.RIVER_FLOOD,
    ):
        return _ModelScience(
            DepthAboveGround(unit=LengthUnit.METRE), DepthAboveGround, InundationContext
        )
    if model is BuiltinImpact.HOT_DAYS:
        return _ModelScience(
            EventCount(event=Event.HOT_DAY, aggregation=Aggregation.COUNT),
            EventCount,
            EmptyContext,
            ("event threshold", "counting duration", "calibration reference period"),
        )
    if model in (BuiltinImpact.FREEZE_THAW_CYCLES, BuiltinImpact.FROST_DAYS):
        event = (
            Event.FREEZE_THAW_CYCLE
            if model is BuiltinImpact.FREEZE_THAW_CYCLES
            else Event.FROST_DAY
        )
        return _ModelScience(
            EventCount(
                event=event,
                aggregation=Aggregation.COUNT,
                threshold=_FREEZING,
                duration=_YEAR,
            ),
            EventCount,
            CellContext,
        )
    if model is BuiltinImpact.MAXIMUM_DAILY_PRECIPITATION:
        return _ModelScience(
            PrecipitationAmount(
                unit=LengthUnit.MILLIMETRE,
                accumulation_duration=Duration(1, unit=TimeUnit.DAY),
            ),
            PrecipitationAmount,
            CellContext,
        )
    if model is BuiltinImpact.CYCLONE:
        # STORM synthetic tracks give 10-minute sustained winds, used without
        # conversion. Cyclone hazard is independent of period and pathway.
        return _ModelScience(
            WindSpeed(
                unit=SpeedUnit.METRES_PER_SECOND,
                averaging_duration=Duration(600, unit=TimeUnit.SECOND),
                aggregation=Aggregation.MAXIMUM,
            ),
            WindSpeed,
            CountryContext,
        )
    if model is BuiltinImpact.MAXIMUM_DAILY_WIND_SPEED:
        # Daily-maximum model wind is used as delivered, with no averaging.
        return _ModelScience(
            WindSpeed(
                unit=SpeedUnit.METRES_PER_SECOND, aggregation=Aggregation.MAXIMUM
            ),
            WindSpeed,
            CountryContext,
        )
    if model is BuiltinImpact.FIRE_WEATHER_INDEX:
        # Fire weather damages only assets in the wildland-urban interface.
        return _ModelScience(
            ClimateIndex(index=ClimateIndexKind.FIRE_WEATHER_INDEX),
            ClimateIndex,
            WildfireContext,
        )
    if model is BuiltinImpact.STANDARDIZED_PRECIPITATION_EVAPOTRANSPIRATION_INDEX:
        # Monthly SPEI standardized over the same series: no fixed reference.
        return _ModelScience(
            ClimateIndex(
                index=ClimateIndexKind.STANDARDIZED_PRECIPITATION_EVAPOTRANSPIRATION_INDEX,
                accumulation_duration=Duration(1, unit=TimeUnit.MONTH),
            ),
            ClimateIndex,
            WaterCostContext,
        )
    if model is BuiltinImpact.COOLING_DEGREE_DAYS:
        # The caller's historic mean context sets the cooling-cost baseline.
        return _ModelScience(
            DegreeDays(
                unit=DegreeDayUnit.KELVIN_DAYS,
                threshold=Threshold(18, unit=TemperatureUnit.CELSIUS),
                duration=_YEAR,
            ),
            DegreeDays,
            CoolingCostContext,
        )
    if model is BuiltinImpact.COOLING_DEGREE_DAYS_CROP_YIELD:
        return _ModelScience(
            Measurement(quantity="degree_days", unit=DegreeDayUnit.KELVIN_DAYS),
            DegreeDays,
            CropContext,
            (
                "temperature threshold",
                "accumulation duration",
                "calibration reference period",
            ),
        )
    return _ModelScience(
        EventCount(event=Event.DRY_DAY, aggregation=Aggregation.MAXIMUM),
        EventCount,
        CropContext,
        (
            "dry-day precipitation threshold",
            "consecutive-event reduction",
            "calibration reference period",
        ),
    )


@cache
def _context_fields(
    model: BuiltinImpact, context_type: type[ImpactContext]
) -> tuple[ContextField, ...]:
    kind = context_type._kind
    fields = _CONTEXT_DEFINITIONS[kind]
    result: list[ContextField] = []
    for name in fields:
        annotation = "ContextValue | " + (
            "int"
            if name == "cell"
            else "float"
            if name == "historic_mean"
            else "FloodRegion"
            if name == "region"
            else "AssetType"
            if name == "asset_type"
            else "bool"
            if name == "wildland_urban_interface"
            else "str"
        )
        choices = tuple(
            cast(list[str], cast(dict[str, object], fields[name]).get("enum", []))
        )
        by_asset_type: dict[AssetType, tuple[str, ...]] = {}
        if name == "country":
            factor = _FACTORS[model]
            if context_type is CoolingCostContext:
                asset_types = cast(dict[str, list[str]], fields["asset_type"])["enum"]
                by_asset_type = {
                    asset: tuple(
                        _core.impact_registry_country_choices(
                            factor, _ASSET_NAMES[asset]
                        )
                    )
                    for asset in map(AssetType, asset_types)
                }
                choices = tuple(
                    sorted({code for codes in by_asset_type.values() for code in codes})
                )
            else:
                fixed_asset_type = _FIXED_ASSET_TYPES.get(model)
                choices = tuple(
                    _core.impact_registry_country_choices(
                        factor,
                        _ASSET_NAMES[fixed_asset_type]
                        if fixed_asset_type is not None
                        else None,
                    )
                )
        result.append(
            ContextField(
                name, annotation, True, choices, choices_by_asset_type=by_asset_type
            )
        )
    return tuple(result)


def _model_context_schema(requirements: ImpactRequirements) -> dict[str, object]:
    schema = _context_schema(requirements.context_type._kind)
    country = next(
        (f for f in requirements.context_fields if f.name == "country"), None
    )
    if country is None:
        return schema

    def country_property(choices: tuple[str, ...]) -> dict[str, object]:
        return {"oneOf": [{"enum": list(choices)}, _BINDING_SCHEMA]}

    properties = cast(dict[str, object], schema["properties"])
    properties["country"] = country_property(country.choices)
    if country.choices_by_asset_type:
        schema["allOf"] = [
            {
                "if": {"properties": {"asset_type": {"const": asset_type.value}}},
                "then": {"properties": {"country": country_property(choices)}},
            }
            for asset_type, choices in country.choices_by_asset_type.items()
        ]
    return schema


class ImpactReference:
    """Pinned built-in identity with model-owned science and exact context schemas."""

    _model: BuiltinImpact
    _version: str
    _manifest: str
    _json: str
    _sha256: str
    __slots__ = ("_json", "_manifest", "_model", "_sha256", "_version")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("impact references are immutable")

    @classmethod
    def _create(cls, model: BuiltinImpact) -> ImpactReference:
        _, version, _, digest = _core.impact_registry_identity()
        document = {
            "schema_version": "2.0",
            "id": model.value,
            "version": version,
            "manifest_sha256": "sha256:" + digest,
        }
        instance = object.__new__(cls)
        object.__setattr__(instance, "_model", model)
        object.__setattr__(instance, "_version", version)
        object.__setattr__(instance, "_manifest", digest)
        object.__setattr__(instance, "_json", _dumps(document))
        object.__setattr__(
            instance,
            "_sha256",
            "sha256:" + hashlib.sha256(instance._json.encode()).hexdigest(),
        )
        return instance

    @property
    def id(self) -> str:
        return self._model.value

    @property
    def version(self) -> str:
        return self._version

    @property
    def sha256(self) -> str:
        return self._sha256

    def _metadata(self) -> dict[str, object]:
        manifest = json.loads(_core.impact_registry_manifest_json())
        return cast(
            dict[str, object],
            next(
                item
                for item in manifest["factors"]
                if item["factor"] == _FACTORS[self._model]
            ),
        )

    @property
    def requirements(self) -> ImpactRequirements:
        science = _model_science(self._model)
        metadata = self._metadata()
        calibration = metadata["calibration"]
        if self._model is BuiltinImpact.COOLING_DEGREE_DAYS_CROP_YIELD:
            calibration = cast(dict[str, object], metadata["branch_calibrations"])[
                "agricultural"
            ]
        schema_kind = {
            DepthAboveGround: "depth_above_ground",
            EventCount: "event_count",
            PrecipitationAmount: "precipitation_amount",
            WindSpeed: "wind_speed",
            ClimateIndex: "climate_index",
            DegreeDays: "degree_days",
        }[science.measurement_type]
        return ImpactRequirements(
            science.input,
            science.measurement_type,
            cast(
                Mapping[str, ScientificValue],
                _freeze(cast(ScientificValue, _measurement_schema(schema_kind))),
            ),
            (cast(Unit | str, science.input.unit),),
            science.context_type,
            _context_fields(self._model, science.context_type),
            science.unestablished,
            self._output(cast(dict[str, str], calibration)),
            _RETIRED.get(self._model),
        )

    def _input(self) -> Measurement:
        return _model_science(self._model).input

    @staticmethod
    def _output(calibration: Mapping[str, str]) -> Measurement:
        if calibration["impact_metric"] == "damage_ratio":
            return Ratio()
        return Measurement(
            quantity=calibration["impact_metric"], unit=RatioUnit.FRACTION
        )

    def resolve(self, context: ImpactContext | None = None) -> Impact:
        required = self.requirements
        if context is None:
            if required.context_type is not EmptyContext:
                raise ImpactError(
                    "missing_context",
                    "/context",
                    f"use {required.context_type.__name__}("
                    + ", ".join(f.name + "=..." for f in required.context_fields)
                    + ")",
                )
            context = EmptyContext()
        if not isinstance(context, required.context_type):
            raise ImpactError(
                "invalid_context",
                "/context",
                f"use {required.context_type.__name__} with fields "
                + ", ".join(f.name for f in required.context_fields),
            )
        values = json.loads(context.to_json())
        values.pop("kind")
        for field in required.context_fields:
            value = values[field.name]
            if isinstance(value, dict):
                raise ImpactError(
                    "unresolved_context",
                    "/context/" + field.name,
                    "resolve asset-column bindings before numerical use",
                )
            if field.name == "country":
                asset_type = values.get("asset_type")
                choices = field.choices
                if field.choices_by_asset_type and isinstance(asset_type, str):
                    choices = field.choices_by_asset_type[AssetType(asset_type)]
                if value not in choices:
                    hint = "Use the country field in model.requirements.context_fields to inspect accepted codes"
                    if field.choices_by_asset_type:
                        hint += " for each AssetType in choices_by_asset_type"
                    raise ImpactError(
                        "missing_table_key",
                        "/context/country",
                        f"{value!r} has no country entry for {self.id}. {hint}.",
                    )
            elif field.choices and value not in field.choices:
                raise ImpactError(
                    "unsupported_context",
                    "/context/" + field.name,
                    "accepted values: " + ", ".join(field.choices),
                )
        provenance = {
            "model": _load(self.to_json()),
            "context": _load(context.to_json()),
        }
        private: dict[str, object] = {}
        if "cell" in values:
            private["cell"] = format(int(values["cell"]), "x")
        if "country" in values:
            private["country"] = values["country"]
        if "historic_mean" in values:
            private["historic_mean"] = values["historic_mean"]
        if "region" in values:
            private["continent"] = _REGION_NAMES[FloodRegion(values["region"])]
        if "asset_type" in values:
            private["building_type"] = _ASSET_NAMES[AssetType(values["asset_type"])]
        if self._model in _FIXED_ASSET_TYPES:
            private["building_type"] = _ASSET_NAMES[_FIXED_ASSET_TYPES[self._model]]
        registry, version, resolver, digest = _core.impact_registry_identity()
        if self._version != version or self._manifest != digest:
            raise ImpactError(
                "registry_unavailable",
                "",
                "the pinned built-in version is not installed",
            )
        if values.get("wildland_urban_interface") is False:
            # Outside the interface the calibrated response is zero damage.
            return Impact._create(
                {"kind": "linear", "slope": 0.0, "intercept": 0.0},
                self._input(),
                required.output,
                provenance=provenance,
            )
        internal = {
            "schema_version": "1.0",
            "registry_id": registry,
            "registry_revision": version,
            "registry_manifest_sha256": digest,
            "resolver_semantics_version": resolver,
            "factor": _FACTORS[self._model],
        }
        try:
            reference = _core.NativeImpactReference.from_json(_dumps(internal))
            native = reference.resolve(
                cast(str | None, private.get("cell")),
                cast(str | None, private.get("country")),
                cast(str | None, private.get("continent")),
                cast(str | None, private.get("building_type")),
                cast(float | None, private.get("historic_mean")),
            )
        except Exception as error:
            raise _error(error) from error
        return Impact._from_native(
            native, self._input(), required.output, provenance=provenance
        )

    def to_json(self) -> str:
        return self._json

    @classmethod
    def from_json(cls, text: str) -> ImpactReference:
        document = _load(text)
        if document.get("schema_version") != "2.0":
            raise ImpactError(
                "unsupported_schema_version",
                "/schema_version",
                "expected impact schema 2.0",
            )
        if set(document) != {"schema_version", "id", "version", "manifest_sha256"}:
            raise ImpactError(
                "invalid_fields", "", "unknown or missing reference fields"
            )
        try:
            reference = cls._create(BuiltinImpact(document["id"]))
        except ValueError as error:
            raise ImpactError(
                "unknown_model", "/id", "unknown BuiltinImpact identity"
            ) from error
        if _dumps(document) != reference.to_json():
            raise ImpactError(
                "registry_unavailable",
                "",
                "the pinned built-in version is not installed",
            )
        return reference

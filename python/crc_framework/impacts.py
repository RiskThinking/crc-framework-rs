"""Calibrated point impacts and exact distribution transformations."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, make_dataclass
from enum import Enum
from itertools import pairwise
from types import MappingProxyType
from typing import cast

import numpy as np

from . import _core
from ._diagnostics import ScientificValue
from ._distribution import ArrayLike, Distribution, ValueResult, Values, _snapshot
from ._measurement import (
    BuildingType,
    ContextValue,
    DepthAboveGround,
    ImpactContext,
    InundationContext,
    LengthUnit,
    Measurement,
    Ratio,
    RatioUnit,
    Unit,
    _measurement,
)


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


def _scaled_square(scale: float, value: float) -> float:
    """Scale nonnegative moments without squaring the scale prematurely."""
    if value == 0.0 or scale == 0.0:
        return 0.0
    if math.isinf(value):
        return value
    if value < 0 or math.isnan(value):
        return math.nan
    scaled = abs(scale) * math.sqrt(value)
    return scaled * scaled


def _dumps(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


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
            "provenance": {"origin": {"kind": "user_defined"}}
            if provenance is None
            else dict(provenance),
        }
        if clamp is not None:
            internal["clamp"] = dict(clamp)
        try:
            native = _core.NativeImpactExpression.from_json(_dumps(internal))
        except Exception as error:
            raise _error(error) from error
        return cls._from_native(native, input, output)

    @classmethod
    def _from_native(
        cls,
        native: _core.NativeImpactExpression,
        input: Measurement,
        output: Measurement,
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
        provenance = {
            key: value
            for key, value in cast(dict[str, object], compiled["provenance"]).items()
            if value != []
        }
        if provenance != {"origin": {"kind": "user_defined"}}:
            document["provenance"] = provenance
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
            np.asarray(edges, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
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
        array = np.asarray(values, dtype=np.float64)
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

    def _segments(self, distribution: Distribution) -> list[tuple[float, float, float]]:
        """Exact finite strata for a piecewise-affine response."""
        document = _load(self.to_json())
        model = cast(dict[str, object], document["model"])
        kind = model["kind"]
        if kind not in ("linear", "tabulated", "binned"):
            raise TypeError(
                "the transformed law has no exact finite piecewise-affine representation"
            )
        cuts: list[float] = []
        if kind == "tabulated":
            cuts = [
                float(point["x"])
                for point in cast(list[dict[str, float]], model["points"])
            ]
        elif kind == "binned":
            cuts = cast(list[float], model["bin_edges"])
        elif model["slope"] != 0:
            slope, intercept = (
                float(cast(float, model["slope"])),
                float(cast(float, model["intercept"])),
            )
            clamp = cast(dict[str, float | None] | None, document.get("clamp"))
            if clamp is not None:
                cuts = [
                    (value - intercept) / slope
                    for value in clamp.values()
                    if value is not None
                ]
        strata: list[tuple[float, float, float]] = []
        for lower, upper, probability in distribution._segments():
            if lower == upper:
                value = float(self.evaluate(lower))
                strata.append((value, value, probability))
                continue
            boundaries = (
                [lower] + sorted(x for x in cuts if lower < x < upper) + [upper]
            )
            for left, right in pairwise(boundaries):
                mass = probability * (right - left) / (upper - lower)
                if kind == "binned":
                    first = last = float(self.evaluate(left + (right - left) / 2))
                else:
                    first, last = (
                        float(self.evaluate(left)),
                        float(self.evaluate(right)),
                    )
                strata.append((min(first, last), max(first, last), mass))
        strata.sort(key=lambda segment: (segment[0], segment[1]))
        return strata

    def _moments(self, source: Distribution) -> tuple[float, float] | None:
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
            mean, variance = source._moments()
            return slope * mean + intercept, _scaled_square(slope, variance)
        clamped_mean = self._tail_integral(source, 0.0, 1.0)
        assert clamped_mean is not None
        if not math.isfinite(clamped_mean):
            return clamped_mean, math.inf if math.isinf(clamped_mean) else math.nan
        clamped_variance = self._integral(
            source, 0.0, 1.0, center=clamped_mean, square=True
        )
        assert clamped_variance is not None
        return clamped_mean, clamped_variance

    def _integral(
        self,
        source: Distribution,
        lo: float,
        hi: float,
        *,
        center: float = 0.0,
        square: bool = False,
    ) -> float | None:
        """Integrate affine ranks through the source's authoritative law.

        This avoids numerical quadrature misreporting a divergent tail as a
        finite value, and handles negative slopes by reversing rank intervals.
        Clamps split rank space into exact constant tails and an affine middle.
        """
        document = _load(self.to_json())
        model = cast(dict[str, object], document["model"])
        if model["kind"] != "linear":
            return None
        slope = float(cast(float, model["slope"]))
        intercept = float(cast(float, model["intercept"]))
        if slope == 0:
            value = float(self.evaluate(0.0)) - center
            return (hi - lo) * (value * value if square else value)

        def affine(left: float, right: float) -> float:
            if right <= left:
                return 0.0
            start, end = (left, right) if slope > 0 else (1 - right, 1 - left)
            if square:
                return _scaled_square(
                    slope,
                    source._integral(
                        start, end, center=(center - intercept) / slope, square=True
                    ),
                )
            return slope * source._integral(start, end) + (intercept - center) * (
                right - left
            )

        clamp = cast(dict[str, float] | None, document.get("clamp"))
        if clamp is None:
            return affine(lo, hi)
        minimum, maximum = clamp.get("minimum"), clamp.get("maximum")
        contributions: list[float] = []
        middle_lo, middle_hi = lo, hi

        def constant(value: float, width: float) -> float:
            value -= center
            return width * (value * value if square else value)

        if minimum is not None:
            threshold = (minimum - intercept) / slope
            probability = float(
                source.cdf(threshold)
                if slope > 0
                else source.exceedance(threshold, inclusive=True)
            )
            width = max(0.0, min(hi, probability) - lo)
            if width > 0:
                contributions.append(constant(minimum, width))
            middle_lo = max(middle_lo, probability)
        if maximum is not None:
            threshold = (maximum - intercept) / slope
            probability = float(
                source.exceedance(threshold)
                if slope > 0
                else source.cdf(threshold, inclusive=False)
            )
            boundary = 1 - probability
            if probability > 0 and boundary == 1.0:
                raise ImpactError(
                    "precision_limit",
                    "/statistics",
                    "the clamped tail lies beyond representable rank resolution",
                )
            width = (
                probability
                if lo <= boundary and hi == 1
                else max(0.0, hi - max(lo, boundary))
            )
            if width > 0:
                contributions.append(constant(maximum, width))
            middle_hi = min(middle_hi, boundary)
        contributions.append(affine(middle_lo, middle_hi))
        try:
            return math.fsum(contributions)
        except ValueError:
            return math.nan

    def _inverse_survival(self, source: Distribution, survival: float) -> float | None:
        document = _load(self.to_json())
        model = cast(dict[str, object], document["model"])
        if model["kind"] != "linear":
            return None
        slope = float(cast(float, model["slope"]))
        intercept = float(cast(float, model["intercept"]))
        if slope == 0:
            return float(self.evaluate(0.0))
        value = (
            slope
            * float(
                source._inverse_survival(survival)
                if slope > 0
                else source.quantile(survival)
            )
            + intercept
        )
        clamp = cast(dict[str, float] | None, document.get("clamp"))
        if clamp is not None:
            if "minimum" in clamp:
                value = max(clamp["minimum"], value)
            if "maximum" in clamp:
                value = min(clamp["maximum"], value)
        return value

    def _tail_integral(
        self,
        source: Distribution,
        s_min: float,
        s_max: float,
        *,
        center: float = 0.0,
        square: bool = False,
    ) -> float | None:
        """Integrate upper ranks using survival coordinates without cancellation."""
        document = _load(self.to_json())
        model = cast(dict[str, object], document["model"])
        if model["kind"] != "linear":
            return None
        slope = float(cast(float, model["slope"]))
        intercept = float(cast(float, model["intercept"]))
        if slope == 0:
            value = float(self.evaluate(0.0)) - center
            return (s_max - s_min) * (value * value if square else value)

        def affine(left: float, right: float) -> float:
            if right <= left:
                return 0.0
            integrate = source._tail_integral if slope > 0 else source._integral
            if square:
                return _scaled_square(
                    slope,
                    integrate(
                        left, right, center=(center - intercept) / slope, square=True
                    ),
                )
            return slope * integrate(left, right) + (intercept - center) * (
                right - left
            )

        clamp = cast(dict[str, float] | None, document.get("clamp"))
        if clamp is None:
            return affine(s_min, s_max)
        contributions: list[float] = []
        middle_lo, middle_hi = s_min, s_max

        def constant(value: float, width: float) -> float:
            value -= center
            return width * (value * value if square else value)

        if "maximum" in clamp:
            maximum = clamp["maximum"]
            threshold = (maximum - intercept) / slope
            probability = float(
                source.exceedance(threshold)
                if slope > 0
                else source.cdf(threshold, inclusive=False)
            )
            width = max(0.0, min(s_max, probability) - s_min)
            if width > 0:
                contributions.append(constant(maximum, width))
            middle_lo = max(middle_lo, probability)
        if "minimum" in clamp:
            minimum = clamp["minimum"]
            threshold = (minimum - intercept) / slope
            probability = float(
                source.cdf(threshold)
                if slope > 0
                else source.exceedance(threshold, inclusive=True)
            )
            boundary = 1 - probability
            if probability > 0 and boundary == 1.0:
                raise ImpactError(
                    "precision_limit",
                    "/statistics",
                    "the clamped lower tail lies beyond representable survival resolution",
                )
            width = (
                probability
                if s_min <= boundary and s_max == 1
                else max(0.0, s_max - max(s_min, boundary))
            )
            if width > 0:
                contributions.append(constant(minimum, width))
            middle_hi = min(middle_hi, boundary)
        contributions.append(affine(middle_lo, middle_hi))
        try:
            return math.fsum(contributions)
        except ValueError:
            return math.nan

    def _threshold_probability(
        self,
        source: Distribution,
        value: float,
        inclusive: bool,
        *,
        exceedance: bool = False,
    ) -> float | None:
        """Invert supported monotone responses without subtracting rare tails."""
        document = _load(self.to_json())
        model = cast(dict[str, object], document["model"])
        kind = model["kind"]
        minimum = maximum = None
        if kind == "linear":
            direction = float(cast(float, model["slope"]))
            intercept = float(cast(float, model["intercept"]))
            clamp = cast(dict[str, float] | None, document.get("clamp"))
            if clamp is not None:
                minimum, maximum = clamp.get("minimum"), clamp.get("maximum")
            if direction == 0:
                response = float(self.evaluate(0.0))
                return float(
                    (response >= value if inclusive else response > value)
                    if exceedance
                    else (response <= value if inclusive else response < value)
                )
            threshold = (value - intercept) / direction
        elif (
            kind == "sigmoid"
            and "zero_below" not in model
            and "origin_adjustment" not in model
        ):
            amplitude = float(cast(float, model["amplitude"]))
            steepness = float(cast(float, model["steepness"]))
            offset = float(cast(float, model["offset"]))
            direction = amplitude * steepness
            if direction == 0:
                response = float(self.evaluate(0.0))
                return float(
                    (response >= value if inclusive else response > value)
                    if exceedance
                    else (response <= value if inclusive else response < value)
                )
            minimum, maximum = (
                min(offset, offset + amplitude),
                max(offset, offset + amplitude),
            )
            fraction = (value - offset) / amplitude
            if fraction <= 0 or fraction >= 1:
                # The sigmoid reaches its endpoint only at infinite input.
                if value <= minimum:
                    return 1.0 if exceedance else 0.0
                return 0.0 if exceedance else 1.0
            threshold = (
                float(cast(float, model["midpoint"]))
                + (math.log(fraction) - math.log1p(-fraction)) / steepness
            )
        else:
            return None
        if minimum is not None and (
            value < minimum
            or value == minimum
            and (inclusive if exceedance else not inclusive)
        ):
            return 1.0 if exceedance else 0.0
        if maximum is not None and (
            value > maximum
            or value == maximum
            and (not inclusive if exceedance else inclusive)
        ):
            return 0.0 if exceedance else 1.0
        if exceedance:
            return float(
                source.exceedance(threshold, inclusive=inclusive)
                if direction > 0
                else source.cdf(threshold, inclusive=inclusive)
            )
        return float(
            source.cdf(threshold, inclusive=inclusive)
            if direction > 0
            else source.exceedance(threshold, inclusive=inclusive)
        )

    def _operation(
        self, distribution: Distribution
    ) -> tuple[_core.NativeImpactExpression, _core.NativeImpactSource]:
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

    def expectation(self, distribution: Distribution) -> float:
        if not isinstance(distribution, Distribution):
            raise TypeError("distribution must be Distribution")
        try:
            native, source = self._operation(distribution)
        except (TypeError, ValueError) as error:
            raise ImpactError("exact_method_unavailable", "", str(error)) from error
        try:
            return float(native.expected_value(source, "exact").value)
        except Exception as error:
            raise _error(error) from error

    def pushforward(self, distribution: Distribution) -> Distribution:
        if not isinstance(distribution, Distribution):
            raise TypeError("distribution must be Distribution")
        data = distribution._data
        try:
            expression, source = self._operation(distribution)
            native = expression.transform_distribution(source)
        except TypeError as error:
            raise ImpactError("exact_method_unavailable", "", str(error)) from error
        except Exception as error:
            raise _error(error) from error
        if data["kind"] in ("discrete", "point_mass"):
            atoms = native.atoms()
            return Distribution.discrete([x for x, _ in atoms], [p for _, p in atoms])
        return Distribution._from_transformed(
            distribution, _load(self.to_json()), native
        )

    @classmethod
    def from_json(cls, text: str) -> Impact:
        document = _load(text)
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
    CONSECUTIVE_DRY_DAYS = "crc:consecutive_dry_days"


_FACTORS = dict(
    zip(
        BuiltinImpact,
        (
            "daily_freezethaw_cycles",
            "frost_days",
            "rx1day",
            "cyclone",
            "wind_max_daily_max",
            "fwi",
            "hot_days",
            "cflood",
            "rflood",
            "inundation",
            "spei",
            "cooling_degree_days",
            "maximum_consecutive_dry_days",
        ),
    )
)
_BUILDING_NAMES = {
    BuildingType.RESIDENTIAL: "Residential buildings",
    BuildingType.COMMERCIAL: "Commercial buildings",
    BuildingType.INDUSTRIAL: "Industrial buildings",
    BuildingType.AGRICULTURAL: "Agricultural buildings",
}


@dataclass(frozen=True)
class ContextField:
    name: str
    annotation: str
    required: bool
    choices: tuple[str, ...]


@dataclass(frozen=True)
class ImpactRequirements:
    measurement_type: type[Measurement]
    accepted_units: tuple[Unit, ...]
    context_type: type[ImpactContext]
    context_fields: tuple[ContextField, ...]
    output: Measurement


class _ContextRecord:
    def __post_init__(self) -> None:
        for name in self.__annotations__:
            value = getattr(self, name)
            if isinstance(value, ContextValue) and not isinstance(value, Enum):
                continue
            if name == "building_type":
                if not isinstance(value, BuildingType):
                    raise TypeError(
                        "building_type must be BuildingType or a context binding"
                    )
            elif name == "historic_mean":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise TypeError("historic_mean must be finite or a context binding")
            elif not isinstance(value, str) or not value or value.strip() != value:
                raise TypeError(
                    f"{name} must be a nonempty identifier or a context binding"
                )

    @property
    def schema(self) -> Mapping[str, ScientificValue]:
        return MappingProxyType(
            {
                field: str(annotation)
                for field, annotation in self.__annotations__.items()
            }
        )

    def to_json(self) -> str:
        values: dict[str, object] = {}
        for name in self.__annotations__:
            value = getattr(self, name)
            if isinstance(value, Enum):
                values[name] = value.value
            elif isinstance(value, ContextValue):
                values[name] = {"column": value.name}
            else:
                values[name] = value
        return _dumps(values)


_CONTEXT_TYPES: dict[tuple[str, ...], type[ImpactContext]] = {}


def _context_type(fields: tuple[str, ...]) -> type[ImpactContext]:
    if fields == ("continent", "building_type"):
        return InundationContext
    if fields not in _CONTEXT_TYPES:
        annotations: dict[str, object] = {
            name: BuildingType | ContextValue
            if name == "building_type"
            else float | ContextValue
            if name == "historic_mean"
            else str | ContextValue
            for name in fields
        }
        generated = make_dataclass(
            "ModelContext",
            list(annotations.items()),
            bases=(_ContextRecord,),
            frozen=True,
            kw_only=True,
            namespace={"__module__": "crc_framework.impacts"},
        )
        _CONTEXT_TYPES[fields] = cast(type[ImpactContext], generated)
    return _CONTEXT_TYPES[fields]


class ImpactReference:
    """Pinned built-in identity. Resolve the model's typed context once."""

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
            "manifest_sha256": digest,
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
        metadata = self._metadata()
        calibration = cast(dict[str, str], metadata["calibration"])
        fields = tuple(cast(list[str], metadata["required_context"]))
        measurement = self._input()
        units = (measurement.unit,) if isinstance(measurement.unit, Enum) else ()
        building_choices = (
            (BuildingType.AGRICULTURAL,)
            if self._model is BuiltinImpact.CONSECUTIVE_DRY_DAYS
            else (
                tuple(BuildingType)
                if self._model is BuiltinImpact.COOLING_DEGREE_DAYS
                else (
                    BuildingType.RESIDENTIAL,
                    BuildingType.COMMERCIAL,
                    BuildingType.INDUSTRIAL,
                )
            )
        )
        context_fields = tuple(
            ContextField(
                name,
                "BuildingType"
                if name == "building_type"
                else "float"
                if name == "historic_mean"
                else "str",
                True,
                tuple(item.value for item in building_choices)
                if name == "building_type"
                else (),
            )
            for name in fields
        )
        return ImpactRequirements(
            type(measurement),
            cast(tuple[Unit, ...], units),
            _context_type(fields),
            context_fields,
            self._output(calibration),
        )

    def _input(self) -> Measurement:
        metadata = self._metadata()
        calibration = cast(dict[str, str], metadata["calibration"])
        if self._model in (
            BuiltinImpact.INUNDATION,
            BuiltinImpact.COASTAL_FLOOD,
            BuiltinImpact.RIVER_FLOOD,
        ):
            return DepthAboveGround(unit=LengthUnit.METRE)
        return _measurement(
            {
                "kind": "measurement",
                "quantity": calibration["exposure_variable"],
                "unit": calibration["exposure_unit"],
                "fields": {},
                "schema": None,
            }
        )

    @staticmethod
    def _output(calibration: Mapping[str, str]) -> Measurement:
        if calibration["impact_metric"] == "damage_ratio":
            return Ratio()
        # These models divide monetary responses by their stated normalization
        # scale; crop response is a fractional yield change. Their numerical
        # definitions therefore declare dimensionless outputs directly.
        quantities = {
            "normalized_water_cost": "normalized_water_cost",
            "normalized_cooling_cost": "normalized_cooling_cost",
            "crop_yield_change": "crop_yield_change",
        }
        quantity = quantities[calibration["impact_metric"]]
        return Measurement(quantity=quantity, unit=RatioUnit.FRACTION)

    def resolve(self, context: ImpactContext | None = None) -> Impact:
        required = self.requirements
        if context is not None and not isinstance(context, required.context_type):
            raise ImpactError(
                "invalid_context",
                "/context",
                f"use {required.context_type.__name__} with fields "
                + ", ".join(field.name for field in required.context_fields),
            )
        values = {} if context is None else json.loads(context.to_json())
        for field in required.context_fields:
            if field.name not in values or values[field.name] is None:
                raise ImpactError(
                    "missing_context",
                    "/context/" + field.name,
                    f"{required.context_type.__name__}.{field.name} is required",
                )
        for name, value in values.items():
            if isinstance(value, dict):
                raise ImpactError(
                    "unresolved_context",
                    "/context/" + name,
                    "resolve asset-column bindings before numerical use",
                )
        building = values.get("building_type")
        if building is not None:
            choices = next(
                field.choices
                for field in required.context_fields
                if field.name == "building_type"
            )
            if building not in choices:
                raise ImpactError(
                    "unsupported_context",
                    "/context/building_type",
                    "accepted BuildingType values: " + ", ".join(choices),
                )
            try:
                values["building_type"] = _BUILDING_NAMES[BuildingType(building)]
            except (KeyError, ValueError) as error:
                raise ImpactError(
                    "unsupported_context",
                    "/context/building_type",
                    "use a supported BuildingType",
                ) from error
        registry, version, resolver, digest = _core.impact_registry_identity()
        if self._version != version or self._manifest != digest:
            raise ImpactError(
                "registry_unavailable",
                "",
                "the pinned built-in version is not installed",
            )
        reference = {
            "schema_version": "1.0",
            "registry_id": registry,
            "registry_revision": version,
            "registry_manifest_sha256": digest,
            "resolver_semantics_version": resolver,
            "factor": _FACTORS[self._model],
            "fallback_policy": "error",
        }
        try:
            native_reference = _core.NativeImpactReference.from_json(_dumps(reference))
            native = native_reference.resolve(
                values.get("cell"),
                values.get("country"),
                values.get("continent"),
                values.get("building_type"),
                values.get("historic_mean"),
            )
        except Exception as error:
            raise _error(error) from error
        calibration = cast(dict[str, str], self._metadata()["calibration"])
        return Impact._from_native(native, self._input(), self._output(calibration))

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

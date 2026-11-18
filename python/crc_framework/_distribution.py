"""Complete probability laws and explicitly incomplete quantile evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy
import numpy.typing

from . import _core
from ._diagnostics import Diagnostic, DiagnosticSeverity

if TYPE_CHECKING:
    from ._impacts import Impact

FloatArray: TypeAlias = numpy.typing.NDArray[numpy.float64]
ArrayLike: TypeAlias = Sequence[float] | FloatArray
Values: TypeAlias = float | ArrayLike
ValueResult: TypeAlias = float | FloatArray


class DistributionFamily(Enum):
    GENERALIZED_EXTREME_VALUE = "genextreme"
    WEIBULL_MINIMUM = "weibull_min"
    WEIBULL_MAXIMUM = "weibull_max"
    SKEW_NORMAL = "skewnorm"
    GUMBEL_RIGHT = "gumbel_r"
    GUMBEL_LEFT = "gumbel_l"
    GENERALIZED_PARETO = "genpareto"


class DistributionKind(Enum):
    PARAMETRIC = "parametric"
    DISCRETE = "discrete"
    HISTOGRAM = "histogram"
    POINT_MASS = "point_mass"
    HURDLE = "hurdle"
    QUANTILE = "quantile"
    TRANSFORMED = "transformed"


class Statistic(Enum):
    MEAN = "mean"
    VARIANCE = "variance"
    STANDARD_DEVIATION = "standard_deviation"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    CVAR = "cvar"


class StatisticStatus(Enum):
    FINITE = "finite"
    POSITIVE_INFINITY = "positive_infinity"
    NEGATIVE_INFINITY = "negative_infinity"
    UNDEFINED = "undefined"


class Tail(Enum):
    UPPER = "upper"
    LOWER = "lower"


@dataclass(frozen=True)
class StatisticValue:
    status: StatisticStatus
    value: float | None
    diagnostics: tuple[Diagnostic, ...]


@dataclass(frozen=True)
class StatisticalSummary:
    values: Mapping[Statistic, StatisticValue]
    level: float | None
    tail: Tail | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True, kw_only=True)
class TailCompletion:
    """Complete finite tails linearly to explicitly declared endpoints."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        _finite(self.lower, "lower endpoint")
        _finite(self.upper, "upper endpoint")
        if self.lower > self.upper:
            raise ValueError("lower endpoint must not exceed upper endpoint")

    @property
    def name(self) -> str:
        return "finite_endpoints"

    def to_json(self) -> str:
        return _dump({"name": self.name, "lower": self.lower, "upper": self.upper})


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, numpy.floating, numpy.integer)
    ):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(values: ArrayLike, label: str) -> FloatArray:
    if numpy.asarray(values).dtype.kind not in "iuf":
        raise TypeError(f"{label} must contain real numbers")
    array = numpy.asarray(values, dtype=numpy.float64)
    if array.ndim != 1 or array.size == 0 or not numpy.isfinite(array).all():
        raise ValueError(f"{label} must be a nonempty finite one-dimensional array")
    return array


def _snapshot(array: FloatArray) -> FloatArray:
    # A bytes-backed snapshot cannot be made writeable by setflags().
    return numpy.frombuffer(array.tobytes(), dtype=numpy.float64).reshape(array.shape)


def _input(values: Values, *, probabilities: bool = False) -> tuple[FloatArray, bool]:
    if numpy.asarray(values).dtype.kind not in "iuf":
        raise TypeError("queries must contain real numbers")
    array = numpy.asarray(values, dtype=numpy.float64)
    if numpy.isnan(array).any():
        raise ValueError("queries must not contain NaN")
    if probabilities and (
        not numpy.isfinite(array).all() or ((array < 0) | (array > 1)).any()
    ):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return array, array.ndim == 0


def _output(
    values: Sequence[float] | FloatArray, shape: tuple[int, ...], scalar: bool
) -> ValueResult:
    result = numpy.asarray(values, dtype=numpy.float64).reshape(shape)
    return float(result) if scalar else _snapshot(result)


def _dump(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _parse(text: str) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("numerical representation must be an object")  # noqa: TRY004
    return cast(dict[str, object], value)


def _fields(
    data: Mapping[str, object],
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
) -> None:
    if set(data) - required - optional or not required <= set(data):
        raise ValueError(
            f"invalid numerical fields; required={sorted(required)}, found={sorted(data)}"
        )


@dataclass(frozen=True, init=False)
class Distribution:
    _data: dict[str, object]
    _native: _core.NativeNumericalLaw | None
    _transformed: _core.NativeImpactDistribution | None
    _record: dict[str, object] | None
    _base: Distribution | None

    def __init__(self) -> None:
        raise TypeError("construct a Distribution with a named constructor")

    @classmethod
    def _create(
        cls, data: dict[str, object], record: dict[str, object] | None = None
    ) -> Distribution:
        result = object.__new__(cls)
        base = None
        native_data = dict(data)
        if data.get("kind") == "hurdle":
            _fields(data, {"kind", "base", "atom_probability", "atom_location"})
            base_data = cast(dict[str, object], data["base"])
            hydrated = cls.from_json(
                _dump({"schema_version": "1.0", "object": "distribution", **base_data})
            )
            native_data["base"] = hydrated._native_data()
            if hydrated._native is None:
                base = hydrated
                probability = _finite(
                    cast(float, data["atom_probability"]), "atom_probability"
                )
                atom = _finite(cast(float, data["atom_location"]), "atom_location")
                if not 0 <= probability <= 1 or (
                    probability < 1 and cast(float, base.exceedance(atom)) <= 0
                ):
                    raise ValueError(
                        "hurdle requires a probability in [0, 1] and a nonempty strict upper tail"
                    )
        object.__setattr__(
            result,
            "_native",
            None if base is not None else _core.NativeNumericalLaw(_dump(native_data)),
        )
        object.__setattr__(result, "_data", data)
        object.__setattr__(result, "_transformed", None)
        object.__setattr__(result, "_record", record)
        object.__setattr__(result, "_base", base)
        return result

    def _native_data(self) -> dict[str, object]:
        data = dict(self._data)
        if self.kind is DistributionKind.HURDLE:
            base_data = cast(dict[str, object], data["base"])
            data["base"] = Distribution.from_json(
                _dump({"schema_version": "1.0", "object": "distribution", **base_data})
            )._native_data()
        return data

    def _serialized_data(self) -> dict[str, object]:
        return {
            **self._data,
            **({"evidence": self._record} if self._record is not None else {}),
        }

    @classmethod
    def parametric(
        cls,
        family: DistributionFamily,
        *,
        location: float,
        scale: float,
        shape: float | None = None,
    ) -> Distribution:
        if not isinstance(family, DistributionFamily):
            raise TypeError("family must be a DistributionFamily")
        if (
            family in (DistributionFamily.GUMBEL_LEFT, DistributionFamily.GUMBEL_RIGHT)
            and shape is not None
        ):
            raise ValueError("Gumbel distributions have no shape parameter")
        parameters: dict[str, object] = {
            "kind": "parametric",
            "family": family.value,
            "location": _finite(location, "location"),
            "scale": _finite(scale, "scale"),
        }
        if shape is not None:
            parameters["shape"] = _finite(shape, "shape")
        return cls._create(parameters)

    @classmethod
    def _from_native(cls, native: _core.NativeFitParameters) -> Distribution:
        if native.family is None or native.location is None or native.scale is None:
            raise ValueError("native fit has no parametric representation")
        return cls.parametric(
            DistributionFamily(native.family),
            location=native.location,
            scale=native.scale,
            shape=native.shape,
        )

    @classmethod
    def empirical(cls, samples: ArrayLike) -> Distribution:
        observed = _vector(samples, "samples")
        values, counts = numpy.unique(observed, return_counts=True)
        return cls.discrete(values, counts.astype(numpy.float64) / observed.size)

    @classmethod
    def discrete(cls, values: ArrayLike, probabilities: ArrayLike) -> Distribution:
        support = _vector(values, "values")
        masses = _vector(probabilities, "probabilities")
        if support.size != masses.size or (masses < 0).any():
            raise ValueError(
                "values and non-negative probability masses must have matching lengths"
            )
        total = math.fsum(masses)
        if abs(total - 1.0) > 1e-12:
            raise ValueError("probability masses must sum to one")
        grouped: dict[float, list[float]] = {}
        for value, mass in zip(support, masses):
            if mass > 0:
                grouped.setdefault(float(value), []).append(float(mass))
        unique = sorted(grouped)
        normalized = [math.fsum(grouped[value]) / total for value in unique]
        return cls._create(
            {"kind": "discrete", "values": unique, "probabilities": normalized}
        )

    @classmethod
    def histogram(cls, edges: ArrayLike, probabilities: ArrayLike) -> Distribution:
        boundaries = _vector(edges, "edges")
        masses = _vector(probabilities, "probabilities")
        total = math.fsum(masses)
        if (masses < 0).any() or abs(total - 1.0) > 1e-12:
            raise ValueError("histogram masses must be non-negative and sum to one")
        return cls._create(
            {
                "kind": "histogram",
                "edges": boundaries.tolist(),
                "probabilities": (masses / total).tolist(),
            }
        )

    @classmethod
    def point_mass(cls, value: float) -> Distribution:
        return cls._create({"kind": "point_mass", "value": _finite(value, "value")})

    @classmethod
    def hurdle(
        cls, base: Distribution, *, atom_probability: float, atom_location: float = 0.0
    ) -> Distribution:
        if not isinstance(base, Distribution):
            raise TypeError("base must be a Distribution")
        return cls._create(
            {
                "kind": "hurdle",
                "base": base._serialized_data(),
                "atom_probability": _finite(atom_probability, "atom_probability"),
                "atom_location": _finite(atom_location, "atom_location"),
            }
        )

    @property
    def kind(self) -> DistributionKind:
        return DistributionKind(self._data["kind"])

    @property
    def support(self) -> tuple[float, float]:
        if self._native is not None:
            return self._native.support()
        if self._base is not None:
            atom = cast(float, self._data["atom_location"])
            p = cast(float, self._data["atom_probability"])
            return (
                (atom, atom)
                if p == 1
                else (
                    atom if p else max(atom, self._base.support[0]),
                    self._base.support[1],
                )
            )
        assert self._transformed is not None
        return self._transformed.quantile(0.0), self._transformed.quantile(1.0)

    def quantile(self, probabilities: Values) -> ValueResult:
        query, scalar = _input(probabilities, probabilities=True)
        if self._native is not None:
            result = self._native.quantile(query.reshape(-1).tolist())
        elif self._base is not None:
            p = cast(float, self._data["atom_probability"])
            atom = cast(float, self._data["atom_location"])
            surviving = cast(float, self._base.exceedance(atom))
            result = [
                atom
                if q <= p
                else self._base._inverse_survival((1 - float(q)) / (1 - p) * surviving)
                for q in query.reshape(-1)
            ]
        else:
            assert self._transformed is not None
            result = self._transformed.quantiles(query.reshape(-1).tolist())
        return _output(result, query.shape, scalar)

    def cdf(self, values: Values, *, inclusive: bool = True) -> ValueResult:
        if type(inclusive) is not bool:
            raise TypeError("inclusive must be bool")
        query, scalar = _input(values)
        if self._native is not None:
            result = self._native.cdf(query.reshape(-1).tolist(), inclusive)
        elif self._base is not None:
            result = [
                self._compound_cdf(float(x), inclusive) for x in query.reshape(-1)
            ]
        else:
            result = [
                self._transformed_cdf(float(x), inclusive) for x in query.reshape(-1)
            ]
        return _output(result, query.shape, scalar)

    def exceedance(self, values: Values, *, inclusive: bool = False) -> ValueResult:
        query, scalar = _input(values)
        if type(inclusive) is not bool:
            raise TypeError("inclusive must be bool")
        if self._native is not None:
            result = self._native.exceedance(query.reshape(-1).tolist(), inclusive)
        elif self._base is not None:
            p, atom = (
                cast(float, self._data["atom_probability"]),
                cast(float, self._data["atom_location"]),
            )
            surviving = cast(float, self._base.exceedance(atom))
            result = [
                1.0
                if x < atom or inclusive and x == atom
                else 0.0
                if p == 1
                else (1 - p)
                * cast(float, self._base.exceedance(float(x), inclusive=inclusive))
                / surviving
                for x in query.reshape(-1)
            ]
        else:
            result = [
                self._transformed_exceedance(float(x), inclusive)
                for x in query.reshape(-1)
            ]
        return _output(result, query.shape, scalar)

    def mass(self, values: Values) -> ValueResult:
        query, scalar = _input(values)
        if self._native is not None:
            result = self._native.mass(query.reshape(-1).tolist())
        elif self._base is not None:
            p, atom = (
                cast(float, self._data["atom_probability"]),
                cast(float, self._data["atom_location"]),
            )
            surviving = cast(float, self._base.exceedance(atom))
            result = [
                p
                if x == atom
                else (1 - p) * cast(float, self._base.mass(float(x))) / surviving
                if x > atom and p < 1
                else 0.0
                for x in query.reshape(-1)
            ]
        else:
            assert self._transformed is not None
            atoms = dict(self._transformed.atoms())
            result = [atoms.get(float(x), 0.0) for x in query.reshape(-1)]
        return _output(result, query.shape, scalar)

    def _point_atoms(self) -> list[tuple[float, float]]:
        if self._native is not None:
            return self._native.atoms()
        if self._transformed is not None:
            return self._transformed.atoms()
        assert self._base is not None
        p = cast(float, self._data["atom_probability"])
        atom = cast(float, self._data["atom_location"])
        result = [(atom, p)] if p else []
        if p < 1:
            survival = float(self._base.exceedance(atom))
            result.extend(
                (x, mass / survival * (1 - p))
                for x, mass in self._base._point_atoms()
                if x > atom
            )
        return result

    def sample(
        self, size: int | tuple[int, ...], *, seed: int | None = None
    ) -> FloatArray:
        dimensions = (size,) if isinstance(size, int) else size
        if not isinstance(dimensions, tuple) or any(
            type(n) is not int or n < 0 for n in dimensions
        ):
            raise ValueError(
                "size must be a non-negative integer or tuple of non-negative integers"
            )
        return cast(
            FloatArray, self.quantile(numpy.random.default_rng(seed).random(dimensions))
        )

    def statistics(
        self,
        *,
        metrics: Sequence[Statistic],
        level: float = 0.95,
        tail: Tail = Tail.UPPER,
    ) -> StatisticalSummary:
        if not isinstance(tail, Tail):
            raise TypeError("tail must be a Tail")
        if any(not isinstance(metric, Statistic) for metric in metrics):
            raise TypeError("metrics must contain Statistic members")
        level = _finite(level, "level")
        if not 0 < level < 1:
            raise ValueError("level must be strictly between zero and one")
        mean, variance = (
            self._moments()
            if any(
                m in metrics
                for m in (
                    Statistic.MEAN,
                    Statistic.VARIANCE,
                    Statistic.STANDARD_DEVIATION,
                )
            )
            else (0.0, 0.0)
        )
        values: dict[Statistic, StatisticValue] = {}
        for metric in metrics:
            if metric is Statistic.MEAN:
                value = mean
            elif metric is Statistic.VARIANCE:
                value = variance
            elif metric is Statistic.STANDARD_DEVIATION:
                value = math.sqrt(variance) if variance >= 0 else math.nan
            elif metric is Statistic.MINIMUM:
                value = self.support[0]
            elif metric is Statistic.MAXIMUM:
                value = self.support[1]
            else:
                lo, hi = (level, 1.0) if tail is Tail.UPPER else (0.0, 1.0 - level)
                value = self._integral(lo, hi) / (hi - lo)
            values[metric] = _statistic_value(metric, value)
        has_cvar = Statistic.CVAR in metrics
        return StatisticalSummary(
            values, level if has_cvar else None, tail if has_cvar else None
        )

    def _moments(self) -> tuple[float, float]:
        if self._native is not None:
            return self._native.moments()
        if self._transformed is not None:
            impact, source = self._impact_source()
            exact = impact._moments(source)
            if exact is not None:
                return exact
        mean = self._integral(0.0, 1.0)
        if not math.isfinite(mean):
            return mean, math.nan
        return mean, self._integral(0.0, 1.0, center=mean, square=True)

    def _integral(
        self, lo: float, hi: float, *, center: float = 0.0, square: bool = False
    ) -> float:
        if self._native is not None:
            return self._native.integral(lo, hi, center, square)
        if self._base is not None:
            return self._tail_integral(1 - hi, 1 - lo, center=center, square=square)
        # Transformed finite laws retain all exact probability atoms. General
        # transformed laws are integrated adaptively by the native engine.
        assert self._transformed is not None
        impact, source = self._impact_source()
        exact = impact._integral(source, lo, hi, center=center, square=square)
        if exact is not None:
            return exact
        return self._transformed.integral(lo, hi, center, square)

    def _impact_source(self) -> tuple[Impact, Distribution]:
        from ._impacts import Impact

        return Impact.from_json(_dump(self._data["impact"])), Distribution.from_json(
            _dump(self._data["source"])
        )

    def _inverse_survival(self, probability: float) -> float:
        if self._native is not None:
            return self._native.inverse_survival(probability)
        if probability == 0:
            return self.support[1]
        if probability == 1:
            return self.support[0]
        if self._base is not None:
            p, atom = (
                cast(float, self._data["atom_probability"]),
                cast(float, self._data["atom_location"]),
            )
            if probability >= 1 - p:
                return atom
            surviving = cast(float, self._base.exceedance(atom))
            return self._base._inverse_survival(probability / (1 - p) * surviving)
        assert self._transformed is not None
        impact, source = self._impact_source()
        exact = impact._inverse_survival(source, probability)
        return (
            exact
            if exact is not None
            else self._transformed.inverse_survival(probability)
        )

    def _tail_integral(
        self, lo: float, hi: float, *, center: float = 0.0, square: bool = False
    ) -> float:
        if self._native is not None:
            return self._native.tail_integral(lo, hi, center, square)
        if self._base is not None:
            p, atom = (
                cast(float, self._data["atom_probability"]),
                cast(float, self._data["atom_location"]),
            )
            w = 1 - p
            value = max(0.0, hi - max(lo, w)) * (
                (atom - center) ** 2 if square else atom - center
            )
            if lo >= w:
                return value
            surviving = cast(float, self._base.exceedance(atom))
            return value + w / surviving * self._base._tail_integral(
                lo / w * surviving,
                min(hi, w) / w * surviving,
                center=center,
                square=square,
            )
        assert self._transformed is not None
        impact, source = self._impact_source()
        exact = impact._tail_integral(source, lo, hi, center=center, square=square)
        return (
            exact
            if exact is not None
            else self._transformed.tail_integral(lo, hi, center, square)
        )

    def _compound_cdf(self, x: float, inclusive: bool) -> float:
        assert self._base is not None
        p, atom = (
            cast(float, self._data["atom_probability"]),
            cast(float, self._data["atom_location"]),
        )
        if x < atom or not inclusive and x == atom:
            return 0.0
        if p == 1:
            return 1.0
        surviving = cast(float, self._base.exceedance(atom))
        return p + (1 - p) * (
            1
            - cast(float, self._base.exceedance(x, inclusive=not inclusive)) / surviving
        )

    def _transformed_cdf(self, x: float, inclusive: bool) -> float:
        assert self._transformed is not None
        exact = self._threshold_probability(x, inclusive, exceedance=False)
        if exact is not None:
            return exact
        lower, upper = self.support
        if x < lower or (not inclusive and x == lower):
            return 0.0
        if x > upper or (inclusive and x == upper):
            return 1.0
        left, right = 0.0, 1.0
        for _ in range(64):
            midpoint = (left + right) / 2
            value = self._transformed.quantile(midpoint)
            if value <= x if inclusive else value < x:
                left = midpoint
            else:
                right = midpoint
        return (left + right) / 2

    def _threshold_probability(
        self, x: float, inclusive: bool, *, exceedance: bool
    ) -> float | None:
        from ._impacts import Impact

        impact = Impact.from_json(_dump(self._data["impact"]))
        source = Distribution.from_json(_dump(self._data["source"]))
        assert self._transformed is not None
        return impact._threshold_probability(
            source,
            x,
            inclusive,
            exceedance=exceedance,
            at_atom=any(value == x for value, _ in self._transformed.atoms()),
        )

    def _transformed_exceedance(self, x: float, inclusive: bool) -> float:
        exact = self._threshold_probability(x, inclusive, exceedance=True)
        return (
            exact if exact is not None else 1 - self._transformed_cdf(x, not inclusive)
        )

    @classmethod
    def _from_transformed(
        cls,
        source: Distribution,
        definition: dict[str, object],
        native: _core.NativeImpactDistribution,
    ) -> Distribution:
        result = object.__new__(cls)
        object.__setattr__(result, "_native", None)
        object.__setattr__(result, "_transformed", native)
        object.__setattr__(result, "_record", None)
        object.__setattr__(result, "_base", None)
        object.__setattr__(
            result,
            "_data",
            {
                "kind": "transformed",
                "source": _parse(source.to_json()),
                "impact": definition,
            },
        )
        return result

    def _source(self) -> _core.NativeImpactSource:
        data = self._data
        if self.kind is DistributionKind.DISCRETE:
            return _core.NativeImpactSource.discrete(
                cast(list[float], data["values"]),
                cast(list[float], data["probabilities"]),
            )
        if self.kind is DistributionKind.HISTOGRAM:
            return _core.NativeImpactSource.histogram(
                cast(list[float], data["edges"]),
                cast(list[float], data["probabilities"]),
            )
        if self.kind is DistributionKind.POINT_MASS:
            return _core.NativeImpactSource.curve(
                "point_mass",
                "point_mass",
                curve_location=cast(float, data["value"]),
                curve_scale=0.0,
                curve_atom_probability=1.0,
                curve_atom_location=cast(float, data["value"]),
            )
        if self.kind is DistributionKind.PARAMETRIC:
            return _core.NativeImpactSource.curve(
                "fitted",
                cast(str, data["family"]),
                curve_location=cast(float, data["location"]),
                curve_scale=cast(float, data["scale"]),
                curve_shape=cast(float | None, data.get("shape")),
            )
        if self.kind is DistributionKind.QUANTILE:
            return _core.NativeImpactSource.curve(
                "tabulated",
                "linear_probability",
                curve_probabilities=cast(list[float], data["probabilities"]),
                curve_values=cast(list[float], data["values"]),
            )
        if self.kind is DistributionKind.HURDLE:
            base = cast(dict[str, object], data["base"])
            if (
                base["kind"] == "parametric"
                and 0 < cast(float, data["atom_probability"]) < 1
            ):
                return _core.NativeImpactSource.curve(
                    "hurdle",
                    cast(str, base["family"]),
                    curve_location=cast(float, base["location"]),
                    curve_scale=cast(float, base["scale"]),
                    curve_shape=cast(float | None, base.get("shape")),
                    curve_atom_probability=cast(float, data["atom_probability"]),
                    curve_atom_location=cast(float, data["atom_location"]),
                )
            if cast(float, data["atom_probability"]) == 1:
                return Distribution.point_mass(
                    cast(float, data["atom_location"])
                )._source()
            return self._segment_source()
        if self.kind is DistributionKind.TRANSFORMED:
            return self._segment_source()
        raise TypeError("this law requires its exact compound representation")

    def _segment_source(self) -> _core.NativeImpactSource:
        return _core.NativeImpactSource.segments(self._segments())

    def _segments(self) -> list[tuple[float, float, float]]:
        """Ordered exact uniform intervals and atoms for a finite law."""
        data = self._data
        if self.kind is DistributionKind.DISCRETE:
            return [
                (x, x, p)
                for x, p in zip(
                    cast(list[float], data["values"]),
                    cast(list[float], data["probabilities"]),
                )
            ]
        if self.kind is DistributionKind.POINT_MASS:
            x = cast(float, data["value"])
            return [(x, x, 1.0)]
        if self.kind is DistributionKind.HISTOGRAM:
            edges = cast(list[float], data["edges"])
            return [
                (a, b, p)
                for a, b, p in zip(
                    edges, edges[1:], cast(list[float], data["probabilities"])
                )
                if p > 0
            ]
        if self.kind is DistributionKind.QUANTILE:
            q = cast(list[float], data["probabilities"])
            knots = cast(list[float], data["values"])
            return [(a, b, q1 - q0) for a, b, q0, q1 in zip(knots, knots[1:], q, q[1:])]
        if self.kind is DistributionKind.HURDLE:
            base = Distribution.from_json(
                _dump(
                    {
                        "schema_version": "1.0",
                        "object": "distribution",
                        **cast(dict[str, object], data["base"]),
                    }
                )
            )
            p, atom = (
                cast(float, data["atom_probability"]),
                cast(float, data["atom_location"]),
            )
            if p == 1:
                return [(atom, atom, 1.0)]
            surviving = cast(float, base.exceedance(atom))
            result = [(atom, atom, p)] if p else []
            for lower, upper, mass in base._segments():
                if upper <= atom:
                    continue
                retained = 1.0 if lower >= atom else (upper - atom) / (upper - lower)
                result.append(
                    (max(lower, atom), upper, mass * retained * (1 - p) / surviving)
                )
            return result
        if self.kind is DistributionKind.TRANSFORMED:
            from ._impacts import Impact

            impact = Impact.from_json(_dump(data["impact"]))
            source = Distribution.from_json(_dump(data["source"]))
            return impact._segments(source)
        raise TypeError("this law has no finite piecewise uniform representation")

    def to_json(self) -> str:
        data = {
            "schema_version": "1.0",
            "object": "distribution",
            **self._serialized_data(),
        }
        return _dump(data)

    @classmethod
    def from_json(cls, text: str) -> Distribution:
        data = _parse(text)
        if (
            data.pop("schema_version", None) != "1.0"
            or data.pop("object", None) != "distribution"
        ):
            raise ValueError("expected numerical distribution schema 1.0")
        has_evidence = "evidence" in data
        record = data.pop("evidence", None)
        if data.get("kind") == "transformed":
            _fields(data, {"kind", "source", "impact"})
            from ._impacts import Impact

            result = Impact.from_json(_dump(data["impact"])).pushforward(
                cls.from_json(_dump(data["source"]))
            )
        else:
            if data.get("kind") == "parametric":
                shapeless = data.get("family") in (
                    DistributionFamily.GUMBEL_LEFT.value,
                    DistributionFamily.GUMBEL_RIGHT.value,
                )
                if shapeless and "shape" in data:
                    raise ValueError("Gumbel representations have no shape field")
                if not shapeless and data.get("shape") is None:
                    raise ValueError("this parametric family requires a shape")
            result = cls._create(data)
        if has_evidence:
            if result.kind is not DistributionKind.QUANTILE or not isinstance(
                record, dict
            ):
                raise ValueError(
                    "evidence applies only to a completed or conditioned quantile law"
                )
            _fields(record, {"source", "operation"}, {"q_min", "q_max", "policy"})
            source = QuantileCurve.from_json(_dump(record["source"]))
            if record["operation"] == "condition":
                _fields(record, {"source", "operation", "q_min", "q_max"})
                expected = source.condition(
                    q_min=cast(float, record["q_min"]),
                    q_max=cast(float, record["q_max"]),
                )
            elif record["operation"] == "complete":
                _fields(record, {"source", "operation", "policy"})
                policy = cast(dict[str, object], record["policy"])
                _fields(policy, {"name", "lower", "upper"})
                if policy["name"] != "finite_endpoints":
                    raise ValueError("unknown tail completion")
                expected = source.complete(
                    TailCompletion(
                        lower=cast(float, policy["lower"]),
                        upper=cast(float, policy["upper"]),
                    )
                )
            else:
                raise ValueError("unknown quantile modelling operation")
            if expected._data != result._data:
                raise ValueError(
                    "quantile law contradicts its recorded modelling choice"
                )
            return expected
        return result


def _statistic_value(metric: Statistic, value: float) -> StatisticValue:
    if math.isfinite(value):
        return StatisticValue(StatisticStatus.FINITE, value, ())
    status = (
        StatisticStatus.UNDEFINED
        if math.isnan(value)
        else StatisticStatus.POSITIVE_INFINITY
        if value > 0
        else StatisticStatus.NEGATIVE_INFINITY
    )
    diagnostic = Diagnostic(
        "undefined_statistic" if math.isnan(value) else "infinite_statistic",
        f"{metric.value} is {status.value}",
        DiagnosticSeverity.INFO,
        f"/statistics/{metric.value}",
        {},
    )
    return StatisticValue(status, None, (diagnostic,))


@dataclass(frozen=True, init=False)
class QuantileCurve:
    _probabilities: FloatArray
    _values: FloatArray

    def __init__(self, probabilities: ArrayLike, values: ArrayLike) -> None:
        q = _vector(probabilities, "probabilities")
        x = _vector(values, "values")
        if (
            len(q) < 2
            or len(q) != len(x)
            or (numpy.diff(q) <= 0).any()
            or (q < 0).any()
            or (q > 1).any()
            or (numpy.diff(x) < 0).any()
        ):
            raise ValueError(
                "quantile evidence requires matching ordered knots, probabilities strictly increasing in [0, 1] and values nondecreasing"
            )
        object.__setattr__(self, "_probabilities", _snapshot(q))
        object.__setattr__(self, "_values", _snapshot(x))

    @property
    def probabilities(self) -> FloatArray:
        return self._probabilities

    @property
    def values(self) -> FloatArray:
        return self._values

    @property
    def probability_interval(self) -> tuple[float, float]:
        return float(self._probabilities[0]), float(self._probabilities[-1])

    def quantile(self, probabilities: Values) -> ValueResult:
        query, scalar = _input(probabilities, probabilities=True)
        lo, hi = self.probability_interval
        if ((query < lo) | (query > hi)).any():
            raise ValueError(
                f"query lies outside supplied probability interval [{lo}, {hi}]"
            )
        return _output(
            numpy.interp(query.reshape(-1), self._probabilities, self._values),
            query.shape,
            scalar,
        )

    def condition(self, *, q_min: float, q_max: float) -> Distribution:
        q_min = _finite(q_min, "q_min")
        q_max = _finite(q_max, "q_max")
        lo, hi = self.probability_interval
        if not lo <= q_min < q_max <= hi:
            raise ValueError(
                "conditioning interval must have positive width within supplied evidence"
            )
        interior = self._probabilities[
            (self._probabilities > q_min) & (self._probabilities < q_max)
        ]
        q = numpy.concatenate(([q_min], interior, [q_max]))
        x = cast(FloatArray, self.quantile(q))
        data: dict[str, object] = {
            "kind": "quantile",
            "probabilities": ((q - q_min) / (q_max - q_min)).tolist(),
            "values": x.tolist(),
        }
        return Distribution._create(
            data,
            {
                "source": _parse(self.to_json()),
                "operation": "condition",
                "q_min": q_min,
                "q_max": q_max,
            },
        )

    def complete(self, policy: TailCompletion) -> Distribution:
        if not isinstance(policy, TailCompletion):
            raise TypeError("policy must be a TailCompletion")
        q = self._probabilities.tolist()
        x = self._values.tolist()
        if policy.lower > x[0] or policy.upper < x[-1]:
            raise ValueError("completion endpoints must contain all supplied values")
        if q[0] == 0 and policy.lower != x[0] or q[-1] == 1 and policy.upper != x[-1]:
            raise ValueError("completion cannot replace an already supplied endpoint")
        if q[0] != 0:
            q.insert(0, 0.0)
            x.insert(0, policy.lower)
        if q[-1] != 1:
            q.append(1.0)
            x.append(policy.upper)
        return Distribution._create(
            {"kind": "quantile", "probabilities": q, "values": x},
            {
                "source": _parse(self.to_json()),
                "operation": "complete",
                "policy": _parse(policy.to_json()),
            },
        )

    @classmethod
    def from_samples(cls, samples: ArrayLike) -> QuantileCurve:
        values = numpy.sort(_vector(samples, "samples"))
        if len(values) == 1:
            values = numpy.repeat(values, 2)
        return cls(numpy.linspace(0, 1, len(values)), values)

    def to_json(self) -> str:
        return _dump(
            {
                "schema_version": "1.0",
                "object": "quantile_curve",
                "probabilities": self._probabilities.tolist(),
                "values": self._values.tolist(),
            }
        )

    @classmethod
    def from_json(cls, text: str) -> QuantileCurve:
        data = _parse(text)
        _fields(data, {"schema_version", "object", "probabilities", "values"})
        if data["schema_version"] != "1.0" or data["object"] != "quantile_curve":
            raise ValueError("expected numerical quantile_curve schema 1.0")
        return cls(
            cast(ArrayLike, data["probabilities"]), cast(ArrayLike, data["values"])
        )

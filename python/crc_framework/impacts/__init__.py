"""Public impact expressions, sources, and evaluation records.

Numerical work is performed by the native expression. These wrappers preserve
array shapes, freeze returned values, and adapt supported source objects.
Arbitrary Python distributions are not inferred.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional, Union, cast

import numpy as np
import numpy.typing as npt

from crc_framework import _core
from crc_framework.constants import RiskFactor
from crc_framework.distributions import (
    Distribution,
    EmpiricalDistribution,
    FittedDistribution,
    HurdleDistribution,
    PointMassDistribution,
    TabulatedDistribution,
)
from crc_framework.models import TransformContext
from crc_framework.transforms.base import ImpactResult, ImpactValues, probability_grid

ImpactError = _core.ImpactError
Calibration = Mapping[str, str]
Clamp = Mapping[str, Optional[float]]
Provenance = Mapping[str, Any]
Snapshot = Mapping[str, Any]

_LEGACY_CALIBRATION: dict[str, str] = {
    "exposure_variable": "value",
    "exposure_unit": "1",
    "impact_metric": "response",
    "impact_unit": "1",
}


def _impact_error(code: str, path: str, message: str) -> ImpactError:
    """Build an `ImpactError` with the `code`/`path` attributes native errors
    always carry, so pure-Python raise sites match `impact_error()` in
    `src/impact.rs`.
    """
    error = ImpactError(f"{code} at {path or '/'}: {message}")
    error.code = code  # type: ignore[attr-defined]
    error.path = path  # type: ignore[attr-defined]
    return error


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _model_definition(
    model: Mapping[str, Any],
    calibration: Calibration,
    clamp: Optional[Clamp],
    provenance: Optional[Provenance],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "1.0",
        "semantics_version": "crc-impact/1",
        "calibration": dict(calibration),
        "model": dict(model),
        "provenance": dict(provenance)
        if provenance is not None
        else {"origin": {"kind": "user_defined"}},
    }
    if clamp is not None:
        document["clamp"] = dict(clamp)
    return document


def _points(points: Sequence[Union[tuple[float, float], Mapping[str, float]]]) -> list[dict[str, float]]:
    converted: list[dict[str, float]] = []
    for point in points:
        if isinstance(point, Mapping):
            converted.append({"x": float(point["x"]), "y": float(point["y"])})
        else:
            converted.append({"x": float(point[0]), "y": float(point[1])})
    return converted


def _cell(value: Union[int, str, None]) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    return f"{int(value):015x}"


def _context_fields(
    context: Union[TransformContext, Mapping[str, Any], None],
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[float]]:
    if context is None:
        return None, None, None, None, None
    if isinstance(context, TransformContext):
        return (
            _cell(context.cell),
            context.country,
            context.continent,
            context.building_type,
            context.historic_mean,
        )
    if isinstance(context, Mapping):
        historic = context.get("historic_mean")
        return (
            _cell(cast(Union[int, str, None], context.get("cell"))),
            cast(Optional[str], context.get("country")),
            cast(Optional[str], context.get("continent")),
            cast(Optional[str], context.get("building_type")),
            None if historic is None else float(historic),
        )
    raise TypeError("context must be a TransformContext or mapping")


def _method_arguments(
    method: Union[str, "GaussKronrodOptions"],
) -> tuple[str, Optional[float], Optional[float], Optional[int]]:
    if method in ("auto", "exact"):
        return str(method), None, None, None
    if isinstance(method, GaussKronrodOptions):
        return (
            "gauss_kronrod",
            method.absolute_tolerance,
            method.relative_tolerance,
            method.max_evaluations,
        )
    raise TypeError("method must be 'auto', 'exact', or GaussKronrodOptions")


def _snapshot_arguments(
    source: Optional[Snapshot],
) -> dict[str, Optional[str]]:
    if source is None:
        return {}
    return {
        "snapshot_uri": str(source["uri"]),
        "snapshot_revision": None
        if source.get("revision") is None
        else str(source["revision"]),
        "snapshot_sha256": str(source["content_sha256"]),
        "snapshot_retention": str(source["retention"]),
    }


def _finite_array(values: ImpactValues) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return array
    return np.ascontiguousarray(array)


def _shaped_result(
    values: npt.NDArray[np.float64], flat: Sequence[float]
) -> ImpactResult:
    result = np.asarray(flat, dtype=np.float64).reshape(values.shape)
    result.setflags(write=False)
    if values.ndim == 0:
        return float(result)
    return result


@dataclass(frozen=True)
class GaussKronrodOptions:
    """Tolerances for adaptive Gauss-Kronrod 15/7 on the probability interval."""

    absolute_tolerance: float = 1e-10
    relative_tolerance: float = 1e-8
    max_evaluations: int = 100_000


@dataclass(frozen=True)
class ImpactAtom:
    value: float
    probability: float


@dataclass(frozen=True)
class ImpactEvaluationResult:
    """Value and the record of how it was obtained."""

    value: Any
    record: "ImpactEvaluationRecord"


class ImpactEvaluationRecord:
    """Canonical evaluation record. It does not choose an output file."""

    def __init__(self, native: _core.NativeImpactEvaluationRecord) -> None:
        self._native = native

    def to_json(self) -> str:
        return self._native.canonical_json()

    def to_dict(self) -> Mapping[str, Any]:
        loaded = json.loads(self.to_json())
        return cast(Mapping[str, Any], _freeze(loaded))


class HistogramExposure:
    """Uniform-within-bin probability masses on strictly increasing edges."""

    def __init__(
        self,
        bin_edges: Union[Sequence[float], npt.NDArray[np.float64]],
        weights: Union[Sequence[float], npt.NDArray[np.float64]],
        *,
        within_bin: str = "uniform",
    ) -> None:
        if within_bin != "uniform":
            raise ValueError("within_bin must be 'uniform'")
        edges = np.asarray(bin_edges, dtype=np.float64).reshape(-1)
        masses = np.asarray(weights, dtype=np.float64).reshape(-1)
        self._native = _core.NativeImpactSource.histogram(edges.tolist(), masses.tolist())
        self.bin_edges = edges.copy()
        self.weights = masses.copy()
        self.bin_edges.setflags(write=False)
        self.weights.setflags(write=False)
        self.within_bin = within_bin

    @property
    def original_mass_sum(self) -> float:
        value = self._native.original_mass_sum
        if value is None:
            raise RuntimeError("histogram diagnostics are missing")
        return float(value)

    @property
    def normalization_factor(self) -> float:
        value = self._native.normalization_factor
        if value is None:
            raise RuntimeError("histogram diagnostics are missing")
        return float(value)


class DiscreteDistribution:
    """Finite support with explicit probability atoms."""

    def __init__(
        self,
        support: Sequence[float],
        probabilities: Sequence[float],
    ) -> None:
        locations = np.asarray(support, dtype=np.float64).reshape(-1)
        masses = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        self._native = _core.NativeImpactSource.discrete(
            locations.tolist(), masses.tolist()
        )
        self.support = locations.copy()
        self.probabilities = masses.copy()
        self.support.setflags(write=False)
        self.probabilities.setflags(write=False)


class ImpactDistribution:
    """Law of ``Y = f(X)``. Tabulation is an explicit approximation."""

    def __init__(self, native: _core.NativeImpactDistribution) -> None:
        self._native = native

    @property
    def definition_hash(self) -> str:
        return self._native.definition_hash

    @property
    def algorithm(self) -> str:
        return self._native.algorithm

    def atoms(self) -> tuple[ImpactAtom, ...]:
        return tuple(
            ImpactAtom(value, probability) for value, probability in self._native.atoms()
        )

    def quantile(self, probability: float) -> float:
        return float(self._native.quantile(float(probability)))

    def quantiles(
        self, probabilities: Union[Sequence[float], npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        grid = np.asarray(probabilities, dtype=np.float64)
        values = self._native.quantiles(grid.reshape(-1).tolist())
        result = np.asarray(values, dtype=np.float64).reshape(grid.shape)
        result.setflags(write=False)
        return result

    def tabulate(
        self,
        probabilities: Sequence[float],
        *,
        interpolation: str,
    ) -> TabulatedDistribution:
        grid = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        values = self._native.tabulate(grid.tolist(), interpolation)
        return TabulatedDistribution(
            grid,
            values,
            interpolation=cast(Any, interpolation),
        )


class ImpactExpression:
    """Immutable compiled impact definition."""

    def __init__(self, native: _core.NativeImpactExpression) -> None:
        self._native = native

    @classmethod
    def from_json(cls, text: str) -> "ImpactExpression":
        return cls(_core.NativeImpactExpression.from_json(text))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ImpactExpression":
        return cls.from_json(_dumps(config))

    @classmethod
    def linear(
        cls,
        slope: float,
        *,
        intercept: float = 0.0,
        calibration: Calibration,
        clamp: Optional[Clamp] = None,
        provenance: Optional[Provenance] = None,
    ) -> "ImpactExpression":
        return cls.from_config(
            _model_definition(
                {"kind": "linear", "slope": slope, "intercept": intercept},
                calibration,
                clamp,
                provenance,
            )
        )

    @classmethod
    def sigmoid(
        cls,
        *,
        midpoint: float,
        steepness: float,
        calibration: Calibration,
        offset: float = 0.0,
        amplitude: float = 1.0,
        origin_adjustment: str = "none",
        zero_below: Optional[float] = None,
        clamp: Optional[Clamp] = None,
        provenance: Optional[Provenance] = None,
    ) -> "ImpactExpression":
        return cls.from_config(
            _model_definition(
                {
                    "kind": "sigmoid",
                    "midpoint": midpoint,
                    "steepness": steepness,
                    "offset": offset,
                    "amplitude": amplitude,
                    "origin_adjustment": origin_adjustment,
                    "zero_below": zero_below,
                },
                calibration,
                clamp,
                provenance,
            )
        )

    @classmethod
    def tabulated(
        cls,
        points: Sequence[Union[tuple[float, float], Mapping[str, float]]],
        *,
        calibration: Calibration,
        interpolation: str = "linear",
        outside_domain: str = "clamp",
        clamp: Optional[Clamp] = None,
        provenance: Optional[Provenance] = None,
    ) -> "ImpactExpression":
        return cls.from_config(
            _model_definition(
                {
                    "kind": "tabulated",
                    "points": _points(points),
                    "interpolation": interpolation,
                    "outside_domain": outside_domain,
                },
                calibration,
                clamp,
                provenance,
            )
        )

    @classmethod
    def binned(
        cls,
        bin_edges: Sequence[float],
        values: Sequence[float],
        *,
        calibration: Calibration,
        outside_domain: str = "clamp",
        clamp: Optional[Clamp] = None,
        provenance: Optional[Provenance] = None,
    ) -> "ImpactExpression":
        return cls.from_config(
            _model_definition(
                {
                    "kind": "binned",
                    "bin_edges": [float(edge) for edge in bin_edges],
                    "values": [float(value) for value in values],
                    "outside_domain": outside_domain,
                },
                calibration,
                clamp,
                provenance,
            )
        )

    @classmethod
    def for_factor(
        cls,
        factor: Union[RiskFactor, str],
        *,
        context: Union[TransformContext, Mapping[str, Any], None] = None,
        overrides: Optional[Mapping[str, Optional[float]]] = None,
        fallback_policy: str = "error",
        registry_revision: Optional[str] = None,
        registry_manifest_sha256: Optional[str] = None,
    ) -> "ImpactExpression":
        registry_id, revision, resolver, manifest = _core.impact_registry_identity()
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "registry_id": registry_id,
            "registry_revision": registry_revision or revision,
            "registry_manifest_sha256": registry_manifest_sha256 or manifest,
            "resolver_semantics_version": resolver,
            "factor": factor.value if isinstance(factor, RiskFactor) else factor,
            "fallback_policy": fallback_policy,
        }
        if overrides is not None:
            payload["overrides"] = dict(overrides)
        reference = _core.NativeImpactReference.from_json(_dumps(payload))
        cell, country, continent, building_type, historic_mean = _context_fields(context)
        return cls(
            reference.resolve(cell, country, continent, building_type, historic_mean)
        )

    def to_json(self) -> str:
        return self._native.canonical_json()

    def to_config(self) -> Mapping[str, Any]:
        loaded = json.loads(self.to_json())
        return cast(Mapping[str, Any], _freeze(loaded))

    @property
    def definition_hash(self) -> str:
        return self._native.definition_hash

    @property
    def model_kind(self) -> str:
        return self._native.model_kind

    @property
    def monotonicity(self) -> str:
        return self._native.monotonicity

    def evaluate(self, values: ImpactValues) -> ImpactResult:
        return self._evaluate(values)

    def evaluate_with_record(
        self,
        values: ImpactValues,
        *,
        source: Optional[Snapshot] = None,
    ) -> ImpactEvaluationResult:
        array = _finite_array(values)
        evaluated, _flat = self._responses(array)
        record = ImpactEvaluationRecord(
            self._native.point_record(
                [int(dimension) for dimension in array.shape],
                array.reshape(-1).tolist(),
                **_snapshot_arguments(source),
            )
        )
        return ImpactEvaluationResult(evaluated, record)

    def expected_value(
        self,
        source: Any,
        *,
        method: Union[str, GaussKronrodOptions] = "auto",
    ) -> float:
        return float(self._expected(source, method).value)

    def expected_value_with_record(
        self,
        source: Any,
        *,
        method: Union[str, GaussKronrodOptions] = "auto",
        snapshot: Optional[Snapshot] = None,
    ) -> ImpactEvaluationResult:
        native_source = _as_source(source)
        result = self._expected(source, method, native_source)
        record = ImpactEvaluationRecord(
            self._native.expectation_record(
                native_source,
                result,
                **_snapshot_arguments(snapshot),
            )
        )
        return ImpactEvaluationResult(float(result.value), record)

    def transform_distribution(self, source: Any) -> ImpactDistribution:
        return ImpactDistribution(
            self._native.transform_distribution(_as_source(source))
        )

    def transform_distribution_with_record(
        self,
        source: Any,
        *,
        snapshot: Optional[Snapshot] = None,
    ) -> ImpactEvaluationResult:
        native_source = _as_source(source)
        distribution = ImpactDistribution(
            self._native.transform_distribution(native_source)
        )
        record = ImpactEvaluationRecord(
            self._native.transform_record(
                native_source, **_snapshot_arguments(snapshot)
            )
        )
        return ImpactEvaluationResult(distribution, record)

    def __call__(
        self,
        distribution: Distribution,
        *,
        probabilities: Optional[Sequence[float]] = None,
        context: Optional[TransformContext] = None,
    ) -> TabulatedDistribution:
        """Approximate quantile table. Decreasing maps reflect each probability."""
        if context is not None:
            raise TypeError("a resolved impact expression does not accept context")
        grid = probability_grid(distribution, probabilities)
        direction = self.monotonicity
        if direction in ("non_monotone", "unknown"):
            raise _impact_error(
                "monotonicity_required",
                "",
                "distribution transformation requires a certified monotone function",
            )
        query = 1.0 - grid if direction == "non_increasing" else grid
        exposure = np.asarray(distribution.quantiles(query), dtype=np.float64)
        values = np.asarray(self.evaluate(exposure), dtype=np.float64)
        return TabulatedDistribution(grid, values)

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return (self.from_json, (self.to_json(),))

    def _evaluate(self, values: ImpactValues) -> ImpactResult:
        array = _finite_array(values)
        evaluated, _flat = self._responses(array)
        return evaluated

    def _responses(
        self, array: npt.NDArray[np.float64]
    ) -> tuple[ImpactResult, list[float]]:
        flat_array = array.reshape(-1)
        if flat_array.size == 0:
            return _shaped_result(array, []), []
        if array.ndim == 0:
            value = self._native.evaluate_value(float(flat_array[0]))
            raw = [value]
        else:
            raw = self._native.evaluate_values(flat_array.tolist())
        return _shaped_result(array, raw), list(raw)

    def _expected(
        self,
        source: Any,
        method: Union[str, GaussKronrodOptions],
        native_source: Optional[_core.NativeImpactSource] = None,
    ) -> _core.NativeExpectation:
        requested, absolute, relative, budget = _method_arguments(method)
        return self._native.expected_value(
            native_source if native_source is not None else _as_source(source),
            requested,
            absolute,
            relative,
            budget,
        )


def _as_source(source: Any) -> _core.NativeImpactSource:
    if isinstance(source, (HistogramExposure, DiscreteDistribution)):
        return source._native
    if isinstance(source, PointMassDistribution):
        return _core.NativeImpactSource.curve(
            "point_mass",
            "point_mass",
            curve_location=source.location,
            curve_scale=0.0,
            curve_atom_probability=1.0,
            curve_atom_location=source.location,
        )
    if isinstance(source, FittedDistribution):
        return _core.NativeImpactSource.curve(
            "fitted",
            source.family,
            curve_shape=source.shape,
            curve_location=source.location,
            curve_scale=source.scale,
        )
    if isinstance(source, HurdleDistribution):
        return _core.NativeImpactSource.curve(
            "hurdle",
            source.base.family,
            curve_shape=source.base.shape,
            curve_location=source.base.location,
            curve_scale=source.base.scale,
            curve_atom_probability=source.atom_probability,
            curve_atom_location=source.atom_location,
        )
    if isinstance(source, TabulatedDistribution):
        if source.interpolation != "linear_probability":
            raise TypeError(
                "only linear-probability tabulated curves are impact sources; "
                "evaluate requested quantiles explicitly"
            )
        return _core.NativeImpactSource.curve(
            "tabulated",
            "linear_probability",
            curve_probabilities=source.probabilities.tolist(),
            curve_values=source.values.tolist(),
        )
    if isinstance(source, EmpiricalDistribution):
        return _core.NativeImpactSource.empirical(source.samples.tolist())
    raise TypeError(
        "unsupported impact source; pass a histogram, discrete distribution, "
        "or canonical CRC distribution"
    )


def legacy_linear_expression(
    slope: float,
    intercept: float,
    minimum: Optional[float],
    maximum: Optional[float],
) -> ImpactExpression:
    """Private adapter. The calibration is not the caller's persisted metadata."""
    return ImpactExpression.linear(
        slope,
        intercept=intercept,
        calibration=_LEGACY_CALIBRATION,
        clamp={"minimum": minimum, "maximum": maximum},
    )


def legacy_sigmoid_expression(
    midpoint: float,
    steepness: float,
    minimum: float,
    maximum: float,
    zero_below: Optional[float],
) -> ImpactExpression:
    return ImpactExpression.sigmoid(
        midpoint=midpoint,
        steepness=steepness,
        offset=minimum,
        amplitude=maximum - minimum,
        zero_below=zero_below,
        calibration=_LEGACY_CALIBRATION,
    )


def legacy_piecewise_expression(
    exposure: Sequence[float], impact: Sequence[float]
) -> ImpactExpression:
    return ImpactExpression.tabulated(
        list(zip(exposure, impact)),
        calibration=_LEGACY_CALIBRATION,
        interpolation="linear",
        outside_domain="clamp",
    )

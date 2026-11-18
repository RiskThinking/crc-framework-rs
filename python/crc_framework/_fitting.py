"""Explicit scientific fitting specifications and authoritative fit records."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, fields
from enum import Enum
from typing import Generic, TypeAlias, TypedDict, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt

from . import _core
from ._distribution import ArrayLike, Distribution, DistributionFamily, QuantileCurve


class SampleEstimator(str, Enum):
    MAXIMUM_LIKELIHOOD = "mle"
    L_MOMENTS = "lmoments"


class SelectionCriterion(str, Enum):
    KOLMOGOROV_SMIRNOV = "ks"
    NORMALIZED_ROOT_MEAN_SQUARE_ERROR = "normalized_rmse"


class FitKind(str, Enum):
    SAMPLES = "samples"
    QUANTILES = "quantiles"
    CENSORED_QUANTILES = "censored_quantiles"
    HURDLE_QUANTILES = "hurdle_quantiles"
    MODEL_SELECTION = "model_selection"


class FitStatus(str, Enum):
    CONVERGED = "converged"
    REJECTED = "rejected"
    FAILED = "failed"


class FitSpec(ABC):
    @property
    @abstractmethod
    def kind(self) -> FitKind: ...

    @abstractmethod
    def to_json(self) -> str: ...

    @classmethod
    def from_json(cls, text: str) -> FitSpec:
        """Decode a current discriminated scientific specification."""
        result = _decode_spec(_load_json(text))
        if cls is not FitSpec and not isinstance(result, cls):
            raise ValueError(f"expected a {cls.__name__} specification")
        return result


def _enum(value: object, expected: type[Enum], name: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be {expected.__name__}")


def _finite(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")


def _minimum(value: int, name: str, floor: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < floor:
        raise ValueError(f"{name} must be an integer of at least {floor}")


@dataclass(frozen=True, kw_only=True)
class FitConstraints:
    """Bound a quantile, or the full support when probability is absent."""

    minimum_value: float | None = None
    maximum_value: float | None = None
    probability: float | None = None
    require_sample_support: bool = False

    def __post_init__(self) -> None:
        for name in ("minimum_value", "maximum_value", "probability"):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)
        if self.probability is not None and not 0 <= self.probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if (
            self.minimum_value is not None
            and self.maximum_value is not None
            and self.minimum_value > self.maximum_value
        ):
            raise ValueError("minimum_value must not exceed maximum_value")
        if not isinstance(self.require_sample_support, bool):
            raise TypeError("require_sample_support must be bool")


def _constraints(value: FitConstraints | None) -> None:
    if value is not None and not isinstance(value, FitConstraints):
        raise TypeError("constraints must be FitConstraints or None")


def _serialize(value: object) -> object:
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(
        value,
        (
            SampleFit,
            QuantileFit,
            CensoredQuantileFit,
            HurdleQuantileFit,
            ModelSelection,
        ),
    ):
        return {
            "kind": value.kind.value,
            **{
                field.name: _serialize(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, FitConstraints):
        return {
            field.name: _serialize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    return value


def _json(specification: FitSpec) -> str:
    return json.dumps(
        _serialize(specification),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True, kw_only=True)
class SampleFit(FitSpec):
    family: DistributionFamily
    estimator: SampleEstimator = SampleEstimator.MAXIMUM_LIKELIHOOD
    minimum_samples: int = 3
    constraints: FitConstraints | None = None
    fallbacks: tuple[SampleFit, ...] = ()

    def __post_init__(self) -> None:
        _enum(self.family, DistributionFamily, "family")
        _enum(self.estimator, SampleEstimator, "estimator")
        _minimum(self.minimum_samples, "minimum_samples", 3)
        _constraints(self.constraints)
        if not isinstance(self.fallbacks, tuple) or any(
            not isinstance(item, SampleFit) for item in self.fallbacks
        ):
            raise TypeError("fallbacks must be a tuple of SampleFit specifications")
        if self.estimator is SampleEstimator.L_MOMENTS and self.family not in (
            DistributionFamily.GENERALIZED_EXTREME_VALUE,
            DistributionFamily.GUMBEL_RIGHT,
            DistributionFamily.GUMBEL_LEFT,
        ):
            raise ValueError("L-moments supports GEV, Gumbel right, and Gumbel left")

    @property
    def kind(self) -> FitKind:
        return FitKind.SAMPLES

    def to_json(self) -> str:
        return _json(self)


@dataclass(frozen=True, kw_only=True)
class QuantileFit(FitSpec):
    family: DistributionFamily
    weights: tuple[float, ...] | None = None
    minimum_knots: int = 4
    constraints: FitConstraints | None = None

    def __post_init__(self) -> None:
        _enum(self.family, DistributionFamily, "family")
        _minimum(self.minimum_knots, "minimum_knots", 4)
        _constraints(self.constraints)
        if self.weights is not None:
            if not isinstance(self.weights, tuple):
                raise TypeError("weights must be a tuple or None")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in self.weights
            ):
                raise ValueError("weights must be finite and nonnegative")
            if not self.weights or sum(self.weights) <= 0:
                raise ValueError("at least one weight must be positive")

    @property
    def kind(self) -> FitKind:
        return FitKind.QUANTILES

    def to_json(self) -> str:
        return _json(self)


@dataclass(frozen=True, kw_only=True)
class CensoredQuantileFit(FitSpec):
    family: DistributionFamily
    censor_location: float
    minimum_knots: int = 4
    residual_scale_floor_ratio: float = 0.001
    constraints: FitConstraints | None = None

    def __post_init__(self) -> None:
        _enum(self.family, DistributionFamily, "family")
        _finite(self.censor_location, "censor_location")
        _finite(self.residual_scale_floor_ratio, "residual_scale_floor_ratio")
        if self.residual_scale_floor_ratio <= 0:
            raise ValueError("residual_scale_floor_ratio must be positive")
        _minimum(self.minimum_knots, "minimum_knots", 4)
        _constraints(self.constraints)

    @property
    def kind(self) -> FitKind:
        return FitKind.CENSORED_QUANTILES

    def to_json(self) -> str:
        return _json(self)


@dataclass(frozen=True, kw_only=True)
class HurdleQuantileFit(FitSpec):
    family: DistributionFamily
    atom_probability: float
    atom_location: float = 0.0
    minimum_knots: int = 4
    constraints: FitConstraints | None = None

    def __post_init__(self) -> None:
        _enum(self.family, DistributionFamily, "family")
        _finite(self.atom_location, "atom_location")
        _finite(self.atom_probability, "atom_probability")
        if not 0 < self.atom_probability < 1:
            raise ValueError("atom_probability must be strictly between zero and one")
        _minimum(self.minimum_knots, "minimum_knots", 4)
        _constraints(self.constraints)

    @property
    def kind(self) -> FitKind:
        return FitKind.HURDLE_QUANTILES

    def to_json(self) -> str:
        return _json(self)


@dataclass(frozen=True, kw_only=True)
class ModelSelection(FitSpec):
    candidates: tuple[FitSpec, ...]
    criterion: SelectionCriterion = SelectionCriterion.KOLMOGOROV_SMIRNOV

    def __post_init__(self) -> None:
        _enum(self.criterion, SelectionCriterion, "criterion")
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ValueError(
                "candidates must be a nonempty tuple of explicit fitting specifications"
            )
        if any(
            not isinstance(
                item, (SampleFit, QuantileFit, CensoredQuantileFit, HurdleQuantileFit)
            )
            for item in self.candidates
        ):
            raise TypeError("candidates must be concrete fitting specifications")
        sample_candidates = all(isinstance(item, SampleFit) for item in self.candidates)
        quantile_candidates = all(
            isinstance(item, (QuantileFit, CensoredQuantileFit, HurdleQuantileFit))
            for item in self.candidates
        )
        if not sample_candidates and not quantile_candidates:
            raise TypeError(
                "model selection cannot mix sample and quantile specifications"
            )
        if (
            sample_candidates
            and self.criterion is not SelectionCriterion.KOLMOGOROV_SMIRNOV
        ):
            raise ValueError(
                "sample model selection requires criterion=SelectionCriterion.KOLMOGOROV_SMIRNOV"
            )
        if (
            quantile_candidates
            and self.criterion
            is not SelectionCriterion.NORMALIZED_ROOT_MEAN_SQUARE_ERROR
        ):
            raise ValueError(
                "quantile model selection requires criterion=SelectionCriterion.NORMALIZED_ROOT_MEAN_SQUARE_ERROR"
            )
        identities = [item.to_json() for item in self.candidates]
        if len(set(identities)) != len(identities):
            raise ValueError("candidate specifications must be distinct")

    @property
    def kind(self) -> FitKind:
        return FitKind.MODEL_SELECTION

    def to_json(self) -> str:
        return _json(self)


@dataclass(frozen=True, slots=True)
class _FitConvergence:
    iterations: int
    evaluations: int

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value):
                raise ValueError(f"fitting diagnostic {field.name} must be finite")


@dataclass(frozen=True, slots=True)
class SampleFitDiagnostic(_FitConvergence):
    ks_statistic: float
    ks_pvalue: float
    rmse: float
    r_squared: float


@dataclass(frozen=True, slots=True)
class _QuantileComparison(_FitConvergence):
    normalized_rmse: float
    normalization_scale: float


@dataclass(frozen=True, slots=True)
class _QuantileResiduals(_QuantileComparison):
    rmse: float
    weighted_r_squared: float
    maximum_absolute_residual: float


@dataclass(frozen=True, slots=True)
class QuantileFitDiagnostic(_QuantileResiduals):
    pass


@dataclass(frozen=True, slots=True)
class CensoredQuantileFitDiagnostic(_QuantileResiduals):
    negative_log_likelihood: float
    residual_scale: float
    residual_scale_floor: float
    censored_count: int
    uncensored_count: int


@dataclass(frozen=True, slots=True)
class HurdleQuantileFitDiagnostic(_QuantileComparison):
    tail_rmse: float
    tail_weighted_r_squared: float
    tail_maximum_absolute_residual: float
    atom_count: int
    tail_count: int


@dataclass(frozen=True, slots=True)
class FitFailureDiagnostic:
    message: str


ConvergedFitDiagnostic: TypeAlias = (
    SampleFitDiagnostic
    | QuantileFitDiagnostic
    | CensoredQuantileFitDiagnostic
    | HurdleQuantileFitDiagnostic
)
DiagnosticT_co = TypeVar("DiagnosticT_co", bound=ConvergedFitDiagnostic, covariant=True)


@dataclass(frozen=True, slots=True)
class FitAttempt:
    specification: FitSpec
    status: FitStatus
    diagnostic: ConvergedFitDiagnostic | FitFailureDiagnostic


@dataclass(frozen=True)
class FitRecord:
    specification: FitSpec
    selected_attempt: int | None
    attempts: tuple[FitAttempt, ...]
    estimator: SampleEstimator | None
    converged: bool
    evidence_count: int
    used_count: int
    excluded_count: int
    weights: tuple[float, ...] | None
    evidence_support: tuple[float, float] | None
    engine_version: str

    def to_json(self) -> str:
        return json.dumps(
            _record_document(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, text: str) -> FitRecord:
        """Validate the shared fitting contract and restore its typed record."""
        _core.validate_document(
            "https://climateriskcommons.org/schemas/fitting/1.0", text
        )
        return _decode_record(json.loads(text))


@dataclass(frozen=True)
class FitResult(Generic[DiagnosticT_co]):
    distribution: Distribution
    record: FitRecord

    @property
    def diagnostic(self) -> DiagnosticT_co:
        selected = self.record.selected_attempt
        assert selected is not None
        diagnostic = self.record.attempts[selected].diagnostic
        assert not isinstance(diagnostic, FitFailureDiagnostic)
        return cast(DiagnosticT_co, diagnostic)


class FitError(ValueError):
    """A requested fit failed, retaining its execution record."""

    record: FitRecord
    code: str
    message: str

    def __init__(self, message: str, *, record: FitRecord) -> None:
        super().__init__(message)
        self.record = record
        self.code = "fit.no_converged_candidate"
        self.message = message


class _NativeBounds(TypedDict):
    minimum_value: float | None
    maximum_value: float | None
    probability: float | None
    require_sample_support: bool


def _record(
    specification: FitSpec,
    attempts: list[FitAttempt],
    estimator: SampleEstimator | None,
    count: int,
    used: int,
    weights: tuple[float, ...] | None,
    support: tuple[float, float] | None,
    converged: bool,
    selected_attempt: int | None = None,
) -> FitRecord:
    return FitRecord(
        specification=specification,
        selected_attempt=selected_attempt,
        attempts=tuple(attempts),
        estimator=estimator,
        converged=converged,
        evidence_count=count,
        used_count=used,
        excluded_count=count - used,
        weights=weights,
        evidence_support=support,
        engine_version=f"crc-framework/{_core.__version__}",
    )


def _label(specification: FitSpec) -> str:
    if isinstance(
        specification, (SampleFit, QuantileFit, CensoredQuantileFit, HurdleQuantileFit)
    ):
        return f"{specification.kind.value}/{specification.family.value}"
    return specification.kind.value


def _normalized_weights(weights: npt.NDArray[np.float64]) -> tuple[float, ...]:
    scaled = weights / float(np.max(weights))
    normalized = tuple(float(value) for value in scaled / float(np.sum(scaled)))
    if any(
        weight == 0 and original > 0 for weight, original in zip(normalized, weights)
    ):
        raise ValueError(
            "relative weights are too small to represent; positive weights cannot be silently dropped"
        )
    return normalized


def _failure(
    specification: FitSpec,
    attempts: list[FitAttempt],
    count: int,
    used: int,
    weights: tuple[float, ...] | None,
    support: tuple[float, float] | None,
) -> FitError:
    failures = "; ".join(
        f"{_label(attempt.specification)}: {attempt.diagnostic.message}"
        for attempt in attempts
        if isinstance(attempt.diagnostic, FitFailureDiagnostic)
    )
    message = f"{specification.kind.value} fitting failed: {failures or 'no requested candidate converged'}"
    return FitError(
        message,
        record=_record(
            specification, attempts, None, count, used, weights, support, False
        ),
    )


def fit_samples(
    samples: ArrayLike, specification: SampleFit | ModelSelection
) -> FitResult[SampleFitDiagnostic]:
    if not isinstance(specification, (SampleFit, ModelSelection)):
        raise TypeError("sample evidence requires SampleFit or ModelSelection")
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("samples must be a one-dimensional observation sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "samples must be finite; missing observations must be handled explicitly"
        )
    candidates = (
        specification.candidates
        if isinstance(specification, ModelSelection)
        else (specification,)
    )
    if any(not isinstance(candidate, SampleFit) for candidate in candidates):
        raise TypeError("sample model selection requires SampleFit candidates")
    attempts: list[FitAttempt] = []
    successes: list[tuple[float, Distribution, SampleFit, int]] = []
    support = (float(np.min(values)), float(np.max(values))) if values.size else None
    used_count = 0

    def run(
        candidate: SampleFit,
    ) -> tuple[float, Distribution, SampleFit, int] | None:
        nonlocal used_count
        try:
            if len(values) < candidate.minimum_samples:
                raise ValueError(
                    f"requires at least {candidate.minimum_samples} observations"
                )
            bounds = candidate.constraints or FitConstraints()
            used_count = len(values)
            native = _core.fit_samples(
                values.tolist(),
                candidate.family.value,
                candidate.estimator.value,
                minimum_value=bounds.minimum_value,
                maximum_value=bounds.maximum_value,
                probability=bounds.probability,
                require_sample_support=bounds.require_sample_support,
            )
            distribution = Distribution._from_native(native.distribution)
            diagnostic = SampleFitDiagnostic(
                iterations=native.iterations,
                evaluations=native.evaluations,
                ks_statistic=native.ks_statistic,
                ks_pvalue=native.ks_pvalue,
                rmse=native.rmse,
                r_squared=native.r_squared,
            )
            attempts.append(FitAttempt(candidate, FitStatus.CONVERGED, diagnostic))
            return (
                native.ks_statistic,
                distribution,
                candidate,
                len(attempts) - 1,
            )
        except ValueError as error:
            status = (
                FitStatus.REJECTED
                if "does not satisfy constraints" in str(error)
                else FitStatus.FAILED
            )
            attempts.append(
                FitAttempt(candidate, status, FitFailureDiagnostic(str(error)))
            )
            for fallback in candidate.fallbacks:
                result = run(fallback)
                if result is not None:
                    return result
            return None

    for candidate in candidates:
        assert isinstance(candidate, SampleFit)
        result = run(candidate)
        if result is not None:
            successes.append(result)
    if not successes:
        raise _failure(specification, attempts, len(values), used_count, None, support)
    _, distribution, chosen, selected = min(successes, key=lambda result: result[0])
    return FitResult(
        distribution,
        _record(
            specification,
            attempts,
            chosen.estimator,
            len(values),
            len(values),
            None,
            support,
            True,
            selected,
        ),
    )


def _quantile_diagnostic(
    native: _core.NativeQuantileFitResult
    | _core.NativeCensoredQuantileFitResult
    | _core.NativeHurdleQuantileFitResult,
    score: float,
    span: float,
) -> (
    QuantileFitDiagnostic | CensoredQuantileFitDiagnostic | HurdleQuantileFitDiagnostic
):
    if isinstance(native, _core.NativeHurdleQuantileFitResult):
        return HurdleQuantileFitDiagnostic(
            iterations=native.iterations,
            evaluations=native.evaluations,
            normalized_rmse=score,
            normalization_scale=span,
            tail_rmse=native.tail_rmse,
            tail_weighted_r_squared=native.tail_weighted_r_squared,
            tail_maximum_absolute_residual=native.tail_maximum_absolute_residual,
            atom_count=native.atom_point_count,
            tail_count=native.tail_point_count,
        )
    if isinstance(native, _core.NativeCensoredQuantileFitResult):
        return CensoredQuantileFitDiagnostic(
            iterations=native.iterations,
            evaluations=native.evaluations,
            normalized_rmse=score,
            normalization_scale=span,
            rmse=native.rmse,
            weighted_r_squared=native.weighted_r_squared,
            maximum_absolute_residual=native.maximum_absolute_residual,
            negative_log_likelihood=native.negative_log_likelihood,
            residual_scale=native.residual_scale,
            residual_scale_floor=native.residual_scale_floor,
            censored_count=native.censored_point_count,
            uncensored_count=native.uncensored_point_count,
        )
    return QuantileFitDiagnostic(
        iterations=native.iterations,
        evaluations=native.evaluations,
        normalized_rmse=score,
        normalization_scale=span,
        rmse=native.rmse,
        weighted_r_squared=native.weighted_r_squared,
        maximum_absolute_residual=native.maximum_absolute_residual,
    )


@overload
def fit_quantiles(
    curve: QuantileCurve, specification: QuantileFit
) -> FitResult[QuantileFitDiagnostic]: ...


@overload
def fit_quantiles(
    curve: QuantileCurve, specification: CensoredQuantileFit
) -> FitResult[CensoredQuantileFitDiagnostic]: ...


@overload
def fit_quantiles(
    curve: QuantileCurve, specification: HurdleQuantileFit
) -> FitResult[HurdleQuantileFitDiagnostic]: ...


@overload
def fit_quantiles(
    curve: QuantileCurve, specification: ModelSelection
) -> FitResult[
    QuantileFitDiagnostic | CensoredQuantileFitDiagnostic | HurdleQuantileFitDiagnostic
]: ...


def fit_quantiles(
    curve: QuantileCurve,
    specification: QuantileFit
    | CensoredQuantileFit
    | HurdleQuantileFit
    | ModelSelection,
) -> FitResult[
    QuantileFitDiagnostic | CensoredQuantileFitDiagnostic | HurdleQuantileFitDiagnostic
]:
    if not isinstance(curve, QuantileCurve):
        raise TypeError("curve must be QuantileCurve; observations use fit_samples")
    if not isinstance(
        specification,
        (QuantileFit, CensoredQuantileFit, HurdleQuantileFit, ModelSelection),
    ):
        raise TypeError("quantile evidence requires a quantile fitting specification")
    candidates = (
        specification.candidates
        if isinstance(specification, ModelSelection)
        else (specification,)
    )
    if any(
        not isinstance(item, (QuantileFit, CensoredQuantileFit, HurdleQuantileFit))
        for item in candidates
    ):
        raise TypeError("quantile model selection requires quantile fitting candidates")
    probabilities, values = curve.probabilities, curve.values
    count = len(values)
    support = curve.probability_interval
    attempts: list[FitAttempt] = []
    successes: list[tuple[float, Distribution, tuple[float, ...], int, int]] = []
    last_used = 0
    last_weights: tuple[float, ...] | None = None
    for candidate in candidates:
        assert isinstance(
            candidate, (QuantileFit, CensoredQuantileFit, HurdleQuantileFit)
        )
        try:
            weights = np.asarray(
                candidate.weights
                if isinstance(candidate, QuantileFit) and candidate.weights is not None
                else np.ones(count),
                dtype=np.float64,
            )
            if weights.shape != values.shape:
                raise ValueError("weights must have one entry per supplied knot")
            active = weights > 0
            if np.any(((probabilities <= 0) | (probabilities >= 1)) & active):
                raise ValueError(
                    "parametric fitting requires interior probabilities; endpoint evidence cannot be silently excluded"
                )
            used = int(np.count_nonzero(weights > 0))
            if used < candidate.minimum_knots:
                raise ValueError(
                    f"requires at least {candidate.minimum_knots} positive-weight knots"
                )
            normalized = _normalized_weights(weights)
            last_used, last_weights = used, normalized
            fit_probabilities, fit_values = probabilities[active], values[active]
            fit_weights = np.asarray(normalized, dtype=np.float64)[active]
            bounds = candidate.constraints or FitConstraints()
            options: _NativeBounds = {
                "minimum_value": bounds.minimum_value,
                "maximum_value": bounds.maximum_value,
                "probability": bounds.probability,
                "require_sample_support": bounds.require_sample_support,
            }
            native: (
                _core.NativeQuantileFitResult
                | _core.NativeCensoredQuantileFitResult
                | _core.NativeHurdleQuantileFitResult
            )
            if isinstance(candidate, CensoredQuantileFit):
                native = _core.fit_censored_quantiles(
                    fit_probabilities.tolist(),
                    fit_values.tolist(),
                    candidate.family.value,
                    censor_location=candidate.censor_location,
                    residual_scale_floor_ratio=candidate.residual_scale_floor_ratio,
                    weights=fit_weights.tolist(),
                    **options,
                )
                distribution = Distribution._from_native(native.distribution)
            elif isinstance(candidate, HurdleQuantileFit):
                native = _core.fit_hurdle_quantiles(
                    fit_probabilities.tolist(),
                    fit_values.tolist(),
                    candidate.family.value,
                    atom_probability=candidate.atom_probability,
                    atom_location=candidate.atom_location,
                    weights=fit_weights.tolist(),
                    **options,
                )
                distribution = Distribution.hurdle(
                    Distribution._from_native(native.base_distribution),
                    atom_probability=candidate.atom_probability,
                    atom_location=candidate.atom_location,
                )
            else:
                native = _core.fit_quantiles(
                    fit_probabilities.tolist(),
                    fit_values.tolist(),
                    candidate.family.value,
                    weights=fit_weights.tolist(),
                    **options,
                )
                distribution = Distribution._from_native(native.distribution)
            predicted = np.asarray(
                distribution.quantile(fit_probabilities), dtype=np.float64
            )
            observed = fit_values
            if isinstance(candidate, CensoredQuantileFit):
                observed = np.maximum(fit_values, candidate.censor_location)
                predicted = np.maximum(predicted, candidate.censor_location)
            effective = fit_weights
            observed_used = observed[effective > 0]
            value_span = float(np.max(observed_used) - np.min(observed_used))
            if value_span <= 0:
                raise ValueError(
                    "normalized RMSE requires nonconstant effective evidence"
                )
            normalized_residuals = (predicted - observed) / value_span
            score = float(np.sqrt(np.sum(effective * normalized_residuals**2)))
            diagnostic = _quantile_diagnostic(native, score, value_span)
            attempts.append(FitAttempt(candidate, FitStatus.CONVERGED, diagnostic))
            successes.append((score, distribution, normalized, used, len(attempts) - 1))
        except ValueError as error:
            attempts.append(
                FitAttempt(
                    candidate, FitStatus.FAILED, FitFailureDiagnostic(str(error))
                )
            )
    if not successes:
        raise _failure(specification, attempts, count, last_used, last_weights, support)
    _, distribution, normalized, used, selected = min(
        successes, key=lambda result: result[0]
    )
    return FitResult(
        distribution,
        _record(
            specification,
            attempts,
            None,
            count,
            used,
            normalized,
            support,
            True,
            selected,
        ),
    )


def _load_json(text: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate fitting field {key!r}")
            result[key] = value
        return result

    value: object = json.loads(text, object_pairs_hook=pairs)
    # Also catches overflowing JSON numbers (for example 1e999), not just NaN.
    json.dumps(value, allow_nan=False)
    return value


def _document(
    value: object, allowed: set[str], required: set[str]
) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError("fitting records must be objects")
    if set(value) - allowed or not required <= set(value):
        raise ValueError(f"invalid fitting fields: expected {sorted(allowed)}")
    return cast(dict[str, object], value)


def _text(value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("expected an exact nonempty string")
    return value


def _count(value: object) -> int:
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        value = int(value)
    if type(value) is not int or value < 0:
        raise ValueError("evidence counts must be nonnegative integers")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("expected a finite number")
    return float(value)


def _items(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("expected a JSON array")
    return value


def _decode_constraints(value: object) -> FitConstraints | None:
    if value is None:
        return None
    fields = {"minimum_value", "maximum_value", "probability", "require_sample_support"}
    data = _document(value, fields, set())
    support = data.get("require_sample_support", False)
    if type(support) is not bool:
        raise ValueError("require_sample_support must be boolean")
    return FitConstraints(
        minimum_value=_optional_float(data.get("minimum_value")),
        maximum_value=_optional_float(data.get("maximum_value")),
        probability=_optional_float(data.get("probability")),
        require_sample_support=support,
    )


def _decode_spec(value: object) -> FitSpec:
    if not isinstance(value, dict):
        raise TypeError("fitting specification must be an object")
    kind = value.get("kind")
    if kind == FitKind.MODEL_SELECTION.value:
        data = _document(
            value, {"kind", "candidates", "criterion"}, {"kind", "candidates"}
        )
        return ModelSelection(
            candidates=tuple(_decode_spec(v) for v in _items(data["candidates"])),
            criterion=SelectionCriterion(_text(data.get("criterion", "ks"))),
        )
    common = {"kind", "family", "constraints"}
    if kind == FitKind.SAMPLES.value:
        fields = common | {"estimator", "minimum_samples", "fallbacks"}
    elif kind == FitKind.QUANTILES.value:
        fields = common | {"weights", "minimum_knots"}
    elif kind == FitKind.CENSORED_QUANTILES.value:
        fields = common | {
            "censor_location",
            "minimum_knots",
            "residual_scale_floor_ratio",
        }
    elif kind == FitKind.HURDLE_QUANTILES.value:
        fields = common | {"atom_probability", "atom_location", "minimum_knots"}
    else:
        raise ValueError("unknown fitting specification kind")
    data = _document(value, fields, {"kind", "family"})
    family = DistributionFamily(_text(data["family"]))
    constraints = _decode_constraints(data.get("constraints"))
    if kind == FitKind.SAMPLES.value:
        fallbacks = tuple(_decode_spec(v) for v in _items(data.get("fallbacks", [])))
        if any(not isinstance(v, SampleFit) for v in fallbacks):
            raise ValueError("sample alternatives must be SampleFit specifications")
        return SampleFit(
            family=family,
            estimator=SampleEstimator(_text(data.get("estimator", "mle"))),
            minimum_samples=_count(data.get("minimum_samples", 3)),
            constraints=constraints,
            fallbacks=cast(tuple[SampleFit, ...], fallbacks),
        )
    minimum = _count(data.get("minimum_knots", 4))
    if kind == FitKind.QUANTILES.value:
        weights = data.get("weights")
        if weights is not None:
            converted = tuple(_optional_float(v) for v in _items(weights))
            if None in converted:
                raise ValueError("weights cannot contain null")
            weights = converted
        return QuantileFit(
            family=family,
            weights=cast(tuple[float, ...] | None, weights),
            minimum_knots=minimum,
            constraints=constraints,
        )
    if kind == FitKind.CENSORED_QUANTILES.value:
        location = _optional_float(data.get("censor_location"))
        floor = _optional_float(data.get("residual_scale_floor_ratio", 0.001))
        if location is None or floor is None:
            raise ValueError(
                "censored fits require censor_location and a finite residual scale floor"
            )
        return CensoredQuantileFit(
            family=family,
            censor_location=location,
            minimum_knots=minimum,
            residual_scale_floor_ratio=floor,
            constraints=constraints,
        )
    probability = _optional_float(data.get("atom_probability"))
    location = _optional_float(data.get("atom_location", 0))
    if probability is None or location is None:
        raise ValueError("hurdle fits require atom_probability and atom_location")
    return HurdleQuantileFit(
        family=family,
        atom_probability=probability,
        atom_location=location,
        minimum_knots=minimum,
        constraints=constraints,
    )


_METRICS = {
    FitKind.SAMPLES: SampleFitDiagnostic,
    FitKind.QUANTILES: QuantileFitDiagnostic,
    FitKind.CENSORED_QUANTILES: CensoredQuantileFitDiagnostic,
    FitKind.HURDLE_QUANTILES: HurdleQuantileFitDiagnostic,
}


def _decode_attempt(value: object) -> FitAttempt:
    # FitRecord.from_json has already validated the attempt's wire variant.
    data = cast(dict[str, object], value)
    specification = _decode_spec(data["specification"])
    status = FitStatus(_text(data["status"]))
    outcome = cast(dict[str, object], data["diagnostic"])
    diagnostic: ConvergedFitDiagnostic | FitFailureDiagnostic
    if status is FitStatus.CONVERGED:
        metric_type = _METRICS[specification.kind]
        build: Callable[..., ConvergedFitDiagnostic] = metric_type
        diagnostic = build(
            **{
                field.name: _count(outcome[field.name])
                if field.type == "int"
                else float(cast(float, outcome[field.name]))
                for field in fields(metric_type)
            }
        )
    else:
        diagnostic = FitFailureDiagnostic(cast(str, outcome["message"]))
    return FitAttempt(specification, status, diagnostic)


def _record_document(record: FitRecord) -> dict[str, object]:
    return {
        "specification": _serialize(record.specification),
        "attempts": [
            {
                "specification": _serialize(a.specification),
                "status": a.status.value,
                "diagnostic": {
                    field.name: getattr(a.diagnostic, field.name)
                    for field in fields(a.diagnostic)
                },
            }
            for a in record.attempts
        ],
        "selected_attempt": record.selected_attempt,
        "estimator": None if record.estimator is None else record.estimator.value,
        "converged": record.converged,
        "evidence_count": record.evidence_count,
        "used_count": record.used_count,
        "excluded_count": record.excluded_count,
        "weights": None if record.weights is None else list(record.weights),
        "evidence_support": None
        if record.evidence_support is None
        else list(record.evidence_support),
        "engine_version": record.engine_version,
    }


def _decode_record(value: object) -> FitRecord:
    fields = {
        "specification",
        "attempts",
        "selected_attempt",
        "estimator",
        "converged",
        "evidence_count",
        "used_count",
        "excluded_count",
        "weights",
        "evidence_support",
        "engine_version",
    }
    data = _document(value, fields, fields)
    support = cast(list[float] | None, data["evidence_support"])
    return FitRecord(
        specification=_decode_spec(data["specification"]),
        attempts=tuple(_decode_attempt(item) for item in _items(data["attempts"])),
        selected_attempt=None
        if data["selected_attempt"] is None
        else _count(data["selected_attempt"]),
        estimator=None
        if data["estimator"] is None
        else SampleEstimator(_text(data["estimator"])),
        converged=cast(bool, data["converged"]),
        evidence_count=_count(data["evidence_count"]),
        used_count=_count(data["used_count"]),
        excluded_count=_count(data["excluded_count"]),
        weights=None
        if data["weights"] is None
        else tuple(float(weight) for weight in cast(list[float], data["weights"])),
        evidence_support=None
        if support is None
        else (float(support[0]), float(support[1])),
        engine_version=_text(data["engine_version"]),
    )

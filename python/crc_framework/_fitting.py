"""Explicit scientific fitting specifications and authoritative fit records."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from enum import Enum
from typing import Protocol, TypedDict

import numpy as np

from . import _core
from ._diagnostics import Diagnostic, DiagnosticSeverity, ScientificValue
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


class FitSpec(Protocol):
    @property
    def kind(self) -> FitKind: ...

    def to_json(self) -> str: ...


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
class SampleFit:
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
class QuantileFit:
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
class CensoredQuantileFit:
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
class HurdleQuantileFit:
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
class ModelSelection:
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


@dataclass(frozen=True)
class FitAttempt:
    specification: FitSpec
    status: FitStatus
    reason: str | None
    diagnostics: tuple[Diagnostic, ...]


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


@dataclass(frozen=True)
class FitResult:
    distribution: Distribution
    diagnostics: tuple[Diagnostic, ...]
    record: FitRecord


class FitError(ValueError):
    """A requested fit failed, retaining its execution record."""

    record: FitRecord
    diagnostics: tuple[Diagnostic, ...]
    code: str
    message: str

    def __init__(
        self, message: str, *, record: FitRecord, diagnostics: tuple[Diagnostic, ...]
    ) -> None:
        super().__init__(message)
        self.record = record
        self.diagnostics = diagnostics
        self.code = "fit.no_converged_candidate"
        self.message = message


def _diagnostic(
    code: str, message: str, details: dict[str, ScientificValue], *, error: bool = False
) -> Diagnostic:
    return Diagnostic(
        code=code,
        message=message,
        severity=DiagnosticSeverity.ERROR if error else DiagnosticSeverity.INFO,
        path="",
        details=details,
    )


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


def _failure(
    specification: FitSpec,
    attempts: list[FitAttempt],
    count: int,
    used: int,
    weights: tuple[float, ...] | None,
    support: tuple[float, float] | None,
) -> FitError:
    diagnostics = tuple(
        diagnostic for attempt in attempts for diagnostic in attempt.diagnostics
    )
    failures = "; ".join(
        f"{_label(attempt.specification)}: {attempt.reason}"
        for attempt in attempts
        if attempt.reason is not None
    )
    message = f"{specification.kind.value} fitting failed: {failures or 'no requested candidate converged'}"
    return FitError(
        message,
        record=_record(
            specification, attempts, None, count, used, weights, support, False
        ),
        diagnostics=diagnostics,
    )


def fit_samples(
    samples: ArrayLike, specification: SampleFit | ModelSelection
) -> FitResult:
    if not isinstance(specification, (SampleFit, ModelSelection)):
        raise TypeError("sample evidence requires SampleFit or ModelSelection")
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("samples must be a one-dimensional observation sequence")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(
            "samples must be nonempty and finite; missing observations must be handled explicitly"
        )
    candidates = (
        specification.candidates
        if isinstance(specification, ModelSelection)
        else (specification,)
    )
    if any(not isinstance(candidate, SampleFit) for candidate in candidates):
        raise TypeError("sample model selection requires SampleFit candidates")
    attempts: list[FitAttempt] = []
    successes: list[
        tuple[float, Distribution, tuple[Diagnostic, ...], SampleFit, int]
    ] = []
    support = float(np.min(values)), float(np.max(values))
    used_count = 0

    def run(
        candidate: SampleFit,
    ) -> tuple[float, Distribution, tuple[Diagnostic, ...], SampleFit, int] | None:
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
            diagnostic = _diagnostic(
                "fit.samples",
                "Requested sample estimator converged",
                {
                    "ks_statistic": native.ks_statistic,
                    "ks_pvalue": native.ks_pvalue,
                    "rmse": native.rmse,
                    "r_squared": native.r_squared,
                    "iterations": native.iterations,
                    "evaluations": native.evaluations,
                },
            )
            diagnostics: tuple[Diagnostic, ...] = (diagnostic,)
            attempts.append(
                FitAttempt(candidate, FitStatus.CONVERGED, None, diagnostics)
            )
            return (
                native.ks_statistic,
                distribution,
                diagnostics,
                candidate,
                len(attempts) - 1,
            )
        except ValueError as error:
            diagnostic = _diagnostic("fit.failed", str(error), {}, error=True)
            status = (
                FitStatus.REJECTED
                if "does not satisfy constraints" in str(error)
                else FitStatus.FAILED
            )
            attempts.append(FitAttempt(candidate, status, str(error), (diagnostic,)))
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
    _, distribution, diagnostics, chosen, selected = min(
        successes, key=lambda result: result[0]
    )
    return FitResult(
        distribution,
        diagnostics,
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


def fit_quantiles(
    curve: QuantileCurve,
    specification: QuantileFit
    | CensoredQuantileFit
    | HurdleQuantileFit
    | ModelSelection,
) -> FitResult:
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
    successes: list[
        tuple[float, Distribution, tuple[Diagnostic, ...], tuple[float, ...], int, int]
    ] = []
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
            scaled_weights = weights / float(np.max(weights))
            normalized = tuple(
                float(value) for value in scaled_weights / float(np.sum(scaled_weights))
            )
            if any(
                weight == 0 and original > 0
                for weight, original in zip(normalized, weights)
            ):
                raise ValueError(
                    "relative weights are too small to represent; positive weights cannot be silently dropped"
                )
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
            diagnostic_details: dict[str, ScientificValue]
            if isinstance(candidate, CensoredQuantileFit):
                censored = _core.fit_censored_quantiles(
                    fit_probabilities.tolist(),
                    fit_values.tolist(),
                    candidate.family.value,
                    censor_location=candidate.censor_location,
                    residual_scale_floor_ratio=candidate.residual_scale_floor_ratio,
                    weights=fit_weights.tolist(),
                    **options,
                )
                distribution = Distribution._from_native(censored.distribution)
                diagnostic_details = {
                    "rmse": censored.rmse,
                    "weighted_r_squared": censored.weighted_r_squared,
                    "maximum_absolute_residual": censored.maximum_absolute_residual,
                    "negative_log_likelihood": censored.negative_log_likelihood,
                    "residual_scale": censored.residual_scale,
                    "residual_scale_floor": censored.residual_scale_floor,
                    "censored_count": censored.censored_point_count,
                    "uncensored_count": censored.uncensored_point_count,
                    "iterations": censored.iterations,
                    "evaluations": censored.evaluations,
                }
            elif isinstance(candidate, HurdleQuantileFit):
                hurdle = _core.fit_hurdle_quantiles(
                    fit_probabilities.tolist(),
                    fit_values.tolist(),
                    candidate.family.value,
                    atom_probability=candidate.atom_probability,
                    atom_location=candidate.atom_location,
                    weights=fit_weights.tolist(),
                    **options,
                )
                distribution = Distribution.hurdle(
                    Distribution._from_native(hurdle.base_distribution),
                    atom_probability=candidate.atom_probability,
                    atom_location=candidate.atom_location,
                )
                diagnostic_details = {
                    "tail_rmse": hurdle.tail_rmse,
                    "tail_weighted_r_squared": hurdle.tail_weighted_r_squared,
                    "tail_maximum_absolute_residual": hurdle.tail_maximum_absolute_residual,
                    "atom_count": hurdle.atom_point_count,
                    "tail_count": hurdle.tail_point_count,
                    "iterations": hurdle.iterations,
                    "evaluations": hurdle.evaluations,
                }
            else:
                quantile = _core.fit_quantiles(
                    fit_probabilities.tolist(),
                    fit_values.tolist(),
                    candidate.family.value,
                    weights=fit_weights.tolist(),
                    **options,
                )
                distribution = Distribution._from_native(quantile.distribution)
                diagnostic_details = {
                    "rmse": quantile.rmse,
                    "weighted_r_squared": quantile.weighted_r_squared,
                    "maximum_absolute_residual": quantile.maximum_absolute_residual,
                    "iterations": quantile.iterations,
                    "evaluations": quantile.evaluations,
                }
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
            diagnostic_details["normalized_rmse"] = score
            diagnostic_details["normalization_scale"] = value_span
            diagnostics: tuple[Diagnostic, ...] = (
                _diagnostic(
                    f"fit.{candidate.kind.value}",
                    "Requested quantile fit converged",
                    diagnostic_details,
                ),
            )
            attempts.append(
                FitAttempt(candidate, FitStatus.CONVERGED, None, diagnostics)
            )
            successes.append(
                (score, distribution, diagnostics, normalized, used, len(attempts) - 1)
            )
        except ValueError as error:
            diagnostic = _diagnostic("fit.failed", str(error), {}, error=True)
            attempts.append(
                FitAttempt(candidate, FitStatus.FAILED, str(error), (diagnostic,))
            )
    if not successes:
        raise _failure(specification, attempts, count, last_used, last_weights, support)
    _, distribution, diagnostics, normalized, used, selected = min(
        successes, key=lambda result: result[0]
    )
    return FitResult(
        distribution,
        diagnostics,
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

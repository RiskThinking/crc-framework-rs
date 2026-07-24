from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union, overload

import numpy as np

from crc_framework import _core

from .base import Distribution
from .empirical import EmpiricalDistribution
from .fitted import DistributionFamily, FittedDistribution
from .hurdle import HurdleDistribution
from .tabulated import TabulatedDistribution


@dataclass(frozen=True)
class FitDiagnostics:
    ks_statistic: float
    ks_pvalue: float
    rmse: float
    r_squared: float


@dataclass(frozen=True)
class FitResult:
    distribution: FittedDistribution
    diagnostics: FitDiagnostics


@dataclass(frozen=True)
class QuantileFitDiagnostics:
    rmse: float
    normalized_rmse: float
    weighted_r_squared: float
    maximum_absolute_residual: float
    point_count: int
    converged: bool
    iterations: int
    evaluations: int


@dataclass(frozen=True)
class QuantileFitResult:
    distribution: FittedDistribution
    diagnostics: QuantileFitDiagnostics


@dataclass(frozen=True)
class HurdleQuantileFitDiagnostics:
    tail: QuantileFitDiagnostics
    atom_probability_lower_bound: float
    atom_probability_upper_bound: float
    atom_point_count: int
    tail_point_count: int


@dataclass(frozen=True)
class HurdleQuantileFitResult:
    distribution: HurdleDistribution
    diagnostics: HurdleQuantileFitDiagnostics


@dataclass(frozen=True)
class FitConstraints:
    probability: float = 0.999
    minimum_value: Optional[float] = None
    maximum_value: Optional[float] = None

    def accepts(self, result: FitResult) -> bool:
        value = float(result.distribution.ppf(self.probability))
        return (
            (self.minimum_value is None or value >= self.minimum_value)
            and (self.maximum_value is None or value <= self.maximum_value)
        )


def _samples(data: Union[Distribution, Sequence[float]]) -> list[float]:
    if isinstance(data, EmpiricalDistribution):
        return [float(value) for value in data.samples]
    if isinstance(data, TabulatedDistribution):
        raise ValueError(
            "fit_distribution accepts observations, not probability/value knots; "
            "use fit_quantiles"
        )
    if isinstance(data, FittedDistribution):
        raise ValueError(
            "fit_distribution accepts observations, not a fitted distribution"
        )
    if isinstance(data, HurdleDistribution):
        raise ValueError(
            "fit_distribution accepts observations, not a hurdle distribution"
        )
    return [float(value) for value in np.asarray(data, dtype=np.float64).reshape(-1)]


def _result(native: _core.NativeFitResult) -> FitResult:
    return FitResult(
        distribution=FittedDistribution._from_native(native.distribution),
        diagnostics=FitDiagnostics(
            ks_statistic=native.ks_statistic,
            ks_pvalue=native.ks_pvalue,
            rmse=native.rmse,
            r_squared=native.r_squared,
        ),
    )


def _quantile_result(native: _core.NativeQuantileFitResult) -> QuantileFitResult:
    return QuantileFitResult(
        distribution=FittedDistribution._from_native(native.distribution),
        diagnostics=QuantileFitDiagnostics(
            rmse=native.rmse,
            normalized_rmse=native.normalized_rmse,
            weighted_r_squared=native.weighted_r_squared,
            maximum_absolute_residual=native.maximum_absolute_residual,
            point_count=native.point_count,
            converged=native.converged,
            iterations=native.iterations,
            evaluations=native.evaluations,
        ),
    )


def _hurdle_result(
    native: _core.NativeHurdleQuantileFitResult,
) -> HurdleQuantileFitResult:
    tail = QuantileFitDiagnostics(
        rmse=native.tail_rmse,
        normalized_rmse=native.tail_normalized_rmse,
        weighted_r_squared=native.tail_weighted_r_squared,
        maximum_absolute_residual=native.tail_maximum_absolute_residual,
        point_count=native.tail_point_count,
        converged=native.converged,
        iterations=native.iterations,
        evaluations=native.evaluations,
    )
    return HurdleQuantileFitResult(
        distribution=HurdleDistribution._from_native(
            native.distribution,
            native.base_distribution,
            atom_probability=native.atom_probability,
            atom_location=native.atom_location,
        ),
        diagnostics=HurdleQuantileFitDiagnostics(
            tail=tail,
            atom_probability_lower_bound=native.atom_probability_lower_bound,
            atom_probability_upper_bound=native.atom_probability_upper_bound,
            atom_point_count=native.atom_point_count,
            tail_point_count=native.tail_point_count,
        ),
    )


def fit_distribution(
    data: Union[EmpiricalDistribution, Sequence[float]],
    family: Union[DistributionFamily, str] = "auto",
    *,
    candidates: Optional[Sequence[DistributionFamily]] = None,
    selector: str = "ks",
    constraints: Optional[Union[FitConstraints, Mapping[str, float]]] = None,
) -> FitResult:
    """Fit explicitly; metric functions never call this automatically."""
    if selector != "ks":
        raise ValueError("only the 'ks' selector is currently supported")
    active_constraints = (
        FitConstraints(**constraints)
        if isinstance(constraints, Mapping)
        else constraints
    )
    if candidates is not None or active_constraints is not None:
        allowed = (
            set(candidates)
            if candidates is not None
            else ({family} if family != "auto" else None)
        )
        results = [
            result
            for result in fit_all(data)
            if (allowed is None or result.distribution.family in allowed)
            and (active_constraints is None or active_constraints.accepts(result))
        ]
        if not results:
            raise ValueError("no fitted candidate satisfies the requested policy")
        return max(results, key=lambda result: result.diagnostics.ks_pvalue)
    return _result(_core.fit(_samples(data), None if family == "auto" else family))


def fit_all(
    data: Union[EmpiricalDistribution, Sequence[float]],
) -> list[FitResult]:
    return [_result(result) for result in _core.fit_candidates(_samples(data))]


def quality_metrics(
    data: Union[EmpiricalDistribution, Sequence[float]],
    distribution: FittedDistribution,
) -> FitDiagnostics:
    values = _core.diagnostic_metrics(_samples(data), distribution._native)
    return FitDiagnostics(**values)


@overload
def fit_quantiles(
    probabilities: TabulatedDistribution,
    values: None = None,
    family: DistributionFamily = "gumbel_r",
    *,
    weights: Optional[Sequence[float]] = None,
) -> QuantileFitResult: ...


@overload
def fit_quantiles(
    probabilities: Sequence[float],
    values: Sequence[float],
    family: DistributionFamily = "gumbel_r",
    *,
    weights: Optional[Sequence[float]] = None,
) -> QuantileFitResult: ...


def fit_quantiles(
    probabilities: Union[TabulatedDistribution, Sequence[float]],
    values: Optional[Sequence[float]] = None,
    family: DistributionFamily = "gumbel_r",
    *,
    weights: Optional[Sequence[float]] = None,
) -> QuantileFitResult:
    knot_probabilities: list[float]
    knot_values: list[float]
    if isinstance(probabilities, TabulatedDistribution):
        if values is not None:
            raise TypeError("values must be omitted when fitting a TabulatedDistribution")
        knot_probabilities = probabilities.probabilities.tolist()
        knot_values = probabilities.values.tolist()
    else:
        if values is None:
            raise TypeError("values are required with explicit probabilities")
        knot_probabilities = [float(value) for value in probabilities]
        knot_values = [float(value) for value in values]
    return _quantile_result(
        _core.fit_quantiles(
            knot_probabilities,
            knot_values,
            family,
            None
            if weights is None
            else np.asarray(weights, dtype=np.float64).reshape(-1).tolist(),
        )
    )


@overload
def fit_hurdle_quantiles(
    probabilities: TabulatedDistribution,
    values: None = None,
    family: DistributionFamily = "gumbel_r",
    *,
    atom_probability: float,
    atom_location: float = 0.0,
    weights: Optional[Sequence[float]] = None,
) -> HurdleQuantileFitResult: ...


@overload
def fit_hurdle_quantiles(
    probabilities: Sequence[float],
    values: Sequence[float],
    family: DistributionFamily = "gumbel_r",
    *,
    atom_probability: float,
    atom_location: float = 0.0,
    weights: Optional[Sequence[float]] = None,
) -> HurdleQuantileFitResult: ...


def fit_hurdle_quantiles(
    probabilities: Union[TabulatedDistribution, Sequence[float]],
    values: Optional[Sequence[float]] = None,
    family: DistributionFamily = "gumbel_r",
    *,
    atom_probability: float,
    atom_location: float = 0.0,
    weights: Optional[Sequence[float]] = None,
) -> HurdleQuantileFitResult:
    knot_probabilities: list[float]
    knot_values: list[float]
    if isinstance(probabilities, TabulatedDistribution):
        if values is not None:
            raise TypeError("values must be omitted when fitting a TabulatedDistribution")
        knot_probabilities = probabilities.probabilities.tolist()
        knot_values = probabilities.values.tolist()
    else:
        if values is None:
            raise TypeError("values are required with explicit probabilities")
        knot_probabilities = [float(value) for value in probabilities]
        knot_values = [float(value) for value in values]
    return _hurdle_result(
        _core.fit_hurdle_quantiles(
            knot_probabilities,
            knot_values,
            family,
            atom_probability,
            atom_location,
            None
            if weights is None
            else np.asarray(weights, dtype=np.float64).reshape(-1).tolist(),
        )
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from crc_framework import _core

from .base import Distribution
from .empirical import EmpiricalDistribution
from .fitted import DistributionFamily, FittedDistribution
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
        return [float(value) for value in data.values]
    if isinstance(data, FittedDistribution):
        return [
            float(value)
            for value in data.quantiles(np.linspace(0.001, 0.999, 1001))
        ]
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


def fit_distribution(
    data: Union[Distribution, Sequence[float]],
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


def fit_all(data: Union[Distribution, Sequence[float]]) -> list[FitResult]:
    return [_result(result) for result in _core.fit_candidates(_samples(data))]


def quality_metrics(
    data: Union[Distribution, Sequence[float]],
    distribution: FittedDistribution,
) -> FitDiagnostics:
    values = _core.diagnostic_metrics(_samples(data), distribution._native)
    return FitDiagnostics(**values)

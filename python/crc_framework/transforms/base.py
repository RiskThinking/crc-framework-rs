from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, Sequence, Union, runtime_checkable

import numpy as np
import numpy.typing as npt

from crc_framework.distributions import Distribution, TabulatedDistribution
from crc_framework.models import TransformContext

ImpactValues = Union[float, Sequence[float], npt.NDArray[np.float64]]
ImpactResult = Union[float, npt.NDArray[np.float64]]


class Transform(Protocol):
    def __call__(
        self,
        distribution: Distribution,
        *,
        probabilities: Optional[Sequence[float]] = None,
        context: Optional[TransformContext] = None,
    ) -> Distribution: ...


@runtime_checkable
class ImpactFunction(Protocol):
    """Evaluate event-aligned exposure values without quantile reordering."""

    def evaluate(
        self,
        values: ImpactValues,
        *,
        context: Optional[TransformContext] = None,
    ) -> ImpactResult: ...


def evaluate_values(
    function: Callable[[npt.NDArray[np.float64]], Any],
    values: ImpactValues,
) -> ImpactResult:
    """Apply point impact math with common shape and finiteness validation."""
    exposure = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(exposure)):
        raise ValueError("impact exposure values must be finite")
    impact = np.asarray(function(exposure), dtype=np.float64)
    if impact.shape != exposure.shape:
        raise ValueError("impact evaluation must preserve the exposure value shape")
    if not np.all(np.isfinite(impact)):
        raise ValueError("impact evaluation produced non-finite values")
    return float(impact) if exposure.ndim == 0 else impact


def probability_grid(
    distribution: Distribution, probabilities: Optional[Sequence[float]]
) -> npt.NDArray[np.float64]:
    if probabilities is not None:
        return np.asarray(probabilities, dtype=np.float64)
    if isinstance(distribution, TabulatedDistribution):
        return distribution.probabilities
    return np.linspace(0.001, 0.999, 1001)


class CallableImpact:
    """Adapt a vectorized Python callable for event-aligned impact evaluation."""

    def __init__(self, function: Callable[[npt.NDArray[np.float64]], Any]):
        self.function = function

    def evaluate(
        self,
        values: ImpactValues,
        *,
        context: Optional[TransformContext] = None,
    ) -> ImpactResult:
        del context
        return evaluate_values(self.function, values)


class CallableTransform(CallableImpact):
    """Adapt a vectorized callable to point and distribution impact interfaces."""

    def __call__(
        self,
        distribution: Distribution,
        *,
        probabilities: Optional[Sequence[float]] = None,
        context: Optional[TransformContext] = None,
    ) -> TabulatedDistribution:
        grid = probability_grid(distribution, probabilities)
        values = np.asarray(
            self.evaluate(distribution.quantiles(grid), context=context), dtype=np.float64
        )
        return TabulatedDistribution(grid, values)

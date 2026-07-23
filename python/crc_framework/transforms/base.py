from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from crc_framework.distributions import Distribution, TabulatedDistribution
from crc_framework.models import TransformContext


class Transform(Protocol):
    def __call__(
        self,
        distribution: Distribution,
        *,
        probabilities: Optional[Sequence[float]] = None,
        context: Optional[TransformContext] = None,
    ) -> Distribution: ...


def probability_grid(
    distribution: Distribution, probabilities: Optional[Sequence[float]]
) -> npt.NDArray[np.float64]:
    if probabilities is not None:
        return np.asarray(probabilities, dtype=np.float64)
    if isinstance(distribution, TabulatedDistribution):
        return distribution.probabilities
    return np.linspace(0.001, 0.999, 1001)


class CallableTransform:
    """Adapt a vectorized Python callable with one callback per full batch."""

    def __init__(self, function: Callable[[npt.NDArray[np.float64]], Sequence[float]]):
        self.function = function

    def __call__(
        self,
        distribution: Distribution,
        *,
        probabilities: Optional[Sequence[float]] = None,
        context: Optional[TransformContext] = None,
    ) -> TabulatedDistribution:
        del context
        grid = probability_grid(distribution, probabilities)
        values = np.asarray(self.function(distribution.quantiles(grid)), dtype=np.float64)
        return TabulatedDistribution(grid, values)

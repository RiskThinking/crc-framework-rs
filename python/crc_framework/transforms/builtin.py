from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from crc_framework.distributions import Distribution, TabulatedDistribution
from crc_framework.models import TransformContext

from .base import probability_grid


@dataclass(frozen=True)
class SigmoidImpact:
    midpoint: float
    steepness: float
    minimum: float = 0.0
    maximum: float = 1.0
    zero_below: Optional[float] = None

    def __call__(
        self,
        distribution: Distribution,
        *,
        probabilities: Optional[Sequence[float]] = None,
        context: Optional[TransformContext] = None,
    ) -> TabulatedDistribution:
        del context
        grid = probability_grid(distribution, probabilities)
        exposure = distribution.quantiles(grid)
        unit = 1.0 / (1.0 + np.exp(-self.steepness * (exposure - self.midpoint)))
        impact = self.minimum + unit * (self.maximum - self.minimum)
        if self.zero_below is not None:
            impact = np.where(exposure < self.zero_below, 0.0, impact)
        if self.steepness < 0.0:
            impact = impact[::-1]
        return TabulatedDistribution(grid, impact)


@dataclass(frozen=True)
class LinearImpact:
    slope: float
    intercept: float = 0.0
    minimum: Optional[float] = 0.0
    maximum: Optional[float] = None

    def __call__(
        self,
        distribution: Distribution,
        *,
        probabilities: Optional[Sequence[float]] = None,
        context: Optional[TransformContext] = None,
    ) -> TabulatedDistribution:
        del context
        grid = probability_grid(distribution, probabilities)
        impact = self.slope * distribution.quantiles(grid) + self.intercept
        if self.minimum is not None:
            impact = np.maximum(impact, self.minimum)
        if self.maximum is not None:
            impact = np.minimum(impact, self.maximum)
        if self.slope < 0.0:
            impact = impact[::-1]
        return TabulatedDistribution(grid, impact)


class PiecewiseLinearImpact:
    def __init__(self, exposure: Sequence[float], impact: Sequence[float]) -> None:
        self.exposure = np.asarray(exposure, dtype=np.float64)
        self.impact = np.asarray(impact, dtype=np.float64)
        if len(self.exposure) != len(self.impact) or len(self.exposure) < 2:
            raise ValueError("exposure and impact must have equal lengths of at least two")
        if np.any(np.diff(self.exposure) <= 0) or np.any(np.diff(self.impact) < 0):
            raise ValueError("exposure must increase and impact must not decrease")

    def __call__(
        self,
        distribution: Distribution,
        *,
        probabilities: Optional[Sequence[float]] = None,
        context: Optional[TransformContext] = None,
    ) -> TabulatedDistribution:
        del context
        grid = probability_grid(distribution, probabilities)
        values = np.interp(distribution.quantiles(grid), self.exposure, self.impact)
        return TabulatedDistribution(grid, values)

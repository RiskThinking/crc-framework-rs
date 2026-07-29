from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

from crc_framework.distributions import Distribution, TabulatedDistribution
from crc_framework.models import TransformContext

from .base import ImpactResult, ImpactValues, evaluate_values, probability_grid


@dataclass(frozen=True)
class SigmoidImpact:
    midpoint: float
    steepness: float
    minimum: float = 0.0
    maximum: float = 1.0
    zero_below: Optional[float] = None

    def evaluate(
        self,
        values: ImpactValues,
        *,
        context: Optional[TransformContext] = None,
    ) -> ImpactResult:
        del context

        def transform(
            exposure: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            with np.errstate(over="ignore"):
                unit = 1.0 / (
                    1.0 + np.exp(-self.steepness * (exposure - self.midpoint))
                )
            impact = self.minimum + unit * (self.maximum - self.minimum)
            if self.zero_below is not None:
                impact = np.where(exposure < self.zero_below, 0.0, impact)
            return impact

        return evaluate_values(transform, values)

    def __call__(
        self,
        distribution: Distribution,
        *,
        probabilities: Optional[Sequence[float]] = None,
        context: Optional[TransformContext] = None,
    ) -> TabulatedDistribution:
        grid = probability_grid(distribution, probabilities)
        impact = np.asarray(
            self.evaluate(distribution.quantiles(grid), context=context), dtype=np.float64
        )
        if self.steepness < 0.0:
            impact = impact[::-1]
        return TabulatedDistribution(grid, impact)


@dataclass(frozen=True)
class LinearImpact:
    slope: float
    intercept: float = 0.0
    minimum: Optional[float] = 0.0
    maximum: Optional[float] = None

    def evaluate(
        self,
        values: ImpactValues,
        *,
        context: Optional[TransformContext] = None,
    ) -> ImpactResult:
        del context

        def transform(
            exposure: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            impact = self.slope * exposure + self.intercept
            if self.minimum is not None:
                impact = np.maximum(impact, self.minimum)
            if self.maximum is not None:
                impact = np.minimum(impact, self.maximum)
            return impact

        return evaluate_values(transform, values)

    def __call__(
        self,
        distribution: Distribution,
        *,
        probabilities: Optional[Sequence[float]] = None,
        context: Optional[TransformContext] = None,
    ) -> TabulatedDistribution:
        grid = probability_grid(distribution, probabilities)
        impact = np.asarray(
            self.evaluate(distribution.quantiles(grid), context=context), dtype=np.float64
        )
        if self.slope < 0.0:
            impact = impact[::-1]
        return TabulatedDistribution(grid, impact)


class PiecewiseLinearImpact:
    def __init__(self, exposure: Sequence[float], impact: Sequence[float]) -> None:
        self.exposure = np.asarray(exposure, dtype=np.float64)
        self.impact = np.asarray(impact, dtype=np.float64)
        if len(self.exposure) != len(self.impact) or len(self.exposure) < 2:
            raise ValueError("exposure and impact must have equal lengths of at least two")
        if (
            np.any(~np.isfinite(self.exposure))
            or np.any(~np.isfinite(self.impact))
            or np.any(np.diff(self.exposure) <= 0)
            or np.any(np.diff(self.impact) < 0)
        ):
            raise ValueError("exposure must increase and impact must not decrease")

    def evaluate(
        self,
        values: ImpactValues,
        *,
        context: Optional[TransformContext] = None,
    ) -> ImpactResult:
        del context
        return evaluate_values(
            lambda exposure: np.interp(exposure, self.exposure, self.impact), values
        )

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

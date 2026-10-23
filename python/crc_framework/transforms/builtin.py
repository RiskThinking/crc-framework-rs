from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from crc_framework.distributions import Distribution, TabulatedDistribution
from crc_framework.impacts import (
    ImpactExpression,
    legacy_linear_expression,
    legacy_piecewise_expression,
    legacy_sigmoid_expression,
)
from crc_framework.models import TransformContext

from .base import ImpactResult, ImpactValues, probability_grid


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
        return self._compiled().evaluate(values)

    def _compiled(self) -> ImpactExpression:
        compiled = getattr(self, "_compiled_expression", None)
        if compiled is None:
            compiled = legacy_sigmoid_expression(
                self.midpoint,
                self.steepness,
                self.minimum,
                self.maximum,
                self.zero_below,
            )
            object.__setattr__(self, "_compiled_expression", compiled)
        return compiled

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
        return self._compiled().evaluate(values)

    def _compiled(self) -> ImpactExpression:
        compiled = getattr(self, "_compiled_expression", None)
        if compiled is None:
            compiled = legacy_linear_expression(
                self.slope, self.intercept, self.minimum, self.maximum
            )
            object.__setattr__(self, "_compiled_expression", compiled)
        return compiled

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
        self._compiled_expression = legacy_piecewise_expression(
            self.exposure.tolist(), self.impact.tolist()
        )

    def evaluate(
        self,
        values: ImpactValues,
        *,
        context: Optional[TransformContext] = None,
    ) -> ImpactResult:
        del context
        return self._compiled_expression.evaluate(values)

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

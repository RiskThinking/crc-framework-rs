from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np

from crc_framework import _core
from crc_framework.constants import RiskFactor
from crc_framework.distributions import Distribution, TabulatedDistribution
from crc_framework.models import TransformContext

from .base import probability_grid


class ClimateImpact:
    def __init__(
        self,
        factor: Union[RiskFactor, str],
        context: TransformContext,
        overrides: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.factor = factor.value if isinstance(factor, RiskFactor) else factor
        self.context = context
        self.overrides = dict(overrides or {})

    def __call__(
        self,
        distribution: Distribution,
        *,
        probabilities: Optional[Sequence[float]] = None,
        context: Optional[TransformContext] = None,
    ) -> TabulatedDistribution:
        active = context or self.context
        grid = probability_grid(distribution, probabilities)
        native_distribution = getattr(distribution, "_native", None)
        if native_distribution is None:
            native_distribution = TabulatedDistribution(
                grid, distribution.quantiles(grid)
            )._native
        native = _core.apply_impact(
            native_distribution,
            grid.tolist(),
            self.factor,
            active.cell,
            active.country,
            active.continent,
            active.building_type,
            active.historic_mean,
            self.overrides,
        )
        return TabulatedDistribution._from_native(native, np.asarray(grid))


class ImpactRegistry:
    def for_factor(
        self,
        factor: Union[RiskFactor, str],
        *,
        context: Optional[TransformContext] = None,
        overrides: Optional[Mapping[str, float]] = None,
    ) -> ClimateImpact:
        return ClimateImpact(factor, context or TransformContext(), overrides)

    resolve = for_factor


impacts = ImpactRegistry()

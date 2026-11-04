from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

from crc_framework.distributions import Distribution


def quantiles(
    distribution: Distribution, probabilities: Sequence[float]
) -> npt.NDArray[np.float64]:
    """Evaluate canonical non-exceedance probabilities."""
    return distribution.quantiles(probabilities)


def exceedance_probability(distribution: Distribution, value: float) -> float:
    return 1.0 - float(distribution.cdf(value))


def value_at_risk(distribution: Distribution, probability: float) -> float:
    return float(distribution.ppf(probability))


def conditional_value_at_risk(
    distribution: Distribution, probability: float, *, points: int = 1000
) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between zero and one")
    if points < 1:
        raise ValueError("points must be positive")
    # Midpoint rule on (probability, 1): never evaluates the quantile function
    # at (or next to) one, where heavy upper tails diverge.
    grid = probability + (1.0 - probability) * (np.arange(points) + 0.5) / points
    return float(np.mean(distribution.quantiles(grid)))

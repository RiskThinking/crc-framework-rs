"""Exact independent binary-risk aggregation and rank scenarios."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

from . import _core
from ._distribution import Distribution, FloatArray


@dataclass(frozen=True, slots=True)
class BinaryOutcome:
    """Two rank states for one independent contribution to aggregate impact."""

    id: str
    lower_rank_mass: float
    lower_impact: float
    upper_rank_mass: float
    upper_impact: float

    def __init__(
        self,
        id: str,
        *,
        lower_rank_mass: float,
        lower_impact: float,
        upper_rank_mass: float,
        upper_impact: float,
    ) -> None:
        if not isinstance(id, str) or not id or id.strip() != id:
            raise ValueError(
                "id must be an exact nonempty string without boundary whitespace"
            )
        values = (lower_rank_mass, lower_impact, upper_rank_mass, upper_impact)
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ValueError("rank masses and impacts must be finite numbers")
        if min(lower_rank_mass, upper_rank_mass) < 0 or not math.isclose(
            lower_rank_mass + upper_rank_mass, 1.0, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("lower_rank_mass and upper_rank_mass must sum to one")
        object.__setattr__(self, "id", id)
        for name, value in zip(
            ("lower_rank_mass", "lower_impact", "upper_rank_mass", "upper_impact"),
            values,
        ):
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True, slots=True)
class RankScenario:
    probability: float
    value: float
    lower_rank_mass: float
    upper_rank_mass: float


@dataclass(frozen=True, slots=True)
class RiskAttribution:
    id: str
    value_at_risk: float
    conditional_value_at_risk: float


@dataclass(frozen=True, slots=True)
class RiskLevel:
    probability: float
    value_at_risk: float
    conditional_value_at_risk: float
    attribution: tuple[RiskAttribution, ...]


@dataclass(frozen=True, slots=True)
class RiskResult:
    levels: tuple[RiskLevel, ...]
    branch_count: int

    def at(self, probability: float) -> RiskLevel:
        for level in self.levels:
            if level.probability == probability:
                return level
        raise KeyError(probability)


def rank_scenarios(
    distribution: Distribution,
    *,
    probabilities: Sequence[float],
) -> tuple[RankScenario, ...]:
    """Query quantiles and label rank masses, without computing summaries.

    Atoms can span a rank split. Use cdf/exceedance/mass for actual threshold
    probabilities instead of treating these rank masses as threshold events.
    """
    if not isinstance(distribution, Distribution):
        raise TypeError("distribution must be a Distribution")
    qs = tuple(probabilities)
    if not qs:
        raise ValueError("at least one probability is required")
    if any(isinstance(q, bool) or not math.isfinite(q) or not 0 <= q <= 1 for q in qs):
        raise ValueError("probabilities must be finite numbers in [0, 1]")
    if any(a >= b for a, b in pairwise(qs)):
        raise ValueError("probabilities must be strictly increasing")
    values = cast(FloatArray, distribution.quantile(qs))
    return tuple(
        RankScenario(float(q), float(value), float(q), float(1 - q))
        for q, value in zip(qs, values)
    )


def compute_risk(
    outcomes: Sequence[BinaryOutcome],
    *,
    levels: Sequence[float],
    max_branches: int | None = None,
) -> RiskResult:
    """Sum independent binary impacts and integrate their upper tail exactly.

    Contributions are added in their supplied units; there is no implicit
    averaging. VaR attribution averages equally valued states by their masses.
    """
    items = tuple(outcomes)
    if not all(isinstance(item, BinaryOutcome) for item in items):
        raise TypeError("outcomes must contain BinaryOutcome objects")
    if max_branches is not None and (
        isinstance(max_branches, bool)
        or not isinstance(max_branches, int)
        or max_branches <= 0
    ):
        raise ValueError("max_branches must be a positive integer")
    qs = tuple(levels)
    if any(isinstance(q, bool) for q in qs):
        raise ValueError("levels must be probabilities, not booleans")
    count, result = _core.rank_risk(
        [
            (o.id, o.lower_rank_mass, o.lower_impact, o.upper_rank_mass, o.upper_impact)
            for o in items
        ],
        qs,
        # Larger Python integers cannot be converted to usize; the native hard
        # limit still applies, so capping this input cannot authorize more work.
        None if max_branches is None else min(max_branches, 1 << 26),
    )
    return RiskResult(
        tuple(
            RiskLevel(
                q, var, cvar, tuple(RiskAttribution(*part) for part in attribution)
            )
            for q, var, cvar, attribution in result
        ),
        count,
    )

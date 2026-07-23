from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from crc_framework import _core

from .microscores import BinaryOutcome


@dataclass(frozen=True)
class RiskAttribution:
    factor: str
    var_impact: float
    cvar_impact: float


@dataclass(frozen=True)
class RiskLevel:
    probability: float
    var: float
    cvar: float
    attribution: tuple[RiskAttribution, ...]


@dataclass(frozen=True)
class RiskResult:
    levels: tuple[RiskLevel, ...]
    branch_count: int

    def at(self, probability: float) -> RiskLevel:
        for level in self.levels:
            if abs(level.probability - probability) < 1.0e-12:
                return level
        raise KeyError(probability)


def compute_risk(
    outcomes: Sequence[BinaryOutcome],
    *,
    levels: Sequence[float] = (0.5, 0.95, 0.99),
    max_branches: Optional[int] = None,
) -> RiskResult:
    native = _core.risk(
        [outcome._native for outcome in outcomes], list(levels), max_branches
    )
    return RiskResult(
        branch_count=native.branch_count,
        levels=tuple(
            RiskLevel(
                probability=level.probability,
                var=level.var,
                cvar=level.cvar,
                attribution=tuple(
                    RiskAttribution(
                        factor=item.factor,
                        var_impact=item.var_impact,
                        cvar_impact=item.cvar_impact,
                    )
                    for item in level.attribution
                ),
            )
            for level in native.levels
        ),
    )

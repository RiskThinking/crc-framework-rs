from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .microscores import BinaryOutcome

from crc_framework import _core


@dataclass(frozen=True)
class SpanningBranch:
    choices: Tuple[bool, ...]
    probability: float
    impact: float
    factor_impacts: Tuple[Tuple[str, float], ...]


def compute_spanning_set(
    outcomes: Sequence[BinaryOutcome], *, max_branches: Optional[int] = None
) -> tuple[SpanningBranch, ...]:
    branches = _core.spanning_set(
        [outcome._native for outcome in outcomes], max_branches
    )
    return tuple(
        SpanningBranch(
            choices=tuple(branch.choices),
            probability=branch.probability,
            impact=branch.impact,
            factor_impacts=tuple(
                (str(factor), float(impact)) for factor, impact in branch.factor_impacts
            ),
        )
        for branch in branches
    )

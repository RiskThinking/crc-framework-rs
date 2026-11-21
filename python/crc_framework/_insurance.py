"""Primary policy terms and empirical summaries for aligned annual losses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from . import _core
from ._diagnostics import Diagnostic, DiagnosticSeverity
from ._distribution import (
    ArrayLike,
    ValueResult,
    Values,
    _finite,
    _output,
    _real_array,
    _vector,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyTerms:
    """Primary terms for one insured interest, in the same currency as its value."""

    deductible: float = 0.0
    limit: float | None = None
    share: float = 1.0
    _native: _core.NativePolicyTerms = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        deductible = _finite(self.deductible, "deductible")
        limit = None if self.limit is None else _finite(self.limit, "limit")
        share = _finite(self.share, "share")
        native = _core.NativePolicyTerms(deductible, limit, share)
        object.__setattr__(self, "deductible", deductible)
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "share", share)
        object.__setattr__(self, "_native", native)

    def insured_loss(
        self, total_insured_value: float, damage_ratio: Values
    ) -> ValueResult:
        """Apply the deductible, limit and share to ground-up loss."""
        value = _finite(total_insured_value, "total insured value")
        ratios = _real_array(damage_ratio, "damage ratio")
        losses = self._native.insured_losses(value, ratios.reshape(-1).tolist())
        return _output(losses, ratios.shape, ratios.ndim == 0)


@dataclass(frozen=True, slots=True)
class AnnualLossLevel:
    """Portfolio and component statistics at a non-exceedance level."""

    level: float
    value_at_risk: float
    conditional_value_at_risk: float
    diversification_benefit: float | None
    contributions: Mapping[str, float] | None
    nominal_tail_years: float
    boundary_years: int
    boundary_weight: float

    def __post_init__(self) -> None:
        if self.contributions is not None:
            object.__setattr__(
                self, "contributions", MappingProxyType(dict(self.contributions))
            )


@dataclass(frozen=True, slots=True)
class AnnualLossExceedance:
    """An empirical annual aggregate loss threshold at a return period."""

    return_period: float
    loss: float
    nominal_exceedance_probability: float
    exceedance_probability: float
    nominal_tail_years: float


@dataclass(frozen=True, slots=True)
class AnnualLossSummary:
    """Empirical loss metrics on a single portfolio and valuation basis."""

    trial_count: int
    total_insured_value: float
    average_annual_loss: float
    standard_deviation: float
    coefficient_of_variation: float
    loss_cost: float | None
    levels: tuple[AnnualLossLevel, ...]
    aep: tuple[AnnualLossExceedance, ...]

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        if self.total_insured_value != 0 or self.loss_cost is not None:
            return ()
        return (
            Diagnostic(
                "zero_total_insured_value",
                "Loss cost is undefined when total insured value is zero.",
                DiagnosticSeverity.INFO,
                "/loss_cost",
                {},
            ),
        )

    def at(self, level: float) -> AnnualLossLevel:
        """Return the statistics for an explicitly requested level."""
        for result in self.levels:
            if result.level == level:
                return result
        raise KeyError(level)


def summarize_annual_losses(
    totals: ArrayLike,
    *,
    total_insured_value: float,
    levels: Sequence[float] = (0.95, 0.99, 0.995),
    return_periods: Sequence[float] = (10, 25, 50, 100, 200, 250),
    components: Mapping[str, ArrayLike] | None = None,
) -> AnnualLossSummary:
    """Summarize insured losses by year, with optional aligned component losses."""
    annual_totals = _vector(totals, "annual losses").tolist()
    insured_value = _finite(total_insured_value, "total insured value")
    qs = tuple(_finite(level, "risk level") for level in levels)
    periods = tuple(_finite(period, "return period") for period in return_periods)
    aligned = None
    if components is not None:
        if not isinstance(components, Mapping):
            raise TypeError("components must map IDs to annual loss arrays")
        aligned = [
            (identity, _vector(losses, f"component {identity!r} losses").tolist())
            for identity, losses in components.items()
        ]

    count, tiv, mean, deviation, cv, cost, risk, aep = _core.summarize_annual_losses(
        annual_totals, insured_value, qs, periods, aligned
    )
    level_results = []
    for (
        level,
        var,
        cvar,
        diversification,
        parts,
        nominal_years,
        boundary_years,
        weight,
    ) in risk:
        level_results.append(
            AnnualLossLevel(
                level,
                var,
                cvar,
                diversification,
                None if parts is None else dict(parts),
                nominal_years,
                boundary_years,
                weight,
            )
        )
    return AnnualLossSummary(
        count,
        tiv,
        mean,
        deviation,
        cv,
        cost,
        tuple(level_results),
        tuple(AnnualLossExceedance(*point) for point in aep),
    )

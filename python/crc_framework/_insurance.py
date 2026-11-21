"""Primary insurance terms, one-peril simulation, and annual loss summaries."""

from __future__ import annotations

import operator
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import numpy

from . import _core
from ._diagnostics import Diagnostic, DiagnosticSeverity
from ._distribution import (
    ArrayLike,
    Distribution,
    DistributionKind,
    FloatArray,
    QuantileCurve,
    ValueResult,
    Values,
    _finite,
    _output,
    _real_array,
    _vector,
)

_U64_MAX = (1 << 64) - 1


def _integer(value: int, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, numpy.bool_)):
        raise TypeError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return result


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


@dataclass(frozen=True, slots=True, kw_only=True)
class AnnualLossSpec:
    """Scientific configuration for a one-peril annual loss simulation."""

    trials: int = 20_000
    seed: int = 20_261_027
    peak_correlation: float = 0.4
    decay_length_km: float = 150.0
    damage_cv: float = 0.3
    repair_tolerance: float | None = None
    calibration_source: str | None = None
    _native: _core.NativeAnnualLossSpec = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        trials = _integer(self.trials, "trials", minimum=1, maximum=sys.maxsize)
        seed = _integer(self.seed, "seed", minimum=0, maximum=_U64_MAX)
        peak = _finite(self.peak_correlation, "peak_correlation")
        decay = _finite(self.decay_length_km, "decay_length_km")
        damage_cv = _finite(self.damage_cv, "damage_cv")
        tolerance = (
            None
            if self.repair_tolerance is None
            else _finite(self.repair_tolerance, "repair_tolerance")
        )
        source = self.calibration_source
        if source is not None:
            if not isinstance(source, str):
                raise TypeError("calibration_source must be a string")
            if not source or source != source.strip():
                raise ValueError("calibration_source must be an exact nonempty label")
        native = _core.NativeAnnualLossSpec(
            trials, seed, peak, decay, damage_cv, tolerance
        )
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "peak_correlation", peak)
        object.__setattr__(self, "decay_length_km", decay)
        object.__setattr__(self, "damage_cv", damage_cv)
        object.__setattr__(self, "repair_tolerance", tolerance)
        object.__setattr__(self, "_native", native)

    @classmethod
    def reference(cls) -> AnnualLossSpec:
        """Use the reference annual simulation parameters."""
        return cls()


@dataclass(frozen=True, slots=True, init=False)
class InsuredInterest:
    """WGS84 location, resolved terms, and bounded mean damage by annual rank."""

    id: str
    longitude: float
    latitude: float
    total_insured_value: float
    damage_curve: QuantileCurve | Distribution
    terms: PolicyTerms
    _native: _core.NativeInsuredInterest = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        id: str,
        *,
        longitude: float,
        latitude: float,
        total_insured_value: float,
        damage_curve: QuantileCurve | Distribution,
        terms: PolicyTerms | None = None,
    ) -> None:
        if not isinstance(id, str):
            raise TypeError("id must be a string")
        if terms is None:
            terms = PolicyTerms()
        elif not isinstance(terms, PolicyTerms):
            raise TypeError("terms must be PolicyTerms")
        longitude = _finite(longitude, "longitude")
        latitude = _finite(latitude, "latitude")
        total_insured_value = _finite(total_insured_value, "total_insured_value")
        if isinstance(damage_curve, QuantileCurve):
            knots = (damage_curve.probabilities.tolist(), damage_curve.values.tolist())
        elif isinstance(damage_curve, Distribution):
            if damage_curve.kind is not DistributionKind.QUANTILE:
                raise ValueError(
                    "damage_curve must interpolate linearly in probability"
                )
            record = damage_curve._record
            if record is not None and record.get("operation") == "condition":
                raise ValueError(
                    "damage_curve must use original annual ranks. "
                    "Use explicit endpoint completion for partial support."
                )
            knots = (
                cast(list[float], damage_curve._data["probabilities"]),
                cast(list[float], damage_curve._data["values"]),
            )
        else:
            raise TypeError("damage_curve must be a QuantileCurve or Distribution")
        native = _core.NativeInsuredInterest(
            id, (longitude, latitude), total_insured_value, knots, terms._native
        )
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "total_insured_value", total_insured_value)
        object.__setattr__(self, "damage_curve", damage_curve)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "_native", native)


@dataclass(frozen=True, slots=True)
class CorrelationRepair:
    """Change to a singular or indefinite latent correlation matrix."""

    minimum_eigenvalue: float
    maximum_change: float


@dataclass(frozen=True, slots=True, init=False, eq=False)
class AnnualLossSimulation:
    """Insured losses indexed by simulated year and optional interest ID."""

    totals: FloatArray
    components: Mapping[str, FloatArray] | None
    total_insured_value: float
    spec: AnnualLossSpec
    secondary_stream: int
    rng_algorithm: str
    correlation_repair: CorrelationRepair | None

    def __init__(self) -> None:
        raise TypeError("construct AnnualLossSimulation with simulate_annual_losses")

    @property
    def trial_ids(self) -> range:
        """Zero-based year identifiers shared by all result arrays."""
        return range(len(self.totals))


def simulate_annual_losses(
    interests: Sequence[InsuredInterest],
    *,
    spec: AnnualLossSpec | None = None,
    include_components: bool = False,
    secondary_stream: int = 0,
    block_size: int = 256,
    memory_limit: int = 536_870_912,
    work_limit: int = 10_000_000_000,
) -> AnnualLossSimulation:
    """Simulate aligned annual losses for one peril in a common valuation basis."""
    if spec is None:
        spec = AnnualLossSpec.reference()
    elif not isinstance(spec, AnnualLossSpec):
        raise TypeError("spec must be an AnnualLossSpec")
    if not isinstance(include_components, bool):
        raise TypeError("include_components must be a bool")
    secondary_stream = _integer(
        secondary_stream, "secondary_stream", minimum=0, maximum=_U64_MAX
    )
    block_size = _integer(block_size, "block_size", minimum=1, maximum=sys.maxsize)
    memory_limit = _integer(
        memory_limit, "memory_limit", minimum=1, maximum=sys.maxsize
    )
    work_limit = _integer(work_limit, "work_limit", minimum=1, maximum=_U64_MAX)
    native_interests = []
    for interest in interests:
        if not isinstance(interest, InsuredInterest):
            raise TypeError("interests must contain InsuredInterest objects")
        native_interests.append(interest._native)

    totals, components, insured_value, algorithm, repair = _core.simulate_annual_losses(
        native_interests,
        spec._native,
        (include_components, secondary_stream, block_size, memory_limit, work_limit),
    )
    result = object.__new__(AnnualLossSimulation)
    object.__setattr__(result, "totals", numpy.frombuffer(totals, dtype=numpy.float64))
    object.__setattr__(
        result,
        "components",
        None
        if components is None
        else MappingProxyType(
            {
                id: numpy.frombuffer(losses, dtype=numpy.float64)
                for id, losses in components
            }
        ),
    )
    object.__setattr__(result, "total_insured_value", insured_value)
    object.__setattr__(result, "spec", spec)
    object.__setattr__(result, "secondary_stream", secondary_stream)
    object.__setattr__(result, "rng_algorithm", algorithm)
    object.__setattr__(
        result,
        "correlation_repair",
        None if repair is None else CorrelationRepair(*repair),
    )
    return result


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

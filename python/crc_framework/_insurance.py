"""Primary insurance terms, annual simulation, and loss summaries."""

from __future__ import annotations

import hashlib
import operator
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from types import MappingProxyType
from typing import TypeAlias, cast

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
    _dump,
    _fields,
    _finite,
    _output,
    _parse,
    _real_array,
    _vector,
)
from ._measurement import _identifier

_U64_MAX = (1 << 64) - 1
MatrixLike: TypeAlias = Sequence[Sequence[float]] | FloatArray
_ExecutionOptions: TypeAlias = tuple[bool, int, int, int, int]
_NativeSimulation: TypeAlias = tuple[
    bytes, list[tuple[str, bytes]] | None, float, str, tuple[float, float] | None
]


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
    """Scientific configuration shared by the simulated perils in an annual run."""

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


def _peril_names(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError("perils must be a sequence of peril IDs")
    names = tuple(_identifier(value, "peril ID") for value in values)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("multi-peril dependence requires at least two distinct perils")
    return names


def _matrix(values: MatrixLike, label: str) -> FloatArray:
    array = _real_array(cast(ArrayLike, values), label)
    if array.ndim != 2 or not numpy.isfinite(array).all():
        raise ValueError(f"{label} must be a finite two-dimensional array")
    return array


@dataclass(frozen=True, slots=True, init=False)
class PerilDependence:
    """Explicit dependence for annual losses with separately applied peril terms."""

    perils: tuple[str, ...]
    correlation: tuple[tuple[float, ...], ...] | None
    calibration_source: str | None
    observation_count: int | None
    observations_sha256: str | None

    def __init__(self) -> None:
        raise TypeError("construct a PerilDependence with a named constructor")

    @classmethod
    def _create(
        cls,
        perils: tuple[str, ...],
        correlation: tuple[tuple[float, ...], ...] | None,
        calibration_source: str | None,
        *,
        observation_count: int | None = None,
        observations_sha256: str | None = None,
    ) -> PerilDependence:
        result = object.__new__(cls)
        object.__setattr__(result, "perils", perils)
        object.__setattr__(result, "correlation", correlation)
        object.__setattr__(result, "calibration_source", calibration_source)
        object.__setattr__(result, "observation_count", observation_count)
        object.__setattr__(result, "observations_sha256", observations_sha256)
        return result

    @classmethod
    def independent(cls) -> PerilDependence:
        """Use independently seeded primary and secondary shocks for each peril."""
        return cls._create((), None, None)

    @classmethod
    def from_correlation(
        cls,
        perils: Sequence[str],
        correlation: MatrixLike,
        *,
        calibration_source: str,
    ) -> PerilDependence:
        """Use a declared latent Gaussian correlation matrix in the given peril order."""
        names = _peril_names(perils)
        source = _identifier(calibration_source, "calibration_source")
        matrix = _matrix(correlation, "correlation")
        if matrix.shape != (len(names), len(names)):
            raise ValueError("correlation must have one row and column per peril")
        order = sorted(range(len(names)), key=names.__getitem__)
        matrix = matrix[numpy.ix_(order, order)]
        native = matrix.tolist()
        _core.validate_peril_correlation(native)
        return cls._create(
            tuple(names[index] for index in order),
            tuple(tuple(row) for row in native),
            source,
        )

    @classmethod
    def calibrate(
        cls,
        perils: Sequence[str],
        annual_ranks: MatrixLike,
        *,
        calibration_source: str,
    ) -> PerilDependence:
        """Fit latent Gaussian correlation to same-year non-exceedance ranks."""
        names = _peril_names(perils)
        source = _identifier(calibration_source, "calibration_source")
        ranks = _matrix(annual_ranks, "annual_ranks")
        count = len(ranks)
        if ranks.shape[1] != len(names) or count <= len(names):
            raise ValueError(
                "annual_ranks must have one column per peril and more years than perils"
            )
        if not ((ranks > 0.0) & (ranks < 1.0)).all():
            raise ValueError("annual_ranks must lie strictly between zero and one")
        if numpy.any(numpy.ptp(ranks, axis=0) == 0.0):
            raise ValueError("annual_ranks must vary over years for every peril")
        order = sorted(range(len(names)), key=names.__getitem__)
        names = tuple(names[index] for index in order)
        ranks = ranks[:, order]
        inverse_normal = NormalDist().inv_cdf
        gaussian = numpy.fromiter(
            (inverse_normal(float(rank)) for rank in ranks.flat),
            dtype=numpy.float64,
            count=ranks.size,
        ).reshape(ranks.shape)
        matrix = cast(FloatArray, numpy.corrcoef(gaussian, rowvar=False))
        # Pearson correlation is symmetric with a unit diagonal. Remove floating
        # point asymmetry without altering its eigenstructure through a repair.
        matrix = (matrix + matrix.T) / 2.0
        numpy.fill_diagonal(matrix, 1.0)
        minimum = numpy.linalg.eigvalsh(matrix)[0]
        if minimum <= numpy.finfo(numpy.float64).eps * len(names):
            raise ValueError(
                "calibration must yield a positive definite correlation matrix"
            )
        resolved = cls.from_correlation(names, matrix, calibration_source=source)
        digest = hashlib.sha256(_dump(names).encode() + b"\n")
        digest.update(ranks.astype("<f8", copy=False).tobytes())
        return cls._create(
            resolved.perils,
            resolved.correlation,
            source,
            observation_count=count,
            observations_sha256=f"sha256:{digest.hexdigest()}",
        )

    def to_json(self) -> str:
        """Serialize the declared peril order, dependence, and calibration evidence."""
        return _dump(
            {
                "perils": self.perils,
                "correlation": self.correlation,
                "calibration_source": self.calibration_source,
                "observation_count": self.observation_count,
                "observations_sha256": self.observations_sha256,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> PerilDependence:
        """Load a declared or calibrated model with the same admission checks."""
        data = _parse(text)
        _fields(
            data,
            {
                "perils",
                "correlation",
                "calibration_source",
                "observation_count",
                "observations_sha256",
            },
        )
        names = data["perils"]
        correlation = data["correlation"]
        source = data["calibration_source"]
        count = data["observation_count"]
        digest = data["observations_sha256"]
        if correlation is None:
            if names != [] or any(
                value is not None for value in (source, count, digest)
            ):
                raise ValueError(
                    "independent peril dependence admits no calibration data"
                )
            return cls.independent()
        if not isinstance(names, list) or not isinstance(correlation, list):
            raise TypeError("expected peril IDs and a correlation matrix")
        if not isinstance(source, str):
            raise TypeError("correlated peril dependence requires a calibration_source")
        resolved = cls.from_correlation(names, correlation, calibration_source=source)
        if count is None and digest is None:
            return resolved
        if (
            type(count) is not int
            or not len(resolved.perils) < count <= sys.maxsize
            or not isinstance(digest, str)
            or re.fullmatch(r"sha256:[a-f0-9]{64}", digest) is None
        ):
            raise ValueError("invalid peril calibration observation count or digest")
        return cls._create(
            resolved.perils,
            resolved.correlation,
            source,
            observation_count=count,
            observations_sha256=digest,
        )


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
    """Insured losses indexed by simulated year and optional component ID."""

    totals: FloatArray
    components: Mapping[str, FloatArray] | None
    total_insured_value: float
    spec: AnnualLossSpec
    secondary_stream: int
    rng_algorithm: str
    correlation_repair: CorrelationRepair | None
    perils: tuple[str, ...]
    dependence: PerilDependence | None

    def __init__(self) -> None:
        raise TypeError("obtain an AnnualLossSimulation from an annual loss simulator")

    @property
    def trial_ids(self) -> range:
        """Zero-based year identifiers shared by all result arrays."""
        return range(len(self.totals))


def _simulation_options(
    *,
    spec: AnnualLossSpec | None,
    include_components: bool,
    secondary_stream: int,
    block_size: int,
    memory_limit: int,
    work_limit: int,
) -> tuple[AnnualLossSpec, _ExecutionOptions]:
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
    return spec, (
        include_components,
        secondary_stream,
        block_size,
        memory_limit,
        work_limit,
    )


def _native_interests(
    interests: Sequence[InsuredInterest],
) -> list[_core.NativeInsuredInterest]:
    native = []
    for interest in interests:
        if not isinstance(interest, InsuredInterest):
            raise TypeError("interests must contain InsuredInterest objects")
        native.append(interest._native)
    return native


def _simulation_result(
    native: _NativeSimulation,
    spec: AnnualLossSpec,
    secondary_stream: int,
    *,
    perils: tuple[str, ...] = (),
    dependence: PerilDependence | None = None,
) -> AnnualLossSimulation:
    totals, components, insured_value, algorithm, repair = native
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
    object.__setattr__(result, "perils", perils)
    object.__setattr__(result, "dependence", dependence)
    return result


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
    spec, execution = _simulation_options(
        spec=spec,
        include_components=include_components,
        secondary_stream=secondary_stream,
        block_size=block_size,
        memory_limit=memory_limit,
        work_limit=work_limit,
    )
    native = _core.simulate_annual_losses(
        _native_interests(interests), spec._native, execution
    )
    return _simulation_result(native, spec, execution[1])


def simulate_separate_peril_losses(
    perils: Mapping[str, Sequence[InsuredInterest]],
    *,
    dependence: PerilDependence,
    spec: AnnualLossSpec | None = None,
    include_components: bool = False,
    secondary_stream: int = 0,
    block_size: int = 256,
    memory_limit: int = 536_870_912,
    work_limit: int = 10_000_000_000,
) -> AnnualLossSimulation:
    """Simulate aligned annual totals for a fixed book with separate peril terms."""
    if not isinstance(perils, Mapping):
        raise TypeError("perils must map peril IDs to sequences of insured interests")
    if not isinstance(dependence, PerilDependence):
        raise TypeError("dependence must be PerilDependence")
    names = tuple(sorted(_peril_names(tuple(perils))))
    if dependence.perils and dependence.perils != names:
        raise ValueError("peril dependence must declare the same peril IDs as the book")
    spec, execution = _simulation_options(
        spec=spec,
        include_components=include_components,
        secondary_stream=secondary_stream,
        block_size=block_size,
        memory_limit=memory_limit,
        work_limit=work_limit,
    )
    matrix = dependence.correlation
    native = _core.simulate_separate_peril_losses(
        [(name, _native_interests(perils[name])) for name in names],
        None if matrix is None else [list(row) for row in matrix],
        spec._native,
        execution,
    )
    return _simulation_result(
        native, spec, execution[1], perils=names, dependence=dependence
    )


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

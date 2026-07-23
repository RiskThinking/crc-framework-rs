from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union, cast

from crc_framework import _core
from crc_framework.distributions import Distribution, TabulatedDistribution
from crc_framework.models import ScenarioMetadata
from crc_framework.transforms import Transform


@dataclass(frozen=True)
class DistributionStatistics:
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    cvar_95: float


@dataclass(frozen=True)
class Microscore:
    probability: float
    exposure: float
    impact: Optional[float]
    probability_below: float
    probability_above: float
    impact_probability_below: Optional[float]
    impact_probability_above: Optional[float]


@dataclass(frozen=True)
class BinaryOutcome:
    factor: str
    downside_probability: float
    downside_impact: float
    upside_probability: Optional[float] = None
    upside_impact: float = 0.0
    weight: float = 1.0

    @property
    def _native(self) -> _core.NativeBinaryOutcome:
        return _core.NativeBinaryOutcome(
            self.factor,
            self.downside_probability,
            self.downside_impact,
            self.upside_probability,
            self.upside_impact,
            self.weight,
        )

    @classmethod
    def _from_native(cls, value: _core.NativeBinaryOutcome) -> "BinaryOutcome":
        return cls(
            factor=value.factor,
            downside_probability=value.downside_probability,
            downside_impact=value.downside_impact,
            upside_probability=value.upside_probability,
            upside_impact=value.upside_impact,
            weight=value.weight,
        )


class MicroscoreSuite:
    def __init__(
        self, native: _core.NativeMicroscoreSuite, metadata: ScenarioMetadata
    ) -> None:
        self._native = native
        self.metadata = metadata

    @property
    def scores(self) -> tuple[Microscore, ...]:
        return tuple(
            Microscore(
                probability=value.probability,
                exposure=value.exposure,
                impact=value.impact,
                probability_below=value.probability_below,
                probability_above=value.probability_above,
                impact_probability_below=value.impact_probability_below,
                impact_probability_above=value.impact_probability_above,
            )
            for value in self._native.scores
        )

    @property
    def exposure_statistics(self) -> DistributionStatistics:
        return DistributionStatistics(**self._native.exposure_statistics())

    @property
    def impact_statistics(self) -> Optional[DistributionStatistics]:
        values = self._native.impact_statistics()
        return DistributionStatistics(**values) if values is not None else None

    def at(self, probability: float) -> BinaryOutcome:
        return BinaryOutcome._from_native(self._native.at(probability))


def generate_microscores(
    exposure: Distribution,
    *,
    impact: Optional[Union[Distribution, Transform]] = None,
    probabilities: Sequence[float],
    statistics: Optional[Sequence[str]] = None,
    metadata: Optional[ScenarioMetadata] = None,
) -> MicroscoreSuite:
    """Sample supplied distributions; fitting is always an explicit upstream step."""
    del statistics  # The native suite currently computes its standard diagnostic set.
    metadata = metadata or ScenarioMetadata()
    impact_distribution: Optional[Distribution] = None
    if impact is not None and not hasattr(impact, "_native"):
        transform = cast(Transform, impact)
        impact_distribution = transform(exposure)
    elif impact is not None:
        impact_distribution = cast(Distribution, impact)
    exposure_native = getattr(exposure, "_native", None)
    if exposure_native is None:
        exposure_native = TabulatedDistribution(
            probabilities, exposure.quantiles(probabilities)
        )._native
    impact_native = (
        None
        if impact_distribution is None
        else getattr(impact_distribution, "_native", None)
    )
    if impact_distribution is not None and impact_native is None:
        impact_native = TabulatedDistribution(
            probabilities, impact_distribution.quantiles(probabilities)
        )._native
    native = _core.microscores(
        exposure_native,
        list(probabilities),
        impact_native,
        metadata.cell,
        metadata.factor,
        metadata.pathway,
        metadata.horizon,
        metadata.country,
        metadata.continent,
        metadata.building_type,
    )
    return MicroscoreSuite(native, metadata)

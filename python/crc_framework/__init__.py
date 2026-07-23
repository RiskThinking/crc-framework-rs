"""Typed Python API for the CRC Framework computation core."""

from .constants import HORIZONS, PATHWAYS, RISK_FACTORS, Pathway, RiskFactor
from .distributions import (
    Distribution,
    EmpiricalDistribution,
    FitConstraints,
    FitDiagnostics,
    FitResult,
    FittedDistribution,
    TabulatedDistribution,
    fit_all,
    fit_distribution,
    quality_metrics,
)
from .metrics import (
    BinaryOutcome,
    Microscore,
    MicroscoreSuite,
    RiskResult,
    compute_risk,
    compute_spanning_set,
    generate_microscores,
)
from .models import Geography, ScenarioMetadata, TransformContext
from .spatial import lookup_continent, lookup_geography, lookup_ipcc_region
from .transforms import (
    CallableTransform,
    ImpactRegistry,
    LinearImpact,
    PiecewiseLinearImpact,
    SigmoidImpact,
    impacts,
)

__all__ = [
    "BinaryOutcome",
    "CallableTransform",
    "Distribution",
    "EmpiricalDistribution",
    "FitDiagnostics",
    "FitConstraints",
    "FitResult",
    "FittedDistribution",
    "Geography",
    "HORIZONS",
    "ImpactRegistry",
    "LinearImpact",
    "Microscore",
    "MicroscoreSuite",
    "PATHWAYS",
    "Pathway",
    "PiecewiseLinearImpact",
    "RISK_FACTORS",
    "RiskFactor",
    "RiskResult",
    "ScenarioMetadata",
    "SigmoidImpact",
    "TabulatedDistribution",
    "TransformContext",
    "compute_risk",
    "compute_spanning_set",
    "fit_all",
    "fit_distribution",
    "generate_microscores",
    "impacts",
    "lookup_continent",
    "lookup_geography",
    "lookup_ipcc_region",
    "quality_metrics",
]

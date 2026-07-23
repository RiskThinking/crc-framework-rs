from typing import Dict, List, Optional, Sequence, Tuple

class NativeDistribution:
    family: Optional[str]
    shape: Optional[float]
    location: Optional[float]
    scale: Optional[float]
    def pdf(self, x: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def ppf(self, probability: float) -> float: ...
    def quantiles(self, probabilities: Sequence[float]) -> List[float]: ...
    def sample(self, size: int, seed: Optional[int] = ...) -> List[float]: ...

class NativeFitResult:
    distribution: NativeDistribution
    ks_statistic: float
    ks_pvalue: float
    rmse: float
    r_squared: float

class NativeMicroscore:
    probability: float
    exposure: float
    impact: Optional[float]
    probability_below: float
    probability_above: float
    impact_probability_below: Optional[float]
    impact_probability_above: Optional[float]

class NativeBinaryOutcome:
    factor: str
    downside_probability: float
    downside_impact: float
    upside_probability: float
    upside_impact: float
    weight: float
    def __init__(
        self,
        factor: str,
        downside_probability: float,
        downside_impact: float,
        upside_probability: Optional[float] = ...,
        upside_impact: float = ...,
        weight: float = ...,
    ) -> None: ...

class NativeMicroscoreSuite:
    scores: List[NativeMicroscore]
    def at(self, probability: float) -> NativeBinaryOutcome: ...
    def exposure_statistics(self) -> Dict[str, float]: ...
    def impact_statistics(self) -> Optional[Dict[str, float]]: ...

class NativeSpanningBranch:
    choices: List[bool]
    probability: float
    impact: float
    factor_impacts: List[Tuple[str, float]]

class NativeRiskAttribution:
    factor: str
    var_impact: float
    cvar_impact: float

class NativeRiskLevel:
    probability: float
    var: float
    cvar: float
    attribution: List[NativeRiskAttribution]

class NativeRiskResult:
    levels: List[NativeRiskLevel]
    branch_count: int

def empirical_distribution(samples: Sequence[float]) -> NativeDistribution: ...
def tabulated_distribution(
    probabilities: Sequence[float],
    values: Sequence[float],
    interpolation: str = ...,
    extrapolate: bool = ...,
) -> NativeDistribution: ...
def return_period_distribution(
    periods: Sequence[float],
    values: Sequence[float],
    tail: str,
    interpolation: str = ...,
    extrapolate: bool = ...,
) -> NativeDistribution: ...
def fitted_distribution(
    family: str, location: float, scale: float, shape: Optional[float] = ...
) -> NativeDistribution: ...
def fit(samples: Sequence[float], family: Optional[str] = ...) -> NativeFitResult: ...
def fit_candidates(samples: Sequence[float]) -> List[NativeFitResult]: ...
def diagnostic_metrics(
    samples: Sequence[float], distribution: NativeDistribution
) -> Dict[str, float]: ...
def apply_impact(
    distribution: NativeDistribution,
    probabilities: Sequence[float],
    factor: str,
    cell: Optional[int] = ...,
    country: Optional[str] = ...,
    continent: Optional[str] = ...,
    building_type: Optional[str] = ...,
    historic_mean: Optional[float] = ...,
    overrides: Optional[Dict[str, float]] = ...,
) -> NativeDistribution: ...
def microscores(
    exposure: NativeDistribution,
    probabilities: Sequence[float],
    impact: Optional[NativeDistribution] = ...,
    cell: Optional[int] = ...,
    factor: Optional[str] = ...,
    pathway: Optional[str] = ...,
    horizon: Optional[int] = ...,
    country: Optional[str] = ...,
    continent: Optional[str] = ...,
    building_type: Optional[str] = ...,
) -> NativeMicroscoreSuite: ...
def spanning_set(
    outcomes: Sequence[NativeBinaryOutcome], max_branches: Optional[int] = ...
) -> List[NativeSpanningBranch]: ...
def risk(
    outcomes: Sequence[NativeBinaryOutcome],
    levels: Sequence[float],
    max_branches: Optional[int] = ...,
) -> NativeRiskResult: ...
def lookup_ipcc_region(cell: object) -> Optional[str]: ...
def lookup_continent(cell: object) -> Optional[str]: ...
def lookup_geography(cell: object) -> Optional[Dict[str, object]]: ...
def pathways() -> List[Tuple[int, str]]: ...
def horizons() -> List[int]: ...
def risk_factors() -> List[Tuple[int, str]]: ...

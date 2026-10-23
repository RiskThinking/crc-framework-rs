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

class NativeQuantileFitResult:
    distribution: NativeDistribution
    rmse: float
    normalized_rmse: float
    weighted_r_squared: float
    maximum_absolute_residual: float
    point_count: int
    converged: bool
    iterations: int
    evaluations: int

class NativeCensoredQuantileFitResult:
    distribution: NativeDistribution
    rmse: float
    normalized_rmse: float
    weighted_r_squared: float
    maximum_absolute_residual: float
    point_count: int
    converged: bool
    iterations: int
    evaluations: int
    negative_log_likelihood: float
    residual_scale: float
    residual_scale_floor: float
    censor_location: float
    censored_point_count: int
    uncensored_point_count: int

class NativeHurdleQuantileFitResult:
    distribution: NativeDistribution
    base_distribution: NativeDistribution
    atom_location: float
    atom_probability: float
    tail_rmse: float
    tail_normalized_rmse: float
    tail_weighted_r_squared: float
    tail_maximum_absolute_residual: float
    tail_point_count: int
    converged: bool
    iterations: int
    evaluations: int
    atom_probability_lower_bound: float
    atom_probability_upper_bound: float
    atom_point_count: int

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
def point_mass_distribution(location: float) -> NativeDistribution: ...
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
def hurdle_distribution(
    base: NativeDistribution,
    atom_probability: float,
    atom_location: float = ...,
) -> NativeDistribution: ...
def fit(samples: Sequence[float], family: Optional[str] = ..., *, method: str = ...) -> NativeFitResult: ...
def fit_candidates(samples: Sequence[float]) -> List[NativeFitResult]: ...
def fit_quantiles(
    probabilities: Sequence[float],
    values: Sequence[float],
    family: str,
    weights: Optional[Sequence[float]] = ...,
) -> NativeQuantileFitResult: ...
def fit_censored_quantiles(
    probabilities: Sequence[float],
    values: Sequence[float],
    family: str,
    censor_location: float = ...,
    residual_scale_floor_ratio: float = ...,
    weights: Optional[Sequence[float]] = ...,
) -> NativeCensoredQuantileFitResult: ...
def fit_hurdle_quantiles(
    probabilities: Sequence[float],
    values: Sequence[float],
    family: str,
    atom_probability: float,
    atom_location: float = ...,
    weights: Optional[Sequence[float]] = ...,
) -> NativeHurdleQuantileFitResult: ...
def diagnostic_metrics(
    samples: Sequence[float], distribution: NativeDistribution
) -> Dict[str, float]: ...
def evaluate_impact(
    values: Sequence[float],
    factor: str,
    cell: Optional[int] = ...,
    country: Optional[str] = ...,
    continent: Optional[str] = ...,
    building_type: Optional[str] = ...,
    historic_mean: Optional[float] = ...,
    overrides: Optional[Dict[str, float]] = ...,
) -> List[float]: ...
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

class ImpactError(Exception):
    code: str
    path: str

class NativeImpactExpression:
    definition_hash: str
    model_kind: str
    monotonicity: str
    @staticmethod
    def from_json(text: str) -> NativeImpactExpression: ...
    def canonical_json(self) -> str: ...
    def evaluate_value(self, value: float) -> float: ...
    def evaluate_values(self, values: Sequence[float]) -> List[float]: ...
    def expected_value(
        self,
        source: NativeImpactSource,
        requested: str,
        absolute_tolerance: Optional[float] = ...,
        relative_tolerance: Optional[float] = ...,
        max_evaluations: Optional[int] = ...,
    ) -> NativeExpectation: ...
    def transform_distribution(self, source: NativeImpactSource) -> NativeImpactDistribution: ...
    def point_record(
        self,
        shape: Sequence[int],
        values: Sequence[float],
        snapshot_uri: Optional[str] = ...,
        snapshot_revision: Optional[str] = ...,
        snapshot_sha256: Optional[str] = ...,
        snapshot_retention: Optional[str] = ...,
    ) -> NativeImpactEvaluationRecord: ...
    def expectation_record(
        self,
        source: NativeImpactSource,
        expectation: NativeExpectation,
        snapshot_uri: Optional[str] = ...,
        snapshot_revision: Optional[str] = ...,
        snapshot_sha256: Optional[str] = ...,
        snapshot_retention: Optional[str] = ...,
    ) -> NativeImpactEvaluationRecord: ...
    def transform_record(
        self,
        source: NativeImpactSource,
        snapshot_uri: Optional[str] = ...,
        snapshot_revision: Optional[str] = ...,
        snapshot_sha256: Optional[str] = ...,
        snapshot_retention: Optional[str] = ...,
    ) -> NativeImpactEvaluationRecord: ...

class NativeImpactSource:
    original_mass_sum: Optional[float]
    normalization_factor: Optional[float]
    @staticmethod
    def histogram(edges: Sequence[float], weights: Sequence[float]) -> NativeImpactSource: ...
    @staticmethod
    def discrete(
        support: Sequence[float], probabilities: Sequence[float]
    ) -> NativeImpactSource: ...
    @staticmethod
    def empirical(samples: Sequence[float]) -> NativeImpactSource: ...
    @staticmethod
    def curve(
        curve_kind: str,
        curve_type: str,
        curve_shape: Optional[float] = ...,
        curve_location: Optional[float] = ...,
        curve_scale: Optional[float] = ...,
        curve_atom_probability: Optional[float] = ...,
        curve_atom_location: Optional[float] = ...,
        curve_probabilities: Optional[Sequence[float]] = ...,
        curve_values: Optional[Sequence[float]] = ...,
    ) -> NativeImpactSource: ...

class NativeExpectation:
    value: float
    requested: str
    method: str
    method_version: str
    absolute_tolerance: Optional[float]
    relative_tolerance: Optional[float]
    max_evaluations: Optional[int]
    evaluations: int
    error_estimate: float
    exact_component: float
    numerical_component: float
    original_mass_sum: Optional[float]
    normalization_factor: Optional[float]

class NativeImpactDistribution:
    definition_hash: str
    algorithm: str
    def atoms(self) -> List[Tuple[float, float]]: ...
    def quantile(self, probability: float) -> float: ...
    def quantiles(self, probabilities: Sequence[float]) -> List[float]: ...
    def tabulate(self, probabilities: Sequence[float], interpolation: str) -> List[float]: ...

class NativeImpactEvaluationRecord:
    def canonical_json(self) -> str: ...

class NativeImpactReference:
    reference_hash: str
    @staticmethod
    def from_json(text: str) -> NativeImpactReference: ...
    def canonical_json(self) -> str: ...
    def resolve(
        self,
        cell: Optional[str] = ...,
        country: Optional[str] = ...,
        continent: Optional[str] = ...,
        building_type: Optional[str] = ...,
        historic_mean: Optional[float] = ...,
    ) -> NativeImpactExpression: ...

def impact_registry_identity() -> Tuple[str, str, str, str]: ...
def impact_engine() -> Tuple[str, str]: ...
def impact_array_digest(shape: Sequence[int], values: Sequence[float]) -> str: ...

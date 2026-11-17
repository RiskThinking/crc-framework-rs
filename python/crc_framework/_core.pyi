from collections.abc import Sequence

__version__: str

def validate_document(schema: str, text: str) -> None: ...

class NativeNumericalDocument:
    def __init__(self, text: str) -> None: ...
    def quantile(self, probability: float) -> float: ...

class NativeNumericalLaw:
    def __init__(self, text: str) -> None: ...
    def support(self) -> tuple[float, float]: ...
    def quantile(self, values: Sequence[float]) -> list[float]: ...
    def cdf(self, values: Sequence[float], inclusive: bool) -> list[float]: ...
    def mass(self, values: Sequence[float]) -> list[float]: ...
    def exceedance(self, values: Sequence[float], inclusive: bool) -> list[float]: ...
    def inverse_survival(self, probability: float) -> float: ...
    def tail_integral(
        self, lo: float, hi: float, center: float, square: bool
    ) -> float: ...
    def integral(self, lo: float, hi: float, center: float, square: bool) -> float: ...
    def moments(self) -> tuple[float, float]: ...

def rank_risk(
    outcomes: Sequence[tuple[str, float, float, float, float]],
    levels: Sequence[float],
    max_branches: int | None = ...,
) -> tuple[int, list[tuple[float, float, float, list[tuple[str, float, float]]]]]: ...

class NativeFitParameters:
    family: str
    shape: float | None
    location: float
    scale: float

class NativeFitResult:
    distribution: NativeFitParameters
    ks_statistic: float
    ks_pvalue: float
    rmse: float
    r_squared: float
    iterations: int
    evaluations: int

class NativeQuantileFitResult:
    distribution: NativeFitParameters
    rmse: float
    weighted_r_squared: float
    maximum_absolute_residual: float
    iterations: int
    evaluations: int

class NativeCensoredQuantileFitResult:
    distribution: NativeFitParameters
    rmse: float
    weighted_r_squared: float
    maximum_absolute_residual: float
    iterations: int
    evaluations: int
    negative_log_likelihood: float
    residual_scale: float
    residual_scale_floor: float
    censored_point_count: int
    uncensored_point_count: int

class NativeHurdleQuantileFitResult:
    base_distribution: NativeFitParameters
    tail_rmse: float
    tail_weighted_r_squared: float
    tail_maximum_absolute_residual: float
    tail_point_count: int
    iterations: int
    evaluations: int
    atom_point_count: int

def fit_samples(
    samples: Sequence[float],
    family: str,
    estimator: str,
    *,
    minimum_value: float | None = ...,
    maximum_value: float | None = ...,
    probability: float | None = ...,
    require_sample_support: bool = ...,
) -> NativeFitResult: ...
def fit_quantiles(
    probabilities: Sequence[float],
    values: Sequence[float],
    family: str,
    weights: Sequence[float] | None = ...,
    *,
    minimum_value: float | None = ...,
    maximum_value: float | None = ...,
    probability: float | None = ...,
    require_sample_support: bool = ...,
) -> NativeQuantileFitResult: ...
def fit_censored_quantiles(
    probabilities: Sequence[float],
    values: Sequence[float],
    family: str,
    censor_location: float = ...,
    residual_scale_floor_ratio: float = ...,
    weights: Sequence[float] | None = ...,
    *,
    minimum_value: float | None = ...,
    maximum_value: float | None = ...,
    probability: float | None = ...,
    require_sample_support: bool = ...,
) -> NativeCensoredQuantileFitResult: ...
def fit_hurdle_quantiles(
    probabilities: Sequence[float],
    values: Sequence[float],
    family: str,
    atom_probability: float,
    atom_location: float = ...,
    weights: Sequence[float] | None = ...,
    *,
    minimum_value: float | None = ...,
    maximum_value: float | None = ...,
    probability: float | None = ...,
    require_sample_support: bool = ...,
) -> NativeHurdleQuantileFitResult: ...
def lookup_ipcc_region(cell: object) -> str | None: ...
def lookup_continent(cell: object) -> str | None: ...
def lookup_geography(cell: object) -> dict[str, object] | None: ...

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
    def evaluate_values(self, values: Sequence[float]) -> list[float]: ...
    def expected_value(
        self,
        source: NativeImpactSource,
        requested: str,
        absolute_tolerance: float | None = ...,
        relative_tolerance: float | None = ...,
        max_evaluations: int | None = ...,
    ) -> NativeExpectation: ...
    def transform_distribution(
        self, source: NativeImpactSource
    ) -> NativeImpactDistribution: ...
    def point_record(
        self,
        shape: Sequence[int],
        values: Sequence[float],
        snapshot_uri: str | None = ...,
        snapshot_revision: str | None = ...,
        snapshot_sha256: str | None = ...,
        snapshot_retention: str | None = ...,
    ) -> NativeImpactEvaluationRecord: ...
    def expectation_record(
        self,
        source: NativeImpactSource,
        expectation: NativeExpectation,
        snapshot_uri: str | None = ...,
        snapshot_revision: str | None = ...,
        snapshot_sha256: str | None = ...,
        snapshot_retention: str | None = ...,
    ) -> NativeImpactEvaluationRecord: ...
    def transform_record(
        self,
        source: NativeImpactSource,
        snapshot_uri: str | None = ...,
        snapshot_revision: str | None = ...,
        snapshot_sha256: str | None = ...,
        snapshot_retention: str | None = ...,
    ) -> NativeImpactEvaluationRecord: ...

class NativeImpactSource:
    original_mass_sum: float | None
    normalization_factor: float | None
    @staticmethod
    def histogram(
        edges: Sequence[float], weights: Sequence[float]
    ) -> NativeImpactSource: ...
    @staticmethod
    def segments(
        probabilities: Sequence[float], values: Sequence[float]
    ) -> NativeImpactSource: ...
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
        curve_shape: float | None = ...,
        curve_location: float | None = ...,
        curve_scale: float | None = ...,
        curve_atom_probability: float | None = ...,
        curve_atom_location: float | None = ...,
        curve_probabilities: Sequence[float] | None = ...,
        curve_values: Sequence[float] | None = ...,
    ) -> NativeImpactSource: ...

class NativeExpectation:
    value: float
    requested: str
    method: str
    method_version: str
    absolute_tolerance: float | None
    relative_tolerance: float | None
    max_evaluations: int | None
    evaluations: int
    error_estimate: float
    exact_component: float
    numerical_component: float
    original_mass_sum: float | None
    normalization_factor: float | None

class NativeImpactDistribution:
    definition_hash: str
    algorithm: str
    def atoms(self) -> list[tuple[float, float]]: ...
    def integral(self, lo: float, hi: float, center: float, square: bool) -> float: ...
    def quantile(self, probability: float) -> float: ...
    def inverse_survival(self, probability: float) -> float: ...
    def tail_integral(
        self, lo: float, hi: float, center: float, square: bool
    ) -> float: ...
    def quantiles(self, probabilities: Sequence[float]) -> list[float]: ...
    def tabulate(
        self, probabilities: Sequence[float], interpolation: str
    ) -> list[float]: ...

class NativeImpactEvaluationRecord:
    def canonical_json(self) -> str: ...

class NativeImpactReference:
    reference_hash: str
    @staticmethod
    def from_json(text: str) -> NativeImpactReference: ...
    def canonical_json(self) -> str: ...
    def resolve(
        self,
        cell: str | None = ...,
        country: str | None = ...,
        continent: str | None = ...,
        building_type: str | None = ...,
        historic_mean: float | None = ...,
    ) -> NativeImpactExpression: ...

def impact_registry_identity() -> tuple[str, str, str, str]: ...
def impact_registry_manifest_json() -> str: ...
def impact_engine() -> tuple[str, str]: ...
def impact_array_digest(shape: Sequence[int], values: Sequence[float]) -> str: ...

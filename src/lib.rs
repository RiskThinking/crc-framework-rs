use std::{collections::HashMap, sync::Arc};

use crc_framework_core::{
    BinaryOutcome, Distribution, DistributionFamily, EmpiricalDistribution, FittedDistribution,
    HurdleDistribution, ImpactContext, ImpactRegistry, Interpolation, PointMassDistribution,
    ScenarioMetadata, TabulatedDistribution, Tail, Transform, compute_risk, compute_spanning_set,
    fit_hurdle_quantiles as core_fit_hurdle_quantiles, fit_quantiles as core_fit_quantiles,
    generate_microscores,
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyDict},
};

fn py_error(error: crc_framework_core::CrcError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(name = "NativeDistribution", frozen)]
#[derive(Clone)]
struct PyDistribution {
    inner: Arc<dyn Distribution>,
    family: Option<String>,
    shape: Option<f64>,
    location: Option<f64>,
    scale: Option<f64>,
}

#[pymethods]
impl PyDistribution {
    fn pdf(&self, x: f64) -> f64 {
        self.inner.pdf(x)
    }

    fn cdf(&self, x: f64) -> f64 {
        self.inner.cdf(x)
    }

    fn ppf(&self, probability: f64) -> PyResult<f64> {
        self.inner.ppf(probability).map_err(py_error)
    }

    fn quantiles(&self, probabilities: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.quantiles(&probabilities).map_err(py_error)
    }

    #[pyo3(signature = (size, seed=None))]
    fn sample(&self, size: usize, seed: Option<u64>) -> PyResult<Vec<f64>> {
        self.inner.sample(size, seed).map_err(py_error)
    }

    #[getter]
    fn family(&self) -> Option<String> {
        self.family.clone()
    }

    #[getter]
    fn shape(&self) -> Option<f64> {
        self.shape
    }

    #[getter]
    fn location(&self) -> Option<f64> {
        self.location
    }

    #[getter]
    fn scale(&self) -> Option<f64> {
        self.scale
    }
}

#[pyclass(name = "NativeFitResult", frozen)]
#[derive(Clone)]
struct PyFitResult {
    #[pyo3(get)]
    distribution: PyDistribution,
    #[pyo3(get)]
    ks_statistic: f64,
    #[pyo3(get)]
    ks_pvalue: f64,
    #[pyo3(get)]
    rmse: f64,
    #[pyo3(get)]
    r_squared: f64,
}

#[pyclass(name = "NativeQuantileFitResult", frozen)]
#[derive(Clone)]
struct PyQuantileFitResult {
    #[pyo3(get)]
    distribution: PyDistribution,
    #[pyo3(get)]
    rmse: f64,
    #[pyo3(get)]
    normalized_rmse: f64,
    #[pyo3(get)]
    weighted_r_squared: f64,
    #[pyo3(get)]
    maximum_absolute_residual: f64,
    #[pyo3(get)]
    point_count: usize,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    evaluations: usize,
}

#[pyclass(name = "NativeHurdleQuantileFitResult", frozen)]
#[derive(Clone)]
struct PyHurdleQuantileFitResult {
    #[pyo3(get)]
    distribution: PyDistribution,
    #[pyo3(get)]
    base_distribution: PyDistribution,
    #[pyo3(get)]
    atom_location: f64,
    #[pyo3(get)]
    atom_probability: f64,
    #[pyo3(get)]
    tail_rmse: f64,
    #[pyo3(get)]
    tail_normalized_rmse: f64,
    #[pyo3(get)]
    tail_weighted_r_squared: f64,
    #[pyo3(get)]
    tail_maximum_absolute_residual: f64,
    #[pyo3(get)]
    tail_point_count: usize,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    evaluations: usize,
    #[pyo3(get)]
    atom_probability_lower_bound: f64,
    #[pyo3(get)]
    atom_probability_upper_bound: f64,
    #[pyo3(get)]
    atom_point_count: usize,
}

fn fitted_py(distribution: FittedDistribution) -> PyDistribution {
    PyDistribution {
        family: Some(distribution.family.name().to_owned()),
        shape: distribution.shape,
        location: Some(distribution.location),
        scale: Some(distribution.scale),
        inner: Arc::new(distribution),
    }
}

fn fitted_from_py(distribution: &PyDistribution) -> PyResult<FittedDistribution> {
    let family_name = distribution
        .family
        .as_deref()
        .ok_or_else(|| PyValueError::new_err("base distribution must be parametric"))?;
    let family = DistributionFamily::from_name(family_name).ok_or_else(|| {
        PyValueError::new_err(format!("unknown distribution family {family_name}"))
    })?;
    FittedDistribution::from_parameters(
        family,
        distribution.shape,
        distribution
            .location
            .ok_or_else(|| PyValueError::new_err("base distribution is missing location"))?,
        distribution
            .scale
            .ok_or_else(|| PyValueError::new_err("base distribution is missing scale"))?,
    )
    .map_err(py_error)
}

fn fit_result_py(result: crc_framework_core::FitResult) -> PyFitResult {
    PyFitResult {
        distribution: fitted_py(result.distribution),
        ks_statistic: result.diagnostics.ks_statistic,
        ks_pvalue: result.diagnostics.ks_pvalue,
        rmse: result.diagnostics.rmse,
        r_squared: result.diagnostics.r_squared,
    }
}

fn quantile_fit_result_py(result: crc_framework_core::QuantileFitResult) -> PyQuantileFitResult {
    PyQuantileFitResult {
        distribution: fitted_py(result.distribution),
        rmse: result.diagnostics.rmse,
        normalized_rmse: result.diagnostics.normalized_rmse,
        weighted_r_squared: result.diagnostics.weighted_r_squared,
        maximum_absolute_residual: result.diagnostics.maximum_absolute_residual,
        point_count: result.diagnostics.point_count,
        converged: result.diagnostics.converged,
        iterations: result.diagnostics.iterations,
        evaluations: result.diagnostics.evaluations,
    }
}

fn hurdle_py(distribution: HurdleDistribution) -> PyDistribution {
    PyDistribution {
        inner: Arc::new(distribution),
        family: None,
        shape: None,
        location: None,
        scale: None,
    }
}

fn hurdle_quantile_fit_result_py(
    result: crc_framework_core::HurdleQuantileFitResult,
) -> PyHurdleQuantileFitResult {
    let atom_location = result.distribution.atom_location();
    let atom_probability = result.distribution.atom_probability();
    let base = result.distribution.base().clone();
    let tail = result.diagnostics.tail;
    PyHurdleQuantileFitResult {
        distribution: hurdle_py(result.distribution),
        base_distribution: fitted_py(base),
        atom_location,
        atom_probability,
        tail_rmse: tail.rmse,
        tail_normalized_rmse: tail.normalized_rmse,
        tail_weighted_r_squared: tail.weighted_r_squared,
        tail_maximum_absolute_residual: tail.maximum_absolute_residual,
        tail_point_count: tail.point_count,
        converged: tail.converged,
        iterations: tail.iterations,
        evaluations: tail.evaluations,
        atom_probability_lower_bound: result.diagnostics.atom_probability_lower_bound,
        atom_probability_upper_bound: result.diagnostics.atom_probability_upper_bound,
        atom_point_count: result.diagnostics.atom_point_count,
    }
}

#[pyfunction]
fn empirical_distribution(samples: Vec<f64>) -> PyResult<PyDistribution> {
    Ok(PyDistribution {
        inner: Arc::new(EmpiricalDistribution::new(samples).map_err(py_error)?),
        family: None,
        shape: None,
        location: None,
        scale: None,
    })
}

#[pyfunction]
fn point_mass_distribution(location: f64) -> PyResult<PyDistribution> {
    let distribution = PointMassDistribution::new(location).map_err(py_error)?;
    Ok(PyDistribution {
        inner: Arc::new(distribution),
        family: None,
        shape: None,
        location: Some(location),
        scale: None,
    })
}

fn interpolation(value: &str) -> PyResult<Interpolation> {
    match value {
        "linear_probability" => Ok(Interpolation::LinearProbability),
        "log_return_period" => Ok(Interpolation::LogReturnPeriod),
        _ => Err(PyValueError::new_err(format!(
            "unknown interpolation {value}"
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (probabilities, values, interpolation="linear_probability", extrapolate=false))]
fn tabulated_distribution(
    probabilities: Vec<f64>,
    values: Vec<f64>,
    interpolation: &str,
    extrapolate: bool,
) -> PyResult<PyDistribution> {
    Ok(PyDistribution {
        inner: Arc::new(
            TabulatedDistribution::new(
                probabilities,
                values,
                self::interpolation(interpolation)?,
                extrapolate,
            )
            .map_err(py_error)?,
        ),
        family: None,
        shape: None,
        location: None,
        scale: None,
    })
}

#[pyfunction]
#[pyo3(signature = (periods, values, tail, interpolation="log_return_period", extrapolate=false))]
fn return_period_distribution(
    periods: Vec<f64>,
    values: Vec<f64>,
    tail: &str,
    interpolation: &str,
    extrapolate: bool,
) -> PyResult<PyDistribution> {
    let tail = match tail {
        "upper" => Tail::Upper,
        "lower" => Tail::Lower,
        _ => return Err(PyValueError::new_err("tail must be 'upper' or 'lower'")),
    };
    Ok(PyDistribution {
        inner: Arc::new(
            TabulatedDistribution::from_return_periods(
                periods,
                values,
                tail,
                self::interpolation(interpolation)?,
                extrapolate,
            )
            .map_err(py_error)?,
        ),
        family: None,
        shape: None,
        location: None,
        scale: None,
    })
}

#[pyfunction]
#[pyo3(signature = (family, location, scale, shape=None))]
fn fitted_distribution(
    family: &str,
    location: f64,
    scale: f64,
    shape: Option<f64>,
) -> PyResult<PyDistribution> {
    let family = DistributionFamily::from_name(family)
        .ok_or_else(|| PyValueError::new_err(format!("unknown distribution family {family}")))?;
    Ok(fitted_py(
        FittedDistribution::from_parameters(family, shape, location, scale).map_err(py_error)?,
    ))
}

#[pyfunction]
#[pyo3(signature = (base, atom_probability, atom_location=0.0))]
fn hurdle_distribution(
    base: PyRef<'_, PyDistribution>,
    atom_probability: f64,
    atom_location: f64,
) -> PyResult<PyDistribution> {
    let base = fitted_from_py(&base)?;
    HurdleDistribution::new(atom_location, atom_probability, base)
        .map(hurdle_py)
        .map_err(py_error)
}

#[pyfunction]
#[pyo3(signature = (samples, family=None))]
fn fit(samples: Vec<f64>, family: Option<&str>) -> PyResult<PyFitResult> {
    let family = family
        .map(|name| {
            DistributionFamily::from_name(name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown distribution family {name}")))
        })
        .transpose()?;
    crc_framework_core::distribution::fit_distribution(&samples, family)
        .map(fit_result_py)
        .map_err(py_error)
}

#[pyfunction]
fn fit_candidates(samples: Vec<f64>) -> PyResult<Vec<PyFitResult>> {
    crc_framework_core::distribution::fit_all(&samples)
        .map(|results| results.into_iter().map(fit_result_py).collect())
        .map_err(py_error)
}

#[pyfunction]
#[pyo3(signature = (probabilities, values, family, weights=None))]
fn fit_quantiles(
    py: Python<'_>,
    probabilities: Vec<f64>,
    values: Vec<f64>,
    family: &str,
    weights: Option<Vec<f64>>,
) -> PyResult<PyQuantileFitResult> {
    let family = DistributionFamily::from_name(family)
        .ok_or_else(|| PyValueError::new_err(format!("unknown distribution family {family}")))?;
    py.detach(move || core_fit_quantiles(&probabilities, &values, weights.as_deref(), family))
        .map(quantile_fit_result_py)
        .map_err(py_error)
}

#[pyfunction]
#[pyo3(signature = (
    probabilities,
    values,
    family,
    atom_probability,
    atom_location=0.0,
    weights=None
))]
fn fit_hurdle_quantiles(
    py: Python<'_>,
    probabilities: Vec<f64>,
    values: Vec<f64>,
    family: &str,
    atom_probability: f64,
    atom_location: f64,
    weights: Option<Vec<f64>>,
) -> PyResult<PyHurdleQuantileFitResult> {
    let family = DistributionFamily::from_name(family)
        .ok_or_else(|| PyValueError::new_err(format!("unknown distribution family {family}")))?;
    py.detach(move || {
        core_fit_hurdle_quantiles(
            &probabilities,
            &values,
            weights.as_deref(),
            family,
            atom_location,
            atom_probability,
        )
    })
    .map(hurdle_quantile_fit_result_py)
    .map_err(py_error)
}

#[pyfunction]
fn diagnostic_metrics(
    samples: Vec<f64>,
    distribution: PyRef<'_, PyDistribution>,
) -> PyResult<HashMap<&'static str, f64>> {
    crc_framework_core::distribution::calculate_diagnostics(&samples, distribution.inner.as_ref())
        .map(|metrics| {
            HashMap::from([
                ("ks_statistic", metrics.ks_statistic),
                ("ks_pvalue", metrics.ks_pvalue),
                ("rmse", metrics.rmse),
                ("r_squared", metrics.r_squared),
            ])
        })
        .map_err(py_error)
}

#[pyfunction]
#[pyo3(signature = (
    values,
    factor,
    cell=None,
    country=None,
    continent=None,
    building_type=None,
    historic_mean=None,
    overrides=None
))]
#[allow(clippy::too_many_arguments)]
fn evaluate_impact(
    values: Vec<f64>,
    factor: String,
    cell: Option<u64>,
    country: Option<String>,
    continent: Option<String>,
    building_type: Option<String>,
    historic_mean: Option<f64>,
    overrides: Option<HashMap<String, f64>>,
) -> PyResult<Vec<f64>> {
    let context = ImpactContext {
        factor,
        cell,
        country,
        continent,
        building_type,
        historic_mean,
    };
    ImpactRegistry::evaluate(&context, &overrides.unwrap_or_default(), &values).map_err(py_error)
}

#[pyfunction]
#[pyo3(signature = (
    distribution,
    probabilities,
    factor,
    cell=None,
    country=None,
    continent=None,
    building_type=None,
    historic_mean=None,
    overrides=None
))]
#[allow(clippy::too_many_arguments)]
fn apply_impact(
    distribution: PyRef<'_, PyDistribution>,
    probabilities: Vec<f64>,
    factor: String,
    cell: Option<u64>,
    country: Option<String>,
    continent: Option<String>,
    building_type: Option<String>,
    historic_mean: Option<f64>,
    overrides: Option<HashMap<String, f64>>,
) -> PyResult<PyDistribution> {
    let context = ImpactContext {
        factor,
        cell,
        country,
        continent,
        building_type,
        historic_mean,
    };
    let transform =
        ImpactRegistry::resolve(&context, &overrides.unwrap_or_default()).map_err(py_error)?;
    let result = transform
        .apply(distribution.inner.as_ref(), &probabilities)
        .map_err(py_error)?;
    Ok(PyDistribution {
        inner: Arc::new(result),
        family: None,
        shape: None,
        location: None,
        scale: None,
    })
}

#[pyclass(name = "NativeMicroscore", frozen)]
#[derive(Clone)]
struct PyMicroscore {
    #[pyo3(get)]
    probability: f64,
    #[pyo3(get)]
    exposure: f64,
    #[pyo3(get)]
    impact: Option<f64>,
    #[pyo3(get)]
    probability_below: f64,
    #[pyo3(get)]
    probability_above: f64,
    #[pyo3(get)]
    impact_probability_below: Option<f64>,
    #[pyo3(get)]
    impact_probability_above: Option<f64>,
}

#[pyclass(name = "NativeBinaryOutcome", frozen)]
#[derive(Clone)]
struct PyBinaryOutcome {
    inner: BinaryOutcome,
}

#[pymethods]
impl PyBinaryOutcome {
    #[new]
    #[pyo3(signature = (
        factor,
        downside_probability,
        downside_impact,
        upside_probability=None,
        upside_impact=0.0,
        weight=1.0
    ))]
    fn new(
        factor: String,
        downside_probability: f64,
        downside_impact: f64,
        upside_probability: Option<f64>,
        upside_impact: f64,
        weight: f64,
    ) -> Self {
        Self {
            inner: BinaryOutcome {
                factor,
                downside_probability,
                downside_impact,
                upside_probability: upside_probability.unwrap_or(1.0 - downside_probability),
                upside_impact,
                weight,
            },
        }
    }

    #[getter]
    fn factor(&self) -> &str {
        &self.inner.factor
    }

    #[getter]
    fn downside_probability(&self) -> f64 {
        self.inner.downside_probability
    }

    #[getter]
    fn downside_impact(&self) -> f64 {
        self.inner.downside_impact
    }

    #[getter]
    fn upside_probability(&self) -> f64 {
        self.inner.upside_probability
    }

    #[getter]
    fn upside_impact(&self) -> f64 {
        self.inner.upside_impact
    }

    #[getter]
    fn weight(&self) -> f64 {
        self.inner.weight
    }
}

#[pyclass(name = "NativeMicroscoreSuite", frozen)]
struct PyMicroscoreSuite {
    suite: crc_framework_core::MicroscoreSuite,
}

#[pymethods]
impl PyMicroscoreSuite {
    #[getter]
    fn scores(&self) -> Vec<PyMicroscore> {
        self.suite
            .scores
            .iter()
            .map(|score| PyMicroscore {
                probability: score.probability,
                exposure: score.exposure,
                impact: score.impact,
                probability_below: score.probability_below,
                probability_above: score.probability_above,
                impact_probability_below: score.impact_probability_below,
                impact_probability_above: score.impact_probability_above,
            })
            .collect()
    }

    fn at(&self, probability: f64) -> PyResult<PyBinaryOutcome> {
        self.suite
            .at(probability)
            .map(|inner| PyBinaryOutcome { inner })
            .map_err(py_error)
    }

    fn exposure_statistics(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        statistics_dict(py, self.suite.exposure_statistics)
    }

    fn impact_statistics(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.suite
            .impact_statistics
            .map(|value| statistics_dict(py, value))
            .transpose()
    }
}

fn statistics_dict(
    py: Python<'_>,
    statistics: crc_framework_core::DistributionStatistics,
) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    result.set_item("mean", statistics.mean)?;
    result.set_item("standard_deviation", statistics.standard_deviation)?;
    result.set_item("minimum", statistics.minimum)?;
    result.set_item("maximum", statistics.maximum)?;
    result.set_item("cvar_95", statistics.cvar_95)?;
    Ok(result.unbind())
}

#[pyfunction]
#[pyo3(signature = (
    exposure,
    probabilities,
    impact=None,
    cell=None,
    factor=None,
    pathway=None,
    horizon=None,
    country=None,
    continent=None,
    building_type=None
))]
#[allow(clippy::too_many_arguments)]
fn microscores(
    exposure: PyRef<'_, PyDistribution>,
    probabilities: Vec<f64>,
    impact: Option<PyRef<'_, PyDistribution>>,
    cell: Option<u64>,
    factor: Option<String>,
    pathway: Option<String>,
    horizon: Option<i32>,
    country: Option<String>,
    continent: Option<String>,
    building_type: Option<String>,
) -> PyResult<PyMicroscoreSuite> {
    let impact_ref = impact
        .as_ref()
        .map(|value| value.inner.as_ref() as &dyn Distribution);
    let metadata = ScenarioMetadata {
        cell,
        factor,
        pathway,
        horizon,
        country,
        continent,
        building_type,
    };
    generate_microscores(
        exposure.inner.as_ref(),
        impact_ref,
        &probabilities,
        metadata,
    )
    .map(|suite| PyMicroscoreSuite { suite })
    .map_err(py_error)
}

#[pyfunction]
#[pyo3(signature = (outcomes, max_branches=None))]
fn spanning_set(
    outcomes: Vec<PyRef<'_, PyBinaryOutcome>>,
    max_branches: Option<usize>,
) -> PyResult<Vec<PySpanningBranch>> {
    let values: Vec<_> = outcomes
        .iter()
        .map(|outcome| outcome.inner.clone())
        .collect();
    compute_spanning_set(&values, max_branches)
        .map(|branches| {
            branches
                .into_iter()
                .map(|branch| PySpanningBranch {
                    choices: branch.choices,
                    probability: branch.probability,
                    impact: branch.impact,
                    factor_impacts: branch.factor_impacts,
                })
                .collect()
        })
        .map_err(py_error)
}

#[pyclass(name = "NativeSpanningBranch", frozen)]
#[derive(Clone)]
struct PySpanningBranch {
    #[pyo3(get)]
    choices: Vec<bool>,
    #[pyo3(get)]
    probability: f64,
    #[pyo3(get)]
    impact: f64,
    #[pyo3(get)]
    factor_impacts: Vec<(String, f64)>,
}

#[pyclass(name = "NativeRiskAttribution", frozen)]
#[derive(Clone)]
struct PyRiskAttribution {
    #[pyo3(get)]
    factor: String,
    #[pyo3(get)]
    var_impact: f64,
    #[pyo3(get)]
    cvar_impact: f64,
}

#[pyclass(name = "NativeRiskLevel", frozen)]
#[derive(Clone)]
struct PyRiskLevel {
    #[pyo3(get)]
    probability: f64,
    #[pyo3(get)]
    var: f64,
    #[pyo3(get)]
    cvar: f64,
    #[pyo3(get)]
    attribution: Vec<PyRiskAttribution>,
}

#[pyclass(name = "NativeRiskResult", frozen)]
#[derive(Clone)]
struct PyRiskResult {
    #[pyo3(get)]
    levels: Vec<PyRiskLevel>,
    #[pyo3(get)]
    branch_count: usize,
}

#[pyfunction]
#[pyo3(signature = (outcomes, levels, max_branches=None))]
fn risk(
    outcomes: Vec<PyRef<'_, PyBinaryOutcome>>,
    levels: Vec<f64>,
    max_branches: Option<usize>,
) -> PyResult<PyRiskResult> {
    let values: Vec<_> = outcomes
        .iter()
        .map(|outcome| outcome.inner.clone())
        .collect();
    let result = compute_risk(&values, &levels, max_branches).map_err(py_error)?;
    Ok(PyRiskResult {
        branch_count: result.branch_count,
        levels: result
            .levels
            .into_iter()
            .map(|level| PyRiskLevel {
                probability: level.probability,
                var: level.var,
                cvar: level.cvar,
                attribution: level
                    .attribution
                    .into_iter()
                    .map(|item| PyRiskAttribution {
                        factor: item.factor,
                        var_impact: item.var_impact,
                        cvar_impact: item.cvar_impact,
                    })
                    .collect(),
            })
            .collect(),
    })
}

fn parse_cell(value: &Bound<'_, PyAny>) -> PyResult<h3o::CellIndex> {
    if let Ok(raw) = value.extract::<u64>() {
        return crc_framework_core::spatial::cell_from_u64(raw).map_err(py_error);
    }
    if let Ok(raw) = value.extract::<String>() {
        return crc_framework_core::spatial::parse_cell(&raw).map_err(py_error);
    }
    Err(PyValueError::new_err(
        "cell must be an H3 integer or hexadecimal string",
    ))
}

#[pyfunction]
fn lookup_ipcc_region(cell: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    crc_framework_core::spatial::lookup_ipcc_region(parse_cell(cell)?)
        .map(|value| value.map(str::to_owned))
        .map_err(py_error)
}

#[pyfunction]
fn lookup_continent(cell: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    crc_framework_core::spatial::lookup_continent(parse_cell(cell)?)
        .map(|value| value.map(str::to_owned))
        .map_err(py_error)
}

#[pyfunction]
fn lookup_geography(py: Python<'_>, cell: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyDict>>> {
    crc_framework_core::spatial::lookup_geography(parse_cell(cell)?)
        .map_err(py_error)?
        .map(|geography| {
            let result = PyDict::new(py);
            result.set_item("continent", geography.continent)?;
            result.set_item("countries", geography.countries)?;
            Ok(result.unbind())
        })
        .transpose()
}

#[pyfunction]
fn pathways() -> Vec<(i32, &'static str)> {
    crc_framework_core::constants::PATHWAYS.to_vec()
}

#[pyfunction]
fn horizons() -> Vec<i32> {
    crc_framework_core::constants::HORIZONS.to_vec()
}

#[pyfunction]
fn risk_factors() -> Vec<(i32, &'static str)> {
    crc_framework_core::constants::RISK_FACTORS.to_vec()
}

#[pymodule(gil_used = false)]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistribution>()?;
    m.add_class::<PyFitResult>()?;
    m.add_class::<PyQuantileFitResult>()?;
    m.add_class::<PyHurdleQuantileFitResult>()?;
    m.add_class::<PyMicroscore>()?;
    m.add_class::<PyMicroscoreSuite>()?;
    m.add_class::<PyBinaryOutcome>()?;
    m.add_class::<PySpanningBranch>()?;
    m.add_class::<PyRiskAttribution>()?;
    m.add_class::<PyRiskLevel>()?;
    m.add_class::<PyRiskResult>()?;
    m.add_function(wrap_pyfunction!(empirical_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(point_mass_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(tabulated_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(return_period_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(fitted_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(hurdle_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(fit_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(fit_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(fit_hurdle_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostic_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_impact, m)?)?;
    m.add_function(wrap_pyfunction!(apply_impact, m)?)?;
    m.add_function(wrap_pyfunction!(microscores, m)?)?;
    m.add_function(wrap_pyfunction!(spanning_set, m)?)?;
    m.add_function(wrap_pyfunction!(risk, m)?)?;
    m.add_function(wrap_pyfunction!(lookup_ipcc_region, m)?)?;
    m.add_function(wrap_pyfunction!(lookup_continent, m)?)?;
    m.add_function(wrap_pyfunction!(lookup_geography, m)?)?;
    m.add_function(wrap_pyfunction!(pathways, m)?)?;
    m.add_function(wrap_pyfunction!(horizons, m)?)?;
    m.add_function(wrap_pyfunction!(risk_factors, m)?)?;
    Ok(())
}

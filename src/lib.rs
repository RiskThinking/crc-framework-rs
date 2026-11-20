mod impact;
mod numerical;
mod rank;

use crc_framework_core::{DistributionFamily, FittedDistribution};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyDict},
};

fn py_error(error: crc_framework_core::CrcError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Fit parameters cross the binding once; all distribution operations use the
/// shared numerical law, never a second native capability surface.
#[pyclass(name = "NativeFitParameters", frozen, from_py_object)]
#[derive(Clone)]
struct PyFitParameters {
    #[pyo3(get)]
    family: String,
    #[pyo3(get)]
    shape: Option<f64>,
    #[pyo3(get)]
    location: f64,
    #[pyo3(get)]
    scale: f64,
}

#[pyclass(name = "NativeFitResult", frozen, from_py_object)]
#[derive(Clone)]
struct PyFitResult {
    #[pyo3(get)]
    distribution: PyFitParameters,
    #[pyo3(get)]
    ks_statistic: f64,
    #[pyo3(get)]
    ks_pvalue: f64,
    #[pyo3(get)]
    rmse: f64,
    #[pyo3(get)]
    r_squared: f64,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    evaluations: usize,
}

#[pyclass(name = "NativeQuantileFitResult", frozen, from_py_object)]
#[derive(Clone)]
struct PyQuantileFitResult {
    #[pyo3(get)]
    distribution: PyFitParameters,
    #[pyo3(get)]
    rmse: f64,
    #[pyo3(get)]
    weighted_r_squared: f64,
    #[pyo3(get)]
    maximum_absolute_residual: f64,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    evaluations: usize,
}

#[pyclass(name = "NativeCensoredQuantileFitResult", frozen, from_py_object)]
#[derive(Clone)]
struct PyCensoredQuantileFitResult {
    #[pyo3(get)]
    distribution: PyFitParameters,
    #[pyo3(get)]
    rmse: f64,
    #[pyo3(get)]
    weighted_r_squared: f64,
    #[pyo3(get)]
    maximum_absolute_residual: f64,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    evaluations: usize,
    #[pyo3(get)]
    negative_log_likelihood: f64,
    #[pyo3(get)]
    residual_scale: f64,
    #[pyo3(get)]
    residual_scale_floor: f64,
    #[pyo3(get)]
    censored_point_count: usize,
    #[pyo3(get)]
    uncensored_point_count: usize,
}

#[pyclass(name = "NativeHurdleQuantileFitResult", frozen, from_py_object)]
#[derive(Clone)]
struct PyHurdleQuantileFitResult {
    #[pyo3(get)]
    base_distribution: PyFitParameters,
    #[pyo3(get)]
    tail_rmse: f64,
    #[pyo3(get)]
    tail_weighted_r_squared: f64,
    #[pyo3(get)]
    tail_maximum_absolute_residual: f64,
    #[pyo3(get)]
    tail_point_count: usize,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    evaluations: usize,
    #[pyo3(get)]
    atom_point_count: usize,
}

fn fitted_py(distribution: FittedDistribution) -> PyFitParameters {
    PyFitParameters {
        family: distribution.family.name().to_owned(),
        shape: distribution.shape,
        location: distribution.location,
        scale: distribution.scale,
    }
}

fn fit_result_py(result: crc_framework_core::FitResult) -> PyFitResult {
    PyFitResult {
        distribution: fitted_py(result.distribution),
        ks_statistic: result.diagnostics.ks_statistic,
        ks_pvalue: result.diagnostics.ks_pvalue,
        rmse: result.diagnostics.rmse,
        r_squared: result.diagnostics.r_squared,
        iterations: result.iterations,
        evaluations: result.evaluations,
    }
}

fn quantile_fit_result_py(result: crc_framework_core::QuantileFitResult) -> PyQuantileFitResult {
    PyQuantileFitResult {
        distribution: fitted_py(result.distribution),
        rmse: result.diagnostics.rmse,
        weighted_r_squared: result.diagnostics.weighted_r_squared,
        maximum_absolute_residual: result.diagnostics.maximum_absolute_residual,
        iterations: result.diagnostics.iterations,
        evaluations: result.diagnostics.evaluations,
    }
}

fn censored_quantile_fit_result_py(
    result: crc_framework_core::CensoredQuantileFitResult,
) -> PyCensoredQuantileFitResult {
    let diagnostics = result.diagnostics;
    let observed = diagnostics.observed;
    PyCensoredQuantileFitResult {
        distribution: fitted_py(result.distribution),
        rmse: observed.rmse,
        weighted_r_squared: observed.weighted_r_squared,
        maximum_absolute_residual: observed.maximum_absolute_residual,
        iterations: observed.iterations,
        evaluations: observed.evaluations,
        negative_log_likelihood: diagnostics.negative_log_likelihood,
        residual_scale: diagnostics.residual_scale,
        residual_scale_floor: diagnostics.residual_scale_floor,
        censored_point_count: diagnostics.censored_point_count,
        uncensored_point_count: diagnostics.uncensored_point_count,
    }
}

fn hurdle_quantile_fit_result_py(
    result: crc_framework_core::HurdleQuantileFitResult,
) -> PyHurdleQuantileFitResult {
    let base = result.distribution.base().clone();
    let tail = result.diagnostics.tail;
    PyHurdleQuantileFitResult {
        base_distribution: fitted_py(base),
        tail_rmse: tail.rmse,
        tail_weighted_r_squared: tail.weighted_r_squared,
        tail_maximum_absolute_residual: tail.maximum_absolute_residual,
        tail_point_count: tail.point_count,
        iterations: tail.iterations,
        evaluations: tail.evaluations,
        atom_point_count: result.diagnostics.atom_point_count,
    }
}

#[pyfunction]
#[pyo3(signature = (samples, family, estimator, *, minimum_value=None, maximum_value=None, probability=None, require_sample_support=false))]
#[allow(clippy::too_many_arguments)] // PyO3 keyword-only fields mirror the typed fitting specification.
fn fit_samples(
    py: Python<'_>,
    samples: Vec<f64>,
    family: &str,
    estimator: &str,
    minimum_value: Option<f64>,
    maximum_value: Option<f64>,
    probability: Option<f64>,
    require_sample_support: bool,
) -> PyResult<PyFitResult> {
    let family = DistributionFamily::from_name(family)
        .ok_or_else(|| PyValueError::new_err(format!("unknown distribution family {family}")))?;
    let bounds = crc_framework_core::distribution::FitBounds {
        minimum_value,
        maximum_value,
        probability,
        require_sample_support,
    };
    bounds.validate().map_err(py_error)?;
    let result = match estimator {
        "mle" => py
            .detach(|| crc_framework_core::distribution::fit_samples_mle(&samples, family, bounds)),
        "lmoments" => py.detach(|| {
            let result = crc_framework_core::distribution::fit_lmoments(&samples, family)?;
            if !bounds.accepts(&result.distribution, &samples) {
                return Err(crc_framework_core::CrcError::InvalidInput(
                    "L-moment estimate does not satisfy constraints".into(),
                ));
            }
            Ok(result)
        }),
        _ => return Err(PyValueError::new_err("unknown sample estimator")),
    };
    result.map(fit_result_py).map_err(py_error)
}

#[pyfunction]
#[pyo3(signature = (probabilities, values, family, weights=None, *, minimum_value=None, maximum_value=None, probability=None, require_sample_support=false))]
#[allow(clippy::too_many_arguments)] // PyO3 keyword-only fields mirror the typed fitting specification.
fn fit_quantiles(
    py: Python<'_>,
    probabilities: Vec<f64>,
    values: Vec<f64>,
    family: &str,
    weights: Option<Vec<f64>>,
    minimum_value: Option<f64>,
    maximum_value: Option<f64>,
    probability: Option<f64>,
    require_sample_support: bool,
) -> PyResult<PyQuantileFitResult> {
    let family = DistributionFamily::from_name(family)
        .ok_or_else(|| PyValueError::new_err(format!("unknown distribution family {family}")))?;
    py.detach(move || {
        crc_framework_core::distribution::fit_quantiles_with_bounds(
            &probabilities,
            &values,
            weights.as_deref(),
            family,
            crc_framework_core::distribution::FitBounds {
                minimum_value,
                maximum_value,
                probability,
                require_sample_support,
            },
        )
    })
    .map(quantile_fit_result_py)
    .map_err(py_error)
}

#[pyfunction]
#[pyo3(signature = (
    probabilities,
    values,
    family,
    censor_location=0.0,
    residual_scale_floor_ratio=1.0e-3,
    weights=None,
    *, minimum_value=None, maximum_value=None, probability=None, require_sample_support=false
))]
#[allow(clippy::too_many_arguments)] // PyO3 keyword-only fields mirror the typed fitting specification.
fn fit_censored_quantiles(
    py: Python<'_>,
    probabilities: Vec<f64>,
    values: Vec<f64>,
    family: &str,
    censor_location: f64,
    residual_scale_floor_ratio: f64,
    weights: Option<Vec<f64>>,
    minimum_value: Option<f64>,
    maximum_value: Option<f64>,
    probability: Option<f64>,
    require_sample_support: bool,
) -> PyResult<PyCensoredQuantileFitResult> {
    let family = DistributionFamily::from_name(family)
        .ok_or_else(|| PyValueError::new_err(format!("unknown distribution family {family}")))?;
    py.detach(move || {
        crc_framework_core::distribution::fit_censored_quantiles_with_bounds(
            &probabilities,
            &values,
            weights.as_deref(),
            family,
            censor_location,
            residual_scale_floor_ratio,
            crc_framework_core::distribution::FitBounds {
                minimum_value,
                maximum_value,
                probability,
                require_sample_support,
            },
        )
    })
    .map(censored_quantile_fit_result_py)
    .map_err(py_error)
}

#[pyfunction]
#[pyo3(signature = (
    probabilities,
    values,
    family,
    atom_probability,
    atom_location=0.0,
    weights=None,
    *, minimum_value=None, maximum_value=None, probability=None, require_sample_support=false
))]
#[allow(clippy::too_many_arguments)] // PyO3 keyword-only fields mirror the typed fitting specification.
fn fit_hurdle_quantiles(
    py: Python<'_>,
    probabilities: Vec<f64>,
    values: Vec<f64>,
    family: &str,
    atom_probability: f64,
    atom_location: f64,
    weights: Option<Vec<f64>>,
    minimum_value: Option<f64>,
    maximum_value: Option<f64>,
    probability: Option<f64>,
    require_sample_support: bool,
) -> PyResult<PyHurdleQuantileFitResult> {
    let family = DistributionFamily::from_name(family)
        .ok_or_else(|| PyValueError::new_err(format!("unknown distribution family {family}")))?;
    py.detach(move || {
        crc_framework_core::distribution::fit_hurdle_quantiles_with_bounds(
            &probabilities,
            &values,
            weights.as_deref(),
            family,
            atom_location,
            atom_probability,
            crc_framework_core::distribution::FitBounds {
                minimum_value,
                maximum_value,
                probability,
                require_sample_support,
            },
        )
    })
    .map(hurdle_quantile_fit_result_py)
    .map_err(py_error)
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

#[pymodule(gil_used = false)]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyFitParameters>()?;
    m.add_class::<PyFitResult>()?;
    m.add_class::<PyQuantileFitResult>()?;
    m.add_class::<PyCensoredQuantileFitResult>()?;
    m.add_class::<PyHurdleQuantileFitResult>()?;
    m.add_function(wrap_pyfunction!(fit_samples, m)?)?;
    m.add_function(wrap_pyfunction!(fit_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(fit_censored_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(fit_hurdle_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(lookup_ipcc_region, m)?)?;
    m.add_function(wrap_pyfunction!(lookup_continent, m)?)?;
    m.add_function(wrap_pyfunction!(lookup_geography, m)?)?;
    impact::register(m)?;
    numerical::register(m)?;
    rank::register(m)?;
    Ok(())
}

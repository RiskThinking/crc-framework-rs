//! Frozen PyO3 expression, source, distribution, and evaluation-record types.

use crc_framework_core::{
    CurveParameters, EmpiricalDistribution, GaussKronrodOptions, HistogramExposure,
    ImpactContractError, ImpactDefinition, ImpactDistribution, ImpactEvaluationRecord,
    ImpactSource, MethodRequest, RecordedSource, RegistryImpactReference, ResolutionContext,
    SnapshotRef, array_content_sha256, manifest_json, registry_identity,
};
use pyo3::prelude::*;

pyo3::create_exception!(crc_framework._core, ImpactError, pyo3::exceptions::PyException);

fn impact_error(py: Python<'_>, error: ImpactContractError) -> PyErr {
    let err = ImpactError::new_err(error.to_string());
    let value = err.value(py);
    let _ = value.setattr("code", error.code);
    let _ = value.setattr("path", error.path);
    err
}

fn value_error(code: &'static str, path: &str, message: &str) -> ImpactContractError {
    ImpactContractError {
        code,
        path: path.to_string(),
        message: message.to_string(),
    }
}

fn engine() -> (&'static str, &'static str) {
    (env!("CARGO_PKG_VERSION"), env!("CRC_TARGET"))
}

#[pyclass(name = "NativeImpactExpression", frozen)]
struct PyExpression {
    inner: ImpactDefinition,
}

#[pymethods]
impl PyExpression {
    #[staticmethod]
    fn from_json(py: Python<'_>, text: &str) -> PyResult<Self> {
        ImpactDefinition::from_json(text)
            .map(|inner| Self { inner })
            .map_err(|error| impact_error(py, error))
    }

    fn canonical_json(&self) -> &str {
        self.inner.canonical_json()
    }

    #[getter]
    fn definition_hash(&self) -> &str {
        self.inner.definition_hash()
    }

    #[getter]
    fn model_kind(&self) -> &str {
        self.inner.model_kind()
    }

    #[getter]
    fn monotonicity(&self) -> &str {
        self.inner.monotonicity()
    }

    fn evaluate_value(&self, py: Python<'_>, value: f64) -> PyResult<f64> {
        self.inner
            .evaluate(value)
            .map_err(|error| impact_error(py, error))
    }

    fn evaluate_values(&self, py: Python<'_>, values: Vec<f64>) -> PyResult<Vec<f64>> {
        let mut responses = Vec::with_capacity(values.len());
        for (index, value) in values.into_iter().enumerate() {
            match self.inner.evaluate(value) {
                Ok(response) => responses.push(response),
                Err(mut error) => {
                    if error.path.is_empty() {
                        error.path = format!("/{index}");
                    }
                    return Err(impact_error(py, error));
                }
            }
        }
        Ok(responses)
    }

    #[pyo3(signature = (
        source,
        requested,
        absolute_tolerance=None,
        relative_tolerance=None,
        max_evaluations=None
    ))]
    fn expected_value(
        &self,
        py: Python<'_>,
        source: PyRef<'_, PyImpactSource>,
        requested: &str,
        absolute_tolerance: Option<f64>,
        relative_tolerance: Option<f64>,
        max_evaluations: Option<u64>,
    ) -> PyResult<PyExpectation> {
        let request = method_request(
            py,
            requested,
            absolute_tolerance,
            relative_tolerance,
            max_evaluations,
        )?;
        self.inner
            .expected_value(&source.inner, request)
            .map(|inner| PyExpectation { inner })
            .map_err(|error| impact_error(py, error))
    }

    fn transform_distribution(
        &self,
        py: Python<'_>,
        source: PyRef<'_, PyImpactSource>,
    ) -> PyResult<PyImpactDistribution> {
        self.inner
            .transform_distribution(&source.inner)
            .map(|inner| PyImpactDistribution { inner })
            .map_err(|error| impact_error(py, error))
    }

    #[pyo3(signature = (
        shape,
        values,
        snapshot_uri=None,
        snapshot_revision=None,
        snapshot_sha256=None,
        snapshot_retention=None
    ))]
    fn point_record(
        &self,
        py: Python<'_>,
        shape: Vec<u64>,
        values: Vec<f64>,
        snapshot_uri: Option<String>,
        snapshot_revision: Option<String>,
        snapshot_sha256: Option<String>,
        snapshot_retention: Option<String>,
    ) -> PyResult<PyRecord> {
        let snapshot = snapshot_from(
            py,
            snapshot_uri,
            snapshot_revision,
            snapshot_sha256,
            snapshot_retention,
        )?;
        let digest = array_content_sha256(&shape, &values);
        let (version, target) = engine();
        Ok(PyRecord {
            inner: ImpactEvaluationRecord::point(
                self.inner.definition_hash(),
                RecordedSource::PointArray {
                    shape,
                    digest: Some(digest),
                },
                snapshot.as_ref(),
                version,
                target,
            ),
        })
    }

    #[pyo3(signature = (
        source,
        expectation,
        snapshot_uri=None,
        snapshot_revision=None,
        snapshot_sha256=None,
        snapshot_retention=None
    ))]
    fn expectation_record(
        &self,
        py: Python<'_>,
        source: PyRef<'_, PyImpactSource>,
        expectation: PyRef<'_, PyExpectation>,
        snapshot_uri: Option<String>,
        snapshot_revision: Option<String>,
        snapshot_sha256: Option<String>,
        snapshot_retention: Option<String>,
    ) -> PyResult<PyRecord> {
        let snapshot = snapshot_from(
            py,
            snapshot_uri,
            snapshot_revision,
            snapshot_sha256,
            snapshot_retention,
        )?;
        let (version, target) = engine();
        Ok(PyRecord {
            inner: ImpactEvaluationRecord::expectation(
                self.inner.definition_hash(),
                source.recorded.clone(),
                snapshot.as_ref(),
                &expectation.inner,
                version,
                target,
            ),
        })
    }

    #[pyo3(signature = (
        source,
        snapshot_uri=None,
        snapshot_revision=None,
        snapshot_sha256=None,
        snapshot_retention=None
    ))]
    fn transform_record(
        &self,
        py: Python<'_>,
        source: PyRef<'_, PyImpactSource>,
        snapshot_uri: Option<String>,
        snapshot_revision: Option<String>,
        snapshot_sha256: Option<String>,
        snapshot_retention: Option<String>,
    ) -> PyResult<PyRecord> {
        let snapshot = snapshot_from(
            py,
            snapshot_uri,
            snapshot_revision,
            snapshot_sha256,
            snapshot_retention,
        )?;
        let (version, target) = engine();
        Ok(PyRecord {
            inner: ImpactEvaluationRecord::transform(
                self.inner.definition_hash(),
                source.recorded.clone(),
                snapshot.as_ref(),
                version,
                target,
            ),
        })
    }
}

#[pyclass(name = "NativeImpactSource", frozen)]
struct PyImpactSource {
    inner: ImpactSource,
    recorded: RecordedSource,
}

#[pymethods]
impl PyImpactSource {
    #[staticmethod]
    fn histogram(py: Python<'_>, edges: Vec<f64>, weights: Vec<f64>) -> PyResult<Self> {
        let histogram = HistogramExposure::new(edges, weights).map_err(|error| impact_error(py, error))?;
        let recorded = RecordedSource::Histogram {
            interval_count: histogram.weights().len() as u64,
            original_mass_sum: histogram.original_mass_sum(),
            normalization_factor: histogram.normalization_factor(),
        };
        Ok(Self {
            inner: ImpactSource::Histogram(histogram),
            recorded,
        })
    }

    #[staticmethod]
    fn discrete(
        py: Python<'_>,
        support: Vec<f64>,
        probabilities: Vec<f64>,
    ) -> PyResult<Self> {
        let discrete = crc_framework_core::DiscreteDistribution::new(support, probabilities)
            .map_err(|error| impact_error(py, error))?;
        Ok(Self {
            inner: ImpactSource::Discrete(discrete),
            recorded: RecordedSource::Untracked {
                description: "discrete distribution".to_string(),
            },
        })
    }

    #[staticmethod]
    fn empirical(py: Python<'_>, samples: Vec<f64>) -> PyResult<Self> {
        let distribution = EmpiricalDistribution::new(samples).map_err(|error| {
            impact_error(
                py,
                value_error("invalid_value", "/samples", &error.to_string()),
            )
        })?;
        Ok(Self {
            inner: ImpactSource::Empirical(distribution),
            recorded: RecordedSource::Untracked {
                description: "empirical distribution".to_string(),
            },
        })
    }

    #[staticmethod]
    #[pyo3(signature = (
        curve_kind,
        curve_type,
        curve_shape=None,
        curve_location=None,
        curve_scale=None,
        curve_atom_probability=None,
        curve_atom_location=None,
        curve_probabilities=None,
        curve_values=None
    ))]
    fn curve(
        py: Python<'_>,
        curve_kind: String,
        curve_type: String,
        curve_shape: Option<f64>,
        curve_location: Option<f64>,
        curve_scale: Option<f64>,
        curve_atom_probability: Option<f64>,
        curve_atom_location: Option<f64>,
        curve_probabilities: Option<Vec<f64>>,
        curve_values: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let parameters = CurveParameters {
            curve_kind,
            curve_type,
            curve_shape,
            curve_location,
            curve_scale,
            curve_atom_probability,
            curve_atom_location,
            curve_probabilities,
            curve_values,
        };
        let curve = parameters.reconstruct().map_err(|error| {
            impact_error(
                py,
                value_error("invalid_value", "", &error.to_string()),
            )
        })?;
        let inner = ImpactSource::from_curve(curve).map_err(|error| impact_error(py, error))?;
        Ok(Self {
            inner,
            recorded: RecordedSource::Canonical(parameters),
        })
    }

    #[getter]
    fn original_mass_sum(&self) -> Option<f64> {
        self.inner.histogram_diagnostics().map(|diagnostics| diagnostics.0)
    }

    #[getter]
    fn normalization_factor(&self) -> Option<f64> {
        self.inner.histogram_diagnostics().map(|diagnostics| diagnostics.1)
    }
}

#[pyclass(name = "NativeExpectation", frozen)]
struct PyExpectation {
    inner: crc_framework_core::Expectation,
}

#[pymethods]
impl PyExpectation {
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value
    }

    #[getter]
    fn requested(&self) -> &'static str {
        match self.inner.requested {
            crc_framework_core::RequestedMethod::Auto => "auto",
            crc_framework_core::RequestedMethod::Exact => "exact",
            crc_framework_core::RequestedMethod::GaussKronrod => "gauss_kronrod",
        }
    }

    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method.kind()
    }

    #[getter]
    fn method_version(&self) -> &'static str {
        match self.inner.method {
            crc_framework_core::ResolvedMethod::Exact { version }
            | crc_framework_core::ResolvedMethod::GaussKronrod { version, .. } => version,
        }
    }

    #[getter]
    fn absolute_tolerance(&self) -> Option<f64> {
        match self.inner.method {
            crc_framework_core::ResolvedMethod::GaussKronrod {
                absolute_tolerance, ..
            } => Some(absolute_tolerance),
            _ => None,
        }
    }

    #[getter]
    fn relative_tolerance(&self) -> Option<f64> {
        match self.inner.method {
            crc_framework_core::ResolvedMethod::GaussKronrod {
                relative_tolerance, ..
            } => Some(relative_tolerance),
            _ => None,
        }
    }

    #[getter]
    fn max_evaluations(&self) -> Option<u64> {
        match self.inner.method {
            crc_framework_core::ResolvedMethod::GaussKronrod {
                max_evaluations, ..
            } => Some(max_evaluations),
            _ => None,
        }
    }

    #[getter]
    fn evaluations(&self) -> u64 {
        self.inner.evaluations
    }

    #[getter]
    fn error_estimate(&self) -> f64 {
        self.inner.error_estimate
    }

    #[getter]
    fn exact_component(&self) -> f64 {
        self.inner.exact_component
    }

    #[getter]
    fn numerical_component(&self) -> f64 {
        self.inner.numerical_component
    }

    #[getter]
    fn original_mass_sum(&self) -> Option<f64> {
        self.inner.original_mass_sum
    }

    #[getter]
    fn normalization_factor(&self) -> Option<f64> {
        self.inner.normalization_factor
    }
}

#[pyclass(name = "NativeImpactDistribution", frozen)]
struct PyImpactDistribution {
    inner: ImpactDistribution,
}

#[pymethods]
impl PyImpactDistribution {
    #[getter]
    fn definition_hash(&self) -> &str {
        self.inner.definition_hash()
    }

    #[getter]
    fn algorithm(&self) -> &'static str {
        self.inner.algorithm()
    }

    fn atoms(&self) -> Vec<(f64, f64)> {
        self.inner
            .atoms()
            .iter()
            .map(|atom| (atom.value, atom.probability))
            .collect()
    }

    fn quantile(&self, py: Python<'_>, probability: f64) -> PyResult<f64> {
        self.inner
            .quantile(probability)
            .map_err(|error| impact_error(py, error))
    }

    fn quantiles(&self, py: Python<'_>, probabilities: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .quantiles(&probabilities)
            .map_err(|error| impact_error(py, error))
    }

    fn tabulate(
        &self,
        py: Python<'_>,
        probabilities: Vec<f64>,
        interpolation: &str,
    ) -> PyResult<Vec<f64>> {
        let interpolation = match interpolation {
            "linear_probability" => crc_framework_core::Interpolation::LinearProbability,
            "log_return_period" => crc_framework_core::Interpolation::LogReturnPeriod,
            _ => {
                return Err(impact_error(
                    py,
                    value_error(
                        "unsupported_value",
                        "/interpolation",
                        "interpolation must be linear_probability or log_return_period",
                    ),
                ));
            }
        };
        self.inner
            .tabulate(&probabilities, interpolation)
            .map(|distribution| distribution.values().to_vec())
            .map_err(|error| impact_error(py, error))
    }
}

#[pyclass(name = "NativeImpactEvaluationRecord", frozen)]
struct PyRecord {
    inner: ImpactEvaluationRecord,
}

#[pymethods]
impl PyRecord {
    fn canonical_json(&self) -> &str {
        self.inner.canonical_json()
    }
}

#[pyclass(name = "NativeImpactReference", frozen)]
struct PyReference {
    inner: RegistryImpactReference,
}

#[pymethods]
impl PyReference {
    #[staticmethod]
    fn from_json(py: Python<'_>, text: &str) -> PyResult<Self> {
        RegistryImpactReference::from_json(text)
            .map(|inner| Self { inner })
            .map_err(|error| impact_error(py, error))
    }

    fn canonical_json(&self) -> &str {
        self.inner.canonical_json()
    }

    #[getter]
    fn reference_hash(&self) -> &str {
        self.inner.reference_hash()
    }

    #[pyo3(signature = (
        cell=None,
        country=None,
        continent=None,
        building_type=None,
        historic_mean=None
    ))]
    fn resolve(
        &self,
        py: Python<'_>,
        cell: Option<String>,
        country: Option<String>,
        continent: Option<String>,
        building_type: Option<String>,
        historic_mean: Option<f64>,
    ) -> PyResult<PyExpression> {
        let context = ResolutionContext {
            cell,
            country,
            continent,
            building_type,
            historic_mean,
        };
        self.inner
            .resolve(&context)
            .map(|inner| PyExpression { inner })
            .map_err(|error| impact_error(py, error))
    }
}

#[pyfunction]
fn impact_registry_identity() -> (&'static str, &'static str, &'static str, &'static str) {
    registry_identity()
}

#[pyfunction]
fn impact_registry_manifest_json() -> &'static str {
    manifest_json()
}

#[pyfunction]
fn impact_engine() -> (&'static str, &'static str) {
    engine()
}

#[pyfunction]
fn impact_array_digest(shape: Vec<u64>, values: Vec<f64>) -> String {
    array_content_sha256(&shape, &values)
}

fn method_request(
    py: Python<'_>,
    requested: &str,
    absolute_tolerance: Option<f64>,
    relative_tolerance: Option<f64>,
    max_evaluations: Option<u64>,
) -> PyResult<MethodRequest> {
    match requested {
        "auto" => Ok(MethodRequest::Auto),
        "exact" => Ok(MethodRequest::Exact),
        "gauss_kronrod" => {
            let absolute = absolute_tolerance.ok_or_else(|| {
                impact_error(
                    py,
                    value_error(
                        "missing_field",
                        "/absolute_tolerance",
                        "Gauss-Kronrod requires an absolute tolerance",
                    ),
                )
            })?;
            let relative = relative_tolerance.ok_or_else(|| {
                impact_error(
                    py,
                    value_error(
                        "missing_field",
                        "/relative_tolerance",
                        "Gauss-Kronrod requires a relative tolerance",
                    ),
                )
            })?;
            let evaluations = max_evaluations.ok_or_else(|| {
                impact_error(
                    py,
                    value_error(
                        "missing_field",
                        "/max_evaluations",
                        "Gauss-Kronrod requires an evaluation budget",
                    ),
                )
            })?;
            GaussKronrodOptions::new(absolute, relative, evaluations)
                .map(MethodRequest::GaussKronrod)
                .map_err(|error| impact_error(py, error))
        }
        _ => Err(impact_error(
            py,
            value_error(
                "unsupported_value",
                "/method",
                "method must be auto, exact, or gauss_kronrod",
            ),
        )),
    }
}

fn snapshot_from(
    py: Python<'_>,
    uri: Option<String>,
    revision: Option<String>,
    sha256: Option<String>,
    retention: Option<String>,
) -> PyResult<Option<SnapshotRef>> {
    if uri.is_none() && revision.is_none() && sha256.is_none() && retention.is_none() {
        return Ok(None);
    }
    SnapshotRef::new(
        uri.unwrap_or_default(),
        revision,
        sha256.unwrap_or_default(),
        retention.unwrap_or_default(),
    )
    .map(Some)
    .map_err(|error| impact_error(py, error))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("ImpactError", py.get_type::<ImpactError>())?;
    module.add_class::<PyExpression>()?;
    module.add_class::<PyImpactSource>()?;
    module.add_class::<PyExpectation>()?;
    module.add_class::<PyImpactDistribution>()?;
    module.add_class::<PyRecord>()?;
    module.add_class::<PyReference>()?;
    module.add_function(wrap_pyfunction!(impact_registry_identity, module)?)?;
    module.add_function(wrap_pyfunction!(impact_registry_manifest_json, module)?)?;
    module.add_function(wrap_pyfunction!(impact_engine, module)?)?;
    module.add_function(wrap_pyfunction!(impact_array_digest, module)?)?;
    Ok(())
}

use crc_framework_core::impact::{KernelCurveParameters, registry_country_choices};
// Frozen native mathematical kernel bindings.

use crc_framework_core::{
    HistogramExposure, ImpactContractError, ImpactDefinition, ImpactDistribution, ImpactSource,
    MethodRequest, RegistryImpactReference, ResolutionContext, manifest_json, registry_identity,
};
use pyo3::prelude::*;

pyo3::create_exception!(
    crc_framework._core,
    ImpactError,
    pyo3::exceptions::PyException
);

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

    fn evaluate_value(&self, py: Python<'_>, value: f64) -> PyResult<f64> {
        self.inner
            .evaluate(value)
            .map_err(|error| impact_error(py, error))
    }

    fn evaluate_values(&self, py: Python<'_>, values: Vec<f64>) -> PyResult<Vec<f64>> {
        let inner = &self.inner;
        py.detach(|| {
            let mut responses = Vec::with_capacity(values.len());
            for (index, value) in values.into_iter().enumerate() {
                match inner.evaluate(value) {
                    Ok(response) => responses.push(response),
                    Err(mut error) => {
                        if error.path.is_empty() {
                            error.path = format!("/{index}");
                        }
                        return Err(error);
                    }
                }
            }
            Ok(responses)
        })
        .map_err(|error| impact_error(py, error))
    }

    fn expectation(
        &self,
        py: Python<'_>,
        source: PyRef<'_, PyImpactSource>,
        exact: bool,
    ) -> PyResult<f64> {
        let (definition, exposure) = (&self.inner, &source.inner);
        let method = if exact {
            MethodRequest::Exact
        } else {
            MethodRequest::Auto
        };
        py.detach(|| definition.expectation(exposure, method))
            .map(|result| result.value)
            .map_err(|error| impact_error(py, error))
    }

    fn pushforward(
        &self,
        py: Python<'_>,
        source: PyRef<'_, PyImpactSource>,
    ) -> PyResult<PyImpactDistribution> {
        let (definition, exposure) = (&self.inner, &source.inner);
        py.detach(|| definition.pushforward(exposure))
            .map(|inner| PyImpactDistribution { inner })
            .map_err(|error| impact_error(py, error))
    }
}

#[pyclass(name = "NativeImpactSource", frozen)]
pub(crate) struct PyImpactSource {
    pub(crate) inner: ImpactSource,
}

#[pymethods]
impl PyImpactSource {
    #[staticmethod]
    fn histogram(py: Python<'_>, edges: Vec<f64>, weights: Vec<f64>) -> PyResult<Self> {
        let histogram =
            HistogramExposure::new(edges, weights).map_err(|error| impact_error(py, error))?;
        Ok(Self {
            inner: ImpactSource::Histogram(histogram),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (support, probabilities, counts=None))]
    fn discrete(
        py: Python<'_>,
        support: Vec<f64>,
        probabilities: Vec<f64>,
        counts: Option<Vec<u64>>,
    ) -> PyResult<Self> {
        let discrete = match counts {
            Some(counts) => {
                let ranks = crc_framework_core::numerical::CountRanks::new(&counts)
                    .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
                if ranks.probabilities() != probabilities {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "discrete probabilities must equal their counts divided by the total",
                    ));
                }
                crc_framework_core::DiscreteDistribution::from_counts(support, &ranks)
            }
            None => crc_framework_core::DiscreteDistribution::new(support, probabilities),
        }
        .map_err(|error| impact_error(py, error))?;
        Ok(Self {
            inner: ImpactSource::Discrete(discrete),
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
    #[allow(clippy::too_many_arguments)]
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
        let parameters = KernelCurveParameters {
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
            impact_error(py, value_error("invalid_value", "", &error.to_string()))
        })?;
        let inner = ImpactSource::from_curve(curve).map_err(|error| impact_error(py, error))?;
        Ok(Self { inner })
    }
}

#[pyclass(name = "NativeImpactDistribution", frozen)]
struct PyImpactDistribution {
    inner: ImpactDistribution,
}

#[pymethods]
impl PyImpactDistribution {
    fn exact_mean_deviation(&self, py: Python<'_>) -> PyResult<Option<(f64, f64)>> {
        py.detach(|| self.inner.exact_mean_deviation())
            .map_err(|error| impact_error(py, error))
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

    fn threshold_probabilities(
        &self,
        py: Python<'_>,
        values: Vec<f64>,
        inclusive: bool,
        exceedance: bool,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            values
                .into_iter()
                .map(|value| {
                    self.inner
                        .threshold_probability(value, inclusive, exceedance)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(|error| impact_error(py, error))
    }

    fn integral(
        &self,
        py: Python<'_>,
        lo: f64,
        hi: f64,
        moment: (f64, bool, f64),
        survival: bool,
    ) -> PyResult<f64> {
        let (center, square, scale) = moment;
        py.detach(|| {
            self.inner
                .scaled_integral(lo, hi, center, square, survival, scale)
        })
        .map_err(|error| impact_error(py, error))
    }

    fn inverse_survival(&self, py: Python<'_>, survival: f64) -> PyResult<f64> {
        self.inner
            .inverse_survival(survival)
            .map_err(|error| impact_error(py, error))
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
#[pyo3(signature = (factor, building_type=None))]
fn impact_registry_country_choices(factor: &str, building_type: Option<&str>) -> Vec<&'static str> {
    registry_country_choices(factor, building_type)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("ImpactError", py.get_type::<ImpactError>())?;
    module.add_class::<PyExpression>()?;
    module.add_class::<PyImpactSource>()?;
    module.add_class::<PyImpactDistribution>()?;
    module.add_class::<PyReference>()?;
    module.add_function(wrap_pyfunction!(impact_registry_identity, module)?)?;
    module.add_function(wrap_pyfunction!(impact_registry_manifest_json, module)?)?;
    module.add_function(wrap_pyfunction!(impact_registry_country_choices, module)?)?;
    Ok(())
}

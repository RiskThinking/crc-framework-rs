use crc_framework_core::numerical::NumericalLaw;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

#[pyfunction]
fn validate_document(py: Python<'_>, schema: &str, text: &str) -> PyResult<()> {
    py.detach(|| crc_framework_core::canonical::validate_document(schema, text))
        .map_err(error)
}

#[pyfunction]
fn unit_conversion(source: &str, target: &str) -> PyResult<(f64, f64)> {
    crc_framework_core::dataset::unit_conversion(source, target).map_err(error)
}

#[pyfunction]
fn validate_unit_conversion(
    py: Python<'_>,
    measurement: &str,
    source: &str,
    target: &str,
    scale: f64,
    offset: f64,
) -> PyResult<()> {
    py.detach(|| {
        let measurement = crc_framework_core::canonical::parse_document(measurement)?;
        crc_framework_core::dataset::reverse_unit_conversion(
            &measurement,
            source,
            target,
            scale,
            offset,
        )
        .map(|_| ())
    })
    .map_err(error)
}

#[pyfunction]
fn validate_fitted_curve(
    py: Python<'_>,
    record: &str,
    curve: &str,
    variable: &str,
    provenance: &str,
) -> PyResult<()> {
    use crc_framework_core::canonical::parse_document;
    py.detach(|| {
        let provenance = parse_document(provenance)?;
        crc_framework_core::dataset::validate_fitted_curve(
            &parse_document(record)?,
            &parse_document(curve)?,
            variable,
            provenance.as_array().ok_or_else(|| {
                crc_framework_core::CrcError::InvalidInput("expected a provenance array".into())
            })?,
        )
    })
    .map_err(error)
}

#[pyclass(name = "NativeNumericalDocument", frozen)]
struct PyNumericalDocument {
    inner: crc_framework_core::Numerical,
}

#[pymethods]
impl PyNumericalDocument {
    #[new]
    fn new(py: Python<'_>, text: &str) -> PyResult<Self> {
        py.detach(|| crc_framework_core::Numerical::from_json(text))
            .map(|inner| Self { inner })
            .map_err(error)
    }
    fn quantile(&self, py: Python<'_>, probability: f64) -> PyResult<f64> {
        py.detach(|| self.inner.quantile(probability))
            .map_err(error)
    }
    fn quantiles(&self, py: Python<'_>, probabilities: Vec<f64>) -> PyResult<Vec<f64>> {
        py.detach(|| {
            probabilities
                .into_iter()
                .map(|q| self.inner.quantile(q))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(error)
    }
    fn inverse_survival(&self, py: Python<'_>, probability: f64) -> PyResult<f64> {
        py.detach(|| self.inner.inverse_survival(probability))
            .map_err(error)
    }
    fn impact_source(&self, py: Python<'_>) -> PyResult<crate::impact::PyImpactSource> {
        py.detach(|| self.inner.impact_source())
            .map(|inner| crate::impact::PyImpactSource { inner })
            .map_err(|failure| match failure {
                crc_framework_core::CrcError::Unsupported(_) => {
                    PyTypeError::new_err(failure.to_string())
                }
                _ => error(failure),
            })
    }
    fn hurdle_atoms(&self, py: Python<'_>) -> PyResult<Vec<(f64, f64)>> {
        py.detach(|| self.inner.hurdle_atoms()).map_err(error)
    }
    fn hurdle_cdf(&self, py: Python<'_>, values: Vec<f64>, inclusive: bool) -> PyResult<Vec<f64>> {
        py.detach(|| {
            values
                .into_iter()
                .map(|x| {
                    self.inner
                        .hurdle_probability(f64::NEG_INFINITY, x, false, inclusive)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(error)
    }
    fn hurdle_exceedance(
        &self,
        py: Python<'_>,
        values: Vec<f64>,
        inclusive: bool,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            values
                .into_iter()
                .map(|x| {
                    self.inner
                        .hurdle_probability(x, f64::INFINITY, inclusive, false)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(error)
    }
    fn hurdle_mass(&self, py: Python<'_>, values: Vec<f64>) -> PyResult<Vec<f64>> {
        py.detach(|| {
            values
                .into_iter()
                .map(|x| self.inner.hurdle_probability(x, x, true, true))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(error)
    }
    fn hurdle_integral(
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
                .scaled_hurdle_integral(lo, hi, center, square, survival, scale)
        })
        .map_err(error)
    }
    fn finite_cvar(&self, py: Python<'_>, level: f64, upper: bool) -> PyResult<Option<f64>> {
        py.detach(|| self.inner.finite_cvar(level, upper))
            .map_err(error)
    }
}

#[pyclass(name = "NativeNumericalLaw", frozen, from_py_object)]
#[derive(Clone)]
struct PyNumericalLaw {
    inner: NumericalLaw,
}

fn error(error: crc_framework_core::CrcError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymethods]
impl PyNumericalLaw {
    #[new]
    fn new(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: NumericalLaw::from_json(text).map_err(error)?,
        })
    }
    fn support(&self) -> (f64, f64) {
        self.inner.support()
    }
    fn atoms(&self, py: Python<'_>) -> Vec<(f64, f64)> {
        py.detach(|| self.inner.atoms())
    }
    fn quantile(&self, py: Python<'_>, values: Vec<f64>) -> PyResult<Vec<f64>> {
        py.detach(|| {
            values
                .into_iter()
                .map(|q| self.inner.quantile(q))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(error)
    }
    fn cdf(&self, py: Python<'_>, values: Vec<f64>, inclusive: bool) -> Vec<f64> {
        py.detach(|| {
            values
                .into_iter()
                .map(|x| self.inner.cdf(x, inclusive))
                .collect()
        })
    }
    fn mass(&self, py: Python<'_>, values: Vec<f64>) -> Vec<f64> {
        py.detach(|| values.into_iter().map(|x| self.inner.mass(x)).collect())
    }
    fn exceedance(&self, py: Python<'_>, values: Vec<f64>, inclusive: bool) -> Vec<f64> {
        py.detach(|| {
            values
                .into_iter()
                .map(|x| self.inner.exceedance(x, inclusive))
                .collect()
        })
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
        .map_err(error)
    }
    fn mean_deviation(&self, py: Python<'_>) -> PyResult<(f64, f64)> {
        py.detach(|| self.inner.mean_deviation()).map_err(error)
    }
    fn finite_cvar(&self, py: Python<'_>, level: f64, upper: bool) -> Option<f64> {
        py.detach(|| self.inner.finite_cvar(level, upper))
    }
    fn inverse_survival(&self, py: Python<'_>, probability: f64) -> PyResult<f64> {
        py.detach(|| self.inner.inverse_survival(probability))
            .map_err(error)
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNumericalLaw>()?;
    module.add_class::<PyNumericalDocument>()?;
    module.add_function(wrap_pyfunction!(validate_document, module)?)?;
    module.add_function(wrap_pyfunction!(unit_conversion, module)?)?;
    module.add_function(wrap_pyfunction!(validate_unit_conversion, module)?)?;
    module.add_function(wrap_pyfunction!(validate_fitted_curve, module)?)?;
    Ok(())
}

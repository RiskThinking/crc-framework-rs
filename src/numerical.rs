use crc_framework_core::numerical::NumericalLaw;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyBytes,
};

#[pyfunction]
fn validate_document(py: Python<'_>, schema: &str, text: &str) -> PyResult<()> {
    py.detach(|| crc_framework_core::canonical::validate_document(schema, text))
        .map_err(error)
}

#[pyfunction]
fn validate_structure(py: Python<'_>, schema: &str, text: &str) -> PyResult<()> {
    use crc_framework_core::canonical::{parse_document, validate_structure};
    py.detach(|| validate_structure(schema, &parse_document(text)?))
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

/// Elements per thread below which a vector query stays on one thread.
const PARALLEL_CHUNK: usize = 1 << 14;

/// Apply a pure per-element query, splitting long inputs across threads.
///
/// Each element's result does not depend on the split, so the output equals
/// the sequential one; the first failing element's error is returned.
fn map<F>(values: &[f64], query: F) -> crc_framework_core::Result<Vec<f64>>
where
    F: Fn(f64) -> crc_framework_core::Result<f64> + Sync,
{
    if values.len() <= PARALLEL_CHUNK {
        return values.iter().map(|&x| query(x)).collect();
    }
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(values.len().div_ceil(PARALLEL_CHUNK));
    if threads <= 1 {
        return values.iter().map(|&x| query(x)).collect();
    }
    let size = values.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let parts: Vec<_> = values
            .chunks(size)
            .map(|chunk| {
                let query = &query;
                scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|&x| query(x))
                        .collect::<crc_framework_core::Result<Vec<f64>>>()
                })
            })
            .collect();
        let mut result = Vec::with_capacity(values.len());
        for part in parts {
            match part.join() {
                Ok(values) => result.extend(values?),
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
        Ok(result)
    })
}

fn map_infallible<F>(values: &[f64], query: F) -> Vec<f64>
where
    F: Fn(f64) -> f64 + Sync,
{
    match map(values, |x| Ok(query(x))) {
        Ok(result) => result,
        Err(_) => unreachable!("infallible queries return no error"),
    }
}

/// Native-endian f64 values from the bytes of a contiguous float64 array.
fn decode(bytes: &[u8]) -> PyResult<Vec<f64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(PyValueError::new_err(
            "expected the bytes of float64 values",
        ));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_ne_bytes(chunk.try_into().expect("eight bytes")))
        .collect())
}

fn floats<'py>(py: Python<'py>, values: &[f64]) -> Bound<'py, PyBytes> {
    PyBytes::new_with(py, values.len() * 8, |bytes| {
        for (target, value) in bytes.chunks_exact_mut(8).zip(values) {
            target.copy_from_slice(&value.to_ne_bytes());
        }
        Ok(())
    })
    .expect("writing f64 bytes cannot fail")
}

fn law(result: crc_framework_core::Result<NumericalLaw>) -> PyResult<PyNumericalLaw> {
    result.map(|inner| PyNumericalLaw { inner }).map_err(error)
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
    /// Typed constructors admit exactly what the JSON representation admits,
    /// without serializing a law only to parse it again.
    #[staticmethod]
    #[pyo3(signature = (family, location, scale, shape=None))]
    fn parametric(family: &str, location: f64, scale: f64, shape: Option<f64>) -> PyResult<Self> {
        law(NumericalLaw::parametric(family, location, scale, shape))
    }
    #[staticmethod]
    fn point_mass(value: f64) -> PyResult<Self> {
        law(NumericalLaw::point_mass(value))
    }
    #[staticmethod]
    fn histogram(edges: Vec<f64>, probabilities: Vec<f64>) -> PyResult<Self> {
        law(NumericalLaw::histogram(edges, probabilities))
    }
    #[staticmethod]
    fn quantile_law(probabilities: Vec<f64>, values: Vec<f64>) -> PyResult<Self> {
        law(NumericalLaw::quantile_law(probabilities, values))
    }
    #[staticmethod]
    fn hurdle(base: &Self, atom_probability: f64, atom_location: f64) -> PyResult<Self> {
        law(NumericalLaw::hurdle(
            base.inner.clone(),
            atom_probability,
            atom_location,
        ))
    }
    fn support(&self) -> (f64, f64) {
        self.inner.support()
    }
    fn atoms(&self, py: Python<'_>) -> Vec<(f64, f64)> {
        py.detach(|| self.inner.atoms())
    }
    /// Vector queries read and return native-endian float64 bytes; NumPy
    /// wraps the result without copying as a read-only array.
    fn quantile<'py>(&self, py: Python<'py>, values: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let values = decode(values)?;
        let result = py
            .detach(|| map(&values, |q| self.inner.quantile(q)))
            .map_err(error)?;
        Ok(floats(py, &result))
    }
    fn cdf<'py>(
        &self,
        py: Python<'py>,
        values: &[u8],
        inclusive: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let values = decode(values)?;
        let result = py.detach(|| map_infallible(&values, |x| self.inner.cdf(x, inclusive)));
        Ok(floats(py, &result))
    }
    fn mass<'py>(&self, py: Python<'py>, values: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let values = decode(values)?;
        let result = py.detach(|| map_infallible(&values, |x| self.inner.mass(x)));
        Ok(floats(py, &result))
    }
    fn exceedance<'py>(
        &self,
        py: Python<'py>,
        values: &[u8],
        inclusive: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let values = decode(values)?;
        let result = py.detach(|| map_infallible(&values, |x| self.inner.exceedance(x, inclusive)));
        Ok(floats(py, &result))
    }
    // Scalar queries skip array conversion for the most common call shape.
    fn quantile_at(&self, probability: f64) -> PyResult<f64> {
        self.inner.quantile(probability).map_err(error)
    }
    fn cdf_at(&self, value: f64, inclusive: bool) -> f64 {
        self.inner.cdf(value, inclusive)
    }
    fn mass_at(&self, value: f64) -> f64 {
        self.inner.mass(value)
    }
    fn exceedance_at(&self, value: f64, inclusive: bool) -> f64 {
        self.inner.exceedance(value, inclusive)
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
    module.add_function(wrap_pyfunction!(validate_structure, module)?)?;
    module.add_function(wrap_pyfunction!(unit_conversion, module)?)?;
    module.add_function(wrap_pyfunction!(validate_unit_conversion, module)?)?;
    module.add_function(wrap_pyfunction!(validate_fitted_curve, module)?)?;
    Ok(())
}

use crc_framework_core::numerical::NumericalLaw;
use pyo3::{exceptions::PyValueError, prelude::*};

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
        center: f64,
        square: bool,
    ) -> PyResult<f64> {
        py.detach(|| self.inner.integral(lo, hi, center, square))
            .map_err(error)
    }
    fn moments(&self, py: Python<'_>) -> PyResult<(f64, f64)> {
        py.detach(|| self.inner.moments()).map_err(error)
    }
    fn inverse_survival(&self, py: Python<'_>, probability: f64) -> PyResult<f64> {
        py.detach(|| self.inner.inverse_survival(probability))
            .map_err(error)
    }
    fn tail_integral(
        &self,
        py: Python<'_>,
        lo: f64,
        hi: f64,
        center: f64,
        square: bool,
    ) -> PyResult<f64> {
        py.detach(|| self.inner.tail_integral(lo, hi, center, square))
            .map_err(error)
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNumericalLaw>()
}

use crc_framework_core::insurance::{PolicyTerms, summarize_annual_losses as summarize};
use pyo3::{prelude::*, types::PyModule};

type Component = (String, Vec<f64>);
type NativeLevel = (
    f64,
    f64,
    f64,
    Option<f64>,
    Option<Vec<(String, f64)>>,
    f64,
    usize,
    f64,
);
type NativeExceedance = (f64, f64, f64, f64, f64);
type NativeSummary = (
    usize,
    f64,
    f64,
    f64,
    f64,
    Option<f64>,
    Vec<NativeLevel>,
    Vec<NativeExceedance>,
);

#[pyclass(name = "NativePolicyTerms", frozen, skip_from_py_object)]
struct PyPolicyTerms(PolicyTerms);

#[pymethods]
impl PyPolicyTerms {
    #[new]
    #[pyo3(signature = (deductible, limit, share))]
    fn new(deductible: f64, limit: Option<f64>, share: f64) -> PyResult<Self> {
        Ok(Self(
            PolicyTerms::new(deductible, limit, share).map_err(super::py_error)?,
        ))
    }

    fn insured_losses(
        &self,
        py: Python<'_>,
        total_insured_value: f64,
        damage_ratios: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| self.0.insured_losses(total_insured_value, &damage_ratios))
            .map_err(super::py_error)
    }
}

#[pyfunction]
#[pyo3(signature = (totals, total_insured_value, levels, return_periods, components=None))]
fn summarize_annual_losses(
    py: Python<'_>,
    totals: Vec<f64>,
    total_insured_value: f64,
    levels: Vec<f64>,
    return_periods: Vec<f64>,
    components: Option<Vec<Component>>,
) -> PyResult<NativeSummary> {
    py.detach(move || {
        let summary = summarize(
            &totals,
            total_insured_value,
            &levels,
            &return_periods,
            components.as_deref(),
        )?;
        let levels = summary
            .levels
            .into_iter()
            .map(|level| {
                (
                    level.level,
                    level.value_at_risk,
                    level.conditional_value_at_risk,
                    level.diversification_benefit,
                    level.contributions,
                    level.nominal_tail_years,
                    level.boundary_years,
                    level.boundary_weight,
                )
            })
            .collect();
        let aep = summary
            .aep
            .into_iter()
            .map(|point| {
                (
                    point.return_period,
                    point.loss,
                    point.nominal_exceedance_probability,
                    point.exceedance_probability,
                    point.nominal_tail_years,
                )
            })
            .collect();
        Ok((
            summary.trial_count,
            summary.total_insured_value,
            summary.average_annual_loss,
            summary.standard_deviation,
            summary.coefficient_of_variation,
            summary.loss_cost,
            levels,
            aep,
        ))
    })
    .map_err(super::py_error)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPolicyTerms>()?;
    module.add_function(wrap_pyfunction!(summarize_annual_losses, module)?)?;
    Ok(())
}

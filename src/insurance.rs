use crc_framework_core::insurance::{
    ANNUAL_LOSS_RNG_ALGORITHM, AnnualLossSimulation, AnnualLossSpec, DamageCurve, InsuredInterest,
    PolicyTerms, SEPARATE_PERIL_RNG_ALGORITHM, SimulationOptions,
    simulate_annual_losses as simulate, simulate_separate_peril_losses as simulate_separate,
    summarize_annual_losses as summarize, validate_peril_correlation as validate_correlation,
};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyModule},
};

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
type NativeSimulationOptions = (bool, u64, usize, usize, u64);
type NativePerils = Vec<(String, Vec<Py<PyInsuredInterest>>)>;
type NativeSimulation<'py> = (
    Bound<'py, PyBytes>,
    Option<Vec<(String, Bound<'py, PyBytes>)>>,
    f64,
    &'static str,
    Option<(f64, f64)>,
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

#[pyclass(name = "NativeAnnualLossSpec", frozen, skip_from_py_object)]
struct PyAnnualLossSpec(AnnualLossSpec);

#[pymethods]
impl PyAnnualLossSpec {
    #[new]
    #[pyo3(signature = (trials, seed, peak_correlation, decay_length_km, damage_cv, repair_tolerance))]
    fn new(
        trials: usize,
        seed: u64,
        peak_correlation: f64,
        decay_length_km: f64,
        damage_cv: f64,
        repair_tolerance: Option<f64>,
    ) -> PyResult<Self> {
        let spec = AnnualLossSpec {
            trials,
            seed,
            peak_correlation,
            decay_length_km,
            damage_cv,
            repair_tolerance,
        };
        spec.validate().map_err(super::py_error)?;
        Ok(Self(spec))
    }
}

#[pyclass(name = "NativeInsuredInterest", frozen, skip_from_py_object)]
struct PyInsuredInterest(InsuredInterest);

#[pymethods]
impl PyInsuredInterest {
    #[new]
    fn new(
        py: Python<'_>,
        id: String,
        coordinates: (f64, f64),
        total_insured_value: f64,
        knots: (Vec<f64>, Vec<f64>),
        terms: PyRef<'_, PyPolicyTerms>,
    ) -> PyResult<Self> {
        let terms = terms.0;
        let (longitude, latitude) = coordinates;
        let (probabilities, ratios) = knots;
        py.detach(move || {
            let interest = InsuredInterest {
                id,
                longitude,
                latitude,
                total_insured_value,
                terms,
                damage_curve: DamageCurve::new(probabilities, ratios)?,
            };
            interest.validate()?;
            Ok(Self(interest))
        })
        .map_err(super::py_error)
    }
}

fn snapshot(py: Python<'_>, values: Vec<f64>) -> PyResult<Bound<'_, PyBytes>> {
    PyBytes::new_with(py, values.len() * size_of::<f64>(), |buffer| {
        py.detach(move || {
            for (word, value) in buffer.chunks_exact_mut(size_of::<f64>()).zip(values) {
                word.copy_from_slice(&value.to_ne_bytes());
            }
        });
        Ok(())
    })
}

#[pyfunction]
fn simulate_annual_losses<'py>(
    py: Python<'py>,
    interests: Vec<Py<PyInsuredInterest>>,
    spec: PyRef<'_, PyAnnualLossSpec>,
    execution: NativeSimulationOptions,
) -> PyResult<NativeSimulation<'py>> {
    let spec = spec.0;
    let options = simulation_options(execution);
    let interests = owned_interests(py, interests);
    let result = py
        .detach(move || simulate(&interests, &spec, &options))
        .map_err(super::py_error)?;
    snapshot_result(py, result, ANNUAL_LOSS_RNG_ALGORITHM)
}

#[pyfunction]
#[pyo3(signature = (perils, cross_peril_correlation, spec, execution))]
fn simulate_separate_peril_losses<'py>(
    py: Python<'py>,
    perils: NativePerils,
    cross_peril_correlation: Option<Vec<Vec<f64>>>,
    spec: PyRef<'_, PyAnnualLossSpec>,
    execution: NativeSimulationOptions,
) -> PyResult<NativeSimulation<'py>> {
    let spec = spec.0;
    let options = simulation_options(execution);
    let perils: Vec<_> = perils
        .into_iter()
        .map(|(name, interests)| (name, owned_interests(py, interests)))
        .collect();
    let result = py
        .detach(move || {
            simulate_separate(&perils, cross_peril_correlation.as_deref(), &spec, &options)
        })
        .map_err(super::py_error)?;
    snapshot_result(py, result, SEPARATE_PERIL_RNG_ALGORITHM)
}

#[pyfunction]
fn validate_peril_correlation(py: Python<'_>, correlation: Vec<Vec<f64>>) -> PyResult<()> {
    py.detach(move || validate_correlation(&correlation))
        .map_err(super::py_error)
}

fn owned_interests(py: Python<'_>, interests: Vec<Py<PyInsuredInterest>>) -> Vec<InsuredInterest> {
    interests
        .into_iter()
        .map(|interest| interest.bind(py).borrow().0.clone())
        .collect()
}

fn simulation_options(execution: NativeSimulationOptions) -> SimulationOptions {
    let (include_components, secondary_stream, block_size, memory_limit, work_limit) = execution;
    SimulationOptions {
        secondary_stream,
        include_components,
        block_size,
        memory_limit,
        work_limit,
    }
}

fn snapshot_result<'py>(
    py: Python<'py>,
    result: AnnualLossSimulation,
    rng_algorithm: &'static str,
) -> PyResult<NativeSimulation<'py>> {
    let totals = snapshot(py, result.totals)?;
    let components = result
        .components
        .map(|series| {
            series
                .into_iter()
                .map(|(id, losses)| Ok((id, snapshot(py, losses)?)))
                .collect::<PyResult<Vec<_>>>()
        })
        .transpose()?;
    let correlation_repair = result
        .correlation_repair
        .map(|repair| (repair.minimum_eigenvalue, repair.maximum_change));
    Ok((
        totals,
        components,
        result.total_insured_value,
        rng_algorithm,
        correlation_repair,
    ))
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
    module.add_class::<PyAnnualLossSpec>()?;
    module.add_class::<PyInsuredInterest>()?;
    module.add_function(wrap_pyfunction!(simulate_annual_losses, module)?)?;
    module.add_function(wrap_pyfunction!(simulate_separate_peril_losses, module)?)?;
    module.add_function(wrap_pyfunction!(validate_peril_correlation, module)?)?;
    module.add_function(wrap_pyfunction!(summarize_annual_losses, module)?)?;
    Ok(())
}

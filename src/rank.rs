use crc_framework_core::rank::{BinaryOutcome, compute_risk};
use pyo3::{prelude::*, types::PyModule};

type NativeOutcome = (String, f64, f64, f64, f64);
type Attribution = (String, f64, f64);
type NativeLevel = (f64, f64, f64, Vec<Attribution>);

#[pyfunction]
#[pyo3(signature = (outcomes, levels, max_branches=None))]
fn rank_risk(
    py: Python<'_>,
    outcomes: Vec<NativeOutcome>,
    levels: Vec<f64>,
    max_branches: Option<usize>,
) -> PyResult<(usize, Vec<NativeLevel>)> {
    let outcomes = outcomes
        .into_iter()
        .map(
            |(id, lower_rank_mass, lower_impact, upper_rank_mass, upper_impact)| BinaryOutcome {
                id,
                lower_rank_mass,
                lower_impact,
                upper_rank_mass,
                upper_impact,
            },
        )
        .collect::<Vec<_>>();
    let result = py
        .detach(move || compute_risk(&outcomes, &levels, max_branches))
        .map_err(super::py_error)?;
    Ok((
        result.branch_count,
        result
            .levels
            .into_iter()
            .map(|level| {
                (
                    level.probability,
                    level.value_at_risk,
                    level.conditional_value_at_risk,
                    level
                        .attribution
                        .into_iter()
                        .map(|part| (part.id, part.value_at_risk, part.conditional_value_at_risk))
                        .collect(),
                )
            })
            .collect(),
    ))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(rank_risk, module)?)?;
    Ok(())
}

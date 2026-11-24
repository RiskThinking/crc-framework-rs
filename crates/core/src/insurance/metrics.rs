//! Empirical portfolio measures and aligned component allocation.

use std::collections::HashSet;

use crate::distribution::empirical_rank;
use crate::numerical::{
    CompensatedSum, affine_segment_mean_deviation, compensated, deviation_scale,
};
use crate::{CrcError, Distribution, EmpiricalDistribution, Result};

/// Report at one non-exceedance confidence level.
#[derive(Clone, Debug, PartialEq)]
pub struct AnnualLossLevel {
    pub level: f64,
    pub value_at_risk: f64,
    pub conditional_value_at_risk: f64,
    pub diversification_benefit: Option<f64>,
    pub contributions: Option<Vec<(String, f64)>>,
    pub nominal_tail_years: f64,
    pub boundary_years: usize,
    pub boundary_weight: f64,
}

/// One point on the annual aggregate exceedance curve.
#[derive(Clone, Debug, PartialEq)]
pub struct AnnualLossExceedance {
    pub return_period: f64,
    pub loss: f64,
    pub nominal_exceedance_probability: f64,
    pub exceedance_probability: f64,
    pub nominal_tail_years: f64,
}

/// Portfolio measures derived from one insured loss per equally weighted year.
#[derive(Clone, Debug, PartialEq)]
pub struct AnnualLossSummary {
    pub trial_count: usize,
    pub total_insured_value: f64,
    pub average_annual_loss: f64,
    pub standard_deviation: f64,
    pub coefficient_of_variation: f64,
    pub loss_cost: Option<f64>,
    pub levels: Vec<AnnualLossLevel>,
    pub aep: Vec<AnnualLossExceedance>,
}

fn invalid(message: impl Into<String>) -> CrcError {
    CrcError::InvalidInput(message.into())
}

fn nonnegative_finite(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn validate_coordinates(values: &[f64], accepts: impl Fn(f64) -> bool) -> bool {
    let mut seen = HashSet::with_capacity(values.len());
    values
        .iter()
        .all(|&value| value.is_finite() && accepts(value) && seen.insert(value.to_bits()))
}

fn validate_components(totals: &[f64], components: &[(String, Vec<f64>)]) -> Result<()> {
    if components.is_empty() {
        return Err(invalid(
            "components must contain at least one aligned series",
        ));
    }
    let mut ids = HashSet::with_capacity(components.len());
    for (id, losses) in components {
        if id.is_empty() || id.trim() != id || !ids.insert(id) {
            return Err(invalid("component IDs must be exact, nonempty, and unique"));
        }
        if losses.len() != totals.len() || losses.iter().any(|&loss| !nonnegative_finite(loss)) {
            return Err(invalid(format!(
                "component {id:?} must provide one finite nonnegative loss per portfolio year"
            )));
        }
    }
    for (trial, &total) in totals.iter().enumerate() {
        let sum = compensated(components.iter().map(|(_, losses)| losses[trial]));
        if !sum.is_finite() || (sum - total).abs() > 1e-12 * sum.max(total) {
            return Err(invalid(format!(
                "component losses do not reconcile with the portfolio total for trial {trial}"
            )));
        }
    }
    Ok(())
}

fn diversification_benefits(
    components: &[(String, Vec<f64>)],
    levels: &[f64],
    portfolio_vars: &[f64],
) -> Result<Vec<f64>> {
    if levels.is_empty() {
        return Ok(Vec::new());
    }
    let mut benefits = portfolio_vars
        .iter()
        .map(|&var| {
            // Subtract the portfolio first. The standalone sum can overflow
            // even when the diversification benefit itself fits.
            let mut sum = CompensatedSum::default();
            sum.add(-var);
            sum
        })
        .collect::<Vec<_>>();
    for (_, losses) in components {
        let law = EmpiricalDistribution::new(losses.clone())?;
        for (sum, &level) in benefits.iter_mut().zip(levels) {
            sum.add(law.ppf(level)?);
        }
    }
    Ok(benefits.into_iter().map(|sum| sum.total()).collect())
}

/// Summarize aligned annual losses. Trial positions in component arrays match
/// the portfolio array, and component losses must sum to each year's total.
pub fn summarize_annual_losses(
    totals: &[f64],
    total_insured_value: f64,
    levels: &[f64],
    return_periods: &[f64],
    components: Option<&[(String, Vec<f64>)]>,
) -> Result<AnnualLossSummary> {
    if totals.is_empty() || totals.iter().any(|&loss| !nonnegative_finite(loss)) {
        return Err(invalid(
            "annual losses must be nonempty, finite, and nonnegative",
        ));
    }
    if !nonnegative_finite(total_insured_value) {
        return Err(invalid(
            "total insured value must be finite and nonnegative",
        ));
    }
    if !validate_coordinates(levels, |q| 0.0 < q && q < 1.0) {
        return Err(invalid(
            "levels must be distinct finite numbers strictly between 0 and 1",
        ));
    }
    if !validate_coordinates(return_periods, |period| period >= 1.0) {
        return Err(invalid(
            "return periods must be distinct finite numbers of at least one year",
        ));
    }
    if let Some(components) = components {
        validate_components(totals, components)?;
    }

    let trial_count = totals.len();
    let sample = EmpiricalDistribution::new(totals.to_vec())?;
    let sorted = sample.samples();
    let scale = deviation_scale(*sorted.last().unwrap());
    let segments = sorted
        .chunk_by(|left, right| left == right)
        .map(|group| (group[0], group[0], empirical_rank(group.len(), trial_count)))
        .collect::<Vec<_>>();
    // Scaling keeps ratios computable when the monetary mean rounds to zero
    // and keeps large intermediate sums inside binary64.
    let (scaled_mean, scaled_deviation) = affine_segment_mean_deviation(&segments, scale, 0.0);
    let average_annual_loss = scaled_mean / scale;
    let standard_deviation = scaled_deviation / scale;
    let coefficient_of_variation = if scaled_mean > 0.0 {
        scaled_deviation / scaled_mean
    } else {
        0.0
    };
    let loss_cost = (total_insured_value > 0.0).then(|| {
        let scaled_tiv = total_insured_value * scale;
        if scaled_tiv > 0.0 && scaled_tiv.is_finite() {
            scaled_mean / scaled_tiv
        } else {
            average_annual_loss / total_insured_value
        }
    });

    let portfolio_vars = levels
        .iter()
        .map(|&level| sample.ppf(level))
        .collect::<Result<Vec<_>>>()?;
    let benefits = components
        .map(|components| diversification_benefits(components, levels, &portfolio_vars))
        .transpose()?;
    let levels = levels
        .iter()
        .zip(portfolio_vars)
        .enumerate()
        .map(|(index, (&level, value_at_risk))| {
            let first = sorted.partition_point(|&loss| loss < value_at_risk);
            let end = sorted.partition_point(|&loss| loss <= value_at_risk);
            let boundary_years = end - first;
            let retained = if level == empirical_rank(end, trial_count) {
                // An exact empirical cutoff excludes the whole boundary atom.
                0.0
            } else {
                // Fused subtraction preserves the fraction at levels just
                // above or below a cutoff, including at the top of the sample.
                (-level)
                    .mul_add(trial_count as f64, end as f64)
                    .clamp(0.0, boundary_years as f64)
            };
            let boundary_weight = retained / boundary_years as f64;
            let included_years = (trial_count - end) as f64 + retained;
            let conditional_value_at_risk = compensated(
                sorted[end..]
                    .iter()
                    .map(|loss| loss * scale)
                    .chain(std::iter::once((value_at_risk * scale) * retained)),
            ) / included_years
                / scale;
            let contributions = components.map(|components| {
                let trial_weights = totals
                    .iter()
                    .map(|&total| {
                        if total > value_at_risk {
                            1.0
                        } else if total == value_at_risk {
                            boundary_weight
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>();
                components
                    .iter()
                    .map(|(id, losses)| {
                        let weighted = compensated(
                            losses
                                .iter()
                                .zip(&trial_weights)
                                .map(|(loss, weight)| (loss * scale) * weight),
                        );
                        (id.clone(), weighted / included_years / scale)
                    })
                    .collect()
            });
            AnnualLossLevel {
                level,
                value_at_risk,
                conditional_value_at_risk,
                diversification_benefit: benefits.as_ref().map(|values| values[index]),
                contributions,
                nominal_tail_years: included_years,
                boundary_years,
                boundary_weight,
            }
        })
        .collect::<Vec<_>>();

    let aep = return_periods
        .iter()
        .map(|&return_period| {
            let nominal_exceedance_probability = return_period.recip();
            let loss = sample.upper_loss_threshold(nominal_exceedance_probability)?;
            let exceeded = trial_count - sorted.partition_point(|&year| year <= loss);
            // At a jump, the fraction strictly exceeding the reported loss
            // need not equal the requested return-period probability.
            Ok(AnnualLossExceedance {
                return_period,
                loss,
                nominal_exceedance_probability,
                exceedance_probability: empirical_rank(exceeded, trial_count),
                nominal_tail_years: trial_count as f64 / return_period,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if !average_annual_loss.is_finite()
        || !standard_deviation.is_finite()
        || !coefficient_of_variation.is_finite()
        || loss_cost.is_some_and(|value| !value.is_finite())
        || levels.iter().any(|report| {
            !report.conditional_value_at_risk.is_finite()
                || report
                    .diversification_benefit
                    .is_some_and(|value| !value.is_finite())
                || report.contributions.as_ref().is_some_and(|contributions| {
                    contributions.iter().any(|(_, value)| !value.is_finite())
                })
        })
    {
        return Err(invalid(
            "annual loss measures exceed the representable range",
        ));
    }
    Ok(AnnualLossSummary {
        trial_count,
        total_insured_value,
        average_annual_loss,
        standard_deviation,
        coefficient_of_variation,
        loss_cost,
        levels,
        aep,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_near(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    #[test]
    fn allocates_the_same_fraction_of_each_boundary_year() {
        let totals = [0.0, 0.0, 0.0, 10.0, 10.0, 30.0, 30.0, 50.0];
        let components = [
            ("A".into(), vec![0.0, 0.0, 0.0, 10.0, 0.0, 20.0, 0.0, 20.0]),
            ("B".into(), vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 30.0, 30.0]),
        ];
        let summary = summarize_annual_losses(
            &totals,
            100.0,
            &[0.8125, 0.875, 0.875_f64.next_up(), 0.99],
            &[100.0],
            Some(&components),
        )
        .unwrap();
        assert_eq!(summary.average_annual_loss, 16.25);
        assert_near(summary.standard_deviation, 298.4375_f64.sqrt());
        assert_near(summary.loss_cost.unwrap(), 0.1625);
        let level = &summary.levels[0];
        assert_eq!(level.value_at_risk, 30.0);
        assert_eq!(level.nominal_tail_years, 1.5);
        assert_eq!(level.boundary_years, 2);
        assert_eq!(level.boundary_weight, 0.25);
        assert_near(level.conditional_value_at_risk, 130.0 / 3.0);
        let parts = level.contributions.as_ref().unwrap();
        assert_near(parts[0].1, 50.0 / 3.0);
        assert_near(parts[1].1, 80.0 / 3.0);
        assert_near(parts[0].1 + parts[1].1, level.conditional_value_at_risk);

        assert_eq!(summary.levels[1].value_at_risk, 30.0);
        assert_eq!(summary.levels[1].boundary_weight, 0.0);
        assert_eq!(summary.levels[1].conditional_value_at_risk, 50.0);
        assert_eq!(summary.levels[2].value_at_risk, 50.0);
        assert_eq!(summary.levels[3].value_at_risk, summary.aep[0].loss);
        assert_eq!(summary.aep[0].nominal_exceedance_probability, 0.01);
        assert_eq!(summary.aep[0].exceedance_probability, 0.0);

        let mut swapped = components.clone();
        for (_, values) in &mut swapped {
            values.swap(5, 6);
        }
        let reordered =
            summarize_annual_losses(&totals, 100.0, &[0.8125], &[], Some(&swapped)).unwrap();
        assert_eq!(&reordered.levels[0], level);
    }

    #[test]
    fn keeps_signed_diversification_and_zero_denominators() {
        let mut a = vec![0.0; 100];
        let mut b = a.clone();
        a[..4].fill(10.0);
        b[4..8].fill(10.0);
        let totals: Vec<f64> = a.iter().zip(&b).map(|(a, b)| a + b).collect();
        let summary = summarize_annual_losses(
            &totals,
            1.0,
            &[0.95],
            &[],
            Some(&[("a".into(), a), ("b".into(), b)]),
        )
        .unwrap();
        assert_eq!(summary.levels[0].diversification_benefit, Some(-10.0));

        let zero = summarize_annual_losses(&[0.0, 0.0], 0.0, &[0.95], &[10.0], None).unwrap();
        assert_eq!(zero.average_annual_loss, 0.0);
        assert_eq!(zero.standard_deviation, 0.0);
        assert_eq!(zero.coefficient_of_variation, 0.0);
        assert_eq!(zero.loss_cost, None);
        assert_eq!(zero.levels[0].conditional_value_at_risk, 0.0);
        assert_eq!(zero.levels[0].contributions, None);
        assert_eq!(zero.aep[0].loss, 0.0);
    }

    #[test]
    fn rejects_invalid_samples_requests_and_components() {
        assert!(summarize_annual_losses(&[f64::NAN], 1.0, &[], &[], None).is_err());
        assert!(summarize_annual_losses(&[1.0], f64::INFINITY, &[], &[], None).is_err());
        assert!(summarize_annual_losses(&[1.0], 1.0, &[1.0], &[], None).is_err());
        assert!(summarize_annual_losses(&[1.0], 1.0, &[0.5, 0.5], &[], None).is_err());
        assert!(summarize_annual_losses(&[1.0], 1.0, &[], &[0.5], None).is_err());
        let conflicting = [("A".into(), vec![0.0, 2.0])];
        let error =
            summarize_annual_losses(&[1.0, 1.0], 1.0, &[], &[], Some(&conflicting)).unwrap_err();
        assert!(error.to_string().contains("reconcile"));
    }

    #[test]
    fn computes_ratios_when_the_mean_amount_underflows() {
        let tiny = f64::from_bits(1);
        let summary = summarize_annual_losses(&[0.0, tiny], tiny, &[0.5], &[], None).unwrap();
        assert_eq!(summary.average_annual_loss, 0.0);
        assert_eq!(summary.coefficient_of_variation, 1.0);
        assert_eq!(summary.loss_cost, Some(0.5));
        assert_eq!(summary.levels[0].conditional_value_at_risk, tiny);
    }
}

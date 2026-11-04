use crate::{
    error::{CrcError, Result},
    metrics::BinaryOutcome,
};

#[doc(hidden)]
pub mod climate {
    pub mod utility {
        pub fn get_hazard_limit(index_id: i32) -> f64 {
            match index_id {
                4 => 300.0,
                8 | 2 | 1 => 366.0,
                _ => f64::INFINITY,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpanningBranch {
    pub choices: Vec<bool>,
    pub probability: f64,
    pub impact: f64,
    pub factor_impacts: Vec<(String, f64)>,
}

#[derive(Debug, Clone)]
pub struct RiskAttribution {
    pub factor: String,
    pub var_impact: f64,
    pub cvar_impact: f64,
}

#[derive(Debug, Clone)]
pub struct RiskLevelResult {
    pub probability: f64,
    pub var: f64,
    pub cvar: f64,
    pub attribution: Vec<RiskAttribution>,
}

#[derive(Debug, Clone)]
pub struct RiskResult {
    pub levels: Vec<RiskLevelResult>,
    pub branch_count: usize,
}

/// Hard ceiling on the number of spanning branches, whatever `max_branches`
/// a caller requests. Each branch needs at least 24 bytes even in the lean
/// risk path, so this keeps a single request below ~1.6 GB instead of
/// aborting the process on an impossible allocation.
pub const MAX_SPANNING_BRANCHES: usize = 1 << 26;

const DEFAULT_MAX_BRANCHES: usize = 1 << 20;

/// One spanning branch without per-branch allocations: the bit `i` of `mask`
/// is set when factor `i` takes its downside outcome.
#[derive(Debug, Clone, Copy)]
struct LeanBranch {
    impact: f64,
    probability: f64,
    mask: usize,
}

struct LeanSpanningSet {
    /// Normalised factor weights.
    weights: Vec<f64>,
    /// Branches sorted by impact; ties keep increasing mask order.
    branches: Vec<LeanBranch>,
}

impl LeanSpanningSet {
    fn weighted_impact(&self, outcomes: &[BinaryOutcome], mask: usize, index: usize) -> f64 {
        let outcome = &outcomes[index];
        let raw_impact = if mask & (1 << index) != 0 {
            outcome.downside_impact
        } else {
            outcome.upside_impact
        };
        raw_impact * self.weights[index]
    }
}

fn validate_outcomes(outcomes: &[BinaryOutcome]) -> Result<()> {
    if outcomes.is_empty() {
        return Err(CrcError::InvalidInput(
            "at least one binary outcome is required".into(),
        ));
    }
    for outcome in outcomes {
        if !outcome.downside_probability.is_finite()
            || !outcome.upside_probability.is_finite()
            || outcome.downside_probability < 0.0
            || outcome.upside_probability < 0.0
            || (outcome.downside_probability + outcome.upside_probability - 1.0).abs() > 1.0e-8
            || !outcome.downside_impact.is_finite()
            || !outcome.upside_impact.is_finite()
            || !outcome.weight.is_finite()
            || outcome.weight < 0.0
        {
            return Err(CrcError::InvalidInput(format!(
                "invalid probability, impact, or weight for factor {}",
                outcome.factor
            )));
        }
    }
    Ok(())
}

fn branch_count(outcomes: &[BinaryOutcome], max_branches: Option<usize>) -> Result<usize> {
    let limit = max_branches
        .unwrap_or(DEFAULT_MAX_BRANCHES)
        .min(MAX_SPANNING_BRANCHES);
    let exceeded = |branches| CrcError::BranchLimitExceeded {
        factors: outcomes.len(),
        branches,
        limit,
    };
    let count = u32::try_from(outcomes.len())
        .ok()
        .and_then(|factors| 1usize.checked_shl(factors))
        .ok_or_else(|| exceeded(usize::MAX))?;
    if count > limit {
        return Err(exceeded(count));
    }
    Ok(count)
}

fn reserve<T>(count: usize, outcomes: &[BinaryOutcome], limit: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| CrcError::BranchLimitExceeded {
            factors: outcomes.len(),
            branches: count,
            limit,
        })?;
    Ok(values)
}

fn lean_spanning_set(
    outcomes: &[BinaryOutcome],
    max_branches: Option<usize>,
) -> Result<LeanSpanningSet> {
    validate_outcomes(outcomes)?;
    let count = branch_count(outcomes, max_branches)?;
    let weight_sum = outcomes.iter().map(|outcome| outcome.weight).sum::<f64>();
    let normalize = if weight_sum > 0.0 {
        weight_sum
    } else {
        outcomes.len() as f64
    };
    let weights = outcomes
        .iter()
        .map(|outcome| {
            if weight_sum > 0.0 {
                outcome.weight / normalize
            } else {
                1.0 / normalize
            }
        })
        .collect::<Vec<_>>();
    let mut branches = reserve::<LeanBranch>(count, outcomes, count)?;
    for mask in 0..count {
        let mut probability = 1.0;
        let mut impact = 0.0;
        for (index, outcome) in outcomes.iter().enumerate() {
            let downside = mask & (1 << index) != 0;
            let (branch_probability, raw_impact) = if downside {
                (outcome.downside_probability, outcome.downside_impact)
            } else {
                (outcome.upside_probability, outcome.upside_impact)
            };
            probability *= branch_probability;
            impact += raw_impact * weights[index];
        }
        branches.push(LeanBranch {
            impact,
            probability,
            mask,
        });
    }
    // Equivalent to a stable sort by impact over the mask-ordered branches.
    branches.sort_unstable_by(|left, right| {
        left.impact
            .total_cmp(&right.impact)
            .then(left.mask.cmp(&right.mask))
    });
    Ok(LeanSpanningSet { weights, branches })
}

pub fn compute_spanning_set(
    outcomes: &[BinaryOutcome],
    max_branches: Option<usize>,
) -> Result<Vec<SpanningBranch>> {
    let set = lean_spanning_set(outcomes, max_branches)?;
    let mut branches = reserve(set.branches.len(), outcomes, set.branches.len())?;
    for branch in &set.branches {
        branches.push(SpanningBranch {
            choices: (0..outcomes.len())
                .map(|index| branch.mask & (1 << index) != 0)
                .collect(),
            probability: branch.probability,
            impact: branch.impact,
            factor_impacts: outcomes
                .iter()
                .enumerate()
                .map(|(index, outcome)| {
                    (
                        outcome.factor.clone(),
                        set.weighted_impact(outcomes, branch.mask, index),
                    )
                })
                .collect(),
        });
    }
    Ok(branches)
}

pub fn compute_risk(
    outcomes: &[BinaryOutcome],
    levels: &[f64],
    max_branches: Option<usize>,
) -> Result<RiskResult> {
    if levels.is_empty()
        || levels
            .iter()
            .any(|level| !level.is_finite() || *level <= 0.0 || *level >= 1.0)
    {
        return Err(CrcError::InvalidInput(
            "risk levels must be finite and strictly between zero and one".into(),
        ));
    }
    let set = lean_spanning_set(outcomes, max_branches)?;
    let branches = &set.branches;
    let total_probability = branches
        .iter()
        .map(|branch| branch.probability)
        .sum::<f64>();
    if total_probability <= 0.0 {
        return Err(CrcError::InvalidInput(
            "spanning branches have zero total probability".into(),
        ));
    }
    let mut results = Vec::with_capacity(levels.len());
    for &level in levels {
        let mut cumulative = 0.0;
        let mut var_branch = branches.last().expect("validated non-empty");
        for branch in branches {
            cumulative += branch.probability / total_probability;
            if cumulative >= level {
                var_branch = branch;
                break;
            }
        }

        let tail_mass = 1.0 - level;
        let mut remaining = tail_mass;
        let mut cvar_sum = 0.0;
        let mut cvar_by_factor = vec![0.0; outcomes.len()];
        for branch in branches.iter().rev() {
            if remaining <= 0.0 {
                break;
            }
            let branch_mass = branch.probability / total_probability;
            let included = branch_mass.min(remaining);
            cvar_sum += branch.impact * included;
            for (index, total) in cvar_by_factor.iter_mut().enumerate() {
                *total += set.weighted_impact(outcomes, branch.mask, index) * included;
            }
            remaining -= included;
        }
        let cvar = cvar_sum / tail_mass;
        let attribution = outcomes
            .iter()
            .enumerate()
            .map(|(index, outcome)| RiskAttribution {
                factor: outcome.factor.clone(),
                var_impact: set.weighted_impact(outcomes, var_branch.mask, index),
                cvar_impact: cvar_by_factor[index] / tail_mass,
            })
            .collect();
        results.push(RiskLevelResult {
            probability: level,
            var: var_branch.impact,
            cvar: cvar.max(var_branch.impact),
            attribution,
        });
    }
    Ok(RiskResult {
        levels: results,
        branch_count: branches.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_factors_create_four_branches() {
        let outcomes = vec![
            BinaryOutcome {
                factor: "a".into(),
                downside_probability: 0.1,
                downside_impact: 1.0,
                upside_probability: 0.9,
                upside_impact: 0.0,
                weight: 1.0,
            },
            BinaryOutcome {
                factor: "b".into(),
                downside_probability: 0.2,
                downside_impact: 2.0,
                upside_probability: 0.8,
                upside_impact: 0.0,
                weight: 1.0,
            },
        ];
        let result = compute_risk(&outcomes, &[0.5, 0.95], None).unwrap();
        assert_eq!(result.branch_count, 4);
        assert!(result.levels[1].cvar >= result.levels[1].var);
    }

    /// The original allocation-heavy implementation, kept to prove the lean
    /// mask-based path produces bit-identical results.
    fn reference_branches(outcomes: &[BinaryOutcome]) -> Vec<SpanningBranch> {
        let weight_sum = outcomes.iter().map(|outcome| outcome.weight).sum::<f64>();
        let normalize = if weight_sum > 0.0 {
            weight_sum
        } else {
            outcomes.len() as f64
        };
        let mut branches = Vec::new();
        for mask in 0..(1usize << outcomes.len()) {
            let mut choices = Vec::new();
            let mut probability = 1.0;
            let mut impact = 0.0;
            let mut factor_impacts = Vec::new();
            for (index, outcome) in outcomes.iter().enumerate() {
                let downside = mask & (1 << index) != 0;
                choices.push(downside);
                probability *= if downside {
                    outcome.downside_probability
                } else {
                    outcome.upside_probability
                };
                let raw_impact = if downside {
                    outcome.downside_impact
                } else {
                    outcome.upside_impact
                };
                let weight = if weight_sum > 0.0 {
                    outcome.weight / normalize
                } else {
                    1.0 / normalize
                };
                impact += raw_impact * weight;
                factor_impacts.push((outcome.factor.clone(), raw_impact * weight));
            }
            branches.push(SpanningBranch {
                choices,
                probability,
                impact,
                factor_impacts,
            });
        }
        branches.sort_by(|left, right| left.impact.total_cmp(&right.impact));
        branches
    }

    /// `(var, cvar, [(var_impact, cvar_impact)])` per level.
    type ReferenceLevel = (f64, f64, Vec<(f64, f64)>);

    fn reference_risk(outcomes: &[BinaryOutcome], levels: &[f64]) -> Vec<ReferenceLevel> {
        let branches = reference_branches(outcomes);
        let total = branches
            .iter()
            .map(|branch| branch.probability)
            .sum::<f64>();
        levels
            .iter()
            .map(|&level| {
                let mut cumulative = 0.0;
                let mut var_branch = branches.last().unwrap();
                for branch in &branches {
                    cumulative += branch.probability / total;
                    if cumulative >= level {
                        var_branch = branch;
                        break;
                    }
                }
                let tail_mass = 1.0 - level;
                let mut remaining = tail_mass;
                let mut sum = 0.0;
                let mut by_factor = vec![0.0; outcomes.len()];
                for branch in branches.iter().rev() {
                    if remaining <= 0.0 {
                        break;
                    }
                    let included = (branch.probability / total).min(remaining);
                    sum += branch.impact * included;
                    for (index, (_, value)) in branch.factor_impacts.iter().enumerate() {
                        by_factor[index] += value * included;
                    }
                    remaining -= included;
                }
                let attribution = (0..outcomes.len())
                    .map(|index| {
                        (
                            var_branch.factor_impacts[index].1,
                            by_factor[index] / tail_mass,
                        )
                    })
                    .collect();
                (
                    var_branch.impact,
                    (sum / tail_mass).max(var_branch.impact),
                    attribution,
                )
            })
            .collect()
    }

    fn sample_outcomes(count: usize, weighted: bool) -> Vec<BinaryOutcome> {
        (0..count)
            .map(|index| {
                let downside_probability = 0.05 + 0.07 * ((index * 7) % 11) as f64 / 11.0;
                BinaryOutcome {
                    factor: format!("factor_{index}"),
                    downside_probability,
                    // Repeated impacts create ties to exercise sort stability.
                    downside_impact: 1.0 + (index % 3) as f64,
                    upside_probability: 1.0 - downside_probability,
                    upside_impact: if index % 4 == 0 { -0.5 } else { 0.0 },
                    weight: if weighted { 0.5 + index as f64 } else { 0.0 },
                }
            })
            .collect()
    }

    #[test]
    fn lean_risk_matches_reference_bit_for_bit() {
        let levels = [0.1, 0.5, 0.9, 0.95, 0.99, 0.999];
        for weighted in [true, false] {
            for count in [1, 2, 5, 9, 12] {
                let outcomes = sample_outcomes(count, weighted);
                let expected_branches = reference_branches(&outcomes);
                let branches = compute_spanning_set(&outcomes, None).unwrap();
                assert_eq!(branches.len(), expected_branches.len());
                for (left, right) in branches.iter().zip(&expected_branches) {
                    assert_eq!(left.choices, right.choices);
                    assert_eq!(left.probability.to_bits(), right.probability.to_bits());
                    assert_eq!(left.impact.to_bits(), right.impact.to_bits());
                    assert_eq!(left.factor_impacts, right.factor_impacts);
                }
                let result = compute_risk(&outcomes, &levels, None).unwrap();
                let expected = reference_risk(&outcomes, &levels);
                for (level, (var, cvar, attribution)) in result.levels.iter().zip(expected) {
                    assert_eq!(level.var.to_bits(), var.to_bits());
                    assert_eq!(level.cvar.to_bits(), cvar.to_bits());
                    for (item, (var_impact, cvar_impact)) in
                        level.attribution.iter().zip(attribution)
                    {
                        assert_eq!(item.var_impact.to_bits(), var_impact.to_bits());
                        assert_eq!(item.cvar_impact.to_bits(), cvar_impact.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn huge_branch_limits_are_rejected_instead_of_aborting() {
        for factors in [27, 40, 63, 64, 70] {
            let outcomes = sample_outcomes(factors, true);
            for limit in [None, Some(usize::MAX), Some(1usize << 40)] {
                let error = compute_risk(&outcomes, &[0.95], limit).unwrap_err();
                assert!(
                    matches!(error, CrcError::BranchLimitExceeded { limit, .. } if limit <= MAX_SPANNING_BRANCHES),
                    "{error:?}"
                );
                assert!(compute_spanning_set(&outcomes, limit).is_err());
            }
        }
    }
}

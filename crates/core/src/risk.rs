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

pub fn compute_spanning_set(
    outcomes: &[BinaryOutcome],
    max_branches: Option<usize>,
) -> Result<Vec<SpanningBranch>> {
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
    let branch_count =
        1usize
            .checked_shl(outcomes.len() as u32)
            .ok_or_else(|| CrcError::BranchLimitExceeded {
                factors: outcomes.len(),
                branches: usize::MAX,
                limit: max_branches.unwrap_or(usize::MAX),
            })?;
    let limit = max_branches.unwrap_or(1 << 20);
    if branch_count > limit {
        return Err(CrcError::BranchLimitExceeded {
            factors: outcomes.len(),
            branches: branch_count,
            limit,
        });
    }
    let weight_sum = outcomes.iter().map(|outcome| outcome.weight).sum::<f64>();
    let normalize = if weight_sum > 0.0 {
        weight_sum
    } else {
        outcomes.len() as f64
    };
    let mut branches = Vec::with_capacity(branch_count);
    for mask in 0..branch_count {
        let mut choices = Vec::with_capacity(outcomes.len());
        let mut probability = 1.0;
        let mut impact = 0.0;
        let mut factor_impacts = Vec::with_capacity(outcomes.len());
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
            let weighted_impact = raw_impact * weight;
            impact += weighted_impact;
            factor_impacts.push((outcome.factor.clone(), weighted_impact));
        }
        branches.push(SpanningBranch {
            choices,
            probability,
            impact,
            factor_impacts,
        });
    }
    branches.sort_by(|left, right| left.impact.total_cmp(&right.impact));
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
    let branches = compute_spanning_set(outcomes, max_branches)?;
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
        for branch in &branches {
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
            for (index, (_, factor_impact)) in branch.factor_impacts.iter().enumerate() {
                cvar_by_factor[index] += factor_impact * included;
            }
            remaining -= included;
        }
        let cvar = cvar_sum / tail_mass;
        let attribution = outcomes
            .iter()
            .enumerate()
            .map(|(index, outcome)| RiskAttribution {
                factor: outcome.factor.clone(),
                var_impact: var_branch.factor_impacts[index].1,
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
}

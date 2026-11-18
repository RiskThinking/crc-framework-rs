//! Independent binary rank outcomes and exact tail-risk attribution.

use crate::error::{CrcError, Result};
use crate::numerical::{CompensatedSum, compensated};
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct BinaryOutcome {
    pub id: String,
    pub lower_rank_mass: f64,
    pub lower_impact: f64,
    pub upper_rank_mass: f64,
    pub upper_impact: f64,
}

#[derive(Clone, Debug)]
pub struct RiskAttribution {
    pub id: String,
    pub value_at_risk: f64,
    pub conditional_value_at_risk: f64,
}

#[derive(Clone, Debug)]
pub struct RiskLevel {
    pub probability: f64,
    pub value_at_risk: f64,
    pub conditional_value_at_risk: f64,
    pub attribution: Vec<RiskAttribution>,
}

#[derive(Clone, Debug)]
pub struct RiskResult {
    pub levels: Vec<RiskLevel>,
    pub branch_count: usize,
}

#[derive(Clone, Copy, Debug)]
struct Branch {
    mask: usize,
    mass: f64,
    value: f64,
}

const DEFAULT_MAX_BRANCHES: usize = 1 << 20;
const HARD_MAX_BRANCHES: usize = 1 << 26;

fn contribution(outcome: &BinaryOutcome, mask: usize, index: usize) -> f64 {
    if mask & (1 << index) == 0 {
        outcome.lower_impact
    } else {
        outcome.upper_impact
    }
}

fn sum_impacts(impacts: impl Iterator<Item = f64> + Clone) -> f64 {
    let total = compensated(impacts.clone());
    if total.is_finite() {
        return total;
    }
    // With at most 26 factors, this power of two prevents intermediate
    // overflow. Keep values that would enter the subnormal range separate so
    // they survive cancellation of the large terms.
    const SCALE: f64 = HARD_MAX_BRANCHES.ilog2().next_power_of_two() as f64;
    const CUTOFF: f64 = f64::MIN_POSITIVE * SCALE;
    let large = compensated(
        impacts
            .clone()
            .filter(|impact| impact.abs() >= CUTOFF)
            .map(|impact| impact / SCALE),
    ) * SCALE;
    large + compensated(impacts.filter(|impact| impact.abs() < CUTOFF))
}

/// Sum independent impacts. There is no implicit weighting or averaging.
pub fn compute_risk(
    outcomes: &[BinaryOutcome],
    levels: &[f64],
    max_branches: Option<usize>,
) -> Result<RiskResult> {
    let invalid = |message: &str| CrcError::InvalidInput(message.into());
    if outcomes.is_empty() {
        return Err(invalid("at least one binary outcome is required"));
    }
    if levels.is_empty()
        || levels
            .iter()
            .any(|q| !q.is_finite() || *q <= 0.0 || *q >= 1.0)
    {
        return Err(invalid(
            "risk levels must be finite and strictly between zero and one",
        ));
    }
    let mut ids = HashSet::new();
    for outcome in outcomes {
        if outcome.id.is_empty() || outcome.id.trim() != outcome.id || !ids.insert(&outcome.id) {
            return Err(invalid(
                "binary outcome ids must be exact, nonempty, and unique",
            ));
        }
        if !outcome.lower_rank_mass.is_finite()
            || !outcome.upper_rank_mass.is_finite()
            || outcome.lower_rank_mass < 0.0
            || outcome.upper_rank_mass < 0.0
            || (outcome.lower_rank_mass + outcome.upper_rank_mass - 1.0).abs() > 1e-12
            || !outcome.lower_impact.is_finite()
            || !outcome.upper_impact.is_finite()
        {
            return Err(invalid(
                "rank masses must sum to one and impacts must be finite",
            ));
        }
    }
    let limit = max_branches
        .unwrap_or(DEFAULT_MAX_BRANCHES)
        .min(HARD_MAX_BRANCHES);
    let exceeded = |count| CrcError::BranchLimitExceeded {
        factors: outcomes.len(),
        branches: count,
        limit,
    };
    let count = u32::try_from(outcomes.len())
        .ok()
        .and_then(|n| 1usize.checked_shl(n))
        .ok_or_else(|| exceeded(usize::MAX))?;
    if count > limit {
        return Err(exceeded(count));
    }
    // Use a fixed enumeration for products, sums and tied branch weights while
    // returning attributions in the order requested by the caller.
    let mut factors: Vec<_> = outcomes.iter().enumerate().collect();
    factors.sort_unstable_by(|(_, a), (_, b)| {
        a.lower_rank_mass
            .total_cmp(&b.lower_rank_mass)
            .then_with(|| a.upper_rank_mass.total_cmp(&b.upper_rank_mass))
            .then_with(|| a.lower_impact.total_cmp(&b.lower_impact))
            .then_with(|| a.upper_impact.total_cmp(&b.upper_impact))
    });
    let mut branches = Vec::new();
    branches
        .try_reserve_exact(count)
        .map_err(|_| exceeded(count))?;
    for mask in 0..count {
        let Some(mass) = factors
            .iter()
            .enumerate()
            .map(|(i, (_, outcome))| {
                if mask & (1 << i) == 0 {
                    outcome.lower_rank_mass
                } else {
                    outcome.upper_rank_mass
                }
            })
            .try_fold(1.0, |mass, state_mass| {
                (state_mass > 0.0).then_some(mass * state_mass)
            })
        else {
            continue;
        };
        let value = sum_impacts(
            factors
                .iter()
                .enumerate()
                .map(|(i, (_, outcome))| contribution(outcome, mask, i)),
        );
        if !value.is_finite() {
            return Err(invalid("aggregated impacts overflowed"));
        }
        if mass > 0.0 {
            branches.push(Branch { mask, mass, value });
        }
    }
    branches.sort_unstable_by(|a, b| a.value.total_cmp(&b.value).then(a.mask.cmp(&b.mask)));
    let total = compensated(branches.iter().map(|b| b.mass));
    let mut result = Vec::with_capacity(levels.len());
    for &q in levels {
        let cutoff = q * total;
        let mut cumulative = CompensatedSum::default();
        let mut quantile = branches.last().expect("positive total mass").value;
        // Compare the compensated prefix and cutoff in the same mass units.
        for branch in &branches {
            cumulative.add(branch.mass);
            if cumulative.total() >= cutoff {
                quantile = branch.value;
                break;
            }
        }
        // Tied VaR states have probability-weighted attribution, independent of order.
        let tied = branches.iter().filter(|b| b.value == quantile);
        let tied_mass = compensated(tied.clone().map(|b| b.mass));
        let mut var_parts = vec![0.0; outcomes.len()];
        for branch in tied {
            for (i, &(index, outcome)) in factors.iter().enumerate() {
                var_parts[index] +=
                    contribution(outcome, branch.mask, i) * (branch.mass / tied_mass);
            }
        }
        let tail_mass = (1.0 - q) * total;
        let mut remaining = tail_mass;
        let mut cvar = 0.0;
        let mut cvar_parts = vec![0.0; outcomes.len()];
        for tied in branches.chunk_by(|a, b| a.value == b.value).rev() {
            let group_mass = compensated(tied.iter().map(|branch| branch.mass));
            let used = group_mass.min(remaining);
            let weight = used / tail_mass;
            cvar += tied[0].value * weight;
            // A partial cutoff includes the same fraction of each tied state.
            for branch in tied {
                let weight = weight * (branch.mass / group_mass);
                for (i, &(index, outcome)) in factors.iter().enumerate() {
                    cvar_parts[index] += contribution(outcome, branch.mask, i) * weight;
                }
            }
            remaining -= used;
            if remaining <= 0.0 {
                break;
            }
        }
        result.push(RiskLevel {
            probability: q,
            value_at_risk: quantile,
            conditional_value_at_risk: cvar,
            attribution: outcomes
                .iter()
                .enumerate()
                .map(|(i, o)| RiskAttribution {
                    id: o.id.clone(),
                    value_at_risk: var_parts[i],
                    conditional_value_at_risk: cvar_parts[i],
                })
                .collect(),
        });
    }
    Ok(RiskResult {
        levels: result,
        branch_count: count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(id: &str, upper_rank_mass: f64, upper_impact: f64) -> BinaryOutcome {
        BinaryOutcome {
            id: id.into(),
            lower_rank_mass: 1.0 - upper_rank_mass,
            lower_impact: 0.0,
            upper_rank_mass,
            upper_impact,
        }
    }

    #[test]
    fn retains_rare_catastrophe() {
        let result = compute_risk(&[outcome("rare", 0.000001, 1e6)], &[0.95], None).unwrap();
        assert!((result.levels[0].conditional_value_at_risk - 20.0).abs() < 1e-10);
    }

    #[test]
    fn sums_without_hidden_weights_and_attribution_agrees() {
        let result = compute_risk(
            &[outcome("a", 0.1, 1.0), outcome("b", 0.2, 2.0)],
            &[0.95],
            None,
        )
        .unwrap();
        let level = &result.levels[0];
        assert_eq!(level.value_at_risk, 2.0);
        assert!((level.conditional_value_at_risk - 2.4).abs() < 1e-12);
        assert!(
            (level
                .attribution
                .iter()
                .map(|p| p.value_at_risk)
                .sum::<f64>()
                - level.value_at_risk)
                .abs()
                < 1e-12
        );
        assert!(
            (level
                .attribution
                .iter()
                .map(|p| p.conditional_value_at_risk)
                .sum::<f64>()
                - level.conditional_value_at_risk)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn tied_var_and_cvar_attribution_use_probability_weights() {
        let a = outcome("a", 0.25, 1.0);
        let b = outcome("b", 0.5, 1.0);
        let result = compute_risk(&[a, b], &[0.625], None).unwrap();
        let level = &result.levels[0];
        assert_eq!(level.value_at_risk, 1.0);
        assert_eq!(level.attribution[0].value_at_risk, 0.25);
        assert_eq!(level.attribution[1].value_at_risk, 0.75);
        assert!((level.conditional_value_at_risk - 4.0 / 3.0).abs() < 1e-12);
        assert_eq!(level.attribution[0].conditional_value_at_risk, 0.5);
        assert!((level.attribution[1].conditional_value_at_risk - 5.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn validates_identity_and_allocation_limits() {
        let a = outcome("a", 0.1, 1.0);
        assert!(compute_risk(&[a.clone(), a], &[0.5], None).is_err());
        let large: Vec<_> = (0..70).map(|i| outcome(&i.to_string(), 0.1, 1.0)).collect();
        assert!(matches!(
            compute_risk(&large, &[0.95], Some(usize::MAX)),
            Err(CrcError::BranchLimitExceeded { .. })
        ));
    }
}

//! Independent binary rank outcomes and exact tail-risk attribution.

use crate::error::{CrcError, Result};
use crate::numerical::{CompensatedSum, compensated, linear_interpolate};
use num_bigint::{BigInt, BigUint};
use num_rational::BigRational;
use num_traits::ToPrimitive;
use std::cell::OnceCell;
use std::cmp::Ordering;
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

type Factors<'a> = [(usize, &'a BinaryOutcome)];

fn upper_state(mask: usize, index: usize) -> usize {
    (mask >> index) & 1
}

fn contribution(outcome: &BinaryOutcome, mask: usize, index: usize) -> f64 {
    if upper_state(mask, index) == 0 {
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

// A binary64 value is an integer times a power of two. Remove trailing zeroes
// so the common integer representations stay small for ordinary inputs.
fn dyadic(value: f64) -> (u64, i32) {
    let bits = value.to_bits();
    let field = ((bits >> 52) & 2047) as i32;
    let fraction = bits & ((1 << 52) - 1);
    let (coefficient, exponent) = if field == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1 << 52), field - 1075)
    };
    let zeroes = coefficient.trailing_zeros().min(63);
    (coefficient >> zeroes, exponent + zeroes as i32)
}

struct ImpactOrder<'a> {
    factors: &'a Factors<'a>,
    rounding_margin: f64,
    varying_mask: usize,
    exact: OnceCell<ExactImpactSums>,
}

impl<'a> ImpactOrder<'a> {
    fn new(factors: &'a Factors<'a>) -> Self {
        let mut maximum = 0.0_f64;
        let mut varying_mask = 0;
        for (i, (_, outcome)) in factors.iter().enumerate() {
            maximum = maximum.max(outcome.lower_impact.abs().max(outcome.upper_impact.abs()));
            if outcome.lower_impact != outcome.upper_impact {
                varying_mask |= 1 << i;
            }
        }
        // Bound the combined rounding errors of two sums of at most 26
        // impacts, including the scaled overflow fallback and subnormals.
        let rounding_margin = (1024.0 * f64::EPSILON) * maximum + 1024.0 * f64::from_bits(1);
        Self {
            factors,
            rounding_margin,
            varying_mask,
            exact: OnceCell::new(),
        }
    }

    fn compare(&self, a: &Branch, b: &Branch) -> Ordering {
        if (a.mask ^ b.mask) & self.varying_mask == 0 {
            Ordering::Equal
        } else if (a.value - b.value).abs() > self.rounding_margin {
            a.value.total_cmp(&b.value)
        } else {
            self.exact
                .get_or_init(|| ExactImpactSums::new(self.factors))
                .compare(a.mask, b.mask)
        }
    }
}

/// Exact branch ordering from each upper state's increment over the common
/// all-lower baseline. Two half tables keep storage below the branch count.
struct ExactImpactSums {
    low: Vec<BigInt>,
    high: Vec<BigInt>,
    split: usize,
}

impl ExactImpactSums {
    fn new(factors: &Factors<'_>) -> Self {
        let states = factors
            .iter()
            .flat_map(|(_, outcome)| [outcome.lower_impact, outcome.upper_impact]);
        let exponent = states
            .clone()
            .filter(|value| *value != 0.0)
            .map(|value| dyadic(value).1)
            .min()
            .unwrap_or(0);
        let integer = |value: f64| {
            let (mantissa, power) = dyadic(value);
            let magnitude = BigInt::from(mantissa) << (power - exponent).max(0) as usize;
            if value.is_sign_negative() {
                -magnitude
            } else {
                magnitude
            }
        };
        let increments: Vec<_> = factors
            .iter()
            .map(|(_, outcome)| integer(outcome.upper_impact) - integer(outcome.lower_impact))
            .collect();
        let split = factors.len() / 2;
        Self {
            low: Self::half_sums(&increments[..split]),
            high: Self::half_sums(&increments[split..]),
            split,
        }
    }

    fn half_sums(increments: &[BigInt]) -> Vec<BigInt> {
        let mut sums = Vec::with_capacity(1 << increments.len());
        sums.push(BigInt::ZERO);
        for increment in increments {
            for i in 0..sums.len() {
                sums.push(&sums[i] + increment);
            }
        }
        sums
    }

    fn compare(&self, a: usize, b: usize) -> Ordering {
        let low_mask = self.low.len() - 1;
        let (a_low, b_low) = (&self.low[a & low_mask], &self.low[b & low_mask]);
        let (a_high, b_high) = (&self.high[a >> self.split], &self.high[b >> self.split]);
        let low = a_low.cmp(b_low);
        let high = a_high.cmp(b_high);
        if low == high || high.is_eq() {
            low
        } else if low.is_eq() {
            high
        } else {
            (a_low - b_low).cmp(&(b_high - a_high))
        }
    }
}

struct ExactMasses {
    factors: Vec<[BigUint; 2]>,
    split: usize,
    low: Vec<BigUint>,
    high: Vec<BigUint>,
    total: BigUint,
}

impl ExactMasses {
    fn new(factors: &Factors<'_>) -> Self {
        let factors: Vec<_> = factors
            .iter()
            .map(|(_, outcome)| {
                let pair = [
                    dyadic(outcome.lower_rank_mass),
                    dyadic(outcome.upper_rank_mass),
                ];
                let exponent = pair
                    .iter()
                    .filter_map(|&(n, e)| (n != 0).then_some(e))
                    .min()
                    .expect("validated positive rank mass");
                pair.map(|(n, e)| BigUint::from(n) << (e - exponent).max(0) as usize)
            })
            .collect();
        let total = factors.iter().map(|[low, high]| low + high).product();
        let split = factors.len() / 2;
        // Meet in the middle: any exact branch mass costs one multiplication,
        // and the two tables use O(sqrt(branch count)) space.
        let low = Self::half_products(&factors[..split]);
        let high = Self::half_products(&factors[split..]);
        Self {
            factors,
            split,
            low,
            high,
            total,
        }
    }

    fn half_products(factors: &[[BigUint; 2]]) -> Vec<BigUint> {
        let mut products = Vec::with_capacity(1 << factors.len());
        products.push(BigUint::from(1u8));
        for [low, high] in factors {
            for i in 0..products.len() {
                let lower = &products[i] * low;
                let upper = &products[i] * high;
                products[i] = lower;
                products.push(upper);
            }
        }
        products
    }

    fn at(&self, mask: usize) -> BigUint {
        &self.low[mask & (self.low.len() - 1)] * &self.high[mask >> self.split]
    }

    fn cutoff(&self, q: f64) -> (BigUint, usize) {
        let (coefficient, exponent) = dyadic(q);
        (&self.total * coefficient, (-exponent) as usize)
    }

    fn quantile_cutoffs(&self, q: f64) -> (BigUint, BigUint, usize) {
        // Compare the normalized prefix with the midpoint between q and the
        // preceding binary64 value without rounding the midpoint itself.
        let previous = f64::from_bits(q.to_bits() - 1);
        let (_, gap_exponent) = dyadic(q - previous);
        let shift = (-gap_exponent + 1) as usize;
        let (cutoff, cutoff_shift) = self.cutoff(q);
        let cutoff = cutoff << (shift - cutoff_shift);
        let midpoint = &cutoff - &self.total;
        (cutoff, midpoint, shift)
    }
}

fn quantile_indices(
    branches: &[Branch],
    total: f64,
    q: f64,
    factors: &Factors<'_>,
    exact: &mut Option<ExactMasses>,
) -> (usize, usize) {
    let cutoff = q * total;
    let mut cumulative = CompensatedSum::default();
    let mut exact_prefix = BigUint::ZERO;
    let mut next_exact = 0;
    let mut exact_cutoffs = None;
    let mut var_index = None;
    let round_midpoint_up = q.to_bits() & 1 == 0;
    // Each joint product has at most 26 factors. The guard accounts for
    // rounding on both sides, including products rounded into subnormals.
    let margin = 64.0 * f64::EPSILON * cutoff + 64.0 * f64::from_bits(1) * branches.len() as f64;
    for (index, branch) in branches.iter().enumerate() {
        cumulative.add(branch.mass);
        let difference = cumulative.total() - cutoff;
        if difference > margin {
            return (var_index.unwrap_or(index), index);
        }
        if difference >= -margin {
            let masses = exact.get_or_insert_with(|| ExactMasses::new(factors));
            let (target, midpoint, shift) =
                exact_cutoffs.get_or_insert_with(|| masses.quantile_cutoffs(q));
            for branch in &branches[next_exact..=index] {
                exact_prefix += masses.at(branch.mask);
            }
            next_exact = index + 1;
            let prefix = &exact_prefix << *shift;
            if prefix >= *target {
                return (var_index.unwrap_or(index), index);
            }
            if var_index.is_none()
                && (prefix > *midpoint || round_midpoint_up && prefix == *midpoint)
            {
                var_index = Some(index);
            }
        }
    }
    let last = branches.len() - 1;
    (var_index.unwrap_or(last), last)
}

fn impact_group(branches: &[Branch], index: usize, order: &ImpactOrder<'_>) -> (usize, usize) {
    let lo = branches[..index].partition_point(|b| order.compare(b, &branches[index]).is_lt());
    let hi =
        index + branches[index..].partition_point(|b| order.compare(b, &branches[index]).is_eq());
    (lo, hi)
}

fn add_state_masses(states: &mut [[CompensatedSum; 2]], mask: usize, mass: f64) {
    for (i, pair) in states.iter_mut().enumerate() {
        pair[upper_state(mask, i)].add(mass);
    }
}

fn exact_factor_mean(factor: &BinaryOutcome, upper_probability: BigRational) -> BigRational {
    let lower = BigRational::from_float(factor.lower_impact).expect("finite impact");
    let upper = BigRational::from_float(factor.upper_impact).expect("finite impact");
    &lower + (upper - &lower) * upper_probability
}

fn conditioned_attribution(
    states: &[[CompensatedSum; 2]],
    factors: &Factors<'_>,
) -> Option<Vec<f64>> {
    let mut parts = vec![0.0; factors.len()];
    for (&(index, factor), pair) in factors.iter().zip(states) {
        let [lower, upper] = pair.each_ref().map(CompensatedSum::total);
        let (a, b) = (factor.lower_impact, factor.upper_impact);
        // Use the smaller state mass as the interpolation weight so a rare
        // state survives even when the other state's weight rounds to one.
        let (start, end) = if upper <= lower { (a, b) } else { (b, a) };
        let weight = lower.min(upper) / (lower + upper);
        let value = linear_interpolate(start, end, weight);
        if value.abs() <= 64.0 * f64::EPSILON * (start * (1.0 - weight)).abs()
            && start.is_sign_negative() != end.is_sign_negative()
            && weight > 0.0
        {
            // Rounding the state fractions separately can erase or invent
            // contributions when two large terms nearly cancel.
            return None;
        }
        parts[index] = value;
    }
    Some(parts)
}

fn var_attribution(
    tied: &[Branch],
    factors: &Factors<'_>,
    exact: &mut Option<ExactMasses>,
) -> Vec<f64> {
    let mut states = Vec::from_iter((0..factors.len()).map(|_| Default::default()));
    if tied.iter().all(|branch| branch.mass >= f64::MIN_POSITIVE) {
        for branch in tied {
            add_state_masses(&mut states, branch.mask, branch.mass);
        }
        if let Some(parts) = conditioned_attribution(&states, factors) {
            return parts;
        }
    }
    let masses = exact.get_or_insert_with(|| ExactMasses::new(factors));
    let mut total = BigUint::ZERO;
    let mut upper = vec![BigUint::ZERO; factors.len()];
    for branch in tied {
        let weight = masses.at(branch.mask);
        for (i, state) in upper.iter_mut().enumerate() {
            if upper_state(branch.mask, i) != 0 {
                *state += &weight;
            }
        }
        total += weight;
    }
    let total = BigInt::from(total);
    let mut parts = vec![0.0; factors.len()];
    for (&(index, factor), upper) in factors.iter().zip(upper) {
        let probability = BigRational::new_raw(BigInt::from(upper), total.clone());
        parts[index] = exact_factor_mean(factor, probability)
            .to_f64()
            .expect("finite VaR attribution");
    }
    parts
}

// Integrate just the excluded prefix in exact binary arithmetic and subtract
// it from the unconditional per-factor means. This preserves tiny excluded
// mass and avoids cancellation from rounding large impact contributions.
fn exact_tail(
    branches: &[Branch],
    below_count: usize,
    q: f64,
    factors: &Factors<'_>,
    masses: &ExactMasses,
) -> (f64, Vec<f64>) {
    let (cutoff, shift) = masses.cutoff(q);
    let mut below_mass = BigUint::ZERO;
    let mut tied_mass = BigUint::ZERO;
    let mut below_upper_mass = vec![BigUint::ZERO; factors.len()];
    let mut tied_upper_mass = vec![BigUint::ZERO; factors.len()];
    for (i, branch) in branches.iter().enumerate() {
        let weight = masses.at(branch.mask);
        let (total, by_factor) = if i < below_count {
            (&mut below_mass, &mut below_upper_mass)
        } else {
            (&mut tied_mass, &mut tied_upper_mass)
        };
        for (i, total) in by_factor.iter_mut().enumerate() {
            if branch.mask & (1 << i) != 0 {
                *total += &weight;
            }
        }
        *total += weight;
    }
    let tied_excluded = &cutoff - (&below_mass << shift);
    let full = &masses.total << shift;
    let tail = BigRational::new_raw(BigInt::from(&full - &cutoff), BigInt::from(full.clone()));
    let denominator = BigInt::from(full * &tied_mass);
    let mut cvar = BigRational::from_integer(BigInt::ZERO);
    let mut parts = vec![0.0; factors.len()];
    for (i, &(index, factor)) in factors.iter().enumerate() {
        let [lower, upper] = &masses.factors[i];
        let upper_probability =
            BigRational::new_raw(BigInt::from(upper.clone()), BigInt::from(lower + upper));
        let excluded_upper_numerator =
            ((&below_upper_mass[i] * &tied_mass) << shift) + &tied_upper_mass[i] * &tied_excluded;
        let excluded_upper =
            BigRational::new_raw(BigInt::from(excluded_upper_numerator), denominator.clone());
        let upper_in_tail = (upper_probability - excluded_upper) / &tail;
        let part = exact_factor_mean(factor, upper_in_tail);
        parts[index] = part.to_f64().expect("finite tail mean");
        cvar += part;
    }
    (cvar.to_f64().expect("finite tail mean"), parts)
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
        branches.push(Branch { mask, mass, value });
    }
    let order = ImpactOrder::new(&factors);
    branches.sort_unstable_by(|a, b| order.compare(a, b).then(a.mask.cmp(&b.mask)));
    let total = compensated(branches.iter().map(|b| b.mass));
    let rounded_mass = branches.iter().any(|b| b.mass < f64::MIN_POSITIVE);
    let mut exact = None;
    let mut result = Vec::with_capacity(levels.len());
    for &q in levels {
        let (index, tail_index) = quantile_indices(&branches, total, q, &factors, &mut exact);
        let quantile = branches[index].value;
        // Tied VaR states have probability-weighted attribution, independent of order.
        let (lo, hi) = impact_group(&branches, index, &order);
        let var_parts = var_attribution(&branches[lo..hi], &factors, &mut exact);
        let approximate = if q < 64.0 * f64::EPSILON || rounded_mass {
            None
        } else {
            upper_tail(&branches, (1.0 - q) * total, &order)
        };
        let (cvar, cvar_parts) = approximate.unwrap_or_else(|| {
            let exact = exact.get_or_insert_with(|| ExactMasses::new(&factors));
            // CVaR integrates at the unrounded probability even if the VaR
            // boundary rounds to q one impact group earlier.
            let (lo, hi) = if tail_index == index {
                (lo, hi)
            } else {
                impact_group(&branches, tail_index, &order)
            };
            exact_tail(&branches[..hi], lo, q, &factors, exact)
        });
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

fn upper_tail(
    branches: &[Branch],
    tail_mass: f64,
    order: &ImpactOrder<'_>,
) -> Option<(f64, Vec<f64>)> {
    let factors = order.factors;
    let mut included = CompensatedSum::default();
    let mut cvar = CompensatedSum::default();
    let mut largest_positive = 0.0_f64;
    let mut largest_negative = 0.0_f64;
    let mut states = Vec::from_iter((0..factors.len()).map(|_| Default::default()));
    let margin = 64.0 * f64::EPSILON * tail_mass;
    for tied in branches.chunk_by(|a, b| order.compare(a, b).is_eq()).rev() {
        let group_mass = compensated(tied.iter().map(|branch| branch.mass));
        let remaining = tail_mass - included.total();
        included.add(group_mass);
        // A rounded boundary can hide positive mass from the next group,
        // whose impacts may dominate the tail mean even at very small weight.
        let difference = tail_mass - included.total();
        if difference.abs() <= margin {
            return None;
        }
        let used = group_mass.min(remaining);
        let weight = used / tail_mass;
        let contribution = tied[0].value * weight;
        largest_positive = largest_positive.max(contribution);
        largest_negative = largest_negative.min(contribution);
        cvar.add(contribution);
        // A partial cutoff includes the same fraction of each tied state.
        for branch in tied {
            add_state_masses(
                &mut states,
                branch.mask,
                weight * (branch.mass / group_mass),
            );
        }
        if difference < 0.0 {
            break;
        }
    }
    let value = cvar.total();
    if !value.is_finite()
        || largest_negative < 0.0
            && largest_positive > 0.0
            && value.abs() <= 64.0 * f64::EPSILON * largest_positive.max(-largest_negative)
    {
        return None;
    }
    Some((value, conditioned_attribution(&states, factors)?))
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

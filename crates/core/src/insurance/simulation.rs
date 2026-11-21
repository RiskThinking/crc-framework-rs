use std::sync::Arc;

use super::PolicyTerms;
use super::dependence::{CorrelationRepair, factor_coordinates, validate_dependence};
use super::sampling::{Draws, SeededDraws, normal_rank};
use crate::numerical::{CompensatedSum, compensated};
use crate::{CrcError, Distribution, Interpolation, Result, TabulatedDistribution};

/// Scientific and random-stream settings for a single peril.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnnualLossSpec {
    pub trials: usize,
    pub seed: u64,
    pub peak_correlation: f64,
    pub decay_length_km: f64,
    pub damage_cv: f64,
    /// Maximum permitted change in any correlation entry. None requires the
    /// original correlation matrix to be positive definite.
    pub repair_tolerance: Option<f64>,
}

impl Default for AnnualLossSpec {
    fn default() -> Self {
        Self {
            trials: 20_000,
            seed: 20_261_027,
            peak_correlation: 0.4,
            decay_length_km: 150.0,
            damage_cv: 0.3,
            repair_tolerance: None,
        }
    }
}

impl AnnualLossSpec {
    pub fn validate(&self) -> Result<()> {
        if self.trials == 0 {
            return Err(invalid("trials must be positive"));
        }
        validate_dependence(
            self.peak_correlation,
            self.decay_length_km,
            self.repair_tolerance,
        )?;
        if !self.damage_cv.is_finite() || self.damage_cv < 0.0 {
            return Err(invalid("damage_cv must be finite and nonnegative"));
        }
        Ok(())
    }
}

/// Mean damage ratio against complete non-exceedance ranks.
#[derive(Clone, Debug)]
pub struct DamageCurve(Arc<TabulatedDistribution>);

impl DamageCurve {
    pub fn new(probabilities: Vec<f64>, ratios: Vec<f64>) -> Result<Self> {
        if probabilities.first() != Some(&0.0) || probabilities.last() != Some(&1.0) {
            return Err(invalid(
                "damage curves must span ranks 0 and 1; supply explicit endpoints",
            ));
        }
        if ratios.iter().any(|ratio| !(0.0..=1.0).contains(ratio)) {
            return Err(invalid("damage ratios must be finite and in [0, 1]"));
        }
        TabulatedDistribution::new(
            probabilities,
            ratios,
            Interpolation::LinearProbability,
            false,
        )
        .map(|curve| Self(Arc::new(curve)))
    }
}

/// One insured interest, with resolved primary terms and its mean damage curve.
#[derive(Clone, Debug)]
pub struct InsuredInterest {
    pub id: String,
    pub longitude: f64,
    pub latitude: f64,
    pub total_insured_value: f64,
    pub terms: PolicyTerms,
    pub damage_curve: DamageCurve,
}

impl InsuredInterest {
    pub fn validate(&self) -> Result<()> {
        if self.id.is_empty() || self.id.trim() != self.id {
            return Err(invalid(
                "interest IDs must be nonempty without boundary whitespace",
            ));
        }
        if !(-180.0..=180.0).contains(&self.longitude) || !(-90.0..=90.0).contains(&self.latitude) {
            return Err(invalid(format!(
                "interest {:?} requires finite WGS84 longitude and latitude",
                self.id,
            )));
        }
        if !self.total_insured_value.is_finite() || self.total_insured_value < 0.0 {
            return Err(invalid(format!(
                "interest {:?} requires a finite nonnegative total insured value",
                self.id,
            )));
        }
        Ok(())
    }
}

/// Execution limits and optional per-interest output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimulationOptions {
    /// Vary this between paired runs to separate secondary uncertainty while
    /// reusing the primary stream.
    pub secondary_stream: u64,
    pub include_components: bool,
    pub block_size: usize,
    pub memory_limit: usize,
    pub work_limit: u64,
}

impl Default for SimulationOptions {
    fn default() -> Self {
        Self {
            secondary_stream: 0,
            include_components: false,
            block_size: 256,
            memory_limit: 512 * 1024 * 1024,
            work_limit: 10_000_000_000,
        }
    }
}

/// Aligned insured losses, in zero-based trial order.
#[derive(Clone, Debug, PartialEq)]
pub struct AnnualLossSimulation {
    pub totals: Vec<f64>,
    pub components: Option<Vec<(String, Vec<f64>)>>,
    pub total_insured_value: f64,
    pub correlation_repair: Option<CorrelationRepair>,
}

fn invalid(message: impl Into<String>) -> CrcError {
    CrcError::InvalidInput(message.into())
}

fn check_budget(count: usize, spec: &AnnualLossSpec, options: &SimulationOptions) -> Result<()> {
    if options.block_size == 0 || options.memory_limit == 0 || options.work_limit == 0 {
        return Err(invalid(
            "block_size, memory_limit, and work_limit must be positive",
        ));
    }

    let n = count as u128;
    let trials = spec.trials as u128;
    let outputs = trials * (1 + if options.include_components { n } else { 0 });
    // Include one output copy for transfer and the reusable vectors, sorted
    // interest references, and per-component vector headers.
    let vectors = 16 * n;
    let dense = n > 1 && spec.peak_correlation > 0.0;
    let (matrix, work) = if dense {
        let square = n * n;
        let (buffers, passes) = if spec.repair_tolerance.is_some() {
            (5, 5)
        } else {
            (2, 1)
        };
        (
            square.saturating_mul(buffers),
            square
                .saturating_mul(n)
                .saturating_mul(passes)
                .saturating_add(trials.saturating_mul(n * (n + 1) / 2)),
        )
    } else {
        (0, 0)
    };
    let bytes = matrix
        .saturating_add(vectors)
        .saturating_add(outputs.saturating_mul(2))
        .saturating_mul(size_of::<f64>() as u128);
    if bytes > options.memory_limit as u128 {
        return Err(invalid(format!(
            "estimated simulation memory {bytes} bytes exceeds memory_limit={}; reduce trials, interests, or component output",
            options.memory_limit,
        )));
    }
    let work = work.saturating_add(n.saturating_mul(trials).saturating_mul(3));
    if work > options.work_limit as u128 {
        return Err(invalid(format!(
            "estimated simulation work {work} exceeds work_limit={}; reduce trials or interests",
            options.work_limit,
        )));
    }
    Ok(())
}

fn reserve_trials(count: usize) -> Result<Vec<f64>> {
    let mut losses = Vec::new();
    losses
        .try_reserve_exact(count)
        .map_err(|_| invalid("could not allocate annual output"))?;
    Ok(losses)
}

/// Simulate a book in one currency and valuation basis for a single peril.
///
/// Interests are sorted by ID before applying dependence, generating draws,
/// or summing a year. The same ordered book and spec replay primary ranks
/// across run blocks. Use separate secondary streams for a paired comparison.
pub fn simulate_annual_losses(
    interests: &[InsuredInterest],
    spec: &AnnualLossSpec,
    options: &SimulationOptions,
) -> Result<AnnualLossSimulation> {
    let mut draws = SeededDraws::new(spec.seed, options.secondary_stream);
    simulate_with_draws(interests, spec, options, &mut draws)
}

fn simulate_with_draws(
    interests: &[InsuredInterest],
    spec: &AnnualLossSpec,
    options: &SimulationOptions,
    draws: &mut impl Draws,
) -> Result<AnnualLossSimulation> {
    spec.validate()?;
    if interests.is_empty() {
        return Err(invalid(
            "annual loss simulation requires at least one insured interest",
        ));
    }
    check_budget(interests.len(), spec, options)?;
    let mut interests = interests.iter().collect::<Vec<_>>();
    interests.sort_unstable_by(|left, right| left.id.cmp(&right.id));
    for interest in &interests {
        interest.validate()?;
    }
    if let Some(pair) = interests.windows(2).find(|pair| pair[0].id == pair[1].id) {
        return Err(invalid(format!(
            "duplicate insured interest ID {:?}",
            pair[0].id
        )));
    }
    let total_insured_value = compensated(interests.iter().map(|item| item.total_insured_value));
    if !total_insured_value.is_finite() {
        return Err(invalid(
            "portfolio total insured value exceeds the representable range",
        ));
    }
    let coordinates = interests
        .iter()
        .map(|item| (item.longitude, item.latitude))
        .collect::<Vec<_>>();
    let (factor, correlation_repair) = factor_coordinates(
        &coordinates,
        spec.peak_correlation,
        spec.decay_length_km,
        spec.repair_tolerance,
    )?;
    let mut totals = reserve_trials(spec.trials)?;
    let mut components = options
        .include_components
        .then(|| {
            interests
                .iter()
                .map(|item| Ok((item.id.clone(), reserve_trials(spec.trials)?)))
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let mut independent = vec![0.0; interests.len()];
    let mut correlated = vec![0.0; interests.len()];
    for first in (0..spec.trials).step_by(options.block_size) {
        let block = (spec.trials - first).min(options.block_size);
        for _ in 0..block {
            independent.fill_with(|| draws.standard_normal());
            factor.apply(&independent, &mut correlated);
            let mut annual = CompensatedSum::default();
            for (position, (interest, &normal)) in interests.iter().zip(&correlated).enumerate() {
                let mean = interest.damage_curve.0.ppf(normal_rank(normal))?;
                let ratio = draws.damage_ratio(mean, spec.damage_cv)?;
                let loss = interest
                    .terms
                    .insured_loss(interest.total_insured_value, ratio)?;
                annual.add(loss);
                if let Some(components) = &mut components {
                    components[position].1.push(loss);
                }
            }
            let loss = annual.total();
            if !loss.is_finite() {
                return Err(invalid(
                    "annual insured loss exceeds the representable range",
                ));
            }
            totals.push(loss);
        }
    }
    Ok(AnnualLossSimulation {
        totals,
        components,
        total_insured_value,
        correlation_repair,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    fn interest(id: &str, value: f64, terms: PolicyTerms) -> InsuredInterest {
        InsuredInterest {
            id: id.to_owned(),
            longitude: 0.0,
            latitude: 0.0,
            total_insured_value: value,
            terms,
            damage_curve: DamageCurve::new(vec![0.0, 0.5, 1.0], vec![0.0, 0.2, 1.0]).unwrap(),
        }
    }

    struct FixedDraws {
        normals: VecDeque<f64>,
        ratios: VecDeque<f64>,
        means: Vec<f64>,
    }

    impl Draws for FixedDraws {
        fn standard_normal(&mut self) -> f64 {
            self.normals.pop_front().unwrap()
        }

        fn damage_ratio(&mut self, mean: f64, cv: f64) -> Result<f64> {
            assert_eq!(cv, 0.3);
            self.means.push(mean);
            Ok(self.ratios.pop_front().unwrap())
        }
    }

    #[test]
    fn injected_draws_follow_ranks_interpolation_terms_and_annual_aggregation() {
        let book = [
            interest("b", 50_000.0, PolicyTerms::new(0.0, None, 1.0).unwrap()),
            interest(
                "a",
                100_000.0,
                PolicyTerms::new(5_000.0, Some(15_000.0), 0.8).unwrap(),
            ),
        ];
        let spec = AnnualLossSpec {
            trials: 2,
            ..AnnualLossSpec::default()
        };
        let spec_options = SimulationOptions {
            include_components: true,
            block_size: 1,
            ..SimulationOptions::default()
        };
        let mut draws = FixedDraws {
            normals: [0.0, 0.6744897501960817 / 0.84_f64.sqrt(), 0.0, 0.0].into(),
            ratios: [0.26, 0.5, 0.0, 0.0].into(),
            means: Vec::new(),
        };
        let result = simulate_with_draws(&book, &spec, &spec_options, &mut draws).unwrap();
        for (actual, expected) in draws.means.into_iter().zip([0.2, 0.6, 0.2, 0.2]) {
            assert!((actual - expected).abs() < 1e-14);
        }
        assert!(draws.normals.is_empty() && draws.ratios.is_empty());
        assert_eq!(result.totals, [37_000.0, 0.0]);
        assert_eq!(result.total_insured_value, 150_000.0);
        assert_eq!(
            result.components.unwrap(),
            [
                ("a".into(), vec![12_000.0, 0.0]),
                ("b".into(), vec![25_000.0, 0.0]),
            ]
        );
        assert!(result.correlation_repair.is_none());
    }

    #[test]
    fn budgets_reject_before_generating_draws() {
        let book = [interest(
            "a",
            100.0,
            PolicyTerms::new(0.0, None, 1.0).unwrap(),
        )];
        let spec = AnnualLossSpec::default();
        let mut draws = FixedDraws {
            normals: VecDeque::new(),
            ratios: VecDeque::new(),
            means: Vec::new(),
        };
        for (options, message) in [
            (
                SimulationOptions {
                    memory_limit: 1,
                    ..SimulationOptions::default()
                },
                "memory_limit",
            ),
            (
                SimulationOptions {
                    work_limit: 1,
                    ..SimulationOptions::default()
                },
                "work_limit",
            ),
        ] {
            assert!(
                simulate_with_draws(&book, &spec, &options, &mut draws)
                    .unwrap_err()
                    .to_string()
                    .contains(message)
            );
        }
    }
}

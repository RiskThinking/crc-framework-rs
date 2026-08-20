use std::f64::consts::{PI, SQRT_2};

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    error::{CrcError, Result},
    reference::curve_fitting as reference_curve_fitting,
};

const EPS: f64 = 1.0e-12;

pub trait Distribution: Send + Sync {
    fn cdf(&self, x: f64) -> f64;
    fn ppf(&self, probability: f64) -> Result<f64>;

    fn probability_support(&self) -> (f64, f64) {
        (0.0, 1.0)
    }

    fn pdf(&self, x: f64) -> f64 {
        let h = 1.0e-5 * x.abs().max(1.0);
        ((self.cdf(x + h) - self.cdf(x - h)) / (2.0 * h)).max(0.0)
    }

    fn quantiles(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        probabilities.iter().map(|&q| self.ppf(q)).collect()
    }

    fn sample(&self, size: usize, seed: Option<u64>) -> Result<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed.unwrap_or_else(rand::random));
        let (minimum, maximum) = self.probability_support();
        let margin = (maximum - minimum) * EPS;
        (0..size)
            .map(|_| self.ppf(rng.random_range(minimum + margin..maximum - margin)))
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tail {
    Upper,
    Lower,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    LinearProbability,
    LogReturnPeriod,
}

#[derive(Debug, Clone)]
pub struct TabulatedDistribution {
    probabilities: Vec<f64>,
    values: Vec<f64>,
    interpolation: Interpolation,
    extrapolate: bool,
    return_period_tail: Tail,
}

impl TabulatedDistribution {
    pub fn new(
        probabilities: Vec<f64>,
        values: Vec<f64>,
        interpolation: Interpolation,
        extrapolate: bool,
    ) -> Result<Self> {
        if probabilities.len() != values.len() || probabilities.len() < 2 {
            return Err(CrcError::InvalidInput(
                "probabilities and values must have the same length of at least two".into(),
            ));
        }
        if probabilities
            .iter()
            .any(|q| !q.is_finite() || *q < 0.0 || *q > 1.0)
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(CrcError::InvalidInput(
                "probabilities and values must be finite, with probabilities in [0, 1]".into(),
            ));
        }
        if probabilities.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(CrcError::InvalidInput(
                "probabilities must be strictly increasing".into(),
            ));
        }
        if values.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(CrcError::InvalidInput(
                "quantile values must be non-decreasing".into(),
            ));
        }
        if interpolation == Interpolation::LogReturnPeriod
            && probabilities.iter().any(|q| *q >= 1.0)
        {
            return Err(CrcError::InvalidInput(
                "log-return-period interpolation requires probabilities below 1".into(),
            ));
        }
        Ok(Self {
            probabilities,
            values,
            interpolation,
            extrapolate,
            return_period_tail: Tail::Upper,
        })
    }

    pub fn from_return_periods(
        periods: Vec<f64>,
        values: Vec<f64>,
        tail: Tail,
        interpolation: Interpolation,
        extrapolate: bool,
    ) -> Result<Self> {
        if periods
            .iter()
            .any(|period| !period.is_finite() || *period <= 1.0)
        {
            return Err(CrcError::InvalidInput(
                "return periods must be finite and greater than one".into(),
            ));
        }
        let mut points: Vec<(f64, f64)> = periods
            .into_iter()
            .zip(values)
            .map(|(period, value)| {
                let q = match tail {
                    Tail::Upper => 1.0 - 1.0 / period,
                    Tail::Lower => 1.0 / period,
                };
                (q, value)
            })
            .collect();
        points.sort_by(|left, right| left.0.total_cmp(&right.0));
        let (probabilities, values) = points.into_iter().unzip();
        let mut distribution = Self::new(probabilities, values, interpolation, extrapolate)?;
        distribution.return_period_tail = tail;
        Ok(distribution)
    }

    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    fn axis(&self, q: f64) -> f64 {
        match self.interpolation {
            Interpolation::LinearProbability => q,
            Interpolation::LogReturnPeriod => match self.return_period_tail {
                Tail::Upper => (1.0 / (1.0 - q)).ln(),
                Tail::Lower => (1.0 / q).ln(),
            },
        }
    }

    fn interpolate(&self, q: f64) -> Result<f64> {
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return Err(CrcError::InvalidInput(
                "probability must be in [0, 1]".into(),
            ));
        }
        let minimum = self.probabilities[0];
        let maximum = *self.probabilities.last().expect("validated non-empty");
        if !self.extrapolate && (q < minimum || q > maximum) {
            return Err(CrcError::OutOfSupport {
                probability: q,
                minimum,
                maximum,
            });
        }
        if q <= minimum {
            return Ok(self.values[0]);
        }
        if q >= maximum {
            return Ok(*self.values.last().expect("validated non-empty"));
        }
        let upper = self
            .probabilities
            .partition_point(|candidate| *candidate < q);
        let lower = upper - 1;
        let x0 = self.axis(self.probabilities[lower]);
        let x1 = self.axis(self.probabilities[upper]);
        let weight = (self.axis(q) - x0) / (x1 - x0);
        Ok(self.values[lower] + weight * (self.values[upper] - self.values[lower]))
    }
}

impl Distribution for TabulatedDistribution {
    fn cdf(&self, x: f64) -> f64 {
        if x < self.values[0] {
            return 0.0;
        }
        if x >= *self.values.last().expect("validated non-empty") {
            return *self.probabilities.last().expect("validated non-empty");
        }
        let upper = self.values.partition_point(|candidate| *candidate < x);
        if upper == 0 {
            return self.probabilities[0];
        }
        let lower = upper - 1;
        let span = self.values[upper] - self.values[lower];
        if span.abs() < EPS {
            return self.probabilities[upper];
        }
        let weight = (x - self.values[lower]) / span;
        self.probabilities[lower] + weight * (self.probabilities[upper] - self.probabilities[lower])
    }

    fn ppf(&self, probability: f64) -> Result<f64> {
        self.interpolate(probability)
    }

    fn probability_support(&self) -> (f64, f64) {
        (
            self.probabilities[0],
            *self.probabilities.last().expect("validated non-empty"),
        )
    }
}

#[derive(Debug, Clone)]
pub struct EmpiricalDistribution {
    samples: Vec<f64>,
}

impl EmpiricalDistribution {
    pub fn new(mut samples: Vec<f64>) -> Result<Self> {
        if samples.len() < 2 || samples.iter().any(|value| !value.is_finite()) {
            return Err(CrcError::InvalidInput(
                "an empirical distribution requires at least two finite samples".into(),
            ));
        }
        samples.sort_by(f64::total_cmp);
        Ok(Self { samples })
    }

    pub fn samples(&self) -> &[f64] {
        &self.samples
    }
}

impl Distribution for EmpiricalDistribution {
    fn cdf(&self, x: f64) -> f64 {
        self.samples.partition_point(|sample| *sample <= x) as f64 / self.samples.len() as f64
    }

    fn ppf(&self, q: f64) -> Result<f64> {
        validate_probability(q)?;
        let position = q * (self.samples.len() - 1) as f64;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        let weight = position - lower as f64;
        Ok(self.samples[lower] + weight * (self.samples[upper] - self.samples[lower]))
    }
}

/// A degenerate distribution whose entire probability mass is at one value.
#[derive(Debug, Clone, Copy)]
pub struct PointMassDistribution {
    location: f64,
}

impl PointMassDistribution {
    pub fn new(location: f64) -> Result<Self> {
        if !location.is_finite() {
            return Err(CrcError::InvalidInput(
                "point-mass location must be finite".into(),
            ));
        }
        Ok(Self { location })
    }

    pub fn location(&self) -> f64 {
        self.location
    }
}

impl Distribution for PointMassDistribution {
    fn pdf(&self, _x: f64) -> f64 {
        // A Dirac mass has no finite density with respect to Lebesgue measure.
        0.0
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < self.location { 0.0 } else { 1.0 }
    }

    fn ppf(&self, probability: f64) -> Result<f64> {
        validate_probability(probability)?;
        Ok(self.location)
    }

    fn sample(&self, size: usize, _seed: Option<u64>) -> Result<Vec<f64>> {
        Ok(vec![self.location; size])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributionFamily {
    GenExtreme,
    WeibullMin,
    WeibullMax,
    SkewNormal,
    GumbelRight,
    GumbelLeft,
    GenPareto,
}

impl DistributionFamily {
    pub const ALL: [Self; 7] = [
        Self::GenExtreme,
        Self::WeibullMin,
        Self::WeibullMax,
        Self::SkewNormal,
        Self::GumbelRight,
        Self::GumbelLeft,
        Self::GenPareto,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::GenExtreme => "genextreme",
            Self::WeibullMin => "weibull_min",
            Self::WeibullMax => "weibull_max",
            Self::SkewNormal => "skewnorm",
            Self::GumbelRight => "gumbel_r",
            Self::GumbelLeft => "gumbel_l",
            Self::GenPareto => "genpareto",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|family| family.name() == name)
    }
}

#[derive(Debug, Clone)]
pub struct FittedDistribution {
    pub family: DistributionFamily,
    pub shape: Option<f64>,
    pub location: f64,
    pub scale: f64,
}

impl FittedDistribution {
    pub fn from_parameters(
        family: DistributionFamily,
        shape: Option<f64>,
        location: f64,
        scale: f64,
    ) -> Result<Self> {
        if !location.is_finite() || !scale.is_finite() || scale <= 0.0 {
            return Err(CrcError::InvalidInput(
                "location must be finite and scale must be positive".into(),
            ));
        }
        if matches!(
            family,
            DistributionFamily::GenExtreme
                | DistributionFamily::WeibullMin
                | DistributionFamily::WeibullMax
                | DistributionFamily::SkewNormal
                | DistributionFamily::GenPareto
        ) && shape.is_none()
        {
            return Err(CrcError::MissingParameter("shape".into()));
        }
        if shape.is_some_and(|value| !value.is_finite()) {
            return Err(CrcError::InvalidInput("shape must be finite".into()));
        }
        Ok(Self {
            family,
            shape,
            location,
            scale,
        })
    }

    fn z(&self, x: f64) -> f64 {
        (x - self.location) / self.scale
    }
}

impl Distribution for FittedDistribution {
    fn pdf(&self, x: f64) -> f64 {
        let z = self.z(x);
        match self.family {
            DistributionFamily::GumbelRight => (-z - (-z).exp()).exp() / self.scale,
            DistributionFamily::GumbelLeft => (z - z.exp()).exp() / self.scale,
            DistributionFamily::WeibullMin => {
                let k = self.shape.expect("validated");
                if z < 0.0 {
                    0.0
                } else {
                    k / self.scale * z.powf(k - 1.0) * (-z.powf(k)).exp()
                }
            }
            DistributionFamily::WeibullMax => {
                let k = self.shape.expect("validated");
                let y = -z;
                if y < 0.0 {
                    0.0
                } else {
                    k / self.scale * y.powf(k - 1.0) * (-y.powf(k)).exp()
                }
            }
            DistributionFamily::GenPareto => {
                let c = self.shape.expect("validated");
                let support = 1.0 + c * z;
                if z < 0.0 || support <= 0.0 {
                    0.0
                } else if c.abs() < EPS {
                    (-z).exp() / self.scale
                } else {
                    support.powf(-1.0 / c - 1.0) / self.scale
                }
            }
            DistributionFamily::GenExtreme => {
                let c = self.shape.expect("validated");
                if c.abs() < EPS {
                    (-z - (-z).exp()).exp() / self.scale
                } else {
                    let support = 1.0 - c * z;
                    if support <= 0.0 {
                        0.0
                    } else {
                        let power = support.powf(1.0 / c);
                        power.powf(1.0 - c) * (-power).exp() / self.scale
                    }
                }
            }
            DistributionFamily::SkewNormal => {
                let alpha = self.shape.expect("validated");
                2.0 * normal_pdf(z) * normal_cdf(alpha * z) / self.scale
            }
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = self.z(x);
        match self.family {
            DistributionFamily::GumbelRight => (-(-z).exp()).exp(),
            DistributionFamily::GumbelLeft => 1.0 - (-z.exp()).exp(),
            DistributionFamily::WeibullMin => {
                if z < 0.0 {
                    0.0
                } else {
                    1.0 - (-z.powf(self.shape.expect("validated"))).exp()
                }
            }
            DistributionFamily::WeibullMax => {
                if z >= 0.0 {
                    1.0
                } else {
                    (-(-z).powf(self.shape.expect("validated"))).exp()
                }
            }
            DistributionFamily::GenPareto => {
                let c = self.shape.expect("validated");
                if z < 0.0 {
                    0.0
                } else if c.abs() < EPS {
                    1.0 - (-z).exp()
                } else {
                    let support = 1.0 + c * z;
                    if support <= 0.0 {
                        1.0
                    } else {
                        1.0 - support.powf(-1.0 / c)
                    }
                }
            }
            DistributionFamily::GenExtreme => {
                let c = self.shape.expect("validated");
                if c.abs() < EPS {
                    (-(-z).exp()).exp()
                } else {
                    let support = 1.0 - c * z;
                    if support <= 0.0 {
                        if c > 0.0 { 1.0 } else { 0.0 }
                    } else {
                        (-support.powf(1.0 / c)).exp()
                    }
                }
            }
            DistributionFamily::SkewNormal => skew_normal_cdf(z, self.shape.expect("validated")),
        }
        .clamp(0.0, 1.0)
    }

    fn ppf(&self, q: f64) -> Result<f64> {
        validate_probability(q)?;
        if q <= 0.0 || q >= 1.0 {
            return Err(CrcError::InvalidInput(
                "parametric ppf requires probability strictly between zero and one".into(),
            ));
        }
        let value = match self.family {
            DistributionFamily::GumbelRight => self.location - self.scale * (-q.ln()).ln(),
            DistributionFamily::GumbelLeft => self.location + self.scale * (-(-q).ln_1p()).ln(),
            DistributionFamily::WeibullMin => {
                self.location
                    + self.scale * (-(-q).ln_1p()).powf(1.0 / self.shape.expect("validated"))
            }
            DistributionFamily::WeibullMax => {
                self.location - self.scale * (-q.ln()).powf(1.0 / self.shape.expect("validated"))
            }
            DistributionFamily::GenPareto => {
                let c = self.shape.expect("validated");
                if c.abs() < EPS {
                    self.location - self.scale * (-q).ln_1p()
                } else {
                    self.location + self.scale * ((1.0 - q).powf(-c) - 1.0) / c
                }
            }
            DistributionFamily::GenExtreme => {
                let c = self.shape.expect("validated");
                if c.abs() < EPS {
                    self.location - self.scale * (-q.ln()).ln()
                } else {
                    self.location + self.scale * (1.0 - (-q.ln()).powf(c)) / c
                }
            }
            DistributionFamily::SkewNormal => inverse_by_bisection(self, q),
        };
        Ok(value)
    }
}

#[derive(Debug, Clone)]
pub struct HurdleDistribution {
    atom_location: f64,
    atom_probability: f64,
    base: FittedDistribution,
    base_cdf_at_atom: f64,
}

impl HurdleDistribution {
    pub fn new(
        atom_location: f64,
        atom_probability: f64,
        base: FittedDistribution,
    ) -> Result<Self> {
        if !atom_location.is_finite() {
            return Err(CrcError::InvalidInput(
                "hurdle atom location must be finite".into(),
            ));
        }
        if !atom_probability.is_finite() || atom_probability <= 0.0 || atom_probability >= 1.0 {
            return Err(CrcError::InvalidInput(
                "hurdle atom probability must be strictly between zero and one".into(),
            ));
        }
        let base_cdf_at_atom = base.cdf(atom_location);
        if !base_cdf_at_atom.is_finite() || 1.0 - base_cdf_at_atom <= EPS {
            return Err(CrcError::InvalidInput(
                "hurdle base distribution must have positive probability above the atom".into(),
            ));
        }
        Ok(Self {
            atom_location,
            atom_probability,
            base,
            base_cdf_at_atom,
        })
    }

    pub fn atom_location(&self) -> f64 {
        self.atom_location
    }

    pub fn atom_probability(&self) -> f64 {
        self.atom_probability
    }

    pub fn base(&self) -> &FittedDistribution {
        &self.base
    }

    pub fn point_mass(&self, x: f64) -> f64 {
        if x == self.atom_location {
            self.atom_probability
        } else {
            0.0
        }
    }
}

impl Distribution for HurdleDistribution {
    fn pdf(&self, x: f64) -> f64 {
        if x <= self.atom_location {
            0.0
        } else {
            (1.0 - self.atom_probability) * self.base.pdf(x) / (1.0 - self.base_cdf_at_atom)
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < self.atom_location {
            0.0
        } else if x == self.atom_location {
            self.atom_probability
        } else {
            (self.atom_probability
                + (1.0 - self.atom_probability) * (self.base.cdf(x) - self.base_cdf_at_atom)
                    / (1.0 - self.base_cdf_at_atom))
                .clamp(self.atom_probability, 1.0)
        }
    }

    fn ppf(&self, q: f64) -> Result<f64> {
        validate_probability(q)?;
        if q <= self.atom_probability {
            return Ok(self.atom_location);
        }
        if q >= 1.0 {
            return Err(CrcError::InvalidInput(
                "hurdle ppf requires probability below one".into(),
            ));
        }
        let conditional_probability = (q - self.atom_probability) / (1.0 - self.atom_probability);
        let base_probability =
            self.base_cdf_at_atom + conditional_probability * (1.0 - self.base_cdf_at_atom);
        self.base.ppf(base_probability.min(1.0 - EPS))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DiagnosticMetrics {
    pub ks_statistic: f64,
    pub ks_pvalue: f64,
    pub rmse: f64,
    pub r_squared: f64,
}

#[derive(Debug, Clone)]
pub struct FitResult {
    pub distribution: FittedDistribution,
    pub diagnostics: DiagnosticMetrics,
}

#[derive(Debug, Clone, Copy)]
pub struct QuantileFitDiagnostics {
    pub rmse: f64,
    pub normalized_rmse: f64,
    pub weighted_r_squared: f64,
    pub maximum_absolute_residual: f64,
    pub point_count: usize,
    pub converged: bool,
    pub iterations: usize,
    pub evaluations: usize,
}

#[derive(Debug, Clone)]
pub struct QuantileFitResult {
    pub distribution: FittedDistribution,
    pub diagnostics: QuantileFitDiagnostics,
}

#[derive(Debug, Clone, Copy)]
pub struct HurdleQuantileFitDiagnostics {
    pub tail: QuantileFitDiagnostics,
    pub atom_probability_lower_bound: f64,
    pub atom_probability_upper_bound: f64,
    pub atom_point_count: usize,
    pub tail_point_count: usize,
}

#[derive(Debug, Clone)]
pub struct HurdleQuantileFitResult {
    pub distribution: HurdleDistribution,
    pub diagnostics: HurdleQuantileFitDiagnostics,
}

pub fn fit_distribution(samples: &[f64], family: Option<DistributionFamily>) -> Result<FitResult> {
    validate_samples(samples)?;
    if let Some(family) = family {
        return fit_family(samples, family);
    }
    fit_all(samples)?
        .into_iter()
        .max_by(|left, right| {
            left.diagnostics
                .ks_pvalue
                .total_cmp(&right.diagnostics.ks_pvalue)
        })
        .ok_or_else(|| CrcError::Unsupported("no candidate distribution could be fitted".into()))
}

pub fn fit_quantiles(
    probabilities: &[f64],
    values: &[f64],
    weights: Option<&[f64]>,
    family: DistributionFamily,
) -> Result<QuantileFitResult> {
    let weights = validate_quantile_points(probabilities, values, weights)?;
    fit_quantile_points_internal(probabilities, values, &weights, family, None)
}

pub fn fit_hurdle_quantiles(
    probabilities: &[f64],
    values: &[f64],
    weights: Option<&[f64]>,
    family: DistributionFamily,
    atom_location: f64,
    atom_probability: f64,
) -> Result<HurdleQuantileFitResult> {
    let weights = validate_quantile_points(probabilities, values, weights)?;
    if !atom_location.is_finite()
        || values
            .iter()
            .any(|value| *value < atom_location || !value.is_finite())
    {
        return Err(CrcError::InvalidInput(
            "hurdle quantile values must be finite and at or above the atom".into(),
        ));
    }
    if !atom_probability.is_finite() || atom_probability <= 0.0 || atom_probability >= 1.0 {
        return Err(CrcError::InvalidInput(
            "hurdle atom probability must be strictly between zero and one".into(),
        ));
    }

    let atom_indices: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value == atom_location).then_some(index))
        .collect();
    let tail_indices: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value > atom_location).then_some(index))
        .collect();
    let lower_bound = atom_indices
        .iter()
        .map(|&index| probabilities[index])
        .fold(0.0, f64::max);
    let upper_bound = tail_indices
        .iter()
        .map(|&index| probabilities[index])
        .fold(1.0, f64::min);
    if atom_probability < lower_bound || atom_probability >= upper_bound {
        return Err(CrcError::InvalidInput(format!(
            "atom probability must be in [{lower_bound}, {upper_bound}) for the supplied knots"
        )));
    }

    let tail_probabilities: Vec<f64> = tail_indices
        .iter()
        .map(|&index| (probabilities[index] - atom_probability) / (1.0 - atom_probability))
        .collect();
    let tail_values: Vec<f64> = tail_indices.iter().map(|&index| values[index]).collect();
    let tail_weights: Vec<f64> = tail_indices.iter().map(|&index| weights[index]).collect();
    let tail_weights =
        validate_quantile_points(&tail_probabilities, &tail_values, Some(&tail_weights))?;
    let tail_result = fit_quantile_points_internal(
        &tail_probabilities,
        &tail_values,
        &tail_weights,
        family,
        Some(atom_location),
    )?;
    let distribution =
        HurdleDistribution::new(atom_location, atom_probability, tail_result.distribution)?;
    Ok(HurdleQuantileFitResult {
        distribution,
        diagnostics: HurdleQuantileFitDiagnostics {
            tail: tail_result.diagnostics,
            atom_probability_lower_bound: lower_bound,
            atom_probability_upper_bound: upper_bound,
            atom_point_count: atom_indices.len(),
            tail_point_count: tail_indices.len(),
        },
    })
}

fn fit_quantile_points_internal(
    probabilities: &[f64],
    values: &[f64],
    weights: &[f64],
    family: DistributionFamily,
    truncation_atom: Option<f64>,
) -> Result<QuantileFitResult> {
    let conditioning_scale = robust_value_scale(values, weights);
    let kind = reference_curve_fitting::DistributionKind::from_name(family.name())
        .expect("public and reference family names are synchronized");
    let optimized = reference_curve_fitting::fit_quantile_points(
        kind,
        probabilities,
        values,
        weights,
        conditioning_scale,
        truncation_atom,
    )
    .ok_or_else(|| CrcError::Unsupported(format!("{} could not be fitted", family.name())))?;
    if !optimized.converged {
        return Err(CrcError::ConvergenceFailed {
            family: family.name().into(),
            iterations: optimized.iterations,
        });
    }
    let distribution = FittedDistribution::from_parameters(
        family,
        optimized.shape,
        optimized.loc,
        optimized.scale,
    )?;
    let fitted_values = probabilities
        .iter()
        .map(|&probability| {
            if let Some(atom) = truncation_atom {
                let atom_cdf = distribution.cdf(atom);
                let base_probability = (atom_cdf + probability * (1.0 - atom_cdf)).min(1.0 - EPS);
                distribution.ppf(base_probability)
            } else {
                distribution.ppf(probability)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let diagnostics = quantile_diagnostics(
        values,
        &fitted_values,
        weights,
        optimized.iterations,
        optimized.evaluations,
    );
    Ok(QuantileFitResult {
        distribution,
        diagnostics,
    })
}

pub fn fit_all(samples: &[f64]) -> Result<Vec<FitResult>> {
    validate_samples(samples)?;
    reference_curve_fitting::fit_all_candidates_diagnostic(samples)
        .into_iter()
        .map(reference_diagnostic)
        .collect()
}

pub fn calculate_diagnostics(
    samples: &[f64],
    distribution: &dyn Distribution,
) -> Result<DiagnosticMetrics> {
    validate_samples(samples)?;
    Ok(diagnostics(samples, distribution))
}

fn fit_family(samples: &[f64], family: DistributionFamily) -> Result<FitResult> {
    reference_curve_fitting::fit_all_candidates_diagnostic(samples)
        .into_iter()
        .find(|candidate| candidate.name == family.name())
        .ok_or_else(|| {
            CrcError::Unsupported(format!(
                "{} could not be fitted to the samples",
                family.name()
            ))
        })
        .and_then(reference_diagnostic)
}

fn reference_diagnostic(diagnostic: reference_curve_fitting::DiagnosticFit) -> Result<FitResult> {
    let family = DistributionFamily::from_name(diagnostic.name).ok_or_else(|| {
        CrcError::Unsupported(format!(
            "reference fitter returned unknown family {}",
            diagnostic.name
        ))
    })?;
    Ok(FitResult {
        distribution: FittedDistribution::from_parameters(
            family,
            diagnostic.shape,
            diagnostic.loc,
            diagnostic.scale,
        )?,
        diagnostics: DiagnosticMetrics {
            ks_statistic: diagnostic.ks_stat,
            ks_pvalue: diagnostic.p_value,
            rmse: diagnostic.rmse,
            r_squared: diagnostic.r_squared,
        },
    })
}

fn diagnostics(samples: &[f64], distribution: &dyn Distribution) -> DiagnosticMetrics {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len() as f64;
    let mut ks: f64 = 0.0;
    let mut squared_error = 0.0;
    for (index, value) in sorted.iter().enumerate() {
        let expected = (index as f64 + 0.5) / count;
        let fitted = distribution.cdf(*value);
        ks = ks.max((fitted - index as f64 / count).abs());
        ks = ks.max(((index + 1) as f64 / count - fitted).abs());
        squared_error += (fitted - expected).powi(2);
    }
    let rmse = (squared_error / count).sqrt();
    let total = sorted
        .iter()
        .enumerate()
        .map(|(index, _)| ((index as f64 + 0.5) / count - 0.5).powi(2))
        .sum::<f64>();
    let r_squared = if total > 0.0 {
        1.0 - squared_error / total
    } else {
        1.0
    };
    let z = (count.sqrt() + 0.12 + 0.11 / count.sqrt()) * ks;
    let ks_pvalue = (1..=100)
        .map(|term| {
            let sign = if term % 2 == 1 { 1.0 } else { -1.0 };
            sign * (-2.0 * (term as f64).powi(2) * z * z).exp()
        })
        .sum::<f64>()
        .mul_add(2.0, 0.0)
        .clamp(0.0, 1.0);
    DiagnosticMetrics {
        ks_statistic: ks,
        ks_pvalue,
        rmse,
        r_squared,
    }
}

fn validate_probability(q: f64) -> Result<()> {
    if q.is_finite() && (0.0..=1.0).contains(&q) {
        Ok(())
    } else {
        Err(CrcError::InvalidInput(
            "probability must be finite and in [0, 1]".into(),
        ))
    }
}

fn validate_samples(samples: &[f64]) -> Result<()> {
    if samples.len() < 4 || samples.iter().any(|value| !value.is_finite()) {
        return Err(CrcError::InvalidInput(
            "fitting requires at least four finite samples".into(),
        ));
    }
    let minimum = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (maximum - minimum).abs() < EPS {
        return Err(CrcError::InvalidInput(
            "samples must not all be equal".into(),
        ));
    }
    Ok(())
}

fn validate_quantile_points(
    probabilities: &[f64],
    values: &[f64],
    weights: Option<&[f64]>,
) -> Result<Vec<f64>> {
    if probabilities.len() != values.len() || probabilities.len() < 4 {
        return Err(CrcError::InvalidInput(
            "probabilities and values must have the same length of at least four".into(),
        ));
    }
    if probabilities
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0 || *value >= 1.0)
    {
        return Err(CrcError::InvalidInput(
            "fit probabilities must be finite and strictly between zero and one".into(),
        ));
    }
    if probabilities.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(CrcError::InvalidInput(
            "fit probabilities must be strictly increasing".into(),
        ));
    }
    if values.iter().any(|value| !value.is_finite())
        || values.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(CrcError::InvalidInput(
            "quantile values must be finite and non-decreasing".into(),
        ));
    }
    let weights = weights.map_or_else(|| vec![1.0; values.len()], <[f64]>::to_vec);
    if weights.len() != values.len()
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(CrcError::InvalidInput(
            "weights must match the knots and be finite and non-negative".into(),
        ));
    }
    let positive: Vec<usize> = weights
        .iter()
        .enumerate()
        .filter_map(|(index, weight)| (*weight > 0.0).then_some(index))
        .collect();
    if positive.len() < 4 {
        return Err(CrcError::InvalidInput(
            "quantile fitting requires at least four positive-weight knots".into(),
        ));
    }
    let minimum = positive
        .iter()
        .map(|&index| values[index])
        .fold(f64::INFINITY, f64::min);
    let maximum = positive
        .iter()
        .map(|&index| values[index])
        .fold(f64::NEG_INFINITY, f64::max);
    if maximum - minimum <= EPS * maximum.abs().max(minimum.abs()).max(1.0) {
        return Err(CrcError::InvalidInput(
            "positive-weight quantile values must have non-zero range".into(),
        ));
    }
    Ok(weights)
}

fn robust_value_scale(values: &[f64], weights: &[f64]) -> f64 {
    let median = weighted_median(
        values
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .collect(),
    );
    let mad = weighted_median(
        values
            .iter()
            .map(|value| (value - median).abs())
            .zip(weights.iter().copied())
            .collect(),
    );
    let range = values.last().expect("validated") - values[0];
    (1.4826 * mad).max(range * 1.0e-3).max(EPS)
}

fn weighted_median(mut points: Vec<(f64, f64)>) -> f64 {
    points.sort_by(|left, right| left.0.total_cmp(&right.0));
    let half = points.iter().map(|point| point.1).sum::<f64>() * 0.5;
    let mut cumulative = 0.0;
    for (value, weight) in points {
        cumulative += weight;
        if cumulative >= half {
            return value;
        }
    }
    unreachable!("validated positive total weight")
}

fn quantile_diagnostics(
    observed: &[f64],
    fitted: &[f64],
    weights: &[f64],
    iterations: usize,
    evaluations: usize,
) -> QuantileFitDiagnostics {
    let weight_sum = weights.iter().sum::<f64>();
    let weighted_mean = observed
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * weight)
        .sum::<f64>()
        / weight_sum;
    let weighted_squared_error = observed
        .iter()
        .zip(fitted)
        .zip(weights)
        .map(|((observed, fitted), weight)| weight * (fitted - observed).powi(2))
        .sum::<f64>();
    let weighted_total = observed
        .iter()
        .zip(weights)
        .map(|(value, weight)| weight * (value - weighted_mean).powi(2))
        .sum::<f64>();
    let rmse = (weighted_squared_error / weight_sum).sqrt();
    let positive_weight_minimum = observed
        .iter()
        .zip(weights)
        .filter_map(|(value, weight)| (*weight > 0.0).then_some(*value))
        .fold(f64::INFINITY, f64::min);
    let positive_weight_maximum = observed
        .iter()
        .zip(weights)
        .filter_map(|(value, weight)| (*weight > 0.0).then_some(*value))
        .fold(f64::NEG_INFINITY, f64::max);
    let range = positive_weight_maximum - positive_weight_minimum;
    QuantileFitDiagnostics {
        rmse,
        normalized_rmse: rmse / range,
        weighted_r_squared: 1.0 - weighted_squared_error / weighted_total,
        maximum_absolute_residual: observed
            .iter()
            .zip(fitted)
            .map(|(observed, fitted)| (fitted - observed).abs())
            .fold(0.0, f64::max),
        point_count: observed.len(),
        converged: true,
        iterations,
        evaluations,
    }
}

fn inverse_by_bisection(distribution: &dyn Distribution, q: f64) -> f64 {
    let mut lower = -1.0;
    let mut upper = 1.0;
    while distribution.cdf(lower) > q {
        lower *= 2.0;
    }
    while distribution.cdf(upper) < q {
        upper *= 2.0;
    }
    for _ in 0..120 {
        let middle = (lower + upper) * 0.5;
        if distribution.cdf(middle) < q {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    (lower + upper) * 0.5
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

fn skew_normal_cdf(z: f64, alpha: f64) -> f64 {
    if z <= -10.0 {
        return 0.0;
    }
    if z >= 10.0 {
        return 1.0;
    }
    let lower = -10.0;
    let steps = 400usize;
    let width = (z - lower) / steps as f64;
    let mut total = 0.0;
    for index in 0..=steps {
        let x = lower + index as f64 * width;
        let weight = if index == 0 || index == steps {
            1.0
        } else if index % 2 == 0 {
            2.0
        } else {
            4.0
        };
        total += weight * 2.0 * normal_pdf(x) * normal_cdf(alpha * x);
    }
    (total * width / 3.0).clamp(0.0, 1.0)
}

fn erf(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial =
        (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t;
    sign * (1.0 - polynomial * (-x * x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_mass_is_exact_at_every_probability() {
        let distribution = PointMassDistribution::new(2.5).unwrap();
        assert_eq!(
            distribution.quantiles(&[0.0, 0.5, 1.0]).unwrap(),
            vec![2.5; 3]
        );
        assert_eq!(distribution.cdf(2.49), 0.0);
        assert_eq!(distribution.cdf(2.5), 1.0);
        assert_eq!(distribution.sample(3, Some(42)).unwrap(), vec![2.5; 3]);
    }

    #[test]
    fn return_periods_convert_to_upper_tail_probabilities() {
        let distribution = TabulatedDistribution::from_return_periods(
            vec![10.0, 100.0],
            vec![1.0, 4.0],
            Tail::Upper,
            Interpolation::LogReturnPeriod,
            false,
        )
        .unwrap();
        assert_eq!(distribution.probabilities(), &[0.9, 0.99]);
        assert!((distribution.ppf(0.99).unwrap() - 4.0).abs() < 1.0e-12);
    }

    #[test]
    fn fitted_gumbel_round_trips_probability() {
        let distribution =
            FittedDistribution::from_parameters(DistributionFamily::GumbelRight, None, 10.0, 2.0)
                .unwrap();
        let value = distribution.ppf(0.95).unwrap();
        assert!((distribution.cdf(value) - 0.95).abs() < 1.0e-10);
    }

    #[test]
    fn every_supported_family_round_trips_probability() {
        let cases = [
            (DistributionFamily::GenExtreme, Some(0.1), 0.0, 1.0),
            (DistributionFamily::WeibullMin, Some(2.0), 0.0, 1.0),
            (DistributionFamily::WeibullMax, Some(2.0), 0.0, 1.0),
            (DistributionFamily::SkewNormal, Some(1.5), 0.0, 1.0),
            (DistributionFamily::GumbelRight, None, 0.0, 1.0),
            (DistributionFamily::GumbelLeft, None, 0.0, 1.0),
            (DistributionFamily::GenPareto, Some(0.1), 0.0, 1.0),
        ];
        for (family, shape, location, scale) in cases {
            let distribution =
                FittedDistribution::from_parameters(family, shape, location, scale).unwrap();
            for probability in [0.1, 0.5, 0.9] {
                let value = distribution.ppf(probability).unwrap();
                assert!(
                    (distribution.cdf(value) - probability).abs() < 2.0e-4,
                    "{} failed at {probability}",
                    family.name()
                );
            }
        }
    }

    #[test]
    fn quantile_fitting_recovers_supported_family_quantiles() {
        let cases = [
            (DistributionFamily::GenExtreme, Some(0.2), 1.0, 2.0),
            (DistributionFamily::WeibullMin, Some(1.8), -0.5, 1.5),
            (DistributionFamily::WeibullMax, Some(2.2), 4.0, 1.2),
            (DistributionFamily::SkewNormal, Some(2.0), 0.5, 1.3),
            (DistributionFamily::GumbelRight, None, 1.0, 0.8),
            (DistributionFamily::GumbelLeft, None, 1.0, 0.8),
            (DistributionFamily::GenPareto, Some(0.15), -0.2, 1.1),
        ];
        let probabilities = [0.1, 0.25, 0.5, 0.7, 0.85, 0.95, 0.99];
        for (family, shape, location, scale) in cases {
            let source =
                FittedDistribution::from_parameters(family, shape, location, scale).unwrap();
            let values = source.quantiles(&probabilities).unwrap();
            let result = fit_quantiles(&probabilities, &values, None, family).unwrap();
            assert!(
                result.diagnostics.normalized_rmse < 1.0e-4,
                "{} normalized RMSE was {}",
                family.name(),
                result.diagnostics.normalized_rmse
            );
            assert!(result.diagnostics.converged);
            assert!((result.distribution.location - location).abs() < 2.0e-3);
            assert!((result.distribution.scale - scale).abs() < 2.0e-3);
            if let (Some(actual), Some(expected)) = (result.distribution.shape, shape) {
                assert!(
                    (actual - expected).abs() < 2.0e-2,
                    "{} shape was {actual}, expected {expected}",
                    family.name()
                );
            }
        }
    }

    #[test]
    fn quantile_fitting_honors_non_uniform_weights() {
        let source =
            FittedDistribution::from_parameters(DistributionFamily::GumbelRight, None, 1.0, 0.8)
                .unwrap();
        let probabilities = [0.1, 0.25, 0.5, 0.7, 0.85, 0.95];
        let mut values = source.quantiles(&probabilities).unwrap();
        values[0] -= 0.8;
        let weights = [0.01, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = fit_quantiles(
            &probabilities,
            &values,
            Some(&weights),
            DistributionFamily::GumbelRight,
        )
        .unwrap();
        assert!((result.distribution.location - 1.0).abs() < 0.02);
        assert!((result.distribution.scale - 0.8).abs() < 0.02);
        assert_eq!(result.diagnostics.point_count, probabilities.len());
    }

    #[test]
    fn quantile_fitting_validates_effective_weights() {
        let probabilities = [0.1, 0.2, 0.3, 0.4];
        let values = [0.0, 1.0, 2.0, 3.0];
        let weights = [1.0, 0.0, 0.0, 0.0];
        assert!(matches!(
            fit_quantiles(
                &probabilities,
                &values,
                Some(&weights),
                DistributionFamily::GumbelRight
            ),
            Err(CrcError::InvalidInput(_))
        ));
    }

    #[test]
    fn sample_mle_gumbel_regression_is_unchanged() {
        let result = fit_distribution(
            &[0.0, 0.2, 0.7, 1.1, 1.9, 2.4, 3.2, 4.8],
            Some(DistributionFamily::GumbelRight),
        )
        .unwrap();
        assert!((result.distribution.location - 1.077_032_290_250_686_6).abs() < 1.0e-12);
        assert!((result.distribution.scale - 1.173_589_689_864_434_2).abs() < 1.0e-12);
        assert!((result.diagnostics.ks_statistic - 0.128_920_449_165_263_24).abs() < 1.0e-12);
    }

    #[test]
    fn hurdle_distribution_normalizes_a_full_support_base() {
        let base =
            FittedDistribution::from_parameters(DistributionFamily::GumbelRight, None, 0.5, 1.2)
                .unwrap();
        let hurdle = HurdleDistribution::new(0.0, 0.4, base).unwrap();
        assert_eq!(hurdle.ppf(0.4).unwrap(), 0.0);
        assert_eq!(hurdle.cdf(0.0), 0.4);
        assert_eq!(hurdle.point_mass(0.0), 0.4);
        for probability in [0.5, 0.75, 0.95] {
            let value = hurdle.ppf(probability).unwrap();
            assert!((hurdle.cdf(value) - probability).abs() < 1.0e-10);
            assert!(value > 0.0);
        }
        let upper = hurdle.ppf(0.999_999).unwrap();
        let steps = 10_000;
        let width = upper / steps as f64;
        let continuous_mass = (0..steps)
            .map(|index| hurdle.pdf((index as f64 + 0.5) * width) * width)
            .sum::<f64>();
        assert!((continuous_mass - 0.6).abs() < 2.0e-3);
        let samples = hurdle.sample(5_000, Some(42)).unwrap();
        let atom_fraction =
            samples.iter().filter(|value| **value == 0.0).count() as f64 / samples.len() as f64;
        assert!((atom_fraction - 0.4).abs() < 0.03);
        let statistics = crate::metrics::calculate_statistics(&hurdle).unwrap();
        assert_eq!(statistics.minimum, 0.0);
        assert!(statistics.mean > 0.0);
    }

    #[test]
    fn hurdle_quantile_fitting_uses_supplied_atom_probability() {
        let base =
            FittedDistribution::from_parameters(DistributionFamily::GumbelRight, None, 0.5, 1.2)
                .unwrap();
        let source = HurdleDistribution::new(0.0, 0.5, base).unwrap();
        let probabilities = [0.2, 0.5, 0.65, 0.75, 0.85, 0.93, 0.98];
        let values = source.quantiles(&probabilities).unwrap();
        let result = fit_hurdle_quantiles(
            &probabilities,
            &values,
            None,
            DistributionFamily::GumbelRight,
            0.0,
            0.5,
        )
        .unwrap();
        assert_eq!(result.distribution.ppf(0.5).unwrap(), 0.0);
        assert!(result.diagnostics.tail.normalized_rmse < 1.0e-4);
        assert_eq!(result.diagnostics.atom_probability_lower_bound, 0.5);
        assert_eq!(result.diagnostics.atom_probability_upper_bound, 0.65);
    }
}

use std::f64::consts::PI;

#[cfg(feature = "sampling")]
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    error::{CrcError, Result},
    numerical::{interval_fraction, linear_interpolate},
    reference::curve_fitting as reference_curve_fitting,
};

mod lmoments;

const EPS: f64 = 1.0e-12;

pub trait Distribution: Send + Sync {
    fn cdf(&self, x: f64) -> f64;
    fn ppf(&self, probability: f64) -> Result<f64>;

    fn probability_support(&self) -> (f64, f64) {
        (0.0, 1.0)
    }

    /// `P(X = x)`. Continuous laws return zero.
    fn probability_atom(&self, _x: f64) -> f64 {
        0.0
    }

    /// Strict lower cdf `P(X < x)`.
    fn cdf_below(&self, x: f64) -> f64 {
        (self.cdf(x) - self.probability_atom(x)).max(0.0)
    }

    /// True when the quantile function is defined on the whole of `[0, 1]`.
    fn has_full_probability_support(&self) -> bool {
        let (lower, upper) = self.probability_support();
        lower <= 0.0 && upper >= 1.0
    }

    fn pdf(&self, x: f64) -> f64 {
        let h = 1.0e-5 * x.abs().max(1.0);
        ((self.cdf(x + h) - self.cdf(x - h)) / (2.0 * h)).max(0.0)
    }

    fn quantiles(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        probabilities.iter().map(|&q| self.ppf(q)).collect()
    }

    #[cfg(feature = "sampling")]
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

    /// Private exact quantile segments. Equal probabilities describe a
    /// zero-mass support gap; equal values describe an atom.
    pub fn from_segments(probabilities: Vec<f64>, values: Vec<f64>) -> Result<Self> {
        if probabilities.len() != values.len()
            || probabilities.len() < 2
            || probabilities.first() != Some(&0.0)
            || probabilities.last() != Some(&1.0)
            || probabilities
                .iter()
                .any(|q| !q.is_finite() || !(0.0..=1.0).contains(q))
            || values.iter().any(|x| !x.is_finite())
            || probabilities.windows(2).any(|p| p[0] > p[1])
            || values.windows(2).any(|p| p[0] > p[1])
        {
            return Err(CrcError::InvalidInput(
                "exact segments require ordered finite full-support coordinates".into(),
            ));
        }
        Ok(Self {
            probabilities,
            values,
            interpolation: Interpolation::LinearProbability,
            extrapolate: false,
            return_period_tail: Tail::Upper,
        })
    }

    /// Right-continuous inverse, which skips zero-mass support gaps.
    pub fn upper_quantile(&self, probability: f64) -> Result<f64> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(CrcError::InvalidInput(
                "probability must be finite and in [0,1]".into(),
            ));
        }
        let after = self.probabilities.partition_point(|q| *q <= probability);
        if after > 0 && self.probabilities[after - 1] == probability {
            return Ok(self.values[after - 1]);
        }
        self.interpolate(probability)
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
        Ok(linear_interpolate(
            self.values[lower],
            self.values[upper],
            weight,
        ))
    }
}

impl Distribution for TabulatedDistribution {
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < self.values[0] {
            return 0.0;
        }
        if x >= *self.values.last().expect("validated non-empty") {
            return *self.probabilities.last().expect("validated non-empty");
        }
        let mut upper = self.values.partition_point(|candidate| *candidate < x);
        if self.values[upper] == x {
            while upper + 1 < self.values.len() && self.values[upper + 1] == x {
                upper += 1;
            }
            return self.probabilities[upper];
        }
        if upper == 0 {
            return self.probabilities[0];
        }
        let lower = upper - 1;
        let (a, b) = (self.values[lower], self.values[upper]);
        linear_interpolate(
            self.probabilities[lower],
            self.probabilities[upper],
            interval_fraction(a, b, a, x),
        )
    }

    fn cdf_below(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= self.values[0] {
            return 0.0;
        }
        if x > *self.values.last().expect("validated non-empty") {
            return *self.probabilities.last().expect("validated non-empty");
        }
        let upper = self.values.partition_point(|candidate| *candidate < x);
        if self.values[upper] == x {
            return self.probabilities[upper];
        }
        if upper == 0 {
            return 0.0;
        }
        let lower = upper - 1;
        let (a, b) = (self.values[lower], self.values[upper]);
        linear_interpolate(
            self.probabilities[lower],
            self.probabilities[upper],
            interval_fraction(a, b, a, x),
        )
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

    fn probability_atom(&self, x: f64) -> f64 {
        let mut mass = 0.0;
        for index in 0..self.values.len() - 1 {
            if self.values[index] == x && self.values[index + 1] == x {
                mass += self.probabilities[index + 1] - self.probabilities[index];
            }
        }
        mass
    }
}

#[derive(Debug, Clone)]
pub struct EmpiricalDistribution {
    samples: Vec<f64>,
}

impl EmpiricalDistribution {
    pub fn new(mut samples: Vec<f64>) -> Result<Self> {
        if samples.is_empty() || samples.iter().any(|value| !value.is_finite()) {
            return Err(CrcError::InvalidInput(
                "an empirical distribution requires nonempty finite samples".into(),
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
        if x.is_nan() {
            return f64::NAN;
        }
        self.samples.partition_point(|sample| *sample <= x) as f64 / self.samples.len() as f64
    }

    fn ppf(&self, q: f64) -> Result<f64> {
        validate_probability(q)?;
        // Generalized inverse of the ECDF: ties remain their observed atoms.
        let index = (q * self.samples.len() as f64).ceil() as usize;
        Ok(self.samples[index.saturating_sub(1).min(self.samples.len() - 1)])
    }

    fn probability_atom(&self, x: f64) -> f64 {
        // Samples are sorted, so the atom is the run of values equal to `x`.
        let upper = self.samples.partition_point(|sample| *sample <= x);
        let lower = self.samples[..upper].partition_point(|sample| *sample < x);
        (upper - lower) as f64 / self.samples.len() as f64
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
        if x.is_nan() {
            f64::NAN
        } else if x < self.location {
            0.0
        } else {
            1.0
        }
    }

    fn ppf(&self, probability: f64) -> Result<f64> {
        validate_probability(probability)?;
        Ok(self.location)
    }

    fn probability_atom(&self, x: f64) -> f64 {
        if x == self.location { 1.0 } else { 0.0 }
    }

    #[cfg(feature = "sampling")]
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

    pub(crate) fn from_reference_kind(kind: reference_curve_fitting::DistributionKind) -> Self {
        use reference_curve_fitting::DistributionKind as Kind;
        match kind {
            Kind::GenExtreme => Self::GenExtreme,
            Kind::WeibullMin => Self::WeibullMin,
            Kind::WeibullMax => Self::WeibullMax,
            Kind::SkewNorm => Self::SkewNormal,
            Kind::GumbelR => Self::GumbelRight,
            Kind::GumbelL => Self::GumbelLeft,
            Kind::GenPareto => Self::GenPareto,
        }
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
        if matches!(
            family,
            DistributionFamily::WeibullMin | DistributionFamily::WeibullMax
        ) && shape.is_some_and(|value| value <= 0.0)
        {
            return Err(CrcError::InvalidInput(
                "weibull shape must be positive".into(),
            ));
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

    /// Upper-tail probability `P(X > x)`.
    ///
    /// Evaluated directly rather than as `1 - cdf(x)`, so it keeps full
    /// relative precision deep in the upper tail where `cdf(x)` rounds to one.
    pub fn sf(&self, x: f64) -> f64 {
        let z = self.z(x);
        match self.family {
            DistributionFamily::GumbelRight => -(-(-z).exp()).exp_m1(),
            DistributionFamily::GumbelLeft => (-z.exp()).exp(),
            DistributionFamily::WeibullMin => {
                if z < 0.0 {
                    1.0
                } else {
                    (-z.powf(self.shape.expect("validated"))).exp()
                }
            }
            DistributionFamily::WeibullMax => {
                if z >= 0.0 {
                    0.0
                } else {
                    -(-(-z).powf(self.shape.expect("validated"))).exp_m1()
                }
            }
            DistributionFamily::GenPareto => {
                let c = self.shape.expect("validated");
                if z < 0.0 {
                    1.0
                } else if c.abs() < EPS {
                    (-z).exp()
                } else {
                    let support = 1.0 + c * z;
                    if support <= 0.0 {
                        0.0
                    } else {
                        support.powf(-1.0 / c)
                    }
                }
            }
            DistributionFamily::GenExtreme => {
                let c = self.shape.expect("validated");
                if c.abs() < EPS {
                    -(-(-z).exp()).exp_m1()
                } else {
                    let support = 1.0 - c * z;
                    if support <= 0.0 {
                        if c > 0.0 { 0.0 } else { 1.0 }
                    } else {
                        -(-support.powf(1.0 / c)).exp_m1()
                    }
                }
            }
            // If X ~ SN(alpha, loc, scale) then -X ~ SN(-alpha, -loc, scale).
            DistributionFamily::SkewNormal => skew_normal_cdf(-z, -self.shape.expect("validated")),
        }
        .clamp(0.0, 1.0)
    }

    /// Logarithmic upper tail, retaining probabilities smaller than binary64 can store.
    pub(crate) fn log_survival(&self, x: f64) -> f64 {
        let z = self.z(x);
        let shape = self.shape.unwrap_or(0.0);
        match self.family {
            DistributionFamily::GumbelRight => log_exponential_complement(-z),
            DistributionFamily::GumbelLeft => -z.exp(),
            DistributionFamily::WeibullMin => -z.max(0.0).powf(shape),
            DistributionFamily::WeibullMax if z < 0.0 => {
                log_exponential_complement(shape * (-z).ln())
            }
            DistributionFamily::WeibullMax => f64::NEG_INFINITY,
            DistributionFamily::GenPareto if z < 0.0 => 0.0,
            DistributionFamily::GenPareto if shape.abs() < EPS => -z,
            DistributionFamily::GenPareto if shape * z <= -1.0 => f64::NEG_INFINITY,
            DistributionFamily::GenPareto => (shape * z).ln_1p() / -shape,
            DistributionFamily::GenExtreme if shape.abs() < EPS => log_exponential_complement(-z),
            DistributionFamily::GenExtreme if 1.0 - shape * z <= 0.0 => {
                if shape > 0.0 {
                    f64::NEG_INFINITY
                } else {
                    0.0
                }
            }
            DistributionFamily::GenExtreme => {
                log_exponential_complement((-shape * z).ln_1p() / shape)
            }
            DistributionFamily::SkewNormal => skew_normal_log_cdf(-z, -shape),
        }
    }

    /// Invert a positive tail even when its linear probability underflows.
    pub(crate) fn inverse_log_survival(&self, logarithm: f64) -> Result<f64> {
        if logarithm > f64::MIN_POSITIVE.ln() {
            return self.isf(logarithm.exp());
        }
        let shape = self.shape.unwrap_or(0.0);
        // Below MIN_POSITIVE, log(-log(1-s)) equals log(s) to binary64 precision.
        let standardized = match self.family {
            DistributionFamily::GumbelRight => -logarithm,
            DistributionFamily::GumbelLeft => (-logarithm).ln(),
            DistributionFamily::WeibullMin => (-logarithm).powf(1.0 / shape),
            DistributionFamily::WeibullMax => -(logarithm / shape).exp(),
            DistributionFamily::GenPareto | DistributionFamily::GenExtreme if shape.abs() < EPS => {
                -logarithm
            }
            DistributionFamily::GenPareto => (-shape * logarithm).exp_m1() / shape,
            DistributionFamily::GenExtreme => -(shape * logarithm).exp_m1() / shape,
            DistributionFamily::SkewNormal => {
                -inverse_by_bisection(|z| skew_normal_log_cdf(z, -shape), logarithm)
            }
        };
        Ok(self.location + self.scale * standardized)
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if x.is_infinite() {
            return f64::NEG_INFINITY;
        }
        let z = self.z(x);
        let shape = self.shape.unwrap_or(0.0);
        let standardized = match self.family {
            DistributionFamily::GumbelRight => -z - (-z).exp(),
            DistributionFamily::GumbelLeft => z - z.exp(),
            DistributionFamily::WeibullMin => weibull_log_pdf(z, shape),
            DistributionFamily::WeibullMax => weibull_log_pdf(-z, shape),
            DistributionFamily::GenPareto if z < 0.0 || shape * z <= -1.0 => f64::NEG_INFINITY,
            DistributionFamily::GenPareto if shape.abs() < EPS => -z,
            DistributionFamily::GenPareto => (-1.0 / shape - 1.0) * (shape * z).ln_1p(),
            DistributionFamily::GenExtreme if shape.abs() < EPS => -z - (-z).exp(),
            DistributionFamily::GenExtreme if 1.0 - shape * z <= 0.0 => f64::NEG_INFINITY,
            DistributionFamily::GenExtreme => {
                let log_power = (-shape * z).ln_1p() / shape;
                (1.0 - shape) * log_power - log_power.exp()
            }
            DistributionFamily::SkewNormal => {
                (2.0 / PI).sqrt().ln() - 0.5 * z * z
                    + reference_curve_fitting::log_standard_normal_cdf(shape * z)
            }
        };
        standardized - self.scale.ln()
    }

    /// Inverse of [`Self::sf`]: the value exceeded with probability `survival`.
    ///
    /// The upper-tail counterpart of `ppf`, accurate for survival
    /// probabilities far below the spacing of `f64` values near one.
    pub fn isf(&self, survival: f64) -> Result<f64> {
        validate_probability(survival)?;
        if survival <= 0.0 || survival >= 1.0 {
            return Err(CrcError::InvalidInput(
                "parametric isf requires probability strictly between zero and one".into(),
            ));
        }
        let value = match self.family {
            DistributionFamily::GumbelRight => {
                self.location - self.scale * (-(-survival).ln_1p()).ln()
            }
            DistributionFamily::GumbelLeft => self.location + self.scale * (-survival.ln()).ln(),
            DistributionFamily::WeibullMin => {
                self.location
                    + self.scale * (-survival.ln()).powf(1.0 / self.shape.expect("validated"))
            }
            DistributionFamily::WeibullMax => {
                self.location
                    - self.scale * (-(-survival).ln_1p()).powf(1.0 / self.shape.expect("validated"))
            }
            DistributionFamily::GenPareto => {
                let c = self.shape.expect("validated");
                if c.abs() < EPS {
                    self.location - self.scale * survival.ln()
                } else {
                    self.location + self.scale * (survival.powf(-c) - 1.0) / c
                }
            }
            DistributionFamily::GenExtreme => {
                let c = self.shape.expect("validated");
                // -ln(cdf), formed from the survival probability without rounding
                // the cdf to one first.
                let neg_log_cdf = -(-survival).ln_1p();
                if c.abs() < EPS {
                    self.location - self.scale * neg_log_cdf.ln()
                } else {
                    self.location + self.scale * (1.0 - neg_log_cdf.powf(c)) / c
                }
            }
            DistributionFamily::SkewNormal => {
                let reflected = Self {
                    shape: self.shape.map(|alpha| -alpha),
                    location: -self.location,
                    ..*self
                };
                -inverse_by_bisection(|x| reflected.cdf(x), survival)
            }
        };
        Ok(value)
    }
}

/// Quantile of `base` truncated to `(atom, inf)` at `conditional_probability`.
///
/// `atom_cdf` and `atom_sf` are `base.cdf(atom)` and `base.sf(atom)`. The
/// target is inverted from whichever tail holds less than half the mass, so
/// the result stays accurate when the atom sits deep in either tail of the
/// base; e.g. an atom with `sf ~ 1e-12` leaves the whole conditional tail
/// within a few ULPs of one in cdf space.
pub(crate) fn truncated_ppf(
    base: &FittedDistribution,
    atom_cdf: f64,
    atom_sf: f64,
    conditional_probability: f64,
) -> Result<f64> {
    let survival = (1.0 - conditional_probability) * atom_sf;
    if survival < 0.5 {
        base.isf(survival)
    } else {
        base.ppf(atom_cdf + conditional_probability * atom_sf)
    }
}

/// Conditional cdf of `base` truncated to `(atom, inf)`, evaluated at `x > atom`.
fn truncated_cdf(base: &FittedDistribution, atom_cdf: f64, atom_sf: f64, x: f64) -> f64 {
    if atom_cdf < atom_sf {
        (base.cdf(x) - atom_cdf) / atom_sf
    } else {
        (atom_sf - base.sf(x)) / atom_sf
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
        if x.is_nan() {
            return f64::NAN;
        }
        let z = self.z(x);
        match self.family {
            DistributionFamily::GumbelRight => (-(-z).exp()).exp(),
            DistributionFamily::GumbelLeft => -(-z.exp()).exp_m1(),
            DistributionFamily::WeibullMin => {
                if z < 0.0 {
                    0.0
                } else {
                    -(-z.powf(self.shape.expect("validated"))).exp_m1()
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
                    -(-z).exp_m1()
                } else {
                    let support = 1.0 + c * z;
                    if support <= 0.0 {
                        1.0
                    } else {
                        // 1 - (1 + c z)^(-1/c), without cancellation for small z.
                        -(-(c * z).ln_1p() / c).exp_m1()
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
        if q == 0.0 || q == 1.0 {
            return Ok(match (self.family, q == 0.0) {
                (DistributionFamily::WeibullMin | DistributionFamily::GenPareto, true) => {
                    self.location
                }
                (DistributionFamily::WeibullMax, false) => self.location,
                (DistributionFamily::GenPareto, false) if self.shape.expect("validated") < 0.0 => {
                    self.location - self.scale / self.shape.expect("validated")
                }
                (DistributionFamily::GenExtreme, true) if self.shape.expect("validated") < 0.0 => {
                    self.location + self.scale / self.shape.expect("validated")
                }
                (DistributionFamily::GenExtreme, false) if self.shape.expect("validated") > 0.0 => {
                    self.location + self.scale / self.shape.expect("validated")
                }
                (_, true) => f64::NEG_INFINITY,
                (_, false) => f64::INFINITY,
            });
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
            DistributionFamily::SkewNormal => inverse_by_bisection(|x| self.cdf(x), q),
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
    base_sf_at_atom: f64,
    base_log_sf_at_atom: f64,
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
        let base_sf_at_atom = base.sf(atom_location);
        let base_log_sf_at_atom = base.log_survival(atom_location);
        if !base_log_sf_at_atom.is_finite() {
            return Err(CrcError::InvalidInput(
                "hurdle base distribution must have positive probability above the atom".into(),
            ));
        }
        Ok(Self {
            atom_location,
            atom_probability,
            base,
            base_cdf_at_atom,
            base_sf_at_atom,
            base_log_sf_at_atom,
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

    pub(crate) fn survival_at(&self, x: f64, inclusive: bool) -> f64 {
        if x < self.atom_location || (inclusive && x == self.atom_location) {
            1.0
        } else {
            let survival = self.base.sf(x);
            let conditional =
                if survival >= f64::MIN_POSITIVE && self.base_sf_at_atom >= f64::MIN_POSITIVE {
                    survival / self.base_sf_at_atom
                } else {
                    (self.base.log_survival(x) - self.base_log_sf_at_atom)
                        .min(0.0)
                        .exp()
                };
            (1.0 - self.atom_probability) * conditional
        }
    }

    pub(crate) fn inverse_survival(&self, survival: f64) -> Result<f64> {
        let weight = 1.0 - self.atom_probability;
        if survival >= weight {
            return Ok(self.atom_location);
        }
        if survival == 0.0 {
            return self.base.ppf(1.0);
        }
        let conditional = survival / weight;
        let target = conditional * self.base_sf_at_atom;
        let quantile = if target >= f64::MIN_POSITIVE {
            self.base.isf(target)
        } else {
            self.base
                .inverse_log_survival(conditional.ln() + self.base_log_sf_at_atom)
        }?;
        Ok(quantile.max(self.atom_location))
    }
}

impl Distribution for HurdleDistribution {
    fn pdf(&self, x: f64) -> f64 {
        if x <= self.atom_location {
            0.0
        } else if self.base_sf_at_atom < f64::MIN_POSITIVE {
            ((-self.atom_probability).ln_1p() + self.base.log_pdf(x) - self.base_log_sf_at_atom)
                .exp()
        } else {
            (self.base.pdf(x) / self.base_sf_at_atom) * (1.0 - self.atom_probability)
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            f64::NAN
        } else if x < self.atom_location {
            0.0
        } else if x == self.atom_location {
            self.atom_probability
        } else {
            let conditional = if self.base_sf_at_atom < f64::MIN_POSITIVE {
                -(self.base.log_survival(x) - self.base_log_sf_at_atom).exp_m1()
            } else {
                truncated_cdf(&self.base, self.base_cdf_at_atom, self.base_sf_at_atom, x)
            };
            (self.atom_probability + (1.0 - self.atom_probability) * conditional)
                .clamp(self.atom_probability, 1.0)
        }
    }

    fn probability_atom(&self, x: f64) -> f64 {
        if x == self.atom_location {
            self.atom_probability
        } else {
            0.0
        }
    }

    fn ppf(&self, q: f64) -> Result<f64> {
        validate_probability(q)?;
        if q <= self.atom_probability {
            return Ok(self.atom_location);
        }
        if q >= 0.5 || self.base_sf_at_atom < 0.5 {
            return self.inverse_survival(1.0 - q);
        }
        let conditional_probability = (q - self.atom_probability) / (1.0 - self.atom_probability);
        truncated_ppf(
            &self.base,
            self.base_cdf_at_atom,
            self.base_sf_at_atom,
            conditional_probability,
        )
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
    pub iterations: usize,
    pub evaluations: usize,
}

/// Constraints on the selected law, enforced inside optimization.
/// With no probability, limits bound the complete support. Otherwise both
/// limits bound the quantile at that probability.
#[derive(Debug, Clone, Copy, Default)]
pub struct FitBounds {
    pub minimum_value: Option<f64>,
    pub maximum_value: Option<f64>,
    pub probability: Option<f64>,
    pub require_sample_support: bool,
}

impl FitBounds {
    pub fn validate(&self) -> Result<()> {
        if self.minimum_value.is_some_and(|x| !x.is_finite())
            || self.maximum_value.is_some_and(|x| !x.is_finite())
            || self
                .probability
                .is_some_and(|q| !q.is_finite() || !(0.0..=1.0).contains(&q))
            || matches!((self.minimum_value, self.maximum_value), (Some(a), Some(b)) if a > b)
        {
            return Err(CrcError::InvalidInput("invalid fitting constraints".into()));
        }
        Ok(())
    }

    pub fn accepts(&self, law: &dyn Distribution, samples: &[f64]) -> bool {
        let lower = law
            .ppf(self.probability.unwrap_or(0.0))
            .unwrap_or(f64::NEG_INFINITY);
        let upper = law
            .ppf(self.probability.unwrap_or(1.0))
            .unwrap_or(f64::INFINITY);
        self.minimum_value.is_none_or(|minimum| lower >= minimum)
            && self.maximum_value.is_none_or(|maximum| upper <= maximum)
            && (!self.require_sample_support
                || samples.iter().all(|&value| {
                    let a = law.ppf(0.0).unwrap_or(f64::NEG_INFINITY);
                    let b = law.ppf(1.0).unwrap_or(f64::INFINITY);
                    value >= a && value <= b
                }))
    }
}

fn project_fit_bounds(
    family: DistributionFamily,
    bounds: FitBounds,
    parameters: &mut [f64],
    log_scale: bool,
) {
    let shaped = !matches!(
        family,
        DistributionFamily::GumbelRight | DistributionFamily::GumbelLeft
    );
    let location_index = usize::from(shaped);
    let shape = shaped.then_some(parameters[0]);
    let scale = if log_scale {
        parameters[location_index + 1].exp()
    } else {
        parameters[location_index + 1]
    };
    // A location shift moves every unconditional quantile by that exact amount.
    // Project before scoring to let the optimizer converge on an active limit.
    if let Ok(law) =
        FittedDistribution::from_parameters(family, shape, parameters[location_index], scale)
    {
        if let Some((probability, target)) = bounds
            .probability
            .zip(bounds.minimum_value)
            .filter(|&(_, target)| bounds.maximum_value == Some(target))
        {
            // The two one-sided projections deliberately move inside their
            // bounds. An equality has no interior, so place it directly.
            let offset = FittedDistribution::from_parameters(family, shape, 0.0, scale)
                .and_then(|centered| centered.ppf(probability));
            if let Ok(offset) = offset {
                if offset.is_finite() {
                    parameters[location_index] = target - offset;
                }
            }
            return;
        }
        let mut shift: f64 = 0.0;
        if let Some(minimum) = bounds.minimum_value {
            if let Ok(q) = law.ppf(bounds.probability.unwrap_or(0.0)) {
                if q.is_finite() && q < minimum {
                    shift = minimum - q + 1e-12 * scale;
                }
            }
        }
        if let Some(maximum) = bounds.maximum_value {
            if let Ok(q) = law.ppf(bounds.probability.unwrap_or(1.0)) {
                if q + shift > maximum {
                    shift = maximum - q - 1e-12 * scale;
                }
            }
        }
        parameters[location_index] += shift;
    }
}

pub fn fit_samples_mle(
    samples: &[f64],
    family: DistributionFamily,
    bounds: FitBounds,
) -> Result<FitResult> {
    if samples.len() < 3 || samples.iter().any(|x| !x.is_finite()) {
        return Err(CrcError::InvalidInput(
            "MLE requires at least three finite observations".into(),
        ));
    }
    bounds.validate()?;
    let kind = reference_curve_fitting::DistributionKind::from_name(family.name())
        .expect("shared family names");
    let optimized = reference_curve_fitting::fit_sample_likelihood(kind, samples, bounds)
        .ok_or_else(|| {
            CrcError::Unsupported(format!(
                "{} has no feasible likelihood fit under the supplied constraints",
                family.name()
            ))
        })?;
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
    Ok(FitResult {
        diagnostics: diagnostics(samples, &distribution),
        distribution,
        iterations: optimized.iterations,
        evaluations: optimized.evaluations,
    })
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
pub struct CensoredQuantileFitDiagnostics {
    pub observed: QuantileFitDiagnostics,
    pub negative_log_likelihood: f64,
    pub residual_scale: f64,
    pub residual_scale_floor: f64,
    pub censor_location: f64,
    pub censored_point_count: usize,
    pub uncensored_point_count: usize,
}

#[derive(Debug, Clone)]
pub struct CensoredQuantileFitResult {
    pub distribution: FittedDistribution,
    pub diagnostics: CensoredQuantileFitDiagnostics,
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

/// Fit GEV or Gumbel observations by unbiased sample L-moments.
///
/// This is a standalone estimator: no likelihood optimization or family selection.
/// GEV shape follows the SciPy convention (`c = -xi`).
pub fn fit_lmoments(samples: &[f64], family: DistributionFamily) -> Result<FitResult> {
    let (distribution, sorted) = lmoments::fit(samples, family)?;
    Ok(FitResult {
        diagnostics: diagnostics_sorted(&sorted, &distribution),
        distribution,
        iterations: 0,
        evaluations: 0,
    })
}

pub fn fit_quantiles(
    probabilities: &[f64],
    values: &[f64],
    weights: Option<&[f64]>,
    family: DistributionFamily,
) -> Result<QuantileFitResult> {
    fit_quantiles_with_bounds(probabilities, values, weights, family, FitBounds::default())
}

pub fn fit_quantiles_with_bounds(
    probabilities: &[f64],
    values: &[f64],
    weights: Option<&[f64]>,
    family: DistributionFamily,
    bounds: FitBounds,
) -> Result<QuantileFitResult> {
    bounds.validate()?;
    let weights = validate_quantile_points(probabilities, values, weights)?;
    fit_quantile_points_internal(probabilities, values, &weights, family, None, None, bounds)
}

pub fn fit_censored_quantiles(
    probabilities: &[f64],
    values: &[f64],
    weights: Option<&[f64]>,
    family: DistributionFamily,
    censor_location: f64,
    residual_scale_floor_ratio: f64,
) -> Result<CensoredQuantileFitResult> {
    fit_censored_quantiles_with_bounds(
        probabilities,
        values,
        weights,
        family,
        censor_location,
        residual_scale_floor_ratio,
        FitBounds::default(),
    )
}

pub fn fit_censored_quantiles_with_bounds(
    probabilities: &[f64],
    values: &[f64],
    weights: Option<&[f64]>,
    family: DistributionFamily,
    censor_location: f64,
    residual_scale_floor_ratio: f64,
    bounds: FitBounds,
) -> Result<CensoredQuantileFitResult> {
    bounds.validate()?;
    let weights = validate_quantile_points(probabilities, values, weights)?;
    if !censor_location.is_finite() {
        return Err(CrcError::InvalidInput(
            "censor location must be finite".into(),
        ));
    }
    if !residual_scale_floor_ratio.is_finite() || residual_scale_floor_ratio <= 0.0 {
        return Err(CrcError::InvalidInput(
            "residual scale floor ratio must be finite and positive".into(),
        ));
    }
    let censored_point_count = values
        .iter()
        .zip(&weights)
        .filter(|(value, weight)| **weight > 0.0 && **value <= censor_location)
        .count();
    let uncensored_point_count = values
        .iter()
        .zip(&weights)
        .filter(|(value, weight)| **weight > 0.0 && **value > censor_location)
        .count();
    if censored_point_count == 0 || uncensored_point_count == 0 {
        return Err(CrcError::InvalidInput(
            "censored quantile fitting requires censored and uncensored knots".into(),
        ));
    }

    let conditioning_scale = censored_conditioning_scale(values, &weights, censor_location);
    let residual_scale_floor = (residual_scale_floor_ratio * conditioning_scale).max(EPS);
    if !residual_scale_floor.is_finite()
        || residual_scale_floor > reference_curve_fitting::MAXIMUM_QUANTILE_SCALE
    {
        return Err(CrcError::InvalidInput(
            "residual scale floor must be finite and no greater than 1e12".into(),
        ));
    }
    let kind = reference_curve_fitting::DistributionKind::from_name(family.name())
        .expect("public and reference family names are synchronized");
    let optimized = reference_curve_fitting::fit_censored_quantile_points(
        kind,
        probabilities,
        values,
        &weights,
        censor_location,
        residual_scale_floor,
        Some(&|law| bounds.accepts(law, values)),
        Some(&|parameters| project_fit_bounds(family, bounds, parameters, true)),
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
    // The likelihood scored this same distribution, so the gate and the
    // residuals describe the curve callers evaluate.
    let latent_values = probabilities
        .iter()
        .map(|&probability| distribution.ppf(probability))
        .collect::<Result<Vec<_>>>()?;
    let transition_is_preserved =
        preserves_censor_transition(values, &latent_values, &weights, censor_location);
    if !transition_is_preserved {
        return Err(CrcError::Unsupported(
            "censored fit does not preserve the observed censor transition".into(),
        ));
    }
    let fitted_values = latent_values
        .iter()
        .map(|value| value.max(censor_location))
        .collect::<Vec<_>>();
    // Left-censored knots are scored at the bound. A placeholder below it is
    // not an observed residual.
    let bounded_values: Vec<f64> = values
        .iter()
        .map(|value| value.max(censor_location))
        .collect();
    let observed = quantile_diagnostics(
        &bounded_values,
        &fitted_values,
        &weights,
        optimized.iterations,
        optimized.evaluations,
    );
    Ok(CensoredQuantileFitResult {
        distribution,
        diagnostics: CensoredQuantileFitDiagnostics {
            observed,
            negative_log_likelihood: optimized.negative_log_likelihood,
            residual_scale: optimized.residual_scale,
            residual_scale_floor,
            censor_location,
            censored_point_count,
            uncensored_point_count,
        },
    })
}

pub fn fit_hurdle_quantiles(
    probabilities: &[f64],
    values: &[f64],
    weights: Option<&[f64]>,
    family: DistributionFamily,
    atom_location: f64,
    atom_probability: f64,
) -> Result<HurdleQuantileFitResult> {
    fit_hurdle_quantiles_with_bounds(
        probabilities,
        values,
        weights,
        family,
        atom_location,
        atom_probability,
        FitBounds::default(),
    )
}

pub fn fit_hurdle_quantiles_with_bounds(
    probabilities: &[f64],
    values: &[f64],
    weights: Option<&[f64]>,
    family: DistributionFamily,
    atom_location: f64,
    atom_probability: f64,
    bounds: FitBounds,
) -> Result<HurdleQuantileFitResult> {
    bounds.validate()?;
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
        Some(atom_probability),
        bounds,
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
    atom_probability: Option<f64>,
    bounds: FitBounds,
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
        Some(&|law| {
            if let Some(atom) = truncation_atom {
                HurdleDistribution::new(
                    atom,
                    atom_probability.expect("hurdle probability"),
                    law.clone(),
                )
                .is_ok_and(|hurdle| bounds.accepts(&hurdle, values))
            } else {
                bounds.accepts(law, values)
            }
        }),
        Some(&|parameters| {
            if truncation_atom.is_none() {
                project_fit_bounds(family, bounds, parameters, false);
            }
        }),
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
                truncated_ppf(
                    &distribution,
                    distribution.cdf(atom),
                    distribution.sf(atom),
                    probability,
                )
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

pub fn calculate_diagnostics(
    samples: &[f64],
    distribution: &dyn Distribution,
) -> Result<DiagnosticMetrics> {
    validate_samples(samples)?;
    Ok(diagnostics(samples, distribution))
}

fn diagnostics(samples: &[f64], distribution: &dyn Distribution) -> DiagnosticMetrics {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    diagnostics_sorted(&sorted, distribution)
}

fn diagnostics_sorted(sorted: &[f64], distribution: &dyn Distribution) -> DiagnosticMetrics {
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
    let ks_pvalue = reference_curve_fitting::ks_pvalue_asymptotic(ks, sorted.len());
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
    // Full endpoint span, including zero-weight knots. This scale divides every
    // residual in the bit-exact quantile objective, so it stays on that formula.
    conditioning_scale(
        values,
        weights,
        values.last().expect("validated") - values[0],
    )
}

fn censored_conditioning_scale(values: &[f64], weights: &[f64], censor_location: f64) -> f64 {
    let bounded: Vec<f64> = values
        .iter()
        .map(|value| value.max(censor_location))
        .collect();
    conditioning_scale(
        &bounded,
        weights,
        reference_curve_fitting::positive_value_span(&bounded, weights, censor_location),
    )
}

fn conditioning_scale(values: &[f64], weights: &[f64], range: f64) -> f64 {
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

fn preserves_censor_transition(
    values: &[f64],
    latent_values: &[f64],
    weights: &[f64],
    censor_location: f64,
) -> bool {
    // `censor + 1e-12` is unchanged once the unit in the last place exceeds EPS,
    // so a one-ulp overshoot has to be measured against the bound itself.
    let slack = EPS.max(censor_location.abs() * f64::EPSILON);
    values
        .iter()
        .zip(latent_values)
        .zip(weights)
        .all(|((&observed, &latent), &weight)| {
            weight == 0.0
                || if observed <= censor_location {
                    latent - censor_location <= slack
                } else {
                    censor_location - latent < slack
                }
        })
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
            .zip(weights)
            .filter(|((_, _), weight)| **weight > 0.0)
            .map(|((observed, fitted), _)| (fitted - observed).abs())
            .fold(0.0, f64::max),
        point_count: weights.iter().filter(|weight| **weight > 0.0).count(),
        converged: true,
        iterations,
        evaluations,
    }
}

fn inverse_by_bisection(cdf: impl Fn(f64) -> f64, q: f64) -> f64 {
    let mut lower = -1.0;
    let mut upper = 1.0;
    while cdf(lower) > q {
        lower *= 2.0;
    }
    while cdf(upper) < q {
        upper *= 2.0;
    }
    for _ in 0..120 {
        let middle = (lower + upper) * 0.5;
        // Once the bounds are adjacent f64 values, further CDF integrations
        // cannot refine the quantile. This preserves the converged result.
        if middle == lower || middle == upper {
            break;
        }
        if cdf(middle) < q {
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

fn weibull_log_pdf(x: f64, shape: f64) -> f64 {
    if x > 0.0 {
        shape.ln() + (shape - 1.0) * x.ln() - x.powf(shape)
    } else if x == 0.0 && shape == 1.0 {
        0.0
    } else if x == 0.0 && shape < 1.0 {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    }
}

fn normal_cdf(x: f64) -> f64 {
    reference_curve_fitting::ndtr_cephes(x)
}

/// Skew-normal cdf `Phi(z) - 2 T(z, alpha)` with Owen's T, as in SciPy.
fn skew_normal_cdf(z: f64, alpha: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    let normal = reference_curve_fitting::ndtr_cephes(z);
    let probability = normal - 2.0 * reference_curve_fitting::owens_t(z, alpha);
    if z <= 0.0 && alpha > 0.0 && probability <= 0.01 * normal {
        skew_normal_log_tail(-z, alpha).exp()
    } else {
        probability.clamp(0.0, 1.0)
    }
}

fn log_exponential_complement(log_rate: f64) -> f64 {
    if log_rate < -37.0 {
        log_rate
    } else {
        (-(-log_rate.exp()).exp_m1()).ln()
    }
}

fn skew_normal_log_cdf(z: f64, alpha: f64) -> f64 {
    if z == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if z == f64::INFINITY {
        return 0.0;
    }
    let normal = reference_curve_fitting::ndtr_cephes(z);
    let probability = normal - 2.0 * reference_curve_fitting::owens_t(z, alpha);
    if z <= 0.0 && (probability < f64::MIN_POSITIVE || probability <= 0.01 * normal) {
        skew_normal_log_tail(-z, alpha)
    } else {
        probability.clamp(0.0, 1.0).ln()
    }
}

fn skew_normal_log_tail(cutoff: f64, alpha: f64) -> f64 {
    if cutoff.is_infinite() {
        return f64::NEG_INFINITY;
    }
    // Integrate the positive density φ(t) erfc(αt/√2) for t >= cutoff.
    // Its natural scale is sqrt(1 + α²) when the skew suppresses this tail.
    let scale = alpha.max(0.0).hypot(1.0);
    let start = scale * cutoff;
    if !start.is_finite() {
        return f64::NEG_INFINITY;
    }
    let width = 1.0 / (1.0 + start);
    let slope = alpha / scale;
    let log_normal = reference_curve_fitting::log_standard_normal_cdf;
    let at_start = log_normal(-slope * start);
    let log_peak = -0.5 * cutoff * cutoff + at_start;
    if !log_peak.is_finite() {
        return f64::NEG_INFINITY;
    }
    let options = crate::impact::quadrature::GaussKronrod {
        absolute_tolerance: 1e-12,
        relative_tolerance: 1e-11,
        ..crate::impact::quadrature::GaussKronrod::defaults()
    };
    let integral = crate::impact::quadrature::integrate(&[(0.0, 1.0)], &options, |u| {
        let remaining = 1.0 - u;
        let step = width * u / remaining;
        let delta = step / scale;
        let log_ratio =
            -delta * (cutoff + 0.5 * delta) + log_normal(-slope * (start + step)) - at_start;
        Ok((log_ratio - 2.0 * remaining.ln()).exp())
    });
    integral.map_or(f64::NEG_INFINITY, |integral| {
        log_peak + width.ln() - scale.ln() + (2.0 / PI).sqrt().ln() + integral.value.ln()
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn empirical_quantile_is_the_generalized_inverse_of_its_ecdf() {
        use super::{Distribution, EmpiricalDistribution};
        let law = EmpiricalDistribution::new(vec![0.0, 0.0, 2.0, 10.0]).unwrap();
        assert_eq!(law.ppf(0.0).unwrap(), 0.0);
        assert_eq!(law.ppf(0.5).unwrap(), 0.0);
        assert_eq!(law.ppf(0.50001).unwrap(), 2.0);
        assert_eq!(law.ppf(1.0).unwrap(), 10.0);
        assert_eq!(law.cdf(0.0), 0.5);
        assert_eq!(law.probability_atom(0.0), 0.5);
        assert_eq!(
            EmpiricalDistribution::new(vec![3.0])
                .unwrap()
                .ppf(0.5)
                .unwrap(),
            3.0
        );
    }
    use super::*;

    #[test]
    fn censor_transition_allows_one_ulp_on_a_large_bound() {
        let censor = 1.0e6;
        assert_eq!(
            censor + EPS,
            censor,
            "absolute EPS must vanish at this magnitude"
        );
        let one_ulp = f64::from_bits(censor.to_bits() + 1);
        let two_ulps = f64::from_bits(censor.to_bits() + 2);
        let values = [
            censor,
            censor,
            censor,
            censor + 1.0,
            censor + 2.0,
            censor + 3.0,
        ];
        let weights = [1.0; 6];
        let one_ulp_overshoot = [
            one_ulp,
            censor,
            censor,
            censor + 1.0,
            censor + 2.0,
            censor + 3.0,
        ];
        assert!(preserves_censor_transition(
            &values,
            &one_ulp_overshoot,
            &weights,
            censor
        ));
        let two_ulp_overshoot = [
            two_ulps,
            censor,
            censor,
            censor + 1.0,
            censor + 2.0,
            censor + 3.0,
        ];
        assert!(!preserves_censor_transition(
            &values,
            &two_ulp_overshoot,
            &weights,
            censor
        ));
        let weights_with_sentinel = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let ignored = [
            censor + 1.0,
            censor,
            censor,
            censor + 1.0,
            censor + 2.0,
            censor + 3.0,
        ];
        assert!(preserves_censor_transition(
            &values,
            &ignored,
            &weights_with_sentinel,
            censor
        ));
    }

    #[test]
    fn point_mass_is_exact_at_every_probability() {
        let distribution = PointMassDistribution::new(2.5).unwrap();
        assert_eq!(
            distribution.quantiles(&[0.0, 0.5, 1.0]).unwrap(),
            vec![2.5; 3]
        );
        assert_eq!(distribution.cdf(2.49), 0.0);
        assert_eq!(distribution.cdf(2.5), 1.0);
        assert_eq!(distribution.cdf_below(2.5), 0.0);
        assert_eq!(distribution.probability_atom(2.5), 1.0);
        #[cfg(feature = "sampling")]
        assert_eq!(distribution.sample(3, Some(42)).unwrap(), vec![2.5; 3]);
    }

    #[test]
    fn tabulated_flat_cdf_includes_the_atom() {
        let distribution = TabulatedDistribution::new(
            vec![0.0, 0.2, 0.8, 1.0],
            vec![0.0, 1.0, 1.0, 2.0],
            Interpolation::LinearProbability,
            false,
        )
        .unwrap();
        assert_eq!(distribution.cdf(1.0), 0.8);
        assert_eq!(distribution.cdf_below(1.0), 0.2);
        assert_eq!(
            distribution.probability_atom(1.0),
            distribution.cdf(1.0) - distribution.cdf_below(1.0)
        );
        assert_eq!(distribution.cdf(0.5), 0.1);
        assert_eq!(distribution.cdf_below(0.5), 0.1);
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
    fn sample_mle_gumbel_solves_the_likelihood_score() {
        let samples = [0.0, 0.2, 0.7, 1.1, 1.9, 2.4, 3.2, 4.8];
        let result = fit_samples_mle(
            &samples,
            DistributionFamily::GumbelRight,
            FitBounds::default(),
        )
        .unwrap();
        let law = result.distribution;
        let score_location = samples
            .iter()
            .map(|x| 1.0 - (-((x - law.location) / law.scale)).exp())
            .sum::<f64>();
        let score_scale = samples
            .iter()
            .map(|x| {
                let z = (x - law.location) / law.scale;
                -1.0 + z * (1.0 - (-z).exp())
            })
            .sum::<f64>();
        assert!(
            score_location.abs() < 1e-5,
            "location score {score_location}"
        );
        assert!(score_scale.abs() < 1e-5, "scale score {score_scale}");
        assert!(result.iterations > 0 && result.evaluations > 0);
    }

    #[test]
    fn fixed_family_mle_enforces_quantile_bounds_during_optimization() {
        let samples = [0.0, 0.2, 0.7, 1.1, 1.9, 2.4, 3.2, 4.8];
        let bounds = FitBounds {
            maximum_value: Some(4.0),
            probability: Some(0.99),
            ..FitBounds::default()
        };
        let result = fit_samples_mle(&samples, DistributionFamily::GumbelRight, bounds).unwrap();
        assert_eq!(result.distribution.family, DistributionFamily::GumbelRight);
        assert!(result.distribution.ppf(0.99).unwrap() <= 4.0);
        assert!(result.evaluations > 0);
    }

    #[test]
    fn infeasible_support_constraint_cannot_return_an_mle() {
        let samples = [0.0, 0.2, 0.7, 1.1, 1.9, 2.4, 3.2, 4.8];
        let bounds = FitBounds {
            minimum_value: Some(0.0),
            ..FitBounds::default()
        };
        assert!(fit_samples_mle(&samples, DistributionFamily::GumbelRight, bounds).is_err());
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
        #[cfg(feature = "sampling")]
        {
            let samples = hurdle.sample(5_000, Some(42)).unwrap();
            let atom_fraction =
                samples.iter().filter(|value| **value == 0.0).count() as f64 / samples.len() as f64;
            assert!((atom_fraction - 0.4).abs() < 0.03);
        }
        let statistics = crate::metrics::calculate_statistics(&hurdle).unwrap();
        assert_eq!(statistics.minimum, 0.0);
        assert!(statistics.mean > 0.0);
    }

    fn tail_test_distributions() -> Vec<FittedDistribution> {
        [
            (DistributionFamily::GumbelRight, None),
            (DistributionFamily::GumbelLeft, None),
            (DistributionFamily::GenExtreme, Some(0.0)),
            (DistributionFamily::GenExtreme, Some(0.2)),
            (DistributionFamily::GenExtreme, Some(-0.2)),
            (DistributionFamily::WeibullMin, Some(1.5)),
            (DistributionFamily::WeibullMax, Some(2.0)),
            (DistributionFamily::GenPareto, Some(0.0)),
            (DistributionFamily::GenPareto, Some(0.2)),
            (DistributionFamily::GenPareto, Some(-0.2)),
            (DistributionFamily::SkewNormal, Some(3.0)),
        ]
        .into_iter()
        .map(|(family, shape)| {
            FittedDistribution::from_parameters(family, shape, 1.5, 2.5).unwrap()
        })
        .collect()
    }

    #[test]
    fn survival_function_complements_cdf_and_inverts_for_every_family() {
        for distribution in tail_test_distributions() {
            let (tolerance, isf_tolerance) = (1.0e-12, 1.0e-8);
            for q in [0.001, 0.05, 0.3, 0.5, 0.7, 0.95, 0.999] {
                let x = distribution.ppf(q).unwrap();
                let survival = distribution.sf(x);
                assert!(
                    (survival + distribution.cdf(x) - 1.0).abs() < tolerance,
                    "{:?} sf+cdf at q={q}",
                    distribution.family
                );
                assert!(
                    (survival - (1.0 - q)).abs() < tolerance,
                    "{:?} sf at q={q}",
                    distribution.family
                );
                let inverted = distribution.isf(1.0 - q).unwrap();
                assert!(
                    (inverted - x).abs() < isf_tolerance * x.abs().max(1.0),
                    "{:?} isf at q={q}: {inverted} vs {x}",
                    distribution.family
                );
            }
            assert!(distribution.isf(0.0).is_err());
            assert!(distribution.isf(1.0).is_err());
        }
    }

    #[test]
    fn survival_function_keeps_relative_precision_deep_in_the_upper_tail() {
        for distribution in tail_test_distributions() {
            if distribution.family == DistributionFamily::SkewNormal {
                continue;
            }
            // Far below the ~1.1e-16 spacing of f64 values just under one.
            for survival in [1.0e-13, 1.0e-20, 1.0e-40] {
                let Ok(x) = distribution.isf(survival) else {
                    panic!("{:?} isf({survival})", distribution.family);
                };
                if !x.is_finite() {
                    // Bounded upper supports (e.g. weibull_max) end first.
                    continue;
                }
                let round_trip = distribution.sf(x);
                if distribution.family == DistributionFamily::WeibullMax
                    || distribution.family == DistributionFamily::GenPareto
                        && distribution.shape == Some(-0.2)
                    || distribution.family == DistributionFamily::GenExtreme
                        && distribution.shape == Some(0.2)
                {
                    // Bounded above: the survival round-trips through a value
                    // squeezed against the upper endpoint.
                    assert!(round_trip <= 1.0e-12, "{:?}", distribution.family);
                    continue;
                }
                assert!(
                    ((round_trip - survival) / survival).abs() < 1.0e-9,
                    "{:?} sf(isf({survival})) = {round_trip}",
                    distribution.family
                );
            }
        }
        let gumbel =
            FittedDistribution::from_parameters(DistributionFamily::GumbelRight, None, 0.0, 1.0)
                .unwrap();
        assert_eq!(gumbel.cdf(40.0), 1.0);
        assert!(((gumbel.sf(40.0) - (-40.0f64).exp()) / (-40.0f64).exp()).abs() < 1.0e-15);
    }

    /// Ottawa hot_days (852b83b3fffffff, ">4 degrees", 2050) as published in the
    /// canonical dataset: the atom sits ~27.6 scales into the base's upper tail,
    /// leaving 1.0232e-12 of base mass above it. Capping the base probability at
    /// `1 - 1e-12` collapsed every return period to ~0.41 days.
    fn ottawa_hot_days() -> HurdleDistribution {
        let base = FittedDistribution::from_parameters(
            DistributionFamily::GumbelRight,
            None,
            -489.120_459_721_483_3,
            17.716_588_490_591_306,
        )
        .unwrap();
        HurdleDistribution::new(0.0, 0.032, base).unwrap()
    }

    #[test]
    fn hurdle_quantiles_are_exact_when_the_atom_is_deep_in_the_base_tail() {
        let hurdle = ottawa_hot_days();
        // Reference values from 60-digit arithmetic on the stored parameters.
        for (return_period, expected) in [
            (10.0, 40.217_752_553_305_5),
            (25.0, 56.451_298_387_663_2),
            (100.0, 81.011_705_110_451_8),
            (1000.0, 121.805_657_667_597),
        ] {
            let probability = 1.0 - 1.0 / return_period;
            let value = hurdle.ppf(probability).unwrap();
            assert!(
                (value - expected).abs() < 1.0e-9,
                "RP{return_period}: {value} vs {expected}"
            );
            assert!((hurdle.cdf(value) - probability).abs() < 1.0e-12);
        }
        assert_eq!(hurdle.ppf(0.032).unwrap(), 0.0);
        let values = hurdle
            .quantiles(&[0.05, 0.2, 0.5, 0.8, 0.95, 0.999, 0.999_999])
            .unwrap();
        assert!(values.windows(2).all(|pair| pair[0] < pair[1]));
        // The continuous part is the base's (near-exponential) upper tail.
        let continuous_mass = {
            let steps = 20_000;
            let upper = values[values.len() - 1];
            let width = upper / steps as f64;
            (0..steps)
                .map(|index| hurdle.pdf((index as f64 + 0.5) * width) * width)
                .sum::<f64>()
        };
        assert!((continuous_mass - 0.968).abs() < 1.0e-4);
    }

    #[test]
    fn hurdle_accepts_an_atom_beyond_the_former_tail_guard() {
        // cdf(atom) rounds to one here, but the survival mass is ~5.2e-17.
        let base =
            FittedDistribution::from_parameters(DistributionFamily::GumbelRight, None, -37.5, 1.0)
                .unwrap();
        assert_eq!(base.cdf(0.0), 1.0);
        let hurdle = HurdleDistribution::new(0.0, 0.25, base).unwrap();
        // The truncated tail is the unit exponential to within ~1e-15.
        for q in [0.3, 0.6, 0.9, 0.99] {
            let conditional: f64 = (q - 0.25) / 0.75;
            let expected = -(-conditional).ln_1p();
            assert!((hurdle.ppf(q).unwrap() - expected).abs() < 1.0e-9);
        }
        let underflow =
            FittedDistribution::from_parameters(DistributionFamily::GumbelRight, None, -800.0, 1.0)
                .unwrap();
        assert_eq!(underflow.sf(0.0), 0.0);
        let hurdle = HurdleDistribution::new(0.0, 0.25, underflow).unwrap();
        assert!((hurdle.ppf(0.75).unwrap() - 3.0_f64.ln()).abs() < 1e-12);
        assert!((hurdle.cdf(1.0) - (1.0 - 0.75 * (-1.0_f64).exp())).abs() < 1e-12);
        assert!((hurdle.pdf(1.0) - 0.75 * (-1.0_f64).exp()).abs() < 1e-12);

        let empty = FittedDistribution::from_parameters(
            DistributionFamily::WeibullMax,
            Some(1.0),
            0.0,
            1.0,
        )
        .unwrap();
        assert!(HurdleDistribution::new(0.0, 0.25, empty).is_err());
    }

    #[test]
    fn hurdle_quantile_fitting_scores_a_deep_tail_atom_on_its_true_quantiles() {
        let source = ottawa_hot_days();
        // The curve_fit_cdf knot grid, from the last plateau knot at the atom.
        let probabilities: Vec<f64> = (32..=999).map(|index| index as f64 / 1000.0).collect();
        let values = source.quantiles(&probabilities).unwrap();
        assert_eq!(values[0], 0.0);
        let result = fit_hurdle_quantiles(
            &probabilities,
            &values,
            None,
            DistributionFamily::GumbelRight,
            0.0,
            0.032,
        )
        .unwrap();
        // The truncated base is only identified up to its tail scale here.
        assert!((result.distribution.base().scale - 17.716_588_490_591_306).abs() < 1.0e-3);
        assert!(result.diagnostics.tail.normalized_rmse < 1.0e-4);
        for probability in [0.5, 0.9, 0.99] {
            let fitted = result.distribution.ppf(probability).unwrap();
            let expected = source.ppf(probability).unwrap();
            assert!(
                (fitted - expected).abs() < 1.0e-2,
                "{probability}: {fitted} vs {expected}"
            );
        }
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

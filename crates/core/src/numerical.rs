//! Complete probability laws and probability-linear quantile evidence.
//!
//! Finite laws integrate their actual probability intervals, including tiny
//! atoms. No quantile grid is substituted for the caller's law.

use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use serde::Deserialize;

use crate::distribution::log_gamma_one_plus;
use crate::error::{CrcError, Result};
use crate::impact::quadrature::{GaussKronrod, integrate};
use crate::{Distribution, DistributionFamily, FittedDistribution};

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum Specification {
    Parametric {
        family: String,
        location: f64,
        scale: f64,
        shape: Option<f64>,
    },
    Discrete {
        values: Vec<f64>,
        probabilities: Vec<f64>,
    },
    Histogram {
        edges: Vec<f64>,
        probabilities: Vec<f64>,
    },
    PointMass {
        value: f64,
    },
    Hurdle {
        base: Box<Specification>,
        atom_probability: f64,
        atom_location: f64,
    },
    Quantile {
        probabilities: Vec<f64>,
        values: Vec<f64>,
    },
}

#[derive(Clone, Debug)]
pub enum NumericalLaw {
    Parametric(FittedDistribution),
    Discrete {
        values: Vec<f64>,
        probabilities: Vec<f64>,
        cumulative: Vec<f64>,
    },
    Histogram {
        edges: Vec<f64>,
        probabilities: Vec<f64>,
        cumulative: Vec<f64>,
    },
    PointMass(f64),
    Hurdle {
        base: Box<NumericalLaw>,
        atom_probability: f64,
        atom_location: f64,
    },
    Quantile {
        probabilities: Vec<f64>,
        values: Vec<f64>,
    },
}

fn invalid(message: &str) -> CrcError {
    CrcError::InvalidInput(message.into())
}

/// Interpolate finite endpoints without overflowing their difference.
pub(crate) fn linear_interpolate(a: f64, b: f64, weight: f64) -> f64 {
    if weight == 0.0 {
        a
    } else if weight == 1.0 {
        b
    } else if (a < 0.0 && b > 0.0) || (b < 0.0 && a > 0.0) {
        a * (1.0 - weight) + b * weight
    } else {
        a + (b - a) * weight
    }
}

/// Fraction of a finite interval covered by the requested bounds.
pub(crate) fn interval_fraction(lo: f64, hi: f64, a: f64, b: f64) -> f64 {
    let (a, b) = (a.max(lo), b.min(hi));
    if a >= b {
        return 0.0;
    }
    let width = hi - lo;
    if width.is_finite() {
        (b - a) / width
    } else {
        (b * 0.5 - a * 0.5) / (hi * 0.5 - lo * 0.5)
    }
}

fn cumulative(probabilities: &[f64]) -> Result<Vec<f64>> {
    if probabilities.is_empty() || probabilities.iter().any(|p| !p.is_finite() || *p < 0.0) {
        return Err(invalid("probabilities must be finite non-negative masses"));
    }
    let sum = compensated(probabilities.iter().copied());
    if (sum - 1.0).abs() > 1e-12 {
        return Err(invalid("probability mass must sum to one"));
    }
    let mut result = vec![0.0];
    for p in probabilities {
        result.push(result.last().unwrap() + p / sum);
    }
    *result.last_mut().unwrap() = 1.0;
    Ok(result)
}

#[derive(Default)]
pub(crate) struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    pub(crate) fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if !next.is_finite() {
            self.sum = next;
            self.correction = 0.0;
            return;
        }
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }

    pub(crate) fn total(&self) -> f64 {
        self.sum + self.correction
    }
}

pub(crate) fn compensated(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = CompensatedSum::default();
    for value in values {
        sum.add(value);
    }
    sum.total()
}

fn scaled_difference(value: f64, center: f64, scale: f64) -> f64 {
    let difference = value - center;
    if difference.is_finite() {
        difference * scale
    } else {
        value * scale - center * scale
    }
}

#[derive(Clone, Copy)]
pub(crate) struct IntegralMoment {
    pub(crate) center: f64,
    pub(crate) square: bool,
    pub(crate) scale: f64,
}

impl IntegralMoment {
    pub(crate) fn new(center: f64, square: bool, scale: f64) -> Self {
        Self {
            center,
            square,
            scale,
        }
    }

    pub(crate) fn value(self, x: f64) -> f64 {
        scaled_difference(x, self.center, self.scale)
    }

    pub(crate) fn segment(self, weight: f64, a: f64, b: f64) -> f64 {
        if weight == 0.0 {
            return 0.0;
        }
        if !self.square {
            return self.weighted_difference(a.midpoint(b), self.center, weight);
        }
        let (mean, deviation) = self.segment_deviation(weight, a, b);
        mean.powi(2) + deviation.powi(2)
    }

    pub(crate) fn linear_quartiles(self, weight: f64, first: f64, third: f64) -> f64 {
        if !self.square {
            return self.weighted_difference(first.midpoint(third), self.center, weight);
        }
        let root_mass = weight.sqrt();
        let mean = self.weighted_difference(first.midpoint(third), self.center, root_mass);
        let deviation = self.weighted_difference(third, first, root_mass / 3.0_f64.sqrt());
        mean.powi(2) + deviation.powi(2)
    }

    fn segment_deviation(self, weight: f64, a: f64, b: f64) -> (f64, f64) {
        let root_mass = weight.sqrt();
        (
            self.weighted_difference(a.midpoint(b), self.center, root_mass),
            self.weighted_difference(b, a, root_mass / 12.0_f64.sqrt()),
        )
    }

    fn weighted_difference(self, value: f64, center: f64, weight: f64) -> f64 {
        // Apply mass and affine scale before squaring. If the factors alone
        // round into the subnormal range, apply the scale to the value first.
        let combined = self.scale * weight;
        if combined < f64::MIN_POSITIVE && self.scale != 1.0 {
            scaled_difference(value, center, self.scale) * weight
        } else {
            scaled_difference(value, center, combined)
        }
    }
}

pub(crate) fn affine_segment_mean_deviation(
    segments: &[ProbabilitySegment],
    slope: f64,
    intercept: f64,
) -> (f64, f64) {
    let first = IntegralMoment::new(0.0, false, slope.abs());
    let mean = compensated(
        segments
            .iter()
            .map(|&(a, b, mass)| first.segment(mass, a, b)),
    );
    let deviation =
        segments
            .iter()
            .filter(|&&(_, _, mass)| mass > 0.0)
            .fold(0.0_f64, |norm, &(a, b, mass)| {
                let a = a * slope.abs();
                let b = b * slope.abs();
                let moment = IntegralMoment::new(mean, true, 1.0);
                let (offset, spread) = moment.segment_deviation(mass, a, b);
                norm.hypot(offset.hypot(spread))
            });
    (mean * slope.signum() + intercept, deviation)
}

// Local probability coordinates avoid subtracting two cumulative values near
// one. In particular an uppermost mass smaller than machine epsilon retains
// its original width instead of becoming a zero-width cumulative interval.
fn mass_intervals(probabilities: &[f64]) -> Vec<(f64, f64, f64)> {
    let total = compensated(probabilities.iter().copied());
    let mut suffix = vec![0.0; probabilities.len() + 1];
    let mut sum: f64 = 0.0;
    let mut correction: f64 = 0.0;
    for i in (0..probabilities.len()).rev() {
        let value = probabilities[i] / total;
        let next = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
        suffix[i] = sum + correction;
    }
    let mut sum: f64 = 0.0;
    let mut correction: f64 = 0.0;
    probabilities
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let mass = p / total;
            let left = sum + correction;
            let next = sum + mass;
            correction += if sum.abs() >= mass.abs() {
                (sum - next) + mass
            } else {
                (mass - next) + sum
            };
            sum = next;
            (left, suffix[i + 1], mass)
        })
        .collect()
}

fn local_interval(lo: f64, hi: f64, left: f64, right: f64, mass: f64) -> (f64, f64, f64) {
    if left > right || (left == right && (lo + hi) * 0.5 > 0.5) {
        let a = ((1.0 - lo) - right).clamp(0.0, mass);
        let b = ((1.0 - hi) - right).clamp(0.0, mass);
        ((a - b).max(0.0), 1.0 - a / mass, 1.0 - b / mass)
    } else {
        let a = (lo - left).clamp(0.0, mass);
        let b = (hi - left).clamp(0.0, mass);
        ((b - a).max(0.0), a / mass, b / mass)
    }
}

/// Gamma(1+r), (Gamma(1+r)-1)/r, and
/// (Gamma(1+2r)-Gamma(1+r)^2)/r^2. The convergent log-Gamma
/// series retains small nonzero shapes instead of replacing their law.
fn gamma_moments(r: f64) -> (f64, f64, f64) {
    if r.abs() >= 0.01 {
        let a = crate::reference::curve_fitting::gamma(1.0 + r);
        return (
            a,
            (a - 1.0) / r,
            (crate::reference::curve_fitting::gamma(1.0 + 2.0 * r) - a * a) / (r * r),
        );
    }
    let zeta = [
        1.644_934_066_848_226_4,
        1.202_056_903_159_594_2,
        1.082_323_233_711_138_1,
        1.036_927_755_143_37,
        1.017_343_061_984_449_2,
        1.008_349_277_381_923,
        1.004_077_356_197_944_4,
        1.002_008_392_826_082_2,
        1.000_994_575_127_818,
    ];
    let mut log_over_r = -0.577_215_664_901_532_9;
    let mut delta_over_r2 = 0.0;
    for (i, z) in zeta.into_iter().enumerate() {
        let n = (i + 2) as i32;
        let coefficient = if n % 2 == 0 {
            z / f64::from(n)
        } else {
            -z / f64::from(n)
        };
        log_over_r += coefficient * r.powi(n - 1);
        delta_over_r2 += coefficient * (2.0_f64.powi(n) - 2.0) * r.powi(n - 2);
    }
    let log = log_over_r * r;
    let delta = delta_over_r2 * r * r;
    let expm1_ratio = |x: f64| if x == 0.0 { 1.0 } else { x.exp_m1() / x };
    (
        log.exp(),
        log_over_r * expm1_ratio(log),
        (2.0 * log).exp() * delta_over_r2 * expm1_ratio(delta),
    )
}

/// Lower incomplete gamma for positive shape. All terms in the convergent
/// series are positive, so an integrable singularity at zero is never
/// sampled or truncated. The complete-gamma limit is below one ulp for
/// shape <= 1 and x >= 40, which are the upper-tail GEV uses below.
fn lower_gamma(shape: f64, x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    if x >= 40.0 {
        return crate::reference::curve_fitting::gamma(shape);
    }
    let mut term = 1.0 / shape;
    let mut sum = term;
    for index in 1..10_000 {
        term *= x / (shape + f64::from(index));
        sum += term;
        if term <= sum * f64::EPSILON {
            break;
        }
    }
    (shape * x.ln() - x).exp() * sum
}

/// Integral over the uppermost `mass` ranks of a finite-moment heavy GEV.
/// With z=-ln(q), the nonconstant term is an incomplete-gamma integral:
/// integral(q0..1,(-ln(q))^c dq)=gamma_lower(1+c,-ln(q0)).
fn gev_upper_integral(distribution: &FittedDistribution, mass: f64, moment: IntegralMoment) -> f64 {
    let c = distribution.shape.expect("GEV has a shape");
    let x = -(-mass).ln_1p();
    let scaled_shape = distribution.scale * moment.scale / c;
    let a = moment.value(distribution.location) + scaled_shape;
    let b = -scaled_shape;
    let first = lower_gamma(1.0 + c, x);
    if moment.square {
        a * a * mass + 2.0 * a * b * first + b * b * lower_gamma(1.0 + 2.0 * c, x)
    } else {
        a * mass + b * first
    }
}

pub(crate) type ProbabilitySegment = (f64, f64, f64);

/// Retain the part of each interval strictly above the hurdle. Endpoint
/// exclusion only affects atoms, since a uniform interval has no endpoint mass.
pub(crate) fn condition_segments(
    segments: Vec<ProbabilitySegment>,
    atom_probability: f64,
    atom_location: f64,
    surviving_mass: f64,
) -> Vec<ProbabilitySegment> {
    let mut result = Vec::with_capacity(segments.len() + 1);
    if atom_probability > 0.0 {
        result.push((atom_location, atom_location, atom_probability));
    }
    if atom_probability < 1.0 {
        result.extend(segments.into_iter().filter_map(|(lo, hi, mass)| {
            if hi <= atom_location {
                return None;
            }
            let retained = if lo >= atom_location {
                1.0
            } else {
                interval_fraction(lo, hi, atom_location, hi)
            };
            let mass = (mass * retained / surviving_mass) * (1.0 - atom_probability);
            (mass > 0.0).then_some((lo.max(atom_location), hi, mass))
        }));
    }
    result
}

pub(crate) fn segment_quantile(segments: &[ProbabilitySegment], q: f64, upper: bool) -> f64 {
    if q == 0.0 {
        return segments[0].0;
    }
    if q == 1.0 {
        return segments.last().unwrap().1;
    }
    let mut left = 0.0;
    for &(a, b, mass) in segments {
        let right = left + mass;
        if q < right || !upper && q == right {
            return linear_interpolate(a, b, ((q - left) / mass).clamp(0.0, 1.0));
        }
        left = right;
    }
    segments.last().unwrap().1
}

pub(crate) fn segment_inverse_survival(
    segments: &[ProbabilitySegment],
    survival: f64,
    upper: bool,
) -> f64 {
    if survival == 0.0 {
        return segments.last().unwrap().1;
    }
    if survival == 1.0 {
        return segments[0].0;
    }
    let mut right = 0.0;
    for &(a, b, mass) in segments.iter().rev() {
        let left = right + mass;
        if survival < left || upper && survival == left {
            return linear_interpolate(b, a, ((survival - right) / mass).clamp(0.0, 1.0));
        }
        right = left;
    }
    segments[0].0
}

pub(crate) fn segment_probability(
    segments: &[ProbabilitySegment],
    lower: f64,
    upper: f64,
    include_lower: bool,
    include_upper: bool,
) -> f64 {
    compensated(segments.iter().map(|&(lo, hi, mass)| {
        if lo == hi {
            let after = lo > lower || (include_lower && lo == lower);
            let before = hi < upper || (include_upper && hi == upper);
            if after && before { mass } else { 0.0 }
        } else {
            mass * interval_fraction(lo, hi, lower, upper)
        }
    }))
    .clamp(0.0, 1.0)
}

pub(crate) fn segment_integral(
    segments: &[ProbabilitySegment],
    lo: f64,
    hi: f64,
    moment: IntegralMoment,
    survival: bool,
) -> f64 {
    let masses: Vec<f64> = segments.iter().map(|&(_, _, p)| p).collect();
    compensated(segments.iter().zip(mass_intervals(&masses)).map(
        |(&(xa, xb, _), (left, right, mass))| {
            let (start, end, xa, xb) = if survival {
                (right, left, xb, xa)
            } else {
                (left, right, xa, xb)
            };
            let (width, a, b) = local_interval(lo, hi, start, end, mass);
            let x = linear_interpolate(xa, xb, a);
            let y = linear_interpolate(xa, xb, b);
            moment.segment(width, x, y)
        },
    ))
}

pub(crate) fn segment_cvar(
    segments: &[ProbabilitySegment],
    level: f64,
    upper: bool,
) -> Option<f64> {
    if !(0.0 < level && level < 1.0) || segments.windows(2).any(|w| w[0].1 > w[1].0) {
        return None;
    }
    // Keep the original masses in exact binary arithmetic: even a partial
    // atom with an almost unchanged width can alter the conditional mean.
    let exact = |value: f64| BigRational::from_float(value).expect("finite segment data");
    let total: BigRational = segments.iter().map(|&(_, _, mass)| exact(mass)).sum();
    let mut excluded = &total * exact(level);
    let included = total - &excluded;
    if included.is_zero() {
        return None;
    }
    let mut moment = BigRational::default();
    for i in 0..segments.len() {
        let (a, b, p) = if upper {
            segments[i]
        } else {
            let (a, b, p) = segments[segments.len() - 1 - i];
            (b, a, p)
        };
        let width = exact(p);
        if excluded >= width {
            excluded -= width;
            continue;
        }
        let a = exact(a);
        let b = exact(b);
        let offset = &excluded / &width;
        moment += (&width - &excluded) * (&a + &b + (&b - &a) * offset);
        excluded = BigRational::default();
    }
    (moment / included / BigRational::from_integer(2.into())).to_f64()
}

impl NumericalLaw {
    pub fn from_json(text: &str) -> Result<Self> {
        // Externally tagged decoding dispatches typed fields directly. Serde's
        // internally tagged buffer otherwise turns arbitrary-precision JSON
        // numbers into private maps before f64 fields can decode them.
        fn tagged(value: serde_json::Value) -> Result<serde_json::Value> {
            let mut data = value
                .as_object()
                .ok_or_else(|| invalid("expected a numerical object"))?
                .clone();
            let kind = data
                .remove("kind")
                .and_then(|v| v.as_str().map(str::to_string))
                .ok_or_else(|| invalid("missing numerical kind"))?;
            if kind == "hurdle" {
                let base = data
                    .remove("base")
                    .ok_or_else(|| invalid("missing hurdle base"))?;
                data.insert("base".into(), tagged(base)?);
            }
            let mut result = serde_json::Map::new();
            result.insert(kind, serde_json::Value::Object(data));
            Ok(serde_json::Value::Object(result))
        }
        let value = serde_json::from_str(text).map_err(|e| invalid(&e.to_string()))?;
        let specification: Specification =
            serde_json::from_value(tagged(value)?).map_err(|e| invalid(&e.to_string()))?;
        Self::from_specification(specification)
    }

    fn from_specification(specification: Specification) -> Result<Self> {
        Ok(match specification {
            Specification::Parametric {
                family,
                location,
                scale,
                shape,
            } => {
                let family = DistributionFamily::from_name(&family)
                    .ok_or_else(|| invalid("unknown distribution family"))?;
                if matches!(
                    family,
                    DistributionFamily::GumbelLeft | DistributionFamily::GumbelRight
                ) && shape.is_some()
                {
                    return Err(invalid("Gumbel distributions have no shape parameter"));
                }
                Self::Parametric(FittedDistribution::from_parameters(
                    family, shape, location, scale,
                )?)
            }
            Specification::PointMass { value } => {
                if !value.is_finite() {
                    return Err(invalid("point mass must be finite"));
                }
                Self::PointMass(value)
            }
            Specification::Discrete {
                values,
                probabilities,
            } => {
                if values.len() != probabilities.len()
                    || values.iter().any(|x| !x.is_finite())
                    || values.windows(2).any(|p| p[0] >= p[1])
                    || probabilities.iter().any(|p| *p <= 0.0)
                {
                    return Err(invalid(
                        "discrete values must increase and have positive probability masses",
                    ));
                }
                let cumulative = cumulative(&probabilities)?;
                Self::Discrete {
                    values,
                    probabilities,
                    cumulative,
                }
            }
            Specification::Histogram {
                edges,
                probabilities,
            } => {
                if edges.len() != probabilities.len() + 1
                    || edges.iter().any(|x| !x.is_finite())
                    || edges.windows(2).any(|p| p[0] >= p[1])
                {
                    return Err(invalid(
                        "histogram requires increasing edges, one more than its masses",
                    ));
                }
                let cumulative = cumulative(&probabilities)?;
                Self::Histogram {
                    edges,
                    probabilities,
                    cumulative,
                }
            }
            Specification::Hurdle {
                base,
                atom_probability,
                atom_location,
            } => Self::hurdle(
                Self::from_specification(*base)?,
                atom_probability,
                atom_location,
            )?,
            Specification::Quantile {
                probabilities,
                values,
            } => {
                if values.len() < 2
                    || values.len() != probabilities.len()
                    || probabilities[0] != 0.0
                    || *probabilities.last().unwrap() != 1.0
                    || probabilities
                        .iter()
                        .any(|q| !q.is_finite() || !(0.0..=1.0).contains(q))
                    || probabilities.windows(2).any(|p| p[0] >= p[1])
                    || values.iter().any(|x| !x.is_finite())
                    || values.windows(2).any(|p| p[0] > p[1])
                {
                    return Err(invalid(
                        "a complete quantile law requires ordered finite knots spanning [0, 1]",
                    ));
                }
                Self::Quantile {
                    probabilities,
                    values,
                }
            }
        })
    }

    pub(crate) fn hurdle(base: Self, atom_probability: f64, atom_location: f64) -> Result<Self> {
        if !atom_location.is_finite() || !(0.0..=1.0).contains(&atom_probability) {
            return Err(invalid(
                "hurdle atom must be finite and its probability in [0, 1]",
            ));
        }
        if atom_probability < 1.0 && !base.log_exceedance(atom_location, false).is_finite() {
            return Err(invalid(
                "hurdle base must have positive probability strictly above its atom",
            ));
        }
        Ok(Self::Hurdle {
            base: Box::new(base),
            atom_probability,
            atom_location,
        })
    }

    pub(crate) fn finite_segments(&self) -> Option<Vec<ProbabilitySegment>> {
        match self {
            Self::PointMass(x) => Some(vec![(*x, *x, 1.0)]),
            Self::Discrete {
                values,
                probabilities,
                ..
            } => Some(
                values
                    .iter()
                    .zip(probabilities)
                    .map(|(x, p)| (*x, *x, *p))
                    .collect(),
            ),
            Self::Histogram {
                edges,
                probabilities,
                ..
            } => Some(
                edges
                    .windows(2)
                    .zip(probabilities)
                    .filter(|(_, p)| **p > 0.0)
                    .map(|(x, p)| (x[0], x[1], *p))
                    .collect(),
            ),
            Self::Quantile {
                probabilities,
                values,
            } => Some(
                values
                    .windows(2)
                    .zip(probabilities.windows(2))
                    .map(|(x, q)| (x[0], x[1], q[1] - q[0]))
                    .collect(),
            ),
            Self::Hurdle {
                base,
                atom_probability: p,
                atom_location: x,
            } => {
                if *p == 1.0 {
                    return Some(vec![(*x, *x, 1.0)]);
                }
                Some(condition_segments(
                    base.finite_segments()?,
                    *p,
                    *x,
                    base.exceedance(*x, false),
                ))
            }
            Self::Parametric(_) => None,
        }
    }

    pub fn support(&self) -> (f64, f64) {
        match self {
            Self::Parametric(d) => {
                let (l, s, c) = (d.location, d.scale, d.shape.unwrap_or(0.0));
                match d.family {
                    DistributionFamily::GenExtreme if c > 0.0 => (f64::NEG_INFINITY, l + s / c),
                    DistributionFamily::GenExtreme if c < 0.0 => (l + s / c, f64::INFINITY),
                    DistributionFamily::GenPareto => {
                        (l, if c < 0.0 { l - s / c } else { f64::INFINITY })
                    }
                    DistributionFamily::WeibullMin => (l, f64::INFINITY),
                    DistributionFamily::WeibullMax => (f64::NEG_INFINITY, l),
                    _ => (f64::NEG_INFINITY, f64::INFINITY),
                }
            }
            Self::PointMass(x) => (*x, *x),
            Self::Discrete { values, .. } | Self::Quantile { values, .. } => {
                (values[0], *values.last().unwrap())
            }
            Self::Histogram {
                edges,
                probabilities,
                ..
            } => {
                let first = probabilities.iter().position(|p| *p > 0.0).unwrap();
                let last = probabilities.iter().rposition(|p| *p > 0.0).unwrap();
                (edges[first], edges[last + 1])
            }
            Self::Hurdle {
                base,
                atom_probability: p,
                atom_location: x,
            } => {
                if *p == 1.0 {
                    return (*x, *x);
                }
                if let Some(segments) = self.finite_segments() {
                    return (segments[0].0, segments.last().unwrap().1);
                }
                let (lo, hi) = base.support();
                (if *p > 0.0 { *x } else { lo.max(*x) }, hi)
            }
        }
    }

    pub fn quantile(&self, q: f64) -> Result<f64> {
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return Err(invalid("probability must be finite and in [0, 1]"));
        }
        if q == 0.0 {
            return Ok(self.support().0);
        }
        if q == 1.0 {
            return Ok(self.support().1);
        }
        Ok(match self {
            Self::Parametric(d) => return d.ppf(q),
            Self::PointMass(x) => *x,
            Self::Discrete {
                values, cumulative, ..
            } => {
                values[cumulative[1..]
                    .partition_point(|p| *p < q)
                    .min(values.len() - 1)]
            }
            Self::Histogram {
                edges,
                cumulative,
                probabilities,
            } => {
                let i = cumulative[1..]
                    .partition_point(|p| *p < q)
                    .min(probabilities.len() - 1);
                linear_interpolate(
                    edges[i],
                    edges[i + 1],
                    (q - cumulative[i]) / (cumulative[i + 1] - cumulative[i]),
                )
            }
            Self::Quantile {
                probabilities,
                values,
            } => {
                let i = probabilities.partition_point(|p| *p < q).max(1) - 1;
                linear_interpolate(
                    values[i],
                    values[i + 1],
                    (q - probabilities[i]) / (probabilities[i + 1] - probabilities[i]),
                )
            }
            Self::Hurdle {
                base,
                atom_probability: p,
                atom_location: x,
            } => {
                if q <= *p {
                    *x
                } else if let Some(segments) = self.finite_segments() {
                    segment_quantile(&segments, q, false)
                } else {
                    return base
                        .conditioned_quantile(*x, (1.0 - q) / (1.0 - p))
                        .map(|quantile| quantile.max(*x));
                }
            }
        })
    }

    pub fn atoms(&self) -> Vec<(f64, f64)> {
        if let Self::Hurdle {
            base,
            atom_probability: p,
            atom_location: x,
        } = self
        {
            let mut atoms = Vec::new();
            if *p > 0.0 {
                atoms.push((*x, *p));
            }
            if *p < 1.0 {
                let surviving = base.exceedance(*x, false);
                atoms.extend(
                    base.atoms()
                        .into_iter()
                        .filter(|&(value, _)| value > *x)
                        .map(|(value, mass)| (value, (mass / surviving) * (1.0 - p))),
                );
            }
            return atoms;
        }
        self.finite_segments()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(lo, hi, mass)| (lo == hi).then_some((lo, mass)))
            .collect()
    }

    /// Private engine primitive for conditioning deep upper tails without
    /// rounding `1-survival` to one.
    pub fn inverse_survival(&self, s: f64) -> Result<f64> {
        if !s.is_finite() || !(0.0..=1.0).contains(&s) {
            return Err(invalid("survival probability must be in [0, 1]"));
        }
        if s == 0.0 {
            return Ok(self.support().1);
        }
        if s == 1.0 {
            return Ok(self.support().0);
        }
        if let Self::Parametric(d) = self {
            return d.isf(s);
        }
        if let Some(segments) = self.finite_segments() {
            return Ok(segment_inverse_survival(&segments, s, false));
        }
        if let Self::Hurdle {
            base,
            atom_probability: p,
            atom_location: x,
        } = self
        {
            if s >= 1.0 - p {
                return Ok(*x);
            }
            return base
                .conditioned_quantile(*x, s / (1.0 - p))
                .map(|quantile| quantile.max(*x));
        }
        unreachable!("all numerical representations covered")
    }

    fn conditioned_quantile(&self, cutoff: f64, survival: f64) -> Result<f64> {
        let target = survival * self.exceedance(cutoff, false);
        if target >= f64::MIN_POSITIVE || survival == 0.0 {
            self.inverse_survival(target)
        } else {
            self.inverse_log_survival(survival.ln() + self.log_exceedance(cutoff, false))
        }
    }

    pub(crate) fn inverse_log_survival(&self, logarithm: f64) -> Result<f64> {
        if !logarithm.is_finite() || logarithm > f64::MIN_POSITIVE.ln() {
            return self.inverse_survival(logarithm.exp());
        }
        match self {
            Self::Parametric(law) => law.inverse_log_survival(logarithm),
            Self::Hurdle {
                base,
                atom_probability: p,
                atom_location: cutoff,
            } if self.finite_segments().is_none() => {
                let log_weight = (-p).ln_1p();
                if logarithm >= log_weight {
                    Ok(*cutoff)
                } else {
                    base.inverse_log_survival(
                        logarithm - log_weight + base.log_exceedance(*cutoff, false),
                    )
                    .map(|quantile| quantile.max(*cutoff))
                }
            }
            _ => self.inverse_survival(logarithm.exp()),
        }
    }

    pub(crate) fn log_exceedance(&self, x: f64, inclusive: bool) -> f64 {
        match self {
            Self::Parametric(law) => law.log_survival(x),
            Self::Hurdle {
                base,
                atom_probability: p,
                atom_location: cutoff,
            } if self.finite_segments().is_none() => {
                if x < *cutoff || (inclusive && x == *cutoff) {
                    0.0
                } else {
                    (-p).ln_1p()
                        + (base.log_exceedance(x, inclusive) - base.log_exceedance(*cutoff, false))
                            .min(0.0)
                }
            }
            _ => self.exceedance(x, inclusive).ln(),
        }
    }

    fn conditional_exceedance(&self, x: f64, inclusive: bool, cutoff: f64) -> f64 {
        let numerator = self.exceedance(x, inclusive);
        let denominator = self.exceedance(cutoff, false);
        if numerator >= f64::MIN_POSITIVE && denominator >= f64::MIN_POSITIVE {
            numerator / denominator
        } else {
            (self.log_exceedance(x, inclusive) - self.log_exceedance(cutoff, false))
                .min(0.0)
                .exp()
        }
    }

    /// Integral in survival coordinates, preserving conditional tail mass
    /// even when both corresponding CDF coordinates round to one.
    pub fn tail_integral(&self, lo: f64, hi: f64, center: f64, square: bool) -> Result<f64> {
        self.scaled_integral(lo, hi, center, square, true, 1.0)
    }

    /// Integrate after applying a scale to each centered quantile.
    pub fn scaled_integral(
        &self,
        lo: f64,
        hi: f64,
        center: f64,
        square: bool,
        survival: bool,
        scale: f64,
    ) -> Result<f64> {
        if let Some(base) = self.unconditioned_hurdle_base() {
            return base.scaled_integral(lo, hi, center, square, survival, scale);
        }
        // Center and scale fitted laws before querying their ranks, which can
        // be outside binary64 even when the scaled integral is representable.
        let scaled = match self {
            Self::Parametric(d) if scale > 0.0 && scale != 1.0 => {
                FittedDistribution::from_parameters(
                    d.family,
                    d.shape,
                    scaled_difference(d.location, center, scale),
                    d.scale * scale,
                )
                .ok()
                .map(Self::Parametric)
            }
            _ => None,
        };
        let (law, moment) = if let Some(scaled) = &scaled {
            (scaled, IntegralMoment::new(0.0, square, 1.0))
        } else {
            (self, IntegralMoment::new(center, square, scale))
        };
        if survival {
            law.tail_integral_moment(lo, hi, moment)
        } else {
            law.integral_moment(lo, hi, moment)
        }
    }

    fn tail_integral_moment(&self, lo: f64, hi: f64, moment: IntegralMoment) -> Result<f64> {
        if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi > 1.0 || lo > hi {
            return Err(invalid("invalid survival integration interval"));
        }
        if lo == hi {
            return Ok(0.0);
        }
        if let Some(segments) = self.finite_segments() {
            return Ok(segment_integral(&segments, lo, hi, moment, true));
        }
        if let Self::Hurdle {
            base,
            atom_probability: p,
            atom_location: x,
        } = self
        {
            let w = 1.0 - p;
            let atom_width = (hi - lo.max(w)).max(0.0);
            let atom = moment.segment(atom_width, *x, *x);
            if lo >= w {
                return Ok(atom);
            }
            let upper = hi.min(w);
            let surviving = base.exceedance(*x, false);
            let lower_base = lo / w * surviving;
            let upper_base = upper / w * surviving;
            let tail =
                if upper_base < f64::MIN_POSITIVE || (lo > 0.0 && lower_base < f64::MIN_POSITIVE) {
                    self.integrate_deep_tail(lo, upper, moment)?
                } else {
                    w / surviving * base.tail_integral_moment(lower_base, upper_base, moment)?
                };
            return Ok(atom + tail);
        }
        if let Self::Parametric(d) = self {
            let c = d.shape.unwrap_or(0.0);
            let order = if moment.square { 2.0 } else { 1.0 };
            if lo == 0.0
                && (matches!(d.family, DistributionFamily::GenPareto) && c * order >= 1.0
                    || matches!(d.family, DistributionFamily::GenExtreme) && c * order <= -1.0)
            {
                return Ok(f64::INFINITY);
            }
            if lo == 0.0 && matches!(d.family, DistributionFamily::GenExtreme) && c < -0.01 {
                return Ok(gev_upper_integral(d, hi, moment));
            }
            if matches!(d.family, DistributionFamily::GenPareto) && c.abs() >= 0.01 {
                let power_integral = |power: f64| {
                    let exponent = 1.0 - power;
                    if exponent == 0.0 {
                        (hi / lo).ln()
                    } else {
                        (hi.powf(exponent) - lo.powf(exponent)) / exponent
                    }
                };
                let b = d.scale * moment.scale / c;
                let a = moment.value(d.location) - b;
                return Ok(if moment.square {
                    a * a * (hi - lo)
                        + 2.0 * a * b * power_integral(c)
                        + b * b * power_integral(2.0 * c)
                } else {
                    a * (hi - lo) + b * power_integral(c)
                });
            }
            return self.integrate_deep_tail(lo, hi, moment);
        }
        unreachable!("all numerical representations covered")
    }

    fn integrate_deep_tail(&self, lo: f64, hi: f64, moment: IntegralMoment) -> Result<f64> {
        if lo == 0.0 && self.upper_tail_moment_diverges(moment.square) {
            return Ok(f64::INFINITY);
        }
        let width = hi - lo;
        let mut options = GaussKronrod::defaults();
        // Apply error control to the conditional expectation, even when the
        // interval itself holds a very small part of the whole law.
        options.absolute_tolerance = (options.absolute_tolerance * width).max(f64::MIN_POSITIVE);
        let result = integrate(&[(0.0, 1.0)], &options, |t| {
            // Integrate in the conditional law's probability coordinates. The
            // original survival can be too small to multiply or to divide by.
            let (log_survival, jacobian) = if lo == 0.0 {
                (hi.ln() + 4.0 * t.ln(), 4.0 * t.powi(3))
            } else {
                ((lo + width * t).ln(), 1.0)
            };
            let value = self.inverse_log_survival(log_survival).map_err(|e| {
                crate::impact::ImpactContractError::new("numerical_failure", "", e.to_string())
            })?;
            Ok(jacobian * moment.segment(width, value, value))
        })
        .map_err(|e| invalid(&e.to_string()))?;
        Ok(result.value)
    }

    fn upper_tail_moment_diverges(&self, square: bool) -> bool {
        match self {
            Self::Hurdle {
                base,
                atom_probability,
                ..
            } => *atom_probability < 1.0 && base.upper_tail_moment_diverges(square),
            Self::Parametric(law) => {
                let shape = law.shape.unwrap_or(0.0);
                let order = if square { 2.0 } else { 1.0 };
                (law.family == DistributionFamily::GenPareto && shape * order >= 1.0)
                    || (law.family == DistributionFamily::GenExtreme && shape * order <= -1.0)
            }
            _ => false,
        }
    }

    pub fn cdf(&self, x: f64, inclusive: bool) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        match self {
            Self::Parametric(d) => {
                if x == f64::NEG_INFINITY {
                    0.0
                } else if x == f64::INFINITY {
                    1.0
                } else {
                    d.cdf(x)
                }
            }
            Self::PointMass(v) => f64::from(if inclusive { *v <= x } else { *v < x }),
            Self::Discrete {
                values,
                probabilities,
                ..
            } => compensated(
                values
                    .iter()
                    .zip(probabilities)
                    .filter_map(|(v, p)| (if inclusive { *v <= x } else { *v < x }).then_some(*p)),
            )
            .clamp(0.0, 1.0),
            Self::Histogram {
                edges,
                probabilities,
                ..
            } => compensated(
                probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, p)| p * interval_fraction(edges[i], edges[i + 1], edges[i], x)),
            )
            .clamp(0.0, 1.0),
            Self::Quantile {
                probabilities,
                values,
            } => {
                let n = values.len();
                if x < values[0] || (!inclusive && x == values[0]) {
                    return 0.0;
                }
                if x > values[n - 1] || (inclusive && x == values[n - 1]) {
                    return 1.0;
                }
                let upper = values.partition_point(|v| if inclusive { *v <= x } else { *v < x });
                if upper == 0 {
                    return 0.0;
                }
                if upper == n {
                    return 1.0;
                }
                let i = upper - 1;
                if values[i] == x {
                    return probabilities[i];
                }
                linear_interpolate(
                    probabilities[i],
                    probabilities[i + 1],
                    interval_fraction(values[i], values[i + 1], values[i], x),
                )
            }
            Self::Hurdle {
                base,
                atom_probability: p,
                atom_location: v,
            } => {
                if let Some(segments) = self.finite_segments() {
                    return segment_probability(&segments, f64::NEG_INFINITY, x, false, inclusive);
                }
                if x < *v || (!inclusive && x == *v) {
                    0.0
                } else if *p == 1.0 {
                    1.0
                } else {
                    p + (1.0 - p) * (1.0 - base.conditional_exceedance(x, !inclusive, *v))
                }
            }
        }
    }

    pub fn mass(&self, x: f64) -> f64 {
        match self {
            Self::Discrete {
                values,
                probabilities,
                ..
            } => {
                let i = values.partition_point(|v| *v < x);
                values
                    .get(i)
                    .filter(|&&v| v == x)
                    .map_or(0.0, |_| probabilities[i])
            }
            Self::PointMass(v) => {
                if *v == x {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Hurdle {
                base,
                atom_probability: p,
                atom_location: v,
            } => {
                if x == *v {
                    *p
                } else if x > *v && *p < 1.0 {
                    let mass = base.mass(x);
                    if mass == 0.0 {
                        0.0
                    } else {
                        (mass.ln() - base.log_exceedance(*v, false)).exp() * (1.0 - p)
                    }
                } else {
                    0.0
                }
            }
            Self::Quantile {
                probabilities,
                values,
            } => compensated(
                values
                    .windows(2)
                    .zip(probabilities.windows(2))
                    .filter_map(|(v, q)| (v[0] == x && v[1] == x).then_some(q[1] - q[0])),
            ),
            _ => 0.0,
        }
    }

    pub fn exceedance(&self, x: f64, inclusive: bool) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        match self {
            Self::Parametric(d) => {
                if x == f64::NEG_INFINITY {
                    1.0
                } else if x == f64::INFINITY {
                    0.0
                } else {
                    d.sf(x)
                }
            }
            Self::Discrete {
                values,
                probabilities,
                ..
            } => compensated(
                values
                    .iter()
                    .zip(probabilities)
                    .filter_map(|(v, p)| (if inclusive { *v >= x } else { *v > x }).then_some(*p)),
            )
            .clamp(0.0, 1.0),
            Self::Histogram {
                edges,
                probabilities,
                ..
            } => compensated(
                probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, p)| p * interval_fraction(edges[i], edges[i + 1], x, edges[i + 1])),
            )
            .clamp(0.0, 1.0),
            Self::Hurdle {
                base,
                atom_probability: p,
                atom_location: v,
            } => {
                if let Some(segments) = self.finite_segments() {
                    return segment_probability(&segments, x, f64::INFINITY, inclusive, false);
                }
                if x < *v || (inclusive && x == *v) {
                    1.0
                } else if *p == 1.0 {
                    0.0
                } else {
                    (1.0 - p) * base.conditional_exceedance(x, inclusive, *v)
                }
            }
            _ => 1.0 - self.cdf(x, !inclusive),
        }
    }

    /// Integral of the quantile over a probability interval. Linear and atomic
    /// representations are analytic; parametric tails use adaptive quadrature.
    pub fn integral(&self, lo: f64, hi: f64, center: f64, square: bool) -> Result<f64> {
        self.scaled_integral(lo, hi, center, square, false, 1.0)
    }

    /// Compute a trimmed mean for a law with finite probability-linear segments.
    pub fn finite_cvar(&self, level: f64, upper: bool) -> Option<f64> {
        segment_cvar(&self.finite_segments()?, level, upper)
    }

    fn integral_moment(&self, lo: f64, hi: f64, moment: IntegralMoment) -> Result<f64> {
        if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi > 1.0 || lo > hi {
            return Err(invalid("invalid integration probability interval"));
        }
        if lo == hi {
            return Ok(0.0);
        }
        if let Self::Hurdle { .. } = self {
            if let Some(segments) = self.finite_segments() {
                return Ok(segment_integral(&segments, lo, hi, moment, false));
            }
        }
        let segment = |qa: f64, qb: f64, xa: f64, xb: f64| {
            let a = lo.max(qa);
            let b = hi.min(qb);
            if a >= b {
                return 0.0;
            }
            let x = linear_interpolate(xa, xb, (a - qa) / (qb - qa));
            let y = linear_interpolate(xa, xb, (b - qa) / (qb - qa));
            moment.segment(b - a, x, y)
        };
        match self {
            Self::PointMass(x) => Ok(moment.segment(hi - lo, *x, *x)),
            Self::Discrete {
                values,
                probabilities,
                ..
            } => Ok(compensated(
                values
                    .iter()
                    .zip(mass_intervals(probabilities))
                    .map(|(x, (left, right, mass))| {
                        let (width, _, _) = local_interval(lo, hi, left, right, mass);
                        moment.segment(width, *x, *x)
                    }),
            )),
            Self::Histogram {
                edges,
                probabilities,
                ..
            } => Ok(compensated(
                edges
                    .windows(2)
                    .zip(mass_intervals(probabilities))
                    .filter(|(_, (_, _, mass))| *mass > 0.0)
                    .map(|(x, (left, right, mass))| {
                        let (width, a, b) = local_interval(lo, hi, left, right, mass);
                        let xa = linear_interpolate(x[0], x[1], a);
                        let xb = linear_interpolate(x[0], x[1], b);
                        moment.segment(width, xa, xb)
                    }),
            )),
            Self::Quantile {
                probabilities,
                values,
            } => Ok(compensated(values.windows(2).enumerate().map(|(i, x)| {
                segment(probabilities[i], probabilities[i + 1], x[0], x[1])
            }))),
            Self::Hurdle {
                atom_probability: p,
                atom_location: x,
                ..
            } => {
                if *p == 1.0 {
                    return Ok(moment.segment(hi - lo, *x, *x));
                }
                let mut terms = vec![segment(0.0, *p, *x, *x)];
                if hi > *p {
                    terms.push(self.tail_integral_moment(1.0 - hi, 1.0 - lo.max(*p), moment)?);
                }
                Ok(compensated(terms.into_iter()))
            }
            Self::Parametric(d) => {
                let c = d.shape.unwrap_or(0.0);
                let order = if moment.square { 2.0 } else { 1.0 };
                let divergent_upper = matches!(d.family, DistributionFamily::GenPareto)
                    && c * order >= 1.0
                    || matches!(d.family, DistributionFamily::GenExtreme) && c * order <= -1.0;
                if hi == 1.0 && divergent_upper {
                    return Ok(f64::INFINITY);
                }
                if hi == 1.0 && matches!(d.family, DistributionFamily::GenExtreme) && c < -0.01 {
                    return Ok(gev_upper_integral(d, 1.0 - lo, moment));
                }
                if matches!(d.family, DistributionFamily::GenPareto) && c.abs() >= 0.01 {
                    // Integrate the actual power tail analytically. An
                    // integrable shape close to one must not depend on a
                    // quadrature budget or a cutoff near probability one.
                    let power_integral = |power: f64| {
                        let exponent = 1.0 - power;
                        if exponent == 0.0 {
                            ((1.0 - lo) / (1.0 - hi)).ln()
                        } else {
                            ((1.0 - lo).powf(exponent) - (1.0 - hi).powf(exponent)) / exponent
                        }
                    };
                    let b = d.scale * moment.scale / c;
                    let a = moment.value(d.location) - b;
                    return Ok(if moment.square {
                        a * a * (hi - lo)
                            + 2.0 * a * b * power_integral(c)
                            + b * b * power_integral(2.0 * c)
                    } else {
                        a * (hi - lo) + b * power_integral(c)
                    });
                }
                // A fourth-power endpoint map improves convergence of integrable
                // singularities without substituting or clipping an endpoint.
                let options = GaussKronrod::defaults();
                let result = integrate(&[(0.0, 1.0)], &options, |t| {
                    let (value, jac) = if hi == 1.0 {
                        let u = 1.0 - t;
                        (d.isf((hi - lo) * u.powi(4)), 4.0 * (hi - lo) * u.powi(3))
                    } else if lo == 0.0 {
                        (
                            self.quantile(lo + (hi - lo) * t.powi(4)),
                            4.0 * (hi - lo) * t.powi(3),
                        )
                    } else {
                        (self.quantile(lo + (hi - lo) * t), hi - lo)
                    };
                    let value = value.map_err(|e| crate::impact::ImpactContractError {
                        code: "numerical_failure",
                        path: "".into(),
                        message: e.to_string(),
                    })?;
                    Ok(moment.segment(jac, value, value))
                })
                .map_err(|e| invalid(&e.to_string()))?;
                Ok(result.value)
            }
        }
    }

    pub fn moments(&self) -> Result<(f64, f64)> {
        let (mean, deviation) = self.mean_deviation()?;
        Ok((mean, deviation.powi(2)))
    }

    /// Compute the standard deviation before squaring so that it can remain
    /// representable even if the variance exceeds the floating-point range.
    pub fn mean_deviation(&self) -> Result<(f64, f64)> {
        if let Some(base) = self.unconditioned_hurdle_base() {
            return base.mean_deviation();
        }
        if let Self::Parametric(d) = self {
            return Ok(affine_fitted_mean_deviation(d, 1.0, 0.0));
        }
        if let Some(segments) = self.finite_segments() {
            return Ok(affine_segment_mean_deviation(&segments, 1.0, 0.0));
        }
        let mean = self.integral(0.0, 1.0, 0.0, false)?;
        if !mean.is_finite() {
            return Ok((mean, f64::NAN));
        }
        // A hurdle can hide most of a large parametric tail in a small
        // probability interval. Use the base scale even when its quartiles
        // coincide with the hurdle.
        let scale = deviation_scale(self.moment_magnitude(mean));
        let variance = self.scaled_integral(0.0, 1.0, mean, true, false, scale)?;
        Ok((mean, variance.sqrt() / scale))
    }

    fn unconditioned_hurdle_base(&self) -> Option<&Self> {
        let Self::Hurdle {
            base,
            atom_probability: 0.0,
            atom_location,
        } = self
        else {
            return None;
        };
        let lower = base.support().0;
        (*atom_location < lower || *atom_location == lower && base.cdf(*atom_location, true) == 0.0)
            .then_some(base)
    }

    fn moment_magnitude(&self, center: f64) -> f64 {
        match self {
            Self::Parametric(d) => d.scale,
            Self::Hurdle {
                base,
                atom_location,
                ..
            } => base
                .moment_magnitude(center)
                .max((center - atom_location).abs()),
            _ => 0.0,
        }
    }
}

pub(crate) fn deviation_scale(magnitude: f64) -> f64 {
    if magnitude > 1e150 {
        2.0_f64.powi(-600)
    } else if magnitude > 0.0 && magnitude < 1e-150 {
        2.0_f64.powi(600)
    } else {
        1.0
    }
}

pub(crate) fn affine_fitted_mean_deviation(
    d: &FittedDistribution,
    slope: f64,
    intercept: f64,
) -> (f64, f64) {
    let l = slope.mul_add(d.location, intercept);
    let s = d.scale * slope;
    let c = d.shape.unwrap_or(0.0);
    let euler = 0.577_215_664_901_532_9;
    let pi = std::f64::consts::PI;
    let gumbel_deviation = pi / 6.0_f64.sqrt();
    let divergence = f64::INFINITY.copysign(slope);
    let scaled = |factor: f64| {
        let value = s * factor;
        if s.is_normal() && value.is_finite() {
            value
        } else {
            let regrouped = d.scale * factor * slope;
            if regrouped.is_normal() {
                regrouped
            } else if factor == 0.0 {
                factor
            } else {
                (d.scale.ln() + slope.abs().ln() + factor.abs().ln())
                    .exp()
                    .copysign(factor * slope.signum())
            }
        }
    };
    let mean = |factor: f64| {
        let value = l + scaled(factor);
        if value.is_finite() {
            value
        } else if s.is_finite() && l.is_finite() {
            s.mul_add(factor, l)
        } else {
            slope.mul_add(d.scale.mul_add(factor, d.location), intercept)
        }
    };
    match d.family {
        DistributionFamily::GumbelRight => (mean(euler), scaled(gumbel_deviation).abs()),
        DistributionFamily::GumbelLeft => (mean(-euler), scaled(gumbel_deviation).abs()),
        DistributionFamily::GenExtreme => {
            if c >= 32.0 {
                // The Gamma terms can exceed binary64 before dimensional scaling.
                let log_scale_over_shape = d.scale.ln() + slope.abs().ln() - c.ln();
                let first = log_gamma_one_plus(c);
                let log_mean = first + (-(-first).exp()).ln_1p();
                return (
                    l + (log_scale_over_shape + log_mean).exp().copysign(-slope),
                    (log_scale_over_shape + 0.5 * log_gamma_variance(c, first)).exp(),
                );
            }
            let mean = if c > -1.0 {
                mean(-gamma_moments(c).1)
            } else {
                divergence
            };
            let deviation = if c > -0.5 {
                scaled(gamma_moments(c).2.sqrt()).abs()
            } else {
                f64::INFINITY
            };
            (mean, deviation)
        }
        DistributionFamily::GenPareto => (
            if c < 1.0 {
                mean(1.0 / (1.0 - c))
            } else {
                divergence
            },
            if c < 0.5 {
                let root = if c >= -f64::MAX / 2.0 {
                    (1.0 - 2.0 * c).sqrt()
                } else {
                    (0.5 - c).sqrt() * 2.0_f64.sqrt()
                };
                let factor = 1.0 / (1.0 - c) / root;
                if factor.is_normal() {
                    scaled(factor).abs()
                } else {
                    (d.scale.ln() + slope.abs().ln() - (1.0 - c).ln() - root.ln()).exp()
                }
            } else {
                f64::INFINITY
            },
        ),
        DistributionFamily::WeibullMin | DistributionFamily::WeibullMax => {
            let reciprocal = 1.0 / c;
            let (log_mean, log_deviation) = if reciprocal < 0.01 {
                // The series retains the variance of a law whose raw
                // moments are almost equal even in log space.
                let (g, _, scaled_variance) = gamma_moments(reciprocal);
                (g.ln(), reciprocal.ln() + 0.5 * scaled_variance.ln())
            } else {
                let first = log_gamma_one_plus(reciprocal);
                (first, 0.5 * log_gamma_variance(reciprocal, first))
            };
            // Scale in log space before exponentiating. The raw moments
            // can overflow even when the dimensional moments are finite.
            let log_scale = d.scale.ln() + slope.abs().ln();
            let sign = if d.family == DistributionFamily::WeibullMin {
                slope
            } else {
                -slope
            };
            (
                l + (log_scale + log_mean).exp().copysign(sign),
                (log_scale + log_deviation).exp(),
            )
        }
        DistributionFamily::SkewNormal => {
            let delta = c / c.hypot(1.0);
            (
                mean(delta * (2.0 / pi).sqrt()),
                scaled((1.0 - 2.0 * delta * delta / pi).sqrt()).abs(),
            )
        }
    }
}

fn log_gamma_variance(shape: f64, first: f64) -> f64 {
    let second = log_gamma_one_plus(2.0 * shape);
    if second.is_infinite() {
        second
    } else {
        second + (-(2.0 * first - second).exp_m1()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rare_mass_survives_moments_and_cvar() {
        let law = NumericalLaw::from_json(
            r#"{"kind":"discrete","values":[0,1000000],"probabilities":[0.999999,0.000001]}"#,
        )
        .unwrap();
        assert!((law.moments().unwrap().0 - 1.0).abs() < 1e-9);
        assert!((law.integral(0.95, 1.0, 0.0, false).unwrap() / 0.05 - 20.0).abs() < 1e-6);
        assert_eq!(law.mass(1e6), 1e-6);
        assert_eq!(law.quantile(1.0).unwrap(), 1e6);
    }
    #[test]
    fn empirical_atoms_and_quantile_plateaus_are_exact() {
        let law = NumericalLaw::from_json(
            r#"{"kind":"discrete","values":[1,2],"probabilities":[0.75,0.25]}"#,
        )
        .unwrap();
        assert_eq!(law.cdf(1.0, true), 0.75);
        assert_eq!(law.cdf(1.0, false), 0.0);
        let curve = NumericalLaw::from_json(
            r#"{"kind":"quantile","probabilities":[0,0.2,0.8,1],"values":[1,2,2,3]}"#,
        )
        .unwrap();
        assert!((curve.mass(2.0) - 0.6).abs() < 1e-15);
        assert_eq!(curve.cdf(2.0, false), 0.2);
        assert_eq!(curve.cdf(2.0, true), 0.8);
    }
    #[test]
    fn sub_epsilon_atoms_and_histogram_bins_survive_integration() {
        let law=NumericalLaw::from_json(r#"{"kind":"discrete","values":[0,100000000000000000000],"probabilities":[1,0.00000000000000000001]}"#).unwrap();
        assert!((law.moments().unwrap().0 - 1.0).abs() < 1e-12);
        assert!((law.integral(0.95, 1.0, 0.0, false).unwrap() / 0.05 - 20.0).abs() < 1e-10);
        let histogram=NumericalLaw::from_json(r#"{"kind":"histogram","edges":[0,1,200000000000000000000],"probabilities":[1,0.00000000000000000001]}"#).unwrap();
        assert!((histogram.moments().unwrap().0 - 1.5).abs() < 1e-12);
    }
    #[test]
    fn conditional_hurdle_moments_preserve_deep_survival_tails() {
        let law=NumericalLaw::from_json(r#"{"kind":"hurdle","base":{"kind":"parametric","family":"gumbel_r","location":-37.5,"scale":1},"atom_probability":0.25,"atom_location":0}"#).unwrap();
        let (mean, variance) = law.moments().unwrap();
        assert!((mean - 0.75).abs() < 1e-8);
        assert!((variance - 0.9375).abs() < 1e-8);
    }

    #[test]
    fn heavy_gev_upper_tail_integrals_are_finite_without_endpoint_sampling() {
        // Independently calculated with 70-digit decimal arithmetic using
        // the alternating expansion of exp(-t), rather than this engine's
        // positive incomplete-gamma series or its quadrature.
        for (shape, expected) in [
            (-0.9, 163.246_833_284_355_37),
            (-0.99, 1_959.087_080_778_165_8),
        ] {
            let law=NumericalLaw::from_json(&format!(r#"{{"kind":"parametric","family":"genextreme","location":0,"scale":1,"shape":{shape}}}"#)).unwrap();
            let upper = law.integral(0.95, 1.0, 0.0, false).unwrap() / 0.05;
            assert!(
                (upper / expected - 1.0).abs() < 1e-12,
                "{shape}: {upper} vs {expected}"
            );
            let survival = law.tail_integral(0.0, 0.05, 0.0, false).unwrap() / 0.05;
            assert!((survival / expected - 1.0).abs() < 1e-12);
        }
        let divergent = NumericalLaw::from_json(
            r#"{"kind":"parametric","family":"genextreme","location":0,"scale":1,"shape":-1}"#,
        )
        .unwrap();
        assert_eq!(
            divergent.integral(0.95, 1.0, 0.0, false).unwrap(),
            f64::INFINITY
        );
    }
}

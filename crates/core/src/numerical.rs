//! Complete probability laws and probability-linear quantile evidence.
//!
//! Finite laws integrate their actual probability intervals, including tiny
//! atoms. No quantile grid is substituted for the caller's law.

use serde::Deserialize;

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

fn compensated(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum: f64 = 0.0;
    let mut correction: f64 = 0.0;
    for value in values {
        let next = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + correction
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
fn gev_upper_integral(
    distribution: &FittedDistribution,
    mass: f64,
    center: f64,
    square: bool,
) -> f64 {
    let c = distribution.shape.expect("GEV has a shape");
    let x = -(-mass).ln_1p();
    let a = distribution.location + distribution.scale / c - center;
    let b = -distribution.scale / c;
    let first = lower_gamma(1.0 + c, x);
    if square {
        a * a * mass + 2.0 * a * b * first + b * b * lower_gamma(1.0 + 2.0 * c, x)
    } else {
        a * mass + b * first
    }
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
            } => {
                if !atom_location.is_finite()
                    || !atom_probability.is_finite()
                    || !(0.0..=1.0).contains(&atom_probability)
                {
                    return Err(invalid(
                        "hurdle atom must be finite and its probability in [0, 1]",
                    ));
                }
                let base = Self::from_specification(*base)?;
                if atom_probability < 1.0 && base.exceedance(atom_location, false) <= 0.0 {
                    return Err(invalid(
                        "hurdle base must have positive probability strictly above its atom",
                    ));
                }
                Self::Hurdle {
                    base: Box::new(base),
                    atom_probability,
                    atom_location,
                }
            }
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

    fn finite_segments(&self) -> Option<Vec<(f64, f64, f64)>> {
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
                let original = base.finite_segments()?;
                let surviving = base.exceedance(*x, false);
                let mut result = if *p > 0.0 { vec![(*x, *x, *p)] } else { vec![] };
                result.extend(original.into_iter().filter(|(_, b, _)| *b > *x).map(
                    |(a, b, mass)| {
                        let fraction = if a >= *x { 1.0 } else { (b - x) / (b - a) };
                        (a.max(*x), b, mass * fraction * (1.0 - p) / surviving)
                    },
                ));
                Some(result)
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
                edges[i]
                    + (edges[i + 1] - edges[i])
                        * ((q - cumulative[i]) / (cumulative[i + 1] - cumulative[i]))
            }
            Self::Quantile {
                probabilities,
                values,
            } => {
                let i = probabilities.partition_point(|p| *p < q).max(1) - 1;
                values[i]
                    + (values[i + 1] - values[i])
                        * ((q - probabilities[i]) / (probabilities[i + 1] - probabilities[i]))
            }
            Self::Hurdle {
                base,
                atom_probability: p,
                atom_location: x,
            } => {
                if *p == 1.0 {
                    return Ok(*x);
                }
                if q <= *p {
                    *x
                } else {
                    let surviving = base.exceedance(*x, false);
                    let survival = (1.0 - q) / (1.0 - p) * surviving;
                    return if let Self::Parametric(d) = base.as_ref() {
                        d.isf(survival)
                    } else {
                        base.quantile(1.0 - survival)
                    };
                }
            }
        })
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
            let total = compensated(segments.iter().map(|(_, _, p)| *p));
            let mut right = 0.0;
            for (a, b, mass) in segments.into_iter().rev() {
                let mass = mass / total;
                if s < right + mass {
                    return Ok(b + (a - b) * ((s - right) / mass));
                }
                right += mass;
            }
            return Ok(self.support().0);
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
            return base.inverse_survival(s / (1.0 - p) * base.exceedance(*x, false));
        }
        unreachable!("all numerical representations covered")
    }

    /// Integral in survival coordinates, preserving conditional tail mass
    /// even when both corresponding CDF coordinates round to one.
    pub fn tail_integral(&self, lo: f64, hi: f64, center: f64, square: bool) -> Result<f64> {
        if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi > 1.0 || lo > hi {
            return Err(invalid("invalid survival integration interval"));
        }
        if lo == hi {
            return Ok(0.0);
        }
        if let Some(segments) = self.finite_segments() {
            let masses: Vec<f64> = segments.iter().map(|(_, _, p)| *p).collect();
            return Ok(compensated(
                segments.iter().zip(mass_intervals(&masses)).map(
                    |((xa, xb, _), (left, right, mass))| {
                        let (width, a, b) = local_interval(lo, hi, right, left, mass);
                        let x = xb + (xa - xb) * a - center;
                        let y = xb + (xa - xb) * b - center;
                        width
                            * if square {
                                (x * x + x * y + y * y) / 3.0
                            } else {
                                (x + y) / 2.0
                            }
                    },
                ),
            ));
        }
        if let Self::Hurdle {
            base,
            atom_probability: p,
            atom_location: x,
        } = self
        {
            let w = 1.0 - p;
            let atom_width = (hi - lo.max(w)).max(0.0);
            let atom = atom_width
                * if square {
                    (x - center).powi(2)
                } else {
                    x - center
                };
            if lo >= w {
                return Ok(atom);
            }
            let surviving = base.exceedance(*x, false);
            return Ok(atom
                + w / surviving
                    * base.tail_integral(
                        lo / w * surviving,
                        hi.min(w) / w * surviving,
                        center,
                        square,
                    )?);
        }
        if let Self::Parametric(d) = self {
            let c = d.shape.unwrap_or(0.0);
            let order = if square { 2.0 } else { 1.0 };
            if lo == 0.0
                && (matches!(d.family, DistributionFamily::GenPareto) && c * order >= 1.0
                    || matches!(d.family, DistributionFamily::GenExtreme) && c * order <= -1.0)
            {
                return Ok(f64::INFINITY);
            }
            if lo == 0.0 && matches!(d.family, DistributionFamily::GenExtreme) && c < -0.01 {
                return Ok(gev_upper_integral(d, hi, center, square));
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
                let a = d.location - d.scale / c - center;
                let b = d.scale / c;
                return Ok(if square {
                    a * a * (hi - lo)
                        + 2.0 * a * b * power_integral(c)
                        + b * b * power_integral(2.0 * c)
                } else {
                    a * (hi - lo) + b * power_integral(c)
                });
            }
            let result = integrate(&[(0.0, 1.0)], &GaussKronrod::defaults(), |t| {
                let (s, jac) = if lo == 0.0 {
                    (hi * t.powi(4), 4.0 * t.powi(3))
                } else {
                    (lo + (hi - lo) * t, 1.0)
                };
                let value = d.isf(s).map_err(|e| crate::impact::ImpactContractError {
                    code: "numerical_failure",
                    path: "".into(),
                    message: e.to_string(),
                })? - center;
                Ok(if square {
                    value * value * jac
                } else {
                    value * jac
                })
            })
            .map_err(|e| invalid(&e.to_string()))?;
            // Error control applies to the conditional expectation. Factoring
            // out the rank width prevents rare tails from satisfying absolute
            // tolerance before their conditional moments are accurate.
            return Ok((hi - lo) * result.value);
        }
        unreachable!("all numerical representations covered")
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
                    .map(|(i, p)| p * ((x - edges[i]) / (edges[i + 1] - edges[i])).clamp(0.0, 1.0)),
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
                probabilities[i]
                    + (probabilities[i + 1] - probabilities[i]) * (x - values[i])
                        / (values[i + 1] - values[i])
            }
            Self::Hurdle {
                base,
                atom_probability: p,
                atom_location: v,
            } => {
                if x < *v || (!inclusive && x == *v) {
                    0.0
                } else if *p == 1.0 {
                    1.0
                } else {
                    p + (1.0 - p)
                        * (1.0 - base.exceedance(x, !inclusive) / base.exceedance(*v, false))
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
            } => values
                .binary_search_by(|v| v.total_cmp(&x))
                .map_or(0.0, |i| probabilities[i]),
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
                    (1.0 - p) * base.mass(x) / base.exceedance(*v, false)
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
            } => compensated(probabilities.iter().enumerate().map(|(i, p)| {
                p * ((edges[i + 1] - x) / (edges[i + 1] - edges[i])).clamp(0.0, 1.0)
            }))
            .clamp(0.0, 1.0),
            Self::Hurdle {
                base,
                atom_probability: p,
                atom_location: v,
            } => {
                if x < *v || (inclusive && x == *v) {
                    1.0
                } else if *p == 1.0 {
                    0.0
                } else {
                    (1.0 - p) * base.exceedance(x, inclusive) / base.exceedance(*v, false)
                }
            }
            _ => 1.0 - self.cdf(x, !inclusive),
        }
    }

    /// Integral of the quantile over a probability interval. Linear and atomic
    /// representations are analytic; parametric tails use adaptive quadrature.
    pub fn integral(&self, lo: f64, hi: f64, center: f64, square: bool) -> Result<f64> {
        if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi > 1.0 || lo > hi {
            return Err(invalid("invalid integration probability interval"));
        }
        if lo == hi {
            return Ok(0.0);
        }
        if let Self::Hurdle { .. } = self {
            if let Some(segments) = self.finite_segments() {
                let masses: Vec<f64> = segments.iter().map(|(_, _, p)| *p).collect();
                return Ok(compensated(
                    segments.iter().zip(mass_intervals(&masses)).map(
                        |((xa, xb, _), (left, right, mass))| {
                            let (width, a, b) = local_interval(lo, hi, left, right, mass);
                            let x = xa + (xb - xa) * a - center;
                            let y = xa + (xb - xa) * b - center;
                            width
                                * if square {
                                    (x * x + x * y + y * y) / 3.0
                                } else {
                                    (x + y) / 2.0
                                }
                        },
                    ),
                ));
            }
        }
        let segment = |qa: f64, qb: f64, xa: f64, xb: f64| {
            let a = lo.max(qa);
            let b = hi.min(qb);
            if a >= b {
                return 0.0;
            }
            let x = xa + (xb - xa) * (a - qa) / (qb - qa) - center;
            let y = xa + (xb - xa) * (b - qa) / (qb - qa) - center;
            (b - a)
                * if square {
                    (x * x + x * y + y * y) / 3.0
                } else {
                    (x + y) / 2.0
                }
        };
        match self {
            Self::PointMass(x) => Ok((hi - lo)
                * if square {
                    (x - center).powi(2)
                } else {
                    x - center
                }),
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
                        width
                            * if square {
                                (x - center).powi(2)
                            } else {
                                x - center
                            }
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
                        let xa = x[0] + (x[1] - x[0]) * a - center;
                        let xb = x[0] + (x[1] - x[0]) * b - center;
                        width
                            * if square {
                                (xa * xa + xa * xb + xb * xb) / 3.0
                            } else {
                                (xa + xb) / 2.0
                            }
                    }),
            )),
            Self::Quantile {
                probabilities,
                values,
            } => Ok(compensated(values.windows(2).enumerate().map(|(i, x)| {
                segment(probabilities[i], probabilities[i + 1], x[0], x[1])
            }))),
            Self::Hurdle {
                base,
                atom_probability: p,
                atom_location: x,
            } => {
                if *p == 1.0 {
                    return Ok((hi - lo)
                        * if square {
                            (x - center).powi(2)
                        } else {
                            x - center
                        });
                }
                let w = 1.0 - p;
                let surviving = base.exceedance(*x, false);
                let mut terms = vec![segment(0.0, *p, *x, *x)];
                if hi > *p {
                    let lower_survival = surviving * (1.0 - hi) / w;
                    let upper_survival = surviving * (1.0 - lo.max(*p)) / w;
                    terms.push(
                        w / surviving
                            * base.tail_integral(lower_survival, upper_survival, center, square)?,
                    );
                }
                Ok(compensated(terms.into_iter()))
            }
            Self::Parametric(d) => {
                let c = d.shape.unwrap_or(0.0);
                let order = if square { 2.0 } else { 1.0 };
                let divergent_upper = matches!(d.family, DistributionFamily::GenPareto)
                    && c * order >= 1.0
                    || matches!(d.family, DistributionFamily::GenExtreme) && c * order <= -1.0;
                if hi == 1.0 && divergent_upper {
                    return Ok(f64::INFINITY);
                }
                if hi == 1.0 && matches!(d.family, DistributionFamily::GenExtreme) && c < -0.01 {
                    return Ok(gev_upper_integral(d, 1.0 - lo, center, square));
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
                    let a = d.location - d.scale / c - center;
                    let b = d.scale / c;
                    return Ok(if square {
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
                    })? - center;
                    Ok(if square {
                        value * value * jac
                    } else {
                        value * jac
                    })
                })
                .map_err(|e| invalid(&e.to_string()))?;
                Ok(result.value)
            }
        }
    }

    pub fn moments(&self) -> Result<(f64, f64)> {
        if let Self::Parametric(d) = self {
            let (l, s, c) = (d.location, d.scale, d.shape.unwrap_or(0.0));
            let euler = 0.577_215_664_901_532_9;
            let pi = std::f64::consts::PI;
            return Ok(match d.family {
                DistributionFamily::GumbelRight => (l + s * euler, s * s * pi * pi / 6.0),
                DistributionFamily::GumbelLeft => (l - s * euler, s * s * pi * pi / 6.0),
                DistributionFamily::GenExtreme => {
                    let mean = if c > -1.0 {
                        l - s * gamma_moments(c).1
                    } else {
                        f64::INFINITY
                    };
                    let variance = if c > -0.5 {
                        s * s * gamma_moments(c).2
                    } else {
                        f64::INFINITY
                    };
                    (mean, variance)
                }
                DistributionFamily::GenPareto => (
                    if c < 1.0 {
                        l + s / (1.0 - c)
                    } else {
                        f64::INFINITY
                    },
                    if c < 0.5 {
                        s * s / ((1.0 - c).powi(2) * (1.0 - 2.0 * c))
                    } else {
                        f64::INFINITY
                    },
                ),
                DistributionFamily::WeibullMin | DistributionFamily::WeibullMax => {
                    let (g, _, scaled_variance) = gamma_moments(1.0 / c);
                    (
                        l + if d.family == DistributionFamily::WeibullMin {
                            s * g
                        } else {
                            -s * g
                        },
                        s * s * scaled_variance / c / c,
                    )
                }
                DistributionFamily::SkewNormal => {
                    let delta = c / c.hypot(1.0);
                    (
                        l + s * delta * (2.0 / pi).sqrt(),
                        s * s * (1.0 - 2.0 * delta * delta / pi),
                    )
                }
            });
        }
        let mean = self.integral(0.0, 1.0, 0.0, false)?;
        let variance = if mean.is_finite() {
            self.integral(0.0, 1.0, mean, true)?
        } else {
            f64::NAN
        };
        Ok((mean, variance))
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

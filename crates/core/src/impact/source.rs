//! Caller-supplied exposure distributions for expectation and pushforward.
//!
//! Histogram bins are uniform. Discrete and point-mass laws keep their atoms.
//! Canonical curves are accepted only as full distributions.

use crate::{
    CanonicalCurve, Distribution, EmpiricalDistribution, FittedDistribution, HurdleDistribution,
    PointMassDistribution, TabulatedDistribution,
};

use super::error::{ContractResult, fail};
use super::quadrature::neumaier;

const MASS_TOLERANCE: f64 = 1e-12;
const EULER: f64 = 0.577_215_664_901_532_9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Tail {
    Finite(f64),
    NegativeInfinity,
    PositiveInfinity,
}

impl Tail {
    pub(crate) fn is_finite(self) -> bool {
        matches!(self, Self::Finite(_))
    }
}

/// Uniform-within-bin histogram whose weights are probability mass.
#[derive(Clone, Debug)]
pub struct HistogramExposure {
    edges: Vec<f64>,
    weights: Vec<f64>,
    normalized: Vec<f64>,
    /// `cumulative[i]` is the left-to-right running sum of the first `i`
    /// normalized weights, without forcing the final entry to one.
    cumulative: Vec<f64>,
    original_mass_sum: f64,
    normalization_factor: f64,
}

impl HistogramExposure {
    pub fn new(edges: Vec<f64>, weights: Vec<f64>) -> ContractResult<Self> {
        if edges.len() < 2 || weights.len() + 1 != edges.len() {
            return Err(fail(
                "invalid_histogram",
                "/bin_edges",
                "histogram edges must be one longer than the weights and contain at least one bin",
            ));
        }
        if edges.iter().any(|edge| !edge.is_finite())
            || edges.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(fail(
                "invalid_histogram",
                "/bin_edges",
                "histogram edges must be finite and strictly increasing",
            ));
        }
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(fail(
                "invalid_histogram",
                "/weights",
                "weights must be finite and non-negative",
            ));
        }
        let original_mass_sum = neumaier(weights.iter().copied());
        if original_mass_sum.is_nan()
            || original_mass_sum <= 0.0
            || (original_mass_sum - 1.0).abs() > MASS_TOLERANCE
        {
            return Err(fail(
                "invalid_histogram",
                "/weights",
                "total mass must be one within 1e-12",
            ));
        }
        let normalization_factor = 1.0 / original_mass_sum;
        let normalized: Vec<f64> = weights
            .iter()
            .map(|weight| weight / original_mass_sum)
            .collect();
        let mut cumulative = Vec::with_capacity(normalized.len() + 1);
        let mut running = 0.0;
        cumulative.push(running);
        for weight in &normalized {
            running += weight;
            cumulative.push(running);
        }
        Ok(Self {
            edges,
            weights,
            normalized,
            cumulative,
            original_mass_sum,
            normalization_factor,
        })
    }

    pub fn bin_edges(&self) -> &[f64] {
        &self.edges
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn normalized_weights(&self) -> &[f64] {
        &self.normalized
    }

    pub fn original_mass_sum(&self) -> f64 {
        self.original_mass_sum
    }

    pub fn normalization_factor(&self) -> f64 {
        self.normalization_factor
    }

    fn cdf_below(&self, x: f64) -> f64 {
        if x <= self.edges[0] {
            return 0.0;
        }
        if x > *self.edges.last().expect("histogram has edges") {
            return 1.0;
        }
        // Number of edges at or below `x`; at least one because `x > edges[0]`.
        let count = self.edges.partition_point(|edge| *edge <= x);
        if count == self.edges.len() {
            return self.cumulative[self.normalized.len()];
        }
        let bin = count - 1;
        let left = self.edges[bin];
        if x > left {
            let width = self.edges[bin + 1] - left;
            self.cumulative[bin] + self.normalized[bin] * (x - left) / width
        } else {
            self.cumulative[bin]
        }
    }

    /// Lower quantile `inf {x : F(x) >= p}`.
    fn quantile(&self, probability: f64) -> f64 {
        if probability <= 0.0 {
            return self.edges[0];
        }
        if probability >= 1.0 {
            return *self.edges.last().expect("histogram has edges");
        }
        // The first bin whose right cumulative reaches `probability` has
        // positive mass: a zero-mass bin repeats the previous cumulative.
        let bin = self.cumulative[1..].partition_point(|cumulative| *cumulative < probability);
        self.interpolate(bin, probability)
    }

    /// Upper quantile `inf {x : F(x) > p}`, which skips zero-mass gaps.
    fn upper_quantile(&self, probability: f64) -> f64 {
        if probability >= 1.0 {
            return *self.edges.last().expect("histogram has edges");
        }
        let probability = probability.max(0.0);
        let bin = self.cumulative[1..].partition_point(|cumulative| *cumulative <= probability);
        self.interpolate(bin, probability)
    }

    fn interpolate(&self, bin: usize, probability: f64) -> f64 {
        if bin >= self.normalized.len() {
            return *self.edges.last().expect("histogram has edges");
        }
        let weight = (probability - self.cumulative[bin]) / self.normalized[bin];
        let left = self.edges[bin];
        left + weight * (self.edges[bin + 1] - left)
    }

    fn knots(&self) -> Vec<f64> {
        let mut knots = self.cumulative.clone();
        *knots.last_mut().expect("knots exist") = 1.0;
        knots
    }
}

/// Finite support with explicit probability atoms.
#[derive(Clone, Debug)]
pub struct DiscreteDistribution {
    pairs: Vec<(f64, f64)>,
    /// Plain running sums of the masses, as used by the quantile.
    cumulative: Vec<f64>,
    /// Compensated (Neumaier) sums of the first `i` masses, as used by the cdf.
    compensated: Vec<f64>,
}

impl DiscreteDistribution {
    pub fn new(support: Vec<f64>, probabilities: Vec<f64>) -> ContractResult<Self> {
        if support.is_empty() || support.len() != probabilities.len() {
            return Err(fail(
                "invalid_value",
                "/support",
                "discrete support and probabilities must be non-empty and the same length",
            ));
        }
        if support.iter().any(|value| !value.is_finite()) {
            return Err(fail(
                "invalid_value",
                "/support",
                "discrete support values must be finite",
            ));
        }
        if probabilities
            .iter()
            .any(|probability| !probability.is_finite() || *probability < 0.0)
        {
            return Err(fail(
                "invalid_value",
                "/probabilities",
                "discrete probabilities must be finite and non-negative",
            ));
        }
        let total = neumaier(probabilities.iter().copied());
        if total.is_nan() || total <= 0.0 || (total - 1.0).abs() > MASS_TOLERANCE {
            return Err(fail(
                "invalid_value",
                "/probabilities",
                "discrete probabilities must sum to one within 1e-12",
            ));
        }
        let factor = 1.0 / total;
        let mut pairs: Vec<(f64, f64)> = support
            .into_iter()
            .zip(probabilities)
            .map(|(location, probability)| (location, probability * factor))
            .collect();
        pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
        let mut merged: Vec<(f64, f64)> = Vec::new();
        for (location, probability) in pairs {
            if merged.last().is_some_and(|last| last.0 == location) {
                merged.last_mut().expect("just checked").1 += probability;
            } else {
                merged.push((location, probability));
            }
        }
        let mut cumulative = Vec::with_capacity(merged.len());
        let mut running = 0.0;
        for (_, mass) in &merged {
            running += mass;
            cumulative.push(running);
        }
        let mut compensated = Vec::with_capacity(merged.len() + 1);
        let mut total = 0.0_f64;
        let mut compensation = 0.0;
        compensated.push(total + compensation);
        for (_, mass) in &merged {
            let candidate = total + mass;
            if total.abs() >= mass.abs() {
                compensation += (total - candidate) + mass;
            } else {
                compensation += (mass - candidate) + total;
            }
            total = candidate;
            compensated.push(total + compensation);
        }
        Ok(Self {
            pairs: merged,
            cumulative,
            compensated,
        })
    }

    pub fn support(&self) -> impl Iterator<Item = f64> + '_ {
        self.pairs.iter().map(|pair| pair.0)
    }

    pub fn probabilities(&self) -> impl Iterator<Item = f64> + '_ {
        self.pairs.iter().map(|pair| pair.1)
    }

    fn quantile(&self, probability: f64) -> f64 {
        let index = self
            .cumulative
            .partition_point(|cumulative| *cumulative < probability);
        self.pairs
            .get(index)
            .unwrap_or_else(|| self.pairs.last().expect("discrete support is non-empty"))
            .0
    }

    /// `P(X <= x)`.
    fn cdf(&self, x: f64) -> f64 {
        self.compensated[self.pairs.partition_point(|(location, _)| *location <= x)]
    }

    /// `P(X < x)`.
    fn cdf_below(&self, x: f64) -> f64 {
        self.compensated[self.pairs.partition_point(|(location, _)| *location < x)]
    }

    fn knots(&self) -> Vec<f64> {
        let mut knots = Vec::with_capacity(self.cumulative.len() + 1);
        knots.push(0.0);
        knots.extend_from_slice(&self.cumulative);
        *knots.last_mut().expect("knots exist") = 1.0;
        knots
    }
}

/// Exposure law accepted by expectation and distribution transformation.
#[derive(Clone, Debug)]
pub enum ImpactSource {
    Histogram(HistogramExposure),
    Discrete(DiscreteDistribution),
    PointMass(PointMassDistribution),
    Tabulated(TabulatedDistribution),
    Fitted(FittedDistribution),
    Hurdle(HurdleDistribution),
    Empirical(EmpiricalDistribution),
}

impl ImpactSource {
    pub fn from_curve(curve: CanonicalCurve) -> ContractResult<Self> {
        match curve {
            CanonicalCurve::Fitted(distribution) => Ok(Self::Fitted(distribution)),
            CanonicalCurve::Hurdle(distribution) => Ok(Self::Hurdle(distribution)),
            CanonicalCurve::PointMass(distribution) => Ok(Self::PointMass(distribution)),
            CanonicalCurve::Tabulated(distribution) => Ok(Self::Tabulated(distribution)),
            CanonicalCurve::NoData => Err(fail(
                "missing_probability_support",
                "",
                "a no-data curve does not define a distribution",
            )),
        }
    }

    pub fn histogram_diagnostics(&self) -> Option<(f64, f64)> {
        match self {
            Self::Histogram(histogram) => Some((
                histogram.original_mass_sum(),
                histogram.normalization_factor(),
            )),
            _ => None,
        }
    }

    pub(crate) fn is_full_probability_support(&self) -> bool {
        match self {
            Self::Histogram(_) | Self::Discrete(_) | Self::PointMass(_) => true,
            Self::Tabulated(distribution) => distribution.has_full_probability_support(),
            Self::Fitted(distribution) => distribution.has_full_probability_support(),
            Self::Hurdle(distribution) => distribution.has_full_probability_support(),
            Self::Empirical(distribution) => distribution.has_full_probability_support(),
        }
    }

    /// Quantile is constant on finitely many probability intervals.
    pub(crate) fn has_piecewise_constant_quantile(&self) -> bool {
        matches!(self, Self::Discrete(_) | Self::PointMass(_))
    }

    /// Quantile is piecewise linear or piecewise constant on `[0, 1]`.
    pub(crate) fn has_piecewise_linear_quantile(&self) -> bool {
        matches!(
            self,
            Self::Histogram(_)
                | Self::Discrete(_)
                | Self::PointMass(_)
                | Self::Tabulated(_)
                | Self::Empirical(_)
        )
    }

    pub(crate) fn quantile(&self, probability: f64) -> ContractResult<f64> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(fail(
                "invalid_value",
                "",
                "probability must be finite and in [0, 1]",
            ));
        }
        let value = match self {
            Self::Histogram(histogram) => histogram.quantile(probability),
            Self::Discrete(discrete) => discrete.quantile(probability),
            Self::PointMass(distribution) => distribution.location(),
            Self::Tabulated(distribution) => {
                distribution.ppf(probability).map_err(distribution_error)?
            }
            Self::Empirical(distribution) => {
                distribution.ppf(probability).map_err(distribution_error)?
            }
            Self::Fitted(distribution) => {
                distribution.ppf(probability).map_err(distribution_error)?
            }
            Self::Hurdle(distribution) => {
                distribution.ppf(probability).map_err(distribution_error)?
            }
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err(fail(
                "non_finite_result",
                "",
                "source quantile is not finite",
            ))
        }
    }

    /// Upper quantile `inf {x : F(x) > p}`. It differs from [`Self::quantile`]
    /// only where the source has a zero-mass gap, and gives the generalized
    /// inverse of a decreasing pushforward there.
    pub(crate) fn upper_quantile(&self, probability: f64) -> ContractResult<f64> {
        match self {
            Self::Histogram(histogram)
                if probability.is_finite() && (0.0..=1.0).contains(&probability) =>
            {
                Ok(histogram.upper_quantile(probability))
            }
            _ => self.quantile(probability),
        }
    }

    pub(crate) fn cdf(&self, x: f64) -> f64 {
        match self {
            Self::Histogram(histogram) => histogram.cdf_below(x),
            Self::Discrete(discrete) => discrete.cdf(x),
            Self::PointMass(distribution) => distribution.cdf(x),
            Self::Tabulated(distribution) => distribution.cdf(x),
            Self::Fitted(distribution) => distribution.cdf(x),
            Self::Hurdle(distribution) => distribution.cdf(x),
            Self::Empirical(distribution) => empirical_cdf(distribution.samples(), x, false),
        }
    }

    pub(crate) fn cdf_below(&self, x: f64) -> f64 {
        match self {
            Self::Histogram(histogram) => histogram.cdf_below(x),
            Self::Discrete(discrete) => discrete.cdf_below(x),
            Self::PointMass(distribution) => distribution.cdf_below(x),
            Self::Tabulated(distribution) => distribution.cdf_below(x),
            Self::Fitted(distribution) => distribution.cdf_below(x),
            Self::Hurdle(distribution) => distribution.cdf_below(x),
            Self::Empirical(distribution) => empirical_cdf(distribution.samples(), x, true),
        }
    }

    pub(crate) fn point_atoms(&self) -> Vec<(f64, f64)> {
        match self {
            Self::Discrete(discrete) => discrete.pairs.clone(),
            Self::PointMass(distribution) => vec![(distribution.location(), 1.0)],
            Self::Hurdle(distribution) => {
                vec![(
                    distribution.atom_location(),
                    distribution.atom_probability(),
                )]
            }
            Self::Tabulated(distribution) => {
                flat_knots(distribution.probabilities(), distribution.values())
            }
            Self::Empirical(distribution) => {
                let samples = distribution.samples();
                let last_gap = (samples.len() - 1) as f64;
                let mut atoms = Vec::new();
                let mut index = 0;
                while index < samples.len() {
                    let location = samples[index];
                    let mut count = 1;
                    while index + count < samples.len() && samples[index + count] == location {
                        count += 1;
                    }
                    if count > 1 {
                        // Under the interpolated quantile, `count` equal order
                        // statistics span `count - 1` of the `n - 1` gaps.
                        atoms.push((location, (count - 1) as f64 / last_gap));
                    }
                    index += count;
                }
                atoms
            }
            Self::Histogram(_) | Self::Fitted(_) => Vec::new(),
        }
    }

    pub(crate) fn probability_knots(&self) -> Vec<f64> {
        match self {
            Self::Histogram(histogram) => histogram.knots(),
            Self::Discrete(discrete) => discrete.knots(),
            Self::PointMass(_) => vec![0.0, 1.0],
            Self::Tabulated(distribution) => {
                let mut knots = distribution.probabilities().to_vec();
                if let Some(first) = knots.first_mut() {
                    *first = 0.0;
                }
                if let Some(last) = knots.last_mut() {
                    *last = 1.0;
                }
                knots
            }
            Self::Empirical(distribution) => {
                let last = (distribution.samples().len() - 1) as f64;
                (0..distribution.samples().len())
                    .map(|index| index as f64 / last)
                    .collect()
            }
            Self::Fitted(_) | Self::Hurdle(_) => vec![0.0, 1.0],
        }
    }

    pub(crate) fn support(&self) -> (Tail, Tail) {
        match self {
            Self::Histogram(histogram) => (
                Tail::Finite(histogram.edges[0]),
                Tail::Finite(*histogram.edges.last().expect("histogram has edges")),
            ),
            Self::Discrete(discrete) => (
                Tail::Finite(discrete.pairs[0].0),
                Tail::Finite(discrete.pairs.last().expect("discrete support").0),
            ),
            Self::PointMass(distribution) => (
                Tail::Finite(distribution.location()),
                Tail::Finite(distribution.location()),
            ),
            Self::Tabulated(distribution) => (
                Tail::Finite(distribution.values()[0]),
                Tail::Finite(*distribution.values().last().expect("tabulated values")),
            ),
            Self::Empirical(distribution) => (
                Tail::Finite(distribution.samples()[0]),
                Tail::Finite(*distribution.samples().last().expect("empirical samples")),
            ),
            Self::Fitted(distribution) => fitted_support(distribution),
            Self::Hurdle(distribution) => {
                let (_, upper) = fitted_support(distribution.base());
                (Tail::Finite(distribution.atom_location()), upper)
            }
        }
    }

    pub(crate) fn mean(&self) -> Option<f64> {
        match self {
            Self::Fitted(distribution) => fitted_mean(distribution),
            // The atom is exact. A closed conditional mean is not required
            // for the families whose unconditional mean is the fixture path.
            Self::Hurdle(_) => None,
            _ => None,
        }
    }

    pub(crate) fn mean_is_finite(&self) -> bool {
        match self {
            Self::Histogram(_)
            | Self::Discrete(_)
            | Self::PointMass(_)
            | Self::Tabulated(_)
            | Self::Empirical(_) => true,
            Self::Fitted(distribution) => fitted_mean_is_finite(distribution),
            Self::Hurdle(distribution) => fitted_mean_is_finite(distribution.base()),
        }
    }
}

/// Cdf of the law whose quantile interpolates the order statistics linearly
/// (Hyndman-Fan type 7, the empirical `ppf`). Order statistic `k` sits at
/// probability `k / (n - 1)`, so tied samples form an atom. `strict` gives
/// `P(X < x)` instead of `P(X <= x)`.
fn empirical_cdf(samples: &[f64], x: f64, strict: bool) -> f64 {
    let count = if strict {
        samples.partition_point(|sample| *sample < x)
    } else {
        samples.partition_point(|sample| *sample <= x)
    };
    if count == 0 {
        return 0.0;
    }
    if count == samples.len() {
        return 1.0;
    }
    let lower = count - 1;
    let (left, right) = (samples[lower], samples[count]);
    let fraction = if x > left {
        ((x - left) / (right - left)).min(1.0)
    } else {
        0.0
    };
    (lower as f64 + fraction) / (samples.len() - 1) as f64
}

fn flat_knots(probabilities: &[f64], values: &[f64]) -> Vec<(f64, f64)> {
    let mut atoms = Vec::new();
    let mut index = 0;
    while index + 1 < values.len() {
        if values[index] == values[index + 1] {
            let start = index;
            while index + 1 < values.len() && values[index + 1] == values[start] {
                index += 1;
            }
            let mass = probabilities[index] - probabilities[start];
            if mass > 0.0 {
                atoms.push((values[start], mass));
            }
        }
        index += 1;
    }
    atoms
}

fn distribution_error(error: crate::CrcError) -> super::error::ImpactContractError {
    match error {
        crate::CrcError::OutOfSupport { .. } => fail(
            "missing_probability_support",
            "",
            "the source quantile is outside the supplied probability support",
        ),
        other => fail("invalid_value", "", other.to_string()),
    }
}

fn fitted_support(distribution: &FittedDistribution) -> (Tail, Tail) {
    use crate::DistributionFamily::{
        GenExtreme, GenPareto, GumbelLeft, GumbelRight, SkewNormal, WeibullMax, WeibullMin,
    };
    let location = distribution.location;
    let scale = distribution.scale;
    let shape = distribution.shape.unwrap_or(0.0);
    match distribution.family {
        GumbelRight | GumbelLeft | SkewNormal => (Tail::NegativeInfinity, Tail::PositiveInfinity),
        WeibullMin => (Tail::Finite(location), Tail::PositiveInfinity),
        WeibullMax => (Tail::NegativeInfinity, Tail::Finite(location)),
        GenPareto => {
            let upper = if shape < 0.0 {
                Tail::Finite(location - scale / shape)
            } else {
                Tail::PositiveInfinity
            };
            (Tail::Finite(location), upper)
        }
        GenExtreme => {
            if shape.abs() < 1e-12 {
                (Tail::NegativeInfinity, Tail::PositiveInfinity)
            } else if shape > 0.0 {
                (
                    Tail::NegativeInfinity,
                    Tail::Finite(location + scale / shape),
                )
            } else {
                (
                    Tail::Finite(location + scale / shape),
                    Tail::PositiveInfinity,
                )
            }
        }
    }
}

fn fitted_mean(distribution: &FittedDistribution) -> Option<f64> {
    use crate::DistributionFamily::{GenExtreme, GenPareto, GumbelLeft, GumbelRight};
    if !fitted_mean_is_finite(distribution) {
        return None;
    }
    let location = distribution.location;
    let scale = distribution.scale;
    let mean = match distribution.family {
        GumbelRight => location + scale * EULER,
        GumbelLeft => location - scale * EULER,
        // SciPy `genextreme`: E[X] = loc + scale * (1 - Gamma(1 + c)) / c.
        GenExtreme => location + scale * gev_mean_factor(distribution.shape?),
        // SciPy `genpareto`: E[X] = loc + scale / (1 - c).
        GenPareto => location + scale / (1.0 - distribution.shape?),
        _ => return None,
    };
    mean.is_finite().then_some(mean)
}

/// `(1 - Gamma(1 + c)) / c` for `c > -1`, with the limit Euler's constant at zero.
///
/// Near zero the direct form cancels, so `ln Gamma(1 + c)` is summed from its
/// Taylor series `-EULER c + sum_k (-1)^k zeta(k) c^k / k` instead.
pub(crate) fn gev_mean_factor(shape: f64) -> f64 {
    if shape.abs() >= 0.1 {
        return (1.0 - crate::reference::curve_fitting::gamma(1.0 + shape)) / shape;
    }
    // zeta(2) through zeta(17); later terms are below 1e-17 for |c| < 0.1.
    const ZETA: [f64; 16] = [
        1.644_934_066_848_226_4,
        1.202_056_903_159_594_3,
        1.082_323_233_711_138_2,
        1.036_927_755_143_37,
        1.017_343_061_984_449,
        1.008_349_277_381_922_8,
        1.004_077_356_197_944_4,
        1.002_008_392_826_082_2,
        1.000_994_575_127_818_1,
        1.000_494_188_604_119_5,
        1.000_246_086_553_308,
        1.000_122_713_347_578_5,
        1.000_061_248_135_058_7,
        1.000_030_588_236_307_1,
        1.000_015_282_259_408_7,
        1.000_007_637_197_637_9,
    ];
    // `log_ratio` is ln Gamma(1 + c) / c.
    let mut log_ratio = -EULER;
    let mut power = 1.0;
    for (offset, zeta) in ZETA.iter().enumerate() {
        let order = (offset + 2) as f64;
        power *= -shape;
        log_ratio -= power * zeta / order;
    }
    let log_gamma = log_ratio * shape;
    let expm1_ratio = if log_gamma == 0.0 {
        1.0
    } else {
        log_gamma.exp_m1() / log_gamma
    };
    -log_ratio * expm1_ratio
}

fn fitted_mean_is_finite(distribution: &FittedDistribution) -> bool {
    use crate::DistributionFamily::{GenExtreme, GenPareto};
    match distribution.family {
        // GEV uses SciPy's `1 - c * z` parameterization: its mean exists
        // exactly when c > -1. GPD instead requires c < 1.
        GenExtreme => distribution.shape.expect("validated shape") > -1.0,
        GenPareto => distribution.shape.expect("validated shape") < 1.0,
        _ => true,
    }
}

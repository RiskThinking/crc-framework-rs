//! Caller-supplied exposure distributions for expectation and pushforward.
//!
//! Histogram bins are uniform. Discrete and point-mass laws keep their atoms.
//! Canonical curves are accepted only as full distributions.

use std::borrow::Cow;

use crate::numerical::{
    CompensatedSum, CountRanks, ProbabilitySegment, affine_fitted_mean_deviation,
    interval_fraction, linear_interpolate, segment_probability,
};
use crate::{
    Distribution, EmpiricalDistribution, FittedDistribution, HurdleDistribution,
    PointMassDistribution, TabulatedDistribution,
};

use super::error::{ContractResult, fail};
use super::kernel_curve::KernelCurve;
use super::quadrature::neumaier;

const MASS_TOLERANCE: f64 = 1e-12;

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
    /// Compensated prefixes of the normalized weights, including zero.
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
        let mut running = CompensatedSum::default();
        cumulative.push(0.0);
        for weight in &normalized {
            running.add(*weight);
            cumulative.push(running.total());
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
            self.cumulative[bin]
                + self.normalized[bin] * interval_fraction(left, self.edges[bin + 1], left, x)
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
        let left = self.cumulative[bin];
        let right = self.cumulative[bin + 1];
        let weight = (probability - left) / (right - left);
        linear_interpolate(self.edges[bin], self.edges[bin + 1], weight)
    }

    fn knots(&self) -> Vec<f64> {
        let mut knots = self.cumulative.clone();
        *knots.last_mut().expect("knots exist") = 1.0;
        knots
    }

    fn inverse_survival(&self, survival: f64, upper: bool) -> f64 {
        let mut suffix = CompensatedSum::default();
        for (edges, &mass) in self.edges.windows(2).zip(&self.normalized).rev() {
            if mass == 0.0 {
                continue;
            }
            let right = suffix.total();
            suffix.add(mass);
            let left = suffix.total();
            if upper && survival == left {
                return edges[0];
            }
            if survival < left || survival == 0.0 {
                let weight = ((survival - right) / mass).clamp(0.0, 1.0);
                return linear_interpolate(edges[1], edges[0], weight);
            }
        }
        self.edges[self.normalized.iter().position(|&p| p > 0.0).unwrap()]
    }
}

/// Finite support with explicit probability atoms.
#[derive(Clone, Debug)]
pub struct DiscreteDistribution {
    pairs: Vec<(f64, f64)>,
    /// Compensated prefixes of the masses, including zero.
    cumulative: Vec<f64>,
    /// Exact ranks when the law was declared from observation counts.
    ranks: Option<CountRanks>,
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
        let mut cumulative = Vec::with_capacity(merged.len() + 1);
        let mut running = CompensatedSum::default();
        cumulative.push(0.0);
        for (_, mass) in &merged {
            running.add(*mass);
            cumulative.push(running.total());
        }
        Ok(Self {
            pairs: merged,
            cumulative,
            ranks: None,
        })
    }

    /// Exact observation ranks, when the law was declared from counts.
    pub fn ranks(&self) -> Option<&CountRanks> {
        self.ranks.as_ref()
    }

    /// An empirical law with exact integer counts at strictly increasing
    /// support points. Cumulative probabilities are the rounded ranks `K / N`.
    pub fn from_counts(support: Vec<f64>, ranks: &CountRanks) -> ContractResult<Self> {
        let probabilities = ranks.probabilities();
        if support.len() != probabilities.len()
            || support.iter().any(|value| !value.is_finite())
            || support.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(fail(
                "invalid_value",
                "/support",
                "counted support values must be finite, increasing, and match the counts",
            ));
        }
        Ok(Self {
            pairs: support.into_iter().zip(probabilities).collect(),
            cumulative: ranks.cumulative(),
            ranks: Some(ranks.clone()),
        })
    }

    pub fn support(&self) -> impl Iterator<Item = f64> + '_ {
        self.pairs.iter().map(|pair| pair.0)
    }

    pub fn probabilities(&self) -> impl Iterator<Item = f64> + '_ {
        self.pairs.iter().map(|pair| pair.1)
    }

    fn quantile(&self, probability: f64) -> f64 {
        let index = self.cumulative[1..].partition_point(|cumulative| *cumulative < probability);
        self.pairs
            .get(index)
            .unwrap_or_else(|| self.pairs.last().expect("discrete support is non-empty"))
            .0
    }

    /// `P(X <= x)`.
    fn cdf(&self, x: f64) -> f64 {
        self.cumulative[self.pairs.partition_point(|(location, _)| *location <= x)]
    }

    /// `P(X < x)`.
    fn cdf_below(&self, x: f64) -> f64 {
        self.cumulative[self.pairs.partition_point(|(location, _)| *location < x)]
    }

    fn knots(&self) -> Vec<f64> {
        let mut knots = self.cumulative.clone();
        *knots.last_mut().expect("knots exist") = 1.0;
        knots
    }
}

/// Ordered finite intervals and atoms with their own probability masses.
#[derive(Clone, Debug)]
pub struct WeightedSegments {
    segments: Vec<ProbabilitySegment>,
    // Independent sums retain small probabilities in both tails.
    prefix: Vec<f64>,
    suffix: Vec<f64>,
}

impl WeightedSegments {
    fn new(segments: Vec<ProbabilitySegment>) -> Self {
        let mut prefix = Vec::with_capacity(segments.len() + 1);
        let mut suffix = vec![0.0; segments.len() + 1];
        let mut forward = CompensatedSum::default();
        let mut backward = CompensatedSum::default();
        prefix.push(0.0);
        for (index, &(_, _, mass)) in segments.iter().enumerate() {
            let reverse = segments.len() - 1 - index;
            forward.add(mass);
            prefix.push(forward.total());
            backward.add(segments[reverse].2);
            suffix[reverse] = backward.total();
        }
        Self {
            segments,
            prefix,
            suffix,
        }
    }

    pub(super) fn intervals(&self) -> impl Iterator<Item = (ProbabilitySegment, f64, f64)> + '_ {
        self.segments
            .iter()
            .enumerate()
            .map(|(index, &segment)| (segment, self.prefix[index], self.suffix[index + 1]))
    }

    fn probability(&self, start: f64, end: f64, start_closed: bool, end_closed: bool) -> f64 {
        let first = self.lower_count(start, !start_closed);
        let end_index = self.upper_start(end, !end_closed);
        if first >= end_index {
            return 0.0;
        }
        // Sum only the intersected pieces, keeping small interior masses
        // representable even when both cumulative boundaries round together.
        segment_probability(
            &self.segments[first..end_index],
            start,
            end,
            start_closed,
            end_closed,
        )
    }

    fn lower_count(&self, x: f64, inclusive: bool) -> usize {
        self.segments
            .partition_point(|&(lo, hi, _)| hi < x || (hi == x && (inclusive || lo < hi)))
    }

    fn upper_start(&self, x: f64, inclusive: bool) -> usize {
        self.segments
            .partition_point(|&(lo, hi, _)| lo < x || (lo == x && !inclusive && lo == hi))
    }

    fn cdf(&self, x: f64, inclusive: bool) -> f64 {
        let index = self.lower_count(x, inclusive);
        let partial = self
            .segments
            .get(index)
            .map_or(0.0, |&(a, b, mass)| mass * interval_fraction(a, b, a, x));
        (self.prefix[index] + partial).min(1.0)
    }

    fn survival(&self, x: f64, inclusive: bool) -> f64 {
        let index = self.upper_start(x, inclusive);
        let partial = index.checked_sub(1).map_or(0.0, |prior| {
            let (a, b, mass) = self.segments[prior];
            mass * interval_fraction(a, b, x, b)
        });
        (self.suffix[index] + partial).min(1.0)
    }

    fn quantile(&self, probability: f64, upper: bool) -> f64 {
        if probability == 0.0 {
            return self.segments[0].0;
        }
        if probability == 1.0 {
            return self.segments.last().unwrap().1;
        }
        let index = self.prefix[1..]
            .partition_point(|&end| end < probability || (upper && end == probability));
        let Some(&(a, b, mass)) = self.segments.get(index) else {
            return self.segments.last().unwrap().1;
        };
        if probability == self.prefix[index + 1] {
            return b;
        }
        linear_interpolate(
            a,
            b,
            ((probability - self.prefix[index]) / mass).clamp(0.0, 1.0),
        )
    }

    fn inverse_survival(&self, survival: f64, upper: bool) -> f64 {
        if survival == 0.0 {
            return self.segments.last().unwrap().1;
        }
        let count = self.suffix[..self.segments.len()]
            .partition_point(|&start| start > survival || (upper && start == survival));
        if survival == 1.0 || count == 0 {
            return self.segments[0].0;
        }
        let index = count - 1;
        let (a, b, mass) = self.segments[index];
        if survival == self.suffix[index] {
            return a;
        }
        linear_interpolate(
            b,
            a,
            ((survival - self.suffix[index + 1]) / mass).clamp(0.0, 1.0),
        )
    }
}

#[derive(Clone, Debug)]
/// Exposure law accepted by expectation and distribution transformation.
pub enum ImpactSource {
    Histogram(HistogramExposure),
    Discrete(DiscreteDistribution),
    PointMass(PointMassDistribution),
    Tabulated(TabulatedDistribution),
    Fitted(FittedDistribution),
    Hurdle(HurdleDistribution),
    Empirical(EmpiricalDistribution),
    Segments(WeightedSegments),
}

impl ImpactSource {
    /// Exact masses for ordered uniform intervals and atoms, including gaps.
    pub fn from_segments(mut segments: Vec<(f64, f64, f64)>) -> ContractResult<Self> {
        let total = neumaier(segments.iter().map(|&(_, _, mass)| mass));
        if !total.is_finite()
            || (total - 1.0).abs() > MASS_TOLERANCE
            || segments.iter().any(|&(lo, hi, mass)| {
                !lo.is_finite() || !hi.is_finite() || lo > hi || !mass.is_finite() || mass < 0.0
            })
        {
            return Err(fail(
                "invalid_value",
                "",
                "finite segments require ordered bounds and masses summing to one",
            ));
        }
        segments.retain(|&(_, _, mass)| mass > 0.0);
        if segments.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(fail(
                "invalid_value",
                "",
                "finite segments must be ordered and non-overlapping",
            ));
        }
        if segments.iter().all(|&(lo, hi, _)| lo == hi) {
            return DiscreteDistribution::new(
                segments.iter().map(|&(x, _, _)| x).collect(),
                segments.iter().map(|&(_, _, mass)| mass).collect(),
            )
            .map(Self::Discrete);
        }
        for (_, _, mass) in &mut segments {
            *mass /= total;
        }
        Ok(Self::Segments(WeightedSegments::new(segments)))
    }

    pub(crate) fn weighted_segments(&self) -> Option<Cow<'_, [ProbabilitySegment]>> {
        match self {
            Self::Segments(segments) => Some(Cow::Borrowed(&segments.segments)),
            Self::Histogram(histogram) => Some(Cow::Owned(
                histogram
                    .edges
                    .windows(2)
                    .zip(&histogram.normalized)
                    .filter(|&(_, &mass)| mass > 0.0)
                    .map(|(edges, &mass)| (edges[0], edges[1], mass))
                    .collect(),
            )),
            _ => None,
        }
    }

    pub(crate) fn interval_probability(
        &self,
        start: f64,
        end: f64,
        start_closed: bool,
        end_closed: bool,
    ) -> f64 {
        if let Self::Segments(segments) = self {
            return segments.probability(start, end, start_closed, end_closed);
        }
        let left = if start_closed {
            self.cdf_below(start)
        } else {
            self.cdf(start)
        };
        if left > 0.5 {
            let left_tail = self.survival_at(start, start_closed);
            let right_tail = self.survival_at(end, !end_closed);
            (left_tail - right_tail).max(0.0)
        } else {
            let right = if end_closed {
                self.cdf(end)
            } else {
                self.cdf_below(end)
            };
            (right - left).max(0.0)
        }
    }

    pub fn from_curve(curve: KernelCurve) -> ContractResult<Self> {
        match curve {
            KernelCurve::Fitted(distribution) => Ok(Self::Fitted(distribution)),
            KernelCurve::Hurdle(distribution) => Ok(Self::Hurdle(distribution)),
            KernelCurve::PointMass(distribution) => Ok(Self::PointMass(distribution)),
            KernelCurve::Tabulated(distribution) => Ok(Self::Tabulated(distribution)),
            KernelCurve::NoData => Err(fail(
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
            Self::Histogram(_) | Self::Discrete(_) | Self::PointMass(_) | Self::Segments(_) => true,
            Self::Tabulated(distribution) => distribution.has_full_probability_support(),
            Self::Fitted(distribution) => distribution.has_full_probability_support(),
            Self::Hurdle(distribution) => distribution.has_full_probability_support(),
            Self::Empirical(distribution) => distribution.has_full_probability_support(),
        }
    }

    /// Quantile is constant on finitely many probability intervals.
    pub(crate) fn has_piecewise_constant_quantile(&self) -> bool {
        matches!(
            self,
            Self::Discrete(_) | Self::PointMass(_) | Self::Empirical(_)
        )
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
                | Self::Segments(_)
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
            Self::Segments(segments) => segments.quantile(probability, false),
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
            Self::Tabulated(distribution) => distribution
                .upper_quantile(probability)
                .map_err(distribution_error),
            Self::Segments(segments)
                if probability.is_finite() && (0.0..=1.0).contains(&probability) =>
            {
                Ok(segments.quantile(probability, true))
            }
            Self::Hurdle(distribution) if probability == distribution.atom_probability() => {
                hurdle_tail_lower_bound(distribution)
            }
            _ => self.quantile(probability),
        }
    }

    /// Direct survival coordinates avoid losing deep tails to `1 - s`.
    pub(crate) fn inverse_survival(&self, survival: f64) -> ContractResult<f64> {
        if !survival.is_finite() || !(0.0..=1.0).contains(&survival) {
            return Err(fail(
                "invalid_value",
                "",
                "survival must be finite and in [0,1]",
            ));
        }
        match self {
            Self::Fitted(distribution) => distribution.isf(survival).map_err(distribution_error),
            Self::Hurdle(distribution) => distribution
                .inverse_survival(survival)
                .map_err(distribution_error),
            Self::Histogram(histogram) => Ok(histogram.inverse_survival(survival, false)),
            Self::Tabulated(distribution) => {
                Ok(tabulated_inverse_survival(distribution, survival, false))
            }
            Self::PointMass(distribution) => Ok(distribution.location()),
            Self::Segments(segments) => Ok(segments.inverse_survival(survival, false)),
            Self::Discrete(discrete) if discrete.ranks.is_some() => {
                let ranks = discrete.ranks.as_ref().expect("checked by the guard");
                Ok(discrete.pairs[ranks.survival_index(survival)].0)
            }
            Self::Discrete(_) | Self::Empirical(_) => {
                let atoms = self.point_atoms();
                let mut suffix = CompensatedSum::default();
                for &(value, mass) in atoms.iter().rev() {
                    suffix.add(mass);
                    if mass > 0.0 && (survival < suffix.total() || survival == 0.0) {
                        return Ok(value);
                    }
                }
                Ok(atoms[0].0)
            }
        }
    }

    pub(crate) fn upper_quantile_from_survival(&self, survival: f64) -> ContractResult<f64> {
        match self {
            Self::Segments(segments) => Ok(segments.inverse_survival(survival, true)),
            Self::Histogram(histogram) => Ok(histogram.inverse_survival(survival, true)),
            Self::Hurdle(distribution) if survival == 1.0 - distribution.atom_probability() => {
                hurdle_tail_lower_bound(distribution)
            }
            Self::Tabulated(distribution) if survival <= 0.5 => {
                Ok(tabulated_inverse_survival(distribution, survival, true))
            }
            Self::Fitted(_) | Self::Hurdle(_) if survival <= 0.5 => self.inverse_survival(survival),
            _ => self.upper_quantile(1.0 - survival),
        }
    }

    pub(crate) fn inverse_log_tail(&self, logarithm: f64, lower: bool) -> ContractResult<f64> {
        match (self, lower) {
            (Self::Fitted(distribution), true) => distribution.inverse_log_cdf(logarithm),
            (Self::Fitted(distribution), false) => distribution.inverse_log_survival(logarithm),
            (Self::Hurdle(distribution), false) => distribution.inverse_log_survival(logarithm),
            (_, true) => return self.upper_quantile(logarithm.exp()),
            (_, false) => return self.inverse_survival(logarithm.exp()),
        }
        .map_err(distribution_error)
    }

    pub(crate) fn log_tail_at(&self, x: f64, inclusive: bool, lower: bool) -> f64 {
        match (self, lower) {
            (Self::Fitted(distribution), true) => distribution.log_cdf(x),
            (Self::Fitted(distribution), false) => distribution.log_survival(x),
            (Self::Hurdle(distribution), false) => distribution.log_survival_at(x, inclusive),
            (_, false) => self.survival_at(x, inclusive).ln(),
            (_, true) if inclusive => self.cdf(x).ln(),
            (_, true) => self.cdf_below(x).ln(),
        }
    }

    /// `P(X > x)`, or `P(X >= x)` when inclusive, without CDF subtraction.
    pub(crate) fn survival_at(&self, x: f64, inclusive: bool) -> f64 {
        match self {
            Self::Fitted(distribution) => distribution.sf(x),
            Self::Hurdle(distribution) => distribution.survival_at(x, inclusive),
            Self::Histogram(histogram) => neumaier(
                histogram
                    .edges
                    .windows(2)
                    .zip(&histogram.normalized)
                    .map(|(bounds, p)| p * interval_fraction(bounds[0], bounds[1], x, bounds[1])),
            ),
            Self::Tabulated(distribution) => neumaier(
                distribution
                    .values()
                    .windows(2)
                    .zip(distribution.probabilities().windows(2))
                    .map(|(bounds, q)| {
                        let fraction = if bounds[0] == bounds[1] {
                            f64::from(if inclusive {
                                bounds[0] >= x
                            } else {
                                bounds[0] > x
                            })
                        } else {
                            interval_fraction(bounds[0], bounds[1], x, bounds[1])
                        };
                        (q[1] - q[0]) * fraction
                    }),
            ),
            Self::Discrete(DiscreteDistribution {
                pairs,
                ranks: Some(ranks),
                ..
            }) => {
                ranks.from(pairs.partition_point(|(v, _)| if inclusive { *v < x } else { *v <= x }))
            }
            Self::Discrete(_) | Self::Empirical(_) | Self::PointMass(_) => neumaier(
                self.point_atoms()
                    .iter()
                    .filter(|(v, _)| if inclusive { *v >= x } else { *v > x })
                    .map(|(_, p)| *p),
            ),
            Self::Segments(segments) => segments.survival(x, inclusive),
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
            Self::Empirical(distribution) => distribution.cdf(x),
            Self::Segments(segments) => segments.cdf(x, true),
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
            Self::Empirical(distribution) => distribution.cdf_below(x),
            Self::Segments(segments) => segments.cdf(x, false),
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
                let count_total = samples.len() as f64;
                let mut atoms = Vec::new();
                let mut index = 0;
                while index < samples.len() {
                    let location = samples[index];
                    let mut count = 1;
                    while index + count < samples.len() && samples[index + count] == location {
                        count += 1;
                    }
                    atoms.push((location, count as f64 / count_total));
                    index += count;
                }
                atoms
            }
            Self::Histogram(_) | Self::Fitted(_) => Vec::new(),
            Self::Segments(segments) => segments
                .segments
                .chunk_by(|&(lo, hi, _), &(next_lo, next_hi, _)| {
                    lo == hi && hi == next_lo && lo == next_hi
                })
                .filter(|group| group[0].0 == group[0].1)
                .map(|group| (group[0].0, neumaier(group.iter().map(|segment| segment.2))))
                .collect(),
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
                let count = distribution.samples().len();
                (0..=count)
                    .map(|index| index as f64 / count as f64)
                    .collect()
            }
            Self::Fitted(_) | Self::Hurdle(_) => vec![0.0, 1.0],
            Self::Segments(segments) => {
                let mut knots = segments.prefix.clone();
                *knots.last_mut().unwrap() = 1.0;
                knots
            }
        }
    }

    pub(super) fn survival_knots(&self) -> Vec<f64> {
        if let Self::Segments(segments) = self {
            let mut knots = segments.suffix.clone();
            knots[0] = 1.0;
            return knots;
        }
        self.probability_knots()
            .into_iter()
            .map(|probability| 1.0 - probability)
            .collect()
    }

    pub(crate) fn support(&self) -> (Tail, Tail) {
        match self {
            Self::Histogram(histogram) => (
                Tail::Finite(
                    histogram.edges[histogram
                        .normalized
                        .iter()
                        .position(|p| *p > 0.0)
                        .expect("positive mass")],
                ),
                Tail::Finite(
                    histogram.edges[histogram
                        .normalized
                        .iter()
                        .rposition(|p| *p > 0.0)
                        .expect("positive mass")
                        + 1],
                ),
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
            Self::Segments(segments) => (
                Tail::Finite(segments.segments[0].0),
                Tail::Finite(segments.segments.last().unwrap().1),
            ),
        }
    }

    pub(crate) fn affine_mean(&self, slope: f64, intercept: f64) -> Option<f64> {
        use crate::DistributionFamily::{GenExtreme, GenPareto, GumbelLeft, GumbelRight};
        let Self::Fitted(distribution) = self else {
            return None;
        };
        if matches!(
            distribution.family,
            GenExtreme | GenPareto | GumbelLeft | GumbelRight
        ) && fitted_mean_is_finite(distribution)
        {
            let mean = affine_fitted_mean_deviation(distribution, slope, intercept).0;
            mean.is_finite().then_some(mean)
        } else {
            None
        }
    }

    pub(crate) fn mean_is_finite(&self) -> bool {
        match self {
            Self::Histogram(_)
            | Self::Discrete(_)
            | Self::PointMass(_)
            | Self::Tabulated(_)
            | Self::Empirical(_)
            | Self::Segments(_) => true,
            Self::Fitted(distribution) => fitted_mean_is_finite(distribution),
            Self::Hurdle(distribution) => fitted_mean_is_finite(distribution.base()),
        }
    }
}

fn hurdle_tail_lower_bound(distribution: &HurdleDistribution) -> ContractResult<f64> {
    distribution
        .base()
        .ppf(0.0)
        .map(|lower| lower.max(distribution.atom_location()))
        .map_err(distribution_error)
}

fn tabulated_inverse_survival(
    distribution: &TabulatedDistribution,
    survival: f64,
    upper: bool,
) -> f64 {
    let q = distribution.probabilities();
    let x = distribution.values();
    for index in (0..q.len() - 1).rev() {
        let mass = q[index + 1] - q[index];
        let right = 1.0 - q[index];
        if mass > 0.0 && (survival < right || upper && survival == right) {
            let weight = ((survival - (1.0 - q[index + 1])) / mass).clamp(0.0, 1.0);
            return linear_interpolate(x[index + 1], x[index], weight);
        }
    }
    x[0]
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
    let (lower, upper) = distribution.support();
    let tail = |value| match value {
        f64::NEG_INFINITY => Tail::NegativeInfinity,
        f64::INFINITY => Tail::PositiveInfinity,
        _ => Tail::Finite(value),
    };
    (tail(lower), tail(upper))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::{segment_inverse_survival, segment_quantile};

    #[test]
    fn indexed_segments_match_scans_at_gaps_atoms_and_tiny_tails() {
        let segments = vec![
            (0.0, 1.0, 0.125),
            (1.0, 1.0, 0.125),
            (1.0, 1.0, 0.25),
            (2.0, 3.0, 0.5),
            (4.0, 4.0, 1e-21),
        ];
        let indexed = WeightedSegments::new(segments.clone());
        for rank in [0.0, 1e-22, 1e-21, 0.125, 0.25, 0.5, 0.75, 1.0] {
            for upper in [false, true] {
                assert_eq!(
                    indexed.quantile(rank, upper),
                    segment_quantile(&segments, rank, upper),
                );
                assert_eq!(
                    indexed.inverse_survival(rank, upper),
                    segment_inverse_survival(&segments, rank, upper),
                );
            }
        }
        let thresholds = [
            f64::NEG_INFINITY,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            4.0,
            f64::INFINITY,
        ];
        for (index, &x) in thresholds.iter().enumerate() {
            for closed in [false, true] {
                assert_eq!(
                    indexed.cdf(x, closed),
                    segment_probability(&segments, f64::NEG_INFINITY, x, false, closed),
                );
                assert_eq!(
                    indexed.survival(x, closed),
                    segment_probability(&segments, x, f64::INFINITY, closed, false),
                );
                for &y in &thresholds[index..] {
                    for end_closed in [false, true] {
                        assert_eq!(
                            indexed.probability(x, y, closed, end_closed),
                            segment_probability(&segments, x, y, closed, end_closed),
                        );
                    }
                }
            }
        }
        assert_eq!(indexed.survival(3.0, false), 1e-21);
    }
}

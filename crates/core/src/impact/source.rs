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
        if !(original_mass_sum > 0.0) || (original_mass_sum - 1.0).abs() > MASS_TOLERANCE {
            return Err(fail(
                "invalid_histogram",
                "/weights",
                "total mass must be one within 1e-12",
            ));
        }
        let normalization_factor = 1.0 / original_mass_sum;
        let normalized = weights
            .iter()
            .map(|weight| weight / original_mass_sum)
            .collect();
        Ok(Self {
            edges,
            weights,
            normalized,
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
        let mut cumulative = 0.0;
        for index in 0..self.normalized.len() {
            let right = self.edges[index + 1];
            if x >= right {
                cumulative += self.normalized[index];
            } else if x > self.edges[index] {
                let width = right - self.edges[index];
                return cumulative + self.normalized[index] * (x - self.edges[index]) / width;
            }
        }
        cumulative
    }

    fn quantile(&self, probability: f64) -> f64 {
        if probability <= 0.0 {
            return self.edges[0];
        }
        if probability >= 1.0 {
            return *self.edges.last().expect("histogram has edges");
        }
        let mut cumulative = 0.0;
        for index in 0..self.normalized.len() {
            let next = cumulative + self.normalized[index];
            if self.normalized[index] > 0.0 && probability <= next {
                let weight = (probability - cumulative) / self.normalized[index];
                let left = self.edges[index];
                return left + weight * (self.edges[index + 1] - left);
            }
            cumulative = next;
        }
        *self.edges.last().expect("histogram has edges")
    }

    fn knots(&self) -> Vec<f64> {
        let mut knots = Vec::with_capacity(self.normalized.len() + 1);
        let mut cumulative = 0.0;
        knots.push(0.0);
        for weight in &self.normalized {
            cumulative += weight;
            knots.push(cumulative);
        }
        *knots.last_mut().expect("knots exist") = 1.0;
        knots
    }
}

/// Finite support with explicit probability atoms.
#[derive(Clone, Debug)]
pub struct DiscreteDistribution {
    pairs: Vec<(f64, f64)>,
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
        if !(total > 0.0) || (total - 1.0).abs() > MASS_TOLERANCE {
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
        Ok(Self { pairs: merged })
    }

    pub fn support(&self) -> impl Iterator<Item = f64> + '_ {
        self.pairs.iter().map(|pair| pair.0)
    }

    pub fn probabilities(&self) -> impl Iterator<Item = f64> + '_ {
        self.pairs.iter().map(|pair| pair.1)
    }

    fn quantile(&self, probability: f64) -> f64 {
        let mut cumulative = 0.0;
        for (location, mass) in &self.pairs {
            cumulative += mass;
            if probability <= cumulative {
                return *location;
            }
        }
        self.pairs.last().expect("discrete support is non-empty").0
    }

    fn knots(&self) -> Vec<f64> {
        let mut knots = vec![0.0];
        let mut cumulative = 0.0;
        for (_, mass) in &self.pairs {
            cumulative += mass;
            knots.push(cumulative);
        }
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

    pub(crate) fn cdf(&self, x: f64) -> f64 {
        match self {
            Self::Histogram(histogram) => histogram.cdf_below(x),
            Self::Discrete(discrete) => neumaier(
                discrete
                    .pairs
                    .iter()
                    .filter(|(location, _)| *location <= x)
                    .map(|(_, mass)| *mass),
            ),
            Self::PointMass(distribution) => distribution.cdf(x),
            Self::Tabulated(distribution) => distribution.cdf(x),
            Self::Fitted(distribution) => distribution.cdf(x),
            Self::Hurdle(distribution) => distribution.cdf(x),
            Self::Empirical(distribution) => distribution.cdf(x),
        }
    }

    pub(crate) fn cdf_below(&self, x: f64) -> f64 {
        match self {
            Self::Histogram(histogram) => histogram.cdf_below(x),
            Self::Discrete(discrete) => neumaier(
                discrete
                    .pairs
                    .iter()
                    .filter(|(location, _)| *location < x)
                    .map(|(_, mass)| *mass),
            ),
            Self::PointMass(distribution) => distribution.cdf_below(x),
            Self::Tabulated(distribution) => distribution.cdf_below(x),
            Self::Fitted(distribution) => distribution.cdf_below(x),
            Self::Hurdle(distribution) => distribution.cdf_below(x),
            Self::Empirical(distribution) => distribution.cdf_below(x),
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
                let mut atoms = Vec::new();
                let mut index = 0;
                while index < samples.len() {
                    let location = samples[index];
                    let mut count = 1;
                    while index + count < samples.len() && samples[index + count] == location {
                        count += 1;
                    }
                    if count > 1 {
                        atoms.push((location, count as f64 / samples.len() as f64));
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
    use crate::DistributionFamily::{GumbelLeft, GumbelRight};
    if !fitted_mean_is_finite(distribution) {
        return None;
    }
    match distribution.family {
        GumbelRight => Some(distribution.location + distribution.scale * EULER),
        GumbelLeft => Some(distribution.location - distribution.scale * EULER),
        _ => None,
    }
}

fn fitted_mean_is_finite(distribution: &FittedDistribution) -> bool {
    use crate::DistributionFamily::{GenExtreme, GenPareto};
    match distribution.family {
        GenPareto | GenExtreme => distribution.shape.expect("validated shape") < 1.0,
        _ => true,
    }
}

//! Distribution of `Y = f(X)` for a certified monotone impact.
//!
//! Continuous responses use `f(Q_X(p))` or `f(Q_X(1-p))`. Piecewise-constant
//! responses become finite atoms and use the generalized inverse, so a tie on
//! a jump returns the lower value. Flats of a continuous response are reported
//! as atoms without replacing that quantile rule.

use crate::Interpolation;
use crate::numerical::{compensated, segment_moment};

use super::definition::{
    Clamp, Interpolation as TableInterpolation, Model, OriginAdjustment, OutsideDomain,
};
use super::error::{ContractResult, fail};
use super::kernel;
use super::quadrature::neumaier;
use super::source::{ImpactSource, Tail};

pub const ALGORITHM: &str = "crc-pushforward/1";
const MASS_COVER: f64 = 1e-12;
const ATOM_MASS_TOLERANCE: f64 = 1e-9;

/// An atom of the pushforward distribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpactAtom {
    pub value: f64,
    pub probability: f64,
}

#[derive(Clone, Debug)]
enum Law {
    Discrete,
    Continuous {
        decreasing: bool,
    },
    /// Response is `floor` on `(-inf, threshold)` and the continuous branch after it.
    LowerThreshold {
        cut: f64,
        survival_cut: f64,
        floor: f64,
        decreasing: bool,
    },
}

/// Native pushforward. It keeps the definition and source instead of tabulating them.
#[derive(Clone, Debug)]
pub struct ImpactDistribution {
    definition_hash: String,
    model: Model,
    clamp: Clamp,
    source: ImpactSource,
    atoms: Vec<ImpactAtom>,
    law: Law,
}

impl ImpactDistribution {
    pub fn definition_hash(&self) -> &str {
        &self.definition_hash
    }

    pub fn algorithm(&self) -> &'static str {
        ALGORITHM
    }

    pub fn atoms(&self) -> &[ImpactAtom] {
        &self.atoms
    }

    pub(crate) fn is_discrete(&self) -> bool {
        matches!(self.law, Law::Discrete)
    }

    pub fn quantile(&self, probability: f64) -> ContractResult<f64> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(fail(
                "invalid_value",
                "",
                "probability must be finite and in [0, 1]",
            ));
        }
        match &self.law {
            Law::Discrete => Ok(discrete_quantile(&self.atoms, probability)),
            Law::Continuous { decreasing } => {
                if probability == 0.0 {
                    return self.essential(true);
                }
                if probability == 1.0 {
                    return self.essential(false);
                }
                // A decreasing map reverses order: `Q_Y(p) = f(Q_X^+(1 - p))`
                // with the upper source quantile, so a zero-mass gap in the
                // source resolves to the lower response value.
                let exposure = if *decreasing {
                    self.source.upper_quantile_from_survival(probability)?
                } else {
                    self.source.quantile(probability)?
                };
                kernel::evaluate(&self.model, &self.clamp, exposure)
            }
            Law::LowerThreshold { cut, floor, .. } => {
                if probability == 0.0 {
                    return self.essential(true);
                }
                if probability == 1.0 {
                    return self.essential(false);
                }
                if probability <= *cut {
                    Ok(*floor)
                } else {
                    let exposure = self.source.quantile(probability)?;
                    kernel::evaluate(&self.model, &self.clamp, exposure)
                }
            }
        }
    }

    pub fn quantiles(&self, probabilities: &[f64]) -> ContractResult<Vec<f64>> {
        probabilities
            .iter()
            .map(|probability| self.quantile(*probability))
            .collect()
    }

    /// Integrate the quantile law, keeping exact atomic slabs and all
    /// source/response breakpoints. Used for moments and rank-tail statistics.
    pub fn integral(&self, lo: f64, hi: f64, center: f64, square: bool) -> ContractResult<f64> {
        if !lo.is_finite()
            || !hi.is_finite()
            || lo < 0.0
            || hi > 1.0
            || lo > hi
            || !center.is_finite()
        {
            return Err(fail(
                "invalid_value",
                "",
                "invalid quantile integration interval",
            ));
        }
        if lo == hi {
            return Ok(0.0);
        }
        let power = |value: f64| {
            let difference = value - center;
            if square {
                difference * difference
            } else {
                difference
            }
        };
        if matches!(self.law, Law::Discrete) {
            let mut start = 0.0;
            let mut contributions = Vec::new();
            for atom in &self.atoms {
                let end = start + atom.probability;
                let width = if lo <= start && hi >= end {
                    atom.probability
                } else {
                    (hi.min(end) - lo.max(start)).max(0.0)
                };
                contributions.push(segment_moment(
                    width, atom.value, atom.value, center, square,
                ));
                start = end;
            }
            return Ok(compensated(contributions.into_iter()));
        }
        let decreasing = matches!(self.law, Law::Continuous { decreasing: true });
        let source_atoms: Vec<(f64, f64, f64, f64)> = self
            .source
            .point_atoms()
            .into_iter()
            .map(|(x, mass)| {
                let (left, right) = (self.source.cdf_below(x), self.source.cdf(x));
                if decreasing {
                    (1.0 - right, 1.0 - left, mass, x)
                } else {
                    (left, right, mass, x)
                }
            })
            .collect();
        let reflect = |q| if decreasing { 1.0 - q } else { q };
        let mut breaks = vec![lo, hi];
        breaks.extend(self.source.probability_knots().into_iter().map(reflect));
        for exposure in super::expectation::response_breakpoints(&self.model, &self.clamp) {
            breaks.push(reflect(self.source.cdf(exposure)));
            breaks.push(reflect(self.source.cdf_below(exposure)));
        }
        for &(left, right, _, _) in &source_atoms {
            breaks.extend([left, right]);
        }
        breaks.retain(|q| q.is_finite() && *q >= lo && *q <= hi);
        breaks.sort_by(f64::total_cmp);
        breaks.dedup();
        let exact_linear = self.source.has_piecewise_linear_quantile()
            && matches!(
                self.model,
                Model::Linear { .. }
                    | Model::Tabulated { .. }
                    | Model::Binned { .. }
                    | Model::Cooling { .. }
            );
        let mut contributions = Vec::new();
        let mut numerical = Vec::new();
        for pair in breaks.windows(2) {
            let (left, right) = (pair[0], pair[1]);
            if right <= left {
                continue;
            }
            if let Some(&(atom_left, atom_right, mass, x)) = source_atoms
                .iter()
                .find(|&&(a, b, _, _)| a <= left && b >= right)
            {
                let width = if left == atom_left && right == atom_right {
                    mass
                } else {
                    right - left
                };
                let response = kernel::evaluate(&self.model, &self.clamp, x)?;
                contributions.push(segment_moment(width, response, response, center, square));
            } else if exact_linear {
                let width = right - left;
                let first = self.quantile(left + width * 0.25)? - center;
                let second = self.quantile(left + width * 0.75)? - center;
                let (a, b) = (1.5 * first - 0.5 * second, 1.5 * second - 0.5 * first);
                contributions.push(
                    width
                        * if square {
                            (a * a + a * b + b * b) / 3.0
                        } else {
                            (a + b) / 2.0
                        },
                );
            } else {
                numerical.push((left, right));
            }
        }
        if !numerical.is_empty() {
            let options = super::quadrature::GaussKronrod::defaults();
            let integral = super::quadrature::integrate(&numerical, &options, |q| {
                Ok(power(self.quantile(q)?))
            })?;
            contributions.push(integral.value);
        }
        Ok(compensated(contributions.into_iter()))
    }

    /// Quantile addressed directly by its upper-tail survival probability.
    pub fn inverse_survival(&self, survival: f64) -> ContractResult<f64> {
        if !survival.is_finite() || !(0.0..=1.0).contains(&survival) {
            return Err(fail(
                "invalid_value",
                "",
                "survival must be finite and in [0,1]",
            ));
        }
        if survival == 0.0 {
            return self.essential(false);
        }
        if survival == 1.0 {
            return self.essential(true);
        }
        if let Law::LowerThreshold {
            survival_cut,
            floor,
            ..
        } = self.law
        {
            if survival >= survival_cut {
                return Ok(floor);
            }
        }
        if matches!(self.law, Law::Discrete) {
            let mut consumed = 0.0;
            for atom in self.atoms.iter().rev() {
                if survival < consumed + atom.probability {
                    return Ok(atom.value);
                }
                consumed += atom.probability;
            }
            return Ok(self.atoms[0].value);
        }
        let decreasing = matches!(self.law, Law::Continuous { decreasing: true });
        let exposure = if decreasing {
            self.source.upper_quantile(survival)?
        } else {
            self.source.inverse_survival(survival)?
        };
        kernel::evaluate(&self.model, &self.clamp, exposure)
    }

    pub(crate) fn inverse_log_survival(&self, logarithm: f64) -> ContractResult<f64> {
        if logarithm >= f64::MIN_POSITIVE.ln()
            || logarithm == f64::NEG_INFINITY
            || self.is_discrete()
        {
            return self.inverse_survival(logarithm.exp());
        }
        let decreasing = matches!(self.law, Law::Continuous { decreasing: true });
        let exposure = self.source.inverse_log_tail(logarithm, decreasing)?;
        if exposure.is_infinite() {
            return Ok(limit_at_infinity(&self.model, &self.clamp, exposure > 0.0));
        }
        kernel::evaluate(&self.model, &self.clamp, exposure)
    }

    /// Query the lower or upper tail directly, with explicit treatment of atoms.
    #[doc(hidden)]
    pub fn threshold_probability(
        &self,
        value: f64,
        inclusive: bool,
        exceedance: bool,
    ) -> ContractResult<f64> {
        self.threshold(value, inclusive, exceedance, false)
    }

    pub(crate) fn log_exceedance(&self, value: f64, inclusive: bool) -> ContractResult<f64> {
        self.threshold(value, inclusive, true, true)
    }

    fn threshold(
        &self,
        value: f64,
        inclusive: bool,
        exceedance: bool,
        logarithmic: bool,
    ) -> ContractResult<f64> {
        let coordinate = |mass: f64| if logarithmic { mass.ln() } else { mass };
        let matches = |response: f64| match (exceedance, inclusive) {
            (true, true) => response >= value,
            (true, false) => response > value,
            (false, true) => response <= value,
            (false, false) => response < value,
        };
        if self.is_discrete() {
            return Ok(coordinate(neumaier(
                self.atoms
                    .iter()
                    .filter(|atom| matches(atom.value))
                    .map(|atom| atom.probability),
            )));
        }
        let (lower, upper) = (self.essential(true)?, self.essential(false)?);
        if lower == upper || value < lower || value > upper {
            return Ok(coordinate(f64::from(matches(lower))));
        }
        let atom_mass = self
            .atoms
            .iter()
            .find(|atom| atom.value == value)
            .map_or(0.0, |atom| atom.probability);
        let at_bound = value == lower || value == upper;
        if at_bound {
            let outside_tail = if exceedance {
                value == upper
            } else {
                value == lower
            };
            if outside_tail {
                if logarithmic && inclusive && atom_mass < f64::MIN_POSITIVE {
                    // A flat at the endpoint can retain positive mass even
                    // when the unconditioned atom cannot be stored linearly.
                    if let Some((cut, increasing)) = self.inverse_cut(value)
                        && (self.clamp.maximum == Some(value) || self.clamp.minimum == Some(value))
                    {
                        return Ok(self.source.log_tail_at(cut, true, increasing != exceedance));
                    }
                    for region in constant_regions(&self.model, &self.clamp)? {
                        if region.y != value {
                            continue;
                        }
                        if region.start == f64::NEG_INFINITY {
                            return Ok(self.source.log_tail_at(
                                region.end,
                                region.end_closed,
                                true,
                            ));
                        }
                        if region.end == f64::INFINITY {
                            return Ok(self.source.log_tail_at(
                                region.start,
                                region.start_closed,
                                false,
                            ));
                        }
                    }
                }
                return Ok(coordinate(if inclusive { atom_mass } else { 0.0 }));
            }
            if inclusive || atom_mass == 0.0 {
                // A rounded-to-zero atom cannot have a small complement.
                return Ok(coordinate(1.0));
            }
            // A boundary atom can round to one even when its complementary
            // tail is positive. Measure the strict tail in source coordinates.
        }

        if let Some((mut threshold, increasing)) = self.inverse_cut(value) {
            let lower_tail = increasing != exceedance;
            if atom_mass > 0.0 {
                // A rounded response can invert to either side of an exact source atom.
                let endpoint = if lower_tail == inclusive {
                    f64::max
                } else {
                    f64::min
                };
                let mut boundary = None;
                for (location, _) in self.source.point_atoms() {
                    if kernel::evaluate(&self.model, &self.clamp, location)? == value {
                        boundary = Some(boundary.map_or(location, |x| endpoint(x, location)));
                    }
                }
                if let Some(location) = boundary {
                    threshold = if at_bound {
                        // A clamp can place many source atoms on the bound.
                        // Keep the inverse cut unless an atom extends it.
                        endpoint(threshold, location)
                    } else {
                        location
                    };
                }
            }
            if logarithmic {
                return Ok(self.source.log_tail_at(threshold, inclusive, lower_tail));
            }
            return Ok(match (lower_tail, inclusive) {
                (true, true) => self.source.cdf(threshold),
                (true, false) => self.source.cdf_below(threshold),
                (false, _) => self.source.survival_at(threshold, inclusive),
            });
        }

        if logarithmic {
            debug_assert!(exceedance);
            // Invert in log-probability space, including tails too small for
            // the ordinary quantile search to reach.
            let (mut left, mut right) = (-1.0, 0.0);
            while !matches(self.inverse_log_survival(left)?) {
                right = left;
                left *= 2.0;
                if !left.is_finite() {
                    return Ok(f64::NEG_INFINITY);
                }
            }
            for _ in 0..64 {
                let mid = left.midpoint(right);
                if mid == left || mid == right {
                    break;
                }
                if matches(self.inverse_log_survival(mid)?) {
                    left = mid;
                } else {
                    right = mid;
                }
            }
            return Ok(left);
        }

        // Positive binary64 bit patterns are ordered by value. Searching them
        // keeps even subnormal tails reachable without subtracting from one.
        let (mut left, mut right) = (0, 1.0f64.to_bits());
        while right - left > 1 {
            let mid = left + (right - left) / 2;
            let probability = f64::from_bits(mid);
            let response = if exceedance {
                self.inverse_survival(probability)?
            } else {
                self.quantile(probability)?
            };
            if matches(response) {
                left = mid;
            } else {
                right = mid;
            }
        }
        Ok(f64::from_bits(left))
    }

    fn inverse_cut(&self, value: f64) -> Option<(f64, bool)> {
        match self.model {
            Model::Linear { slope, intercept } if slope != 0.0 => {
                Some(((value - intercept) / slope, slope > 0.0))
            }
            Model::Sigmoid {
                midpoint,
                steepness,
                offset,
                amplitude,
                origin_adjustment: OriginAdjustment::None,
                zero_below: None,
            } if amplitude != 0.0 && steepness != 0.0 => {
                let fraction = (value - offset) / amplitude;
                (fraction > 0.0 && fraction < 1.0).then(|| {
                    (
                        midpoint + (fraction.ln() - (-fraction).ln_1p()) / steepness,
                        (amplitude > 0.0) == (steepness > 0.0),
                    )
                })
            }
            _ => None,
        }
    }

    /// Quantile integration in direct survival coordinates.
    pub fn tail_integral(
        &self,
        s_min: f64,
        s_max: f64,
        center: f64,
        square: bool,
    ) -> ContractResult<f64> {
        if !s_min.is_finite()
            || !s_max.is_finite()
            || s_min < 0.0
            || s_max > 1.0
            || s_min > s_max
            || !center.is_finite()
        {
            return Err(fail(
                "invalid_value",
                "",
                "invalid survival integration interval",
            ));
        }
        if s_min == s_max {
            return Ok(0.0);
        }
        let power = |x: f64| {
            let x = x - center;
            if square { x * x } else { x }
        };
        if matches!(self.law, Law::Discrete) {
            let mut contributions = Vec::new();
            let mut consumed = 0.0;
            for atom in self.atoms.iter().rev() {
                let end = consumed + atom.probability;
                let width = if s_min <= consumed && s_max >= end {
                    atom.probability
                } else {
                    (s_max.min(end) - s_min.max(consumed)).max(0.0)
                };
                contributions.push(segment_moment(
                    width, atom.value, atom.value, center, square,
                ));
                consumed = end;
            }
            return Ok(compensated(contributions.into_iter()));
        }
        let decreasing = matches!(self.law, Law::Continuous { decreasing: true });
        let source_atoms: Vec<_> = self
            .source
            .point_atoms()
            .into_iter()
            .map(|(x, mass)| {
                let start = if decreasing {
                    self.source.cdf_below(x)
                } else {
                    self.source.survival_at(x, false)
                };
                (start, start + mass, mass, x)
            })
            .collect();
        let mut breaks = vec![s_min, s_max];
        for q in self.source.probability_knots() {
            breaks.push(if decreasing { q } else { 1.0 - q });
        }
        for x in super::expectation::response_breakpoints(&self.model, &self.clamp) {
            breaks.push(if decreasing {
                self.source.cdf(x)
            } else {
                self.source.survival_at(x, false)
            });
            breaks.push(if decreasing {
                self.source.cdf_below(x)
            } else {
                self.source.survival_at(x, true)
            });
        }
        for &(a, b, _, _) in &source_atoms {
            breaks.extend([a, b]);
        }
        breaks.retain(|s| s.is_finite() && *s >= s_min && *s <= s_max);
        breaks.sort_by(f64::total_cmp);
        breaks.dedup();
        let exact_linear = self.source.has_piecewise_linear_quantile()
            && matches!(
                self.model,
                Model::Linear { .. }
                    | Model::Tabulated { .. }
                    | Model::Binned { .. }
                    | Model::Cooling { .. }
            );
        let mut contributions = Vec::new();
        for pair in breaks.windows(2) {
            let (left, right) = (pair[0], pair[1]);
            if right <= left {
                continue;
            }
            if let Some(&(a, b, mass, x)) = source_atoms
                .iter()
                .find(|&&(a, b, _, _)| a <= left && b >= right)
            {
                let width = if left == a && right == b {
                    mass
                } else {
                    right - left
                };
                let response = kernel::evaluate(&self.model, &self.clamp, x)?;
                contributions.push(segment_moment(width, response, response, center, square));
            } else if exact_linear {
                let width = right - left;
                let first = self.inverse_survival(left + width * 0.25)? - center;
                let second = self.inverse_survival(left + width * 0.75)? - center;
                let (a, b) = (1.5 * first - 0.5 * second, 1.5 * second - 0.5 * first);
                contributions.push(
                    width
                        * if square {
                            (a * a + a * b + b * b) / 3.0
                        } else {
                            (a + b) / 2.0
                        },
                );
            } else {
                let integrated = super::quadrature::integrate(
                    &[(0.0, 1.0)],
                    &super::quadrature::GaussKronrod::defaults(),
                    |t| {
                        let width = right - left;
                        let (s, jac) = if left == 0.0 {
                            (width * t.powi(4), 4.0 * t.powi(3))
                        } else {
                            (left + width * t, 1.0)
                        };
                        Ok(power(self.inverse_survival(s)?) * jac)
                    },
                )?;
                contributions.push((right - left) * integrated.value);
            }
        }
        Ok(compensated(contributions.into_iter()))
    }

    /// Explicit approximate quantile table. The interpolation is part of the approximation.
    pub fn tabulate(
        &self,
        probabilities: &[f64],
        interpolation: Interpolation,
    ) -> ContractResult<crate::TabulatedDistribution> {
        if probabilities.len() < 2 {
            return Err(fail(
                "invalid_value",
                "/probabilities",
                "tabulation requires at least two probabilities",
            ));
        }
        if probabilities
            .iter()
            .any(|probability| !probability.is_finite() || !(0.0..=1.0).contains(probability))
            || probabilities.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(fail(
                "invalid_value",
                "/probabilities",
                "tabulation probabilities must be finite and strictly increasing in [0, 1]",
            ));
        }
        let values = self.quantiles(probabilities)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(fail(
                "unsupported_value",
                "/probabilities",
                "an infinite essential endpoint cannot be stored in a finite quantile table",
            ));
        }
        crate::TabulatedDistribution::new(probabilities.to_vec(), values, interpolation, false)
            .map_err(|error| fail("invalid_value", "/probabilities", error.to_string()))
    }

    fn essential(&self, infimum: bool) -> ContractResult<f64> {
        if let Law::Discrete = self.law {
            return Ok(discrete_quantile(
                &self.atoms,
                if infimum { 0.0 } else { 1.0 },
            ));
        }
        let decreasing = matches!(
            self.law,
            Law::Continuous { decreasing: true }
                | Law::LowerThreshold {
                    decreasing: true,
                    ..
                }
        );
        let at_lower_support = if decreasing { !infimum } else { infimum };
        let (lower, upper) = self.source.support();
        let tail = if at_lower_support { lower } else { upper };
        match tail {
            Tail::Finite(exposure) => kernel::evaluate(&self.model, &self.clamp, exposure),
            Tail::NegativeInfinity => Ok(limit_at_infinity(&self.model, &self.clamp, false)),
            Tail::PositiveInfinity => Ok(limit_at_infinity(&self.model, &self.clamp, true)),
        }
    }
}

pub(crate) fn transform(
    model: &Model,
    clamp: &Clamp,
    definition_hash: &str,
    source: &ImpactSource,
) -> ContractResult<ImpactDistribution> {
    if !source.is_full_probability_support() {
        return Err(fail(
            "missing_probability_support",
            "",
            "distribution transformation requires a full probability distribution",
        ));
    }
    // Any finite atomic source has an exact image, including under a
    // non-monotone response. Preserve every positive atom rather than sampling
    // the quantile law or discarding small probabilities.
    if source.has_piecewise_constant_quantile() {
        let mut atoms = Vec::new();
        for (location, probability) in source.point_atoms() {
            if probability > 0.0 {
                atoms.push(ImpactAtom {
                    value: kernel::evaluate(model, clamp, location)?,
                    probability,
                });
            }
        }
        return Ok(ImpactDistribution {
            definition_hash: definition_hash.to_string(),
            model: model.clone(),
            clamp: clamp.clone(),
            source: source.clone(),
            atoms: merge_atoms(atoms),
            law: Law::Discrete,
        });
    }
    // A piecewise-constant response also has an exact finite image; its bin
    // order need not be monotone.
    let piecewise_constant = matches!(
        model,
        Model::Binned { .. }
            | Model::Tabulated {
                interpolation: TableInterpolation::StepLeft | TableInterpolation::StepRight,
                ..
            }
    );
    let direction = kernel::monotonicity(model, clamp);
    let decreasing = match direction {
        "constant" => false,
        "non_decreasing" => false,
        "non_increasing" => true,
        "non_monotone" if piecewise_constant => false,
        "non_monotone" | "unknown" => {
            return Err(fail(
                "monotonicity_required",
                "",
                "distribution transformation requires a certified monotone function",
            ));
        }
        other => {
            return Err(fail(
                "monotonicity_required",
                "",
                format!("unrecognized monotonicity class {other}"),
            ));
        }
    };
    if super::expectation::restricted_domain_exited(model, source) {
        return Err(fail(
            "out_of_domain",
            "",
            "the source support leaves a table domain whose outside policy is error",
        ));
    }
    let mut regions = constant_regions(model, clamp)?;
    // Regions are built left to right. Two flat pieces that meet at a knot
    // must not both own it, or the knot's mass would be counted twice.
    for index in 1..regions.len() {
        let previous = regions[index - 1];
        let current = &mut regions[index];
        if previous.end_closed && current.start_closed && previous.end == current.start {
            current.start_closed = false;
        }
    }
    let mut atoms = Vec::new();
    for region in &regions {
        let mass = region.probability(source);
        if mass > 0.0 {
            atoms.push(ImpactAtom {
                value: region.y,
                probability: mass,
            });
        }
    }
    for (location, mass) in source.point_atoms() {
        if mass <= 0.0 || regions.iter().any(|region| region.contains(location)) {
            continue;
        }
        atoms.push(ImpactAtom {
            value: kernel::evaluate(model, clamp, location)?,
            probability: mass,
        });
    }
    let atoms = merge_atoms(atoms);
    let covered = neumaier(atoms.iter().map(|atom| atom.probability));
    if covered > 1.0 + ATOM_MASS_TOLERANCE {
        return Err(fail(
            "invalid_value",
            "",
            "pushforward atoms carry more than the total probability",
        ));
    }
    if piecewise_constant && covered < 1.0 - MASS_COVER {
        return Err(fail(
            "out_of_domain",
            "",
            "the source support leaves a piecewise-constant table domain",
        ));
    }
    // Nearly complete atom coverage is not complete coverage. A tiny remaining
    // continuous tail is scientifically meaningful and keeps its quantile law.
    let law = if piecewise_constant {
        Law::Discrete
    } else if let Some((cut, survival_cut, floor)) = lower_threshold(model, clamp, source)? {
        if direction == "non_decreasing" {
            Law::LowerThreshold {
                cut,
                survival_cut,
                floor,
                decreasing,
            }
        } else {
            Law::Continuous { decreasing }
        }
    } else {
        Law::Continuous { decreasing }
    };
    Ok(ImpactDistribution {
        definition_hash: definition_hash.to_string(),
        model: model.clone(),
        clamp: clamp.clone(),
        source: source.clone(),
        atoms,
        law,
    })
}

fn discrete_quantile(atoms: &[ImpactAtom], probability: f64) -> f64 {
    if atoms.is_empty() {
        return f64::NAN;
    }
    if probability <= 0.0 {
        return atoms[0].value;
    }
    if probability >= 1.0 {
        return atoms.last().expect("atoms are non-empty").value;
    }
    let mut cumulative = 0.0;
    for atom in atoms {
        cumulative += atom.probability;
        if probability <= cumulative {
            return atom.value;
        }
    }
    atoms.last().expect("atoms are non-empty").value
}

fn merge_atoms(atoms: Vec<ImpactAtom>) -> Vec<ImpactAtom> {
    let mut ordered = atoms;
    ordered.sort_by(|left, right| left.value.total_cmp(&right.value));
    let mut merged: Vec<ImpactAtom> = Vec::new();
    for atom in ordered {
        if merged.last().is_some_and(|last| last.value == atom.value) {
            merged.last_mut().expect("atom group exists").probability += atom.probability;
        } else {
            merged.push(atom);
        }
    }
    merged
}

#[derive(Clone, Copy)]
struct Region {
    start: f64,
    end: f64,
    start_closed: bool,
    end_closed: bool,
    y: f64,
}

impl Region {
    fn contains(self, x: f64) -> bool {
        let after_start = if self.start.is_infinite() {
            true
        } else if self.start_closed {
            x >= self.start
        } else {
            x > self.start
        };
        let before_end = if self.end.is_infinite() {
            true
        } else if self.end_closed {
            x <= self.end
        } else {
            x < self.end
        };
        after_start && before_end
    }

    fn probability(self, source: &ImpactSource) -> f64 {
        source.interval_probability(self.start, self.end, self.start_closed, self.end_closed)
    }
}

fn constant_regions(model: &Model, clamp: &Clamp) -> ContractResult<Vec<Region>> {
    let mut regions = Vec::new();
    match model {
        Model::Linear { slope, intercept } => {
            linear_regions(*slope, *intercept, clamp, &mut regions)?;
        }
        Model::Tabulated {
            points,
            interpolation,
            outside_domain,
        } => match interpolation {
            TableInterpolation::Linear => {
                tabulated_linear_regions(points, *outside_domain, clamp, &mut regions)?;
            }
            TableInterpolation::StepLeft => {
                step_regions(points, *outside_domain, clamp, true, &mut regions)?;
            }
            TableInterpolation::StepRight => {
                step_regions(points, *outside_domain, clamp, false, &mut regions)?;
            }
        },
        Model::Binned {
            edges,
            values,
            outside_domain,
        } => binned_regions(edges, values, *outside_domain, clamp, &mut regions)?,
        Model::Sigmoid { zero_below, .. } => {
            if let Some(threshold) = zero_below {
                // The raw response is exactly zero below the threshold.
                let floor = apply_clamp(clamp, 0.0);
                regions.push(Region {
                    start: f64::NEG_INFINITY,
                    end: *threshold,
                    start_closed: false,
                    end_closed: false,
                    y: floor,
                });
            }
        }
        Model::Cooling { .. } => {
            regions.push(Region {
                start: f64::NEG_INFINITY,
                end: 1.0,
                start_closed: false,
                end_closed: true,
                y: kernel::evaluate(model, clamp, 1.0)?,
            });
        }
        Model::Spei { .. } => {
            let level = kernel::evaluate(model, clamp, 0.0)?;
            regions.push(Region {
                start: 0.0,
                end: f64::INFINITY,
                start_closed: true,
                end_closed: false,
                y: level,
            });
        }
        Model::Crop { .. } => {}
    }
    Ok(regions)
}

fn linear_regions(
    slope: f64,
    intercept: f64,
    clamp: &Clamp,
    regions: &mut Vec<Region>,
) -> ContractResult<()> {
    let model = Model::Linear { slope, intercept };
    if slope == 0.0 {
        regions.push(Region {
            start: f64::NEG_INFINITY,
            end: f64::INFINITY,
            start_closed: false,
            end_closed: false,
            y: kernel::evaluate(&model, clamp, 0.0)?,
        });
        return Ok(());
    }
    let mut cuts = Vec::new();
    for limit in [clamp.minimum, clamp.maximum].into_iter().flatten() {
        cuts.push((limit - intercept) / slope);
    }
    cuts.retain(|cut| cut.is_finite());
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|left, right| *left == *right);
    let mut bounds = vec![f64::NEG_INFINITY];
    bounds.extend(cuts);
    bounds.push(f64::INFINITY);
    for pair in bounds.windows(2) {
        let (first, second) = probes(pair[0], pair[1]);
        let y1 = kernel::evaluate(&model, clamp, first)?;
        let y2 = kernel::evaluate(&model, clamp, second)?;
        if y1 == y2 {
            regions.push(Region {
                start: pair[0],
                end: pair[1],
                start_closed: pair[0].is_finite(),
                end_closed: false,
                y: y1,
            });
        }
    }
    Ok(())
}

fn probes(start: f64, end: f64) -> (f64, f64) {
    if start.is_finite() && end.is_finite() {
        let width = end - start;
        (start + width * 0.25, start + width * 0.75)
    } else if start.is_finite() {
        (start, start + 1.0)
    } else if end.is_finite() {
        (end - 1.0, end - 2.0)
    } else {
        (0.0, 1.0)
    }
}

fn tabulated_linear_regions(
    points: &[(f64, f64)],
    outside: OutsideDomain,
    clamp: &Clamp,
    regions: &mut Vec<Region>,
) -> ContractResult<()> {
    if outside == OutsideDomain::Clamp {
        regions.push(Region {
            start: f64::NEG_INFINITY,
            end: points[0].0,
            start_closed: false,
            end_closed: false,
            y: apply_clamp(clamp, points[0].1),
        });
    }
    for pair in points.windows(2) {
        let (x0, y0) = pair[0];
        let (x1, y1) = pair[1];
        let mut cuts = vec![x0, x1];
        if y0 != y1 {
            for limit in [clamp.minimum, clamp.maximum].into_iter().flatten() {
                let low = y0.min(y1);
                let high = y0.max(y1);
                if limit > low && limit < high {
                    cuts.push(x0 + (x1 - x0) * (limit - y0) / (y1 - y0));
                }
            }
        }
        cuts.sort_by(f64::total_cmp);
        cuts.dedup_by(|left, right| *left == *right);
        for window in cuts.windows(2) {
            let mid = 0.5 * (window[0] + window[1]);
            let y_mid = apply_clamp(clamp, lerp(x0, y0, x1, y1, mid));
            let y_left = apply_clamp(clamp, lerp(x0, y0, x1, y1, window[0]));
            if y_left == y_mid {
                regions.push(Region {
                    start: window[0],
                    end: window[1],
                    start_closed: true,
                    end_closed: window[1] == x1,
                    y: y_left,
                });
            }
        }
    }
    if outside == OutsideDomain::Clamp {
        let last = points.last().expect("tabulated curve has points");
        regions.push(Region {
            start: last.0,
            end: f64::INFINITY,
            start_closed: false,
            end_closed: false,
            y: apply_clamp(clamp, last.1),
        });
    }
    Ok(())
}

fn step_regions(
    points: &[(f64, f64)],
    outside: OutsideDomain,
    clamp: &Clamp,
    step_left: bool,
    regions: &mut Vec<Region>,
) -> ContractResult<()> {
    let first = points[0];
    let last = *points.last().expect("step curve has points");
    if outside == OutsideDomain::Clamp {
        regions.push(Region {
            start: f64::NEG_INFINITY,
            end: first.0,
            start_closed: false,
            end_closed: false,
            y: apply_clamp(clamp, first.1),
        });
    }
    for pair in points.windows(2) {
        let y = if step_left { pair[0].1 } else { pair[1].1 };
        regions.push(Region {
            start: pair[0].0,
            end: pair[1].0,
            start_closed: step_left || pair[0].0 == first.0,
            end_closed: !step_left,
            y: apply_clamp(clamp, y),
        });
    }
    if outside == OutsideDomain::Clamp {
        regions.push(Region {
            start: last.0,
            end: f64::INFINITY,
            start_closed: false,
            end_closed: false,
            y: apply_clamp(clamp, last.1),
        });
    } else if step_left {
        regions.push(Region {
            start: last.0,
            end: last.0,
            start_closed: true,
            end_closed: true,
            y: apply_clamp(clamp, last.1),
        });
    }
    Ok(())
}

fn binned_regions(
    edges: &[f64],
    values: &[f64],
    outside: OutsideDomain,
    clamp: &Clamp,
    regions: &mut Vec<Region>,
) -> ContractResult<()> {
    let first = edges[0];
    if values.len() == 1 && outside == OutsideDomain::Clamp {
        regions.push(Region {
            start: f64::NEG_INFINITY,
            end: f64::INFINITY,
            start_closed: false,
            end_closed: false,
            y: apply_clamp(clamp, values[0]),
        });
        return Ok(());
    }
    if outside == OutsideDomain::Clamp {
        regions.push(Region {
            start: f64::NEG_INFINITY,
            end: edges[1],
            start_closed: false,
            end_closed: false,
            y: apply_clamp(clamp, values[0]),
        });
    } else {
        regions.push(Region {
            start: first,
            end: edges[1],
            start_closed: true,
            end_closed: false,
            y: apply_clamp(clamp, values[0]),
        });
    }
    for index in 1..values.len() {
        let end = edges[index + 1];
        let last_bin = index + 1 == values.len();
        regions.push(Region {
            start: edges[index],
            end: if last_bin && outside == OutsideDomain::Clamp {
                f64::INFINITY
            } else {
                end
            },
            start_closed: true,
            end_closed: last_bin && outside == OutsideDomain::Error,
            y: apply_clamp(clamp, values[index]),
        });
    }
    Ok(())
}

fn lower_threshold(
    model: &Model,
    clamp: &Clamp,
    source: &ImpactSource,
) -> ContractResult<Option<(f64, f64, f64)>> {
    let Model::Sigmoid {
        zero_below: Some(threshold),
        ..
    } = model
    else {
        return Ok(None);
    };
    let floor = apply_clamp(clamp, 0.0);
    let at_threshold = kernel::evaluate(model, clamp, *threshold)?;
    if floor == at_threshold {
        return Ok(None);
    }
    Ok(Some((
        source.cdf_below(*threshold),
        source.survival_at(*threshold, true),
        floor,
    )))
}

fn lerp(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
    if x1 == x0 {
        y0
    } else {
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }
}

fn apply_clamp(clamp: &Clamp, mut value: f64) -> f64 {
    if let Some(minimum) = clamp.minimum {
        value = value.max(minimum);
    }
    if let Some(maximum) = clamp.maximum {
        value = value.min(maximum);
    }
    if value == 0.0 { 0.0 } else { value }
}

fn limit_at_infinity(model: &Model, clamp: &Clamp, positive: bool) -> f64 {
    let value = match model {
        Model::Linear { slope, intercept } => {
            if *slope == 0.0 {
                *intercept
            } else if (*slope > 0.0) == positive {
                return clamp.maximum.unwrap_or(f64::INFINITY);
            } else {
                return clamp.minimum.unwrap_or(f64::NEG_INFINITY);
            }
        }
        Model::Sigmoid {
            steepness,
            offset,
            amplitude,
            origin_adjustment,
            zero_below,
            midpoint,
        } => {
            if !positive && zero_below.is_some() {
                0.0
            } else {
                let mut unit = if *steepness == 0.0 {
                    0.5
                } else if (*steepness > 0.0) == positive {
                    1.0
                } else {
                    0.0
                };
                if *origin_adjustment == super::definition::OriginAdjustment::SubtractAtZero {
                    let at_zero = if *steepness == 0.0 {
                        0.5
                    } else {
                        kernel_logistic(*steepness * (0.0 - midpoint))
                    };
                    unit -= at_zero;
                }
                offset + amplitude * unit
            }
        }
        Model::Tabulated { points, .. } => {
            if positive {
                points.last().expect("points").1
            } else {
                points[0].1
            }
        }
        Model::Binned { values, .. } => {
            if positive {
                *values.last().expect("values")
            } else {
                values[0]
            }
        }
        Model::Cooling { cooling_cost, .. } => {
            if !positive || *cooling_cost == 0.0 {
                return apply_clamp(clamp, 0.0);
            }
            return clamp.maximum.unwrap_or(f64::INFINITY);
        }
        Model::Spei {
            water_consumption_asset,
            original_water_unit_cost,
        } => {
            if positive {
                original_water_unit_cost * water_consumption_asset / 10_000_000.0
            } else {
                original_water_unit_cost * 11.0 * water_consumption_asset / 10_000_000.0
            }
        }
        Model::Crop { weights, index, .. } => {
            let mut finite = 0.0;
            let mut negative = false;
            for crop in 0..3 {
                if weights[crop] == 0.0 || index[crop] == 0.0 {
                    continue;
                }
                let grows = if positive {
                    index[crop] > 0.0
                } else {
                    index[crop] < 0.0
                };
                if grows {
                    negative = true;
                } else {
                    finite += weights[crop];
                }
            }
            if negative {
                return clamp.minimum.unwrap_or(f64::NEG_INFINITY);
            }
            finite
        }
    };
    apply_clamp(clamp, value)
}

fn kernel_logistic(value: f64) -> f64 {
    if value >= 0.0 {
        let exponential = (-value).exp();
        1.0 / (1.0 + exponential)
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

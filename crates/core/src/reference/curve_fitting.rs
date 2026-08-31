// Copyright (C) 2026 Riskthinking.AI
//
// This file contains Rust adaptations of numerical routines from SciPy,
// including coefficients and algorithms derived from the Cephes Math Library.
// Upstream portions remain subject to the BSD 3-Clause terms reproduced in
// THIRD_PARTY_NOTICES.md. Modified by Riskthinking.AI in 2026.
//
// Riskthinking.AI's modifications are licensed under AGPL-3.0-or-later.
// SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

use std::f64::consts::{PI, SQRT_2};

use crate::risk::climate::utility::get_hazard_limit;

// Numeric constants
const EPS: f64 = 2.220446049250313e-16;
const LOGXMAX: f64 = 709.782712893384; // approx ln(max double)
const EULER: f64 = 0.577215664901532860606512090082402431042; // -psi(1)
const SQRTPI: f64 = 1.772453850905516027298167483341145182797549456; // sqrt(pi)
const MAXGAM: f64 = 171.6243769563027; // Cephes overflow threshold for Gamma
const M_SQRT1_2: f64 = 0.707106781186547524400844362104849039284835938; // 1/sqrt(2)
const M_SQRT2: f64 = 1.414213562373095048801688724209698078569671875; // sqrt(2)

// General interface for a continuous distribution to evaluate
pub trait ContinuousDistribution {
    fn pdf(&self, x: f64) -> f64;
    fn cdf(&self, x: f64) -> f64;

    // Inverse CDF. Default implementation uses binary search on cdf
    fn ppf(&self, p: f64) -> f64 {
        // Guard inputs
        let q = p.clamp(0.0, 1.0);
        if q <= 0.0 {
            return self.support_lower();
        }
        if q >= 1.0 {
            return self.support_upper();
        }

        // Expand bounds until they bracket the desired quantile.
        let mut low = self.support_lower_bound_guess();
        let mut high = self.support_upper_bound_guess();
        while self.cdf(high) < q {
            high *= 2.0;
            if !high.is_finite() || high.abs() > 1e16 {
                break;
            }
        }
        while self.cdf(low) > q {
            low *= 2.0;
            if !low.is_finite() || low.abs() > 1e16 {
                break;
            }
        }

        // Binary search
        let mut lo = low.min(high);
        let mut hi = low.max(high);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            let cm = self.cdf(mid);
            if (cm - q).abs() < 1e-10 {
                return mid;
            }
            if cm < q {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    // Lower/upper soft support bounds for default ppf
    fn support_lower(&self) -> f64 {
        f64::NEG_INFINITY
    }
    fn support_upper(&self) -> f64 {
        f64::INFINITY
    }

    // Guesses to initialize ppf search. Defaults are generic.
    fn support_lower_bound_guess(&self) -> f64 {
        -1.0
    }
    fn support_upper_bound_guess(&self) -> f64 {
        1.0
    }
}

// Trait for types that can be fitted from sample data
pub trait FittableDistribution: Sized + ContinuousDistribution {
    fn fit(data: &[f64]) -> Option<Self>;
}

// Registry of supported distributions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistributionKind {
    GenExtreme,
    WeibullMin,
    WeibullMax,
    SkewNorm,
    GumbelR,
    GumbelL,
    GenPareto,
}

impl DistributionKind {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "genextreme" => Some(Self::GenExtreme),
            "weibull_min" => Some(Self::WeibullMin),
            "weibull_max" => Some(Self::WeibullMax),
            "skewnorm" => Some(Self::SkewNorm),
            "gumbel_r" => Some(Self::GumbelR),
            "gumbel_l" => Some(Self::GumbelL),
            "genpareto" => Some(Self::GenPareto),
            _ => None,
        }
    }
}

// Unified wrapper
#[derive(Debug, Clone, Copy)]
pub enum FittedDistribution {
    GenExtreme(GEV),
    WeibullMin(WeibullMin),
    WeibullMax(WeibullMax),
    SkewNorm(SkewNormal),
    GumbelR(Gumbel),
    GumbelL(GumbelL),
    GenPareto(GeneralizedPareto),
}

impl ContinuousDistribution for FittedDistribution {
    fn ppf(&self, p: f64) -> f64 {
        match self {
            Self::GenExtreme(d) => d.ppf(p),
            Self::WeibullMin(d) => d.ppf(p),
            Self::WeibullMax(d) => d.ppf(p),
            Self::SkewNorm(d) => d.ppf(p),
            Self::GumbelR(d) => d.ppf(p),
            Self::GumbelL(d) => d.ppf(p),
            Self::GenPareto(d) => d.ppf(p),
        }
    }
    fn pdf(&self, x: f64) -> f64 {
        match self {
            Self::GenExtreme(d) => d.pdf(x),
            Self::WeibullMin(d) => d.pdf(x),
            Self::WeibullMax(d) => d.pdf(x),
            Self::SkewNorm(d) => d.pdf(x),
            Self::GumbelR(d) => d.pdf(x),
            Self::GumbelL(d) => d.pdf(x),
            Self::GenPareto(d) => d.pdf(x),
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        match self {
            Self::GenExtreme(d) => d.cdf(x),
            Self::WeibullMin(d) => d.cdf(x),
            Self::WeibullMax(d) => d.cdf(x),
            Self::SkewNorm(d) => d.cdf(x),
            Self::GumbelR(d) => d.cdf(x),
            Self::GumbelL(d) => d.cdf(x),
            Self::GenPareto(d) => d.cdf(x),
        }
    }
    fn support_lower(&self) -> f64 {
        match self {
            Self::GenExtreme(d) => d.support_lower(),
            Self::WeibullMin(d) => d.support_lower(),
            Self::WeibullMax(d) => d.support_lower(),
            Self::SkewNorm(d) => d.support_lower(),
            Self::GumbelR(d) => d.support_lower(),
            Self::GumbelL(d) => d.support_lower(),
            Self::GenPareto(d) => d.support_lower(),
        }
    }
    fn support_upper(&self) -> f64 {
        match self {
            Self::GenExtreme(d) => d.support_upper(),
            Self::WeibullMin(d) => d.support_upper(),
            Self::WeibullMax(d) => d.support_upper(),
            Self::SkewNorm(d) => d.support_upper(),
            Self::GumbelR(d) => d.support_upper(),
            Self::GumbelL(d) => d.support_upper(),
            Self::GenPareto(d) => d.support_upper(),
        }
    }
}

// Fit a selected distribution to data using simple, robust estimators
pub fn fit_distribution(kind: DistributionKind, data: &[f64]) -> Option<FittedDistribution> {
    match kind {
        DistributionKind::GenExtreme => GEV::fit(data).map(FittedDistribution::GenExtreme),
        DistributionKind::WeibullMin => WeibullMin::fit(data).map(FittedDistribution::WeibullMin),
        DistributionKind::WeibullMax => WeibullMax::fit(data).map(FittedDistribution::WeibullMax),
        DistributionKind::SkewNorm => SkewNormal::fit(data).map(FittedDistribution::SkewNorm),
        DistributionKind::GumbelR => Gumbel::fit(data).map(FittedDistribution::GumbelR),
        DistributionKind::GumbelL => GumbelL::fit(data).map(FittedDistribution::GumbelL),
        DistributionKind::GenPareto => {
            GeneralizedPareto::fit(data).map(FittedDistribution::GenPareto)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct QuantileOptimization {
    pub shape: Option<f64>,
    pub loc: f64,
    pub scale: f64,
    pub converged: bool,
    pub iterations: usize,
    pub evaluations: usize,
}

#[derive(Debug, Clone)]
struct SimplexResult {
    parameters: Vec<f64>,
    objective: f64,
    converged: bool,
    iterations: usize,
    evaluations: usize,
}

pub(crate) fn fit_quantile_points(
    kind: DistributionKind,
    probabilities: &[f64],
    values: &[f64],
    weights: &[f64],
    value_scale: f64,
    truncation_atom: Option<f64>,
) -> Option<QuantileOptimization> {
    let starts = quantile_starts(kind, probabilities, values, weights);
    let parameter_count = if matches!(kind, DistributionKind::GumbelR | DistributionKind::GumbelL) {
        2
    } else {
        3
    };
    let value_range = values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - values.iter().copied().fold(f64::INFINITY, f64::min);
    let mut best: Option<SimplexResult> = None;

    for start in starts {
        let mut steps = vec![0.0; parameter_count];
        if parameter_count == 2 {
            steps[0] = 0.1 * value_range.max(value_scale);
            steps[1] = 0.1 * start[1].abs().max(value_scale);
        } else {
            steps[0] = match kind {
                DistributionKind::WeibullMin | DistributionKind::WeibullMax => {
                    0.15 * start[0].abs().max(1.0)
                }
                DistributionKind::SkewNorm => 0.5,
                _ => 0.15,
            };
            steps[1] = 0.1 * value_range.max(value_scale);
            steps[2] = 0.1 * start[2].abs().max(value_scale);
        }
        let result = quantile_nelder_mead(
            start,
            steps,
            |parameters| {
                quantile_objective(
                    kind,
                    parameters,
                    probabilities,
                    values,
                    weights,
                    value_scale,
                    truncation_atom,
                )
            },
            |parameters| constrain_quantile_parameters(kind, parameters),
            2_000,
            1.0e-8 * value_scale.max(1.0),
            1.0e-12,
        );
        if result.objective.is_finite()
            && best.as_ref().is_none_or(|candidate| {
                (result.converged && !candidate.converged)
                    || (result.converged == candidate.converged
                        && result.objective < candidate.objective)
            })
        {
            best = Some(result);
        }
    }

    let best = best?;
    let (shape, loc, scale) = if parameter_count == 2 {
        (None, best.parameters[0], best.parameters[1])
    } else {
        (
            Some(best.parameters[0]),
            best.parameters[1],
            best.parameters[2],
        )
    };
    Some(QuantileOptimization {
        shape,
        loc,
        scale,
        converged: best.converged,
        iterations: best.iterations,
        evaluations: best.evaluations,
    })
}

fn quantile_starts(
    kind: DistributionKind,
    probabilities: &[f64],
    values: &[f64],
    weights: &[f64],
) -> Vec<Vec<f64>> {
    let shapes: &[f64] = match kind {
        DistributionKind::GenExtreme => &[-0.5, -0.15, 0.0, 0.15, 0.5],
        DistributionKind::WeibullMin | DistributionKind::WeibullMax => &[0.7, 1.0, 2.0, 4.0],
        DistributionKind::SkewNorm => &[-5.0, -1.0, 0.0, 1.0, 5.0],
        DistributionKind::GenPareto => &[-0.5, -0.15, 0.0, 0.15, 0.5],
        DistributionKind::GumbelR | DistributionKind::GumbelL => &[0.0],
    };
    let mut starts = Vec::new();
    for &shape in shapes {
        let standardized: Option<Vec<f64>> = probabilities
            .iter()
            .map(|&probability| standardized_quantile(kind, shape, probability))
            .collect();
        let Some(standardized) = standardized else {
            continue;
        };
        if let Some((loc, scale)) = weighted_location_scale(&standardized, values, weights) {
            if matches!(kind, DistributionKind::GumbelR | DistributionKind::GumbelL) {
                starts.push(vec![loc, scale]);
            } else {
                starts.push(vec![shape, loc, scale]);
            }
        }
    }
    starts
}

fn standardized_quantile(kind: DistributionKind, shape: f64, probability: f64) -> Option<f64> {
    let distribution = FittedDistribution::from_kind_and_params(
        kind,
        (!matches!(kind, DistributionKind::GumbelR | DistributionKind::GumbelL)).then_some(shape),
        0.0,
        1.0,
    )?;
    let value = distribution.ppf(probability);
    value.is_finite().then_some(value)
}

fn weighted_location_scale(
    standardized: &[f64],
    values: &[f64],
    weights: &[f64],
) -> Option<(f64, f64)> {
    let weight_sum = weights.iter().sum::<f64>();
    let z_mean = standardized
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * weight)
        .sum::<f64>()
        / weight_sum;
    let x_mean = values
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * weight)
        .sum::<f64>()
        / weight_sum;
    let covariance = standardized
        .iter()
        .zip(values)
        .zip(weights)
        .map(|((z, x), weight)| weight * (z - z_mean) * (x - x_mean))
        .sum::<f64>();
    let variance = standardized
        .iter()
        .zip(weights)
        .map(|(z, weight)| weight * (z - z_mean).powi(2))
        .sum::<f64>();
    let scale = covariance / variance;
    let loc = x_mean - scale * z_mean;
    (loc.is_finite() && scale.is_finite() && scale > 0.0).then_some((loc, scale))
}

fn constrain_quantile_parameters(kind: DistributionKind, parameters: &mut [f64]) {
    let scale_index = parameters.len() - 1;
    parameters[scale_index] = parameters[scale_index].clamp(1.0e-12, 1.0e12);
    if parameters.len() == 3 {
        parameters[0] = match kind {
            DistributionKind::WeibullMin | DistributionKind::WeibullMax => {
                parameters[0].clamp(0.05, 50.0)
            }
            DistributionKind::SkewNorm => parameters[0].clamp(-50.0, 50.0),
            DistributionKind::GenExtreme | DistributionKind::GenPareto => {
                parameters[0].clamp(-2.0, 2.0)
            }
            DistributionKind::GumbelR | DistributionKind::GumbelL => parameters[0],
        };
    }
}

fn quantile_objective(
    kind: DistributionKind,
    parameters: &[f64],
    probabilities: &[f64],
    values: &[f64],
    weights: &[f64],
    value_scale: f64,
    truncation_atom: Option<f64>,
) -> f64 {
    let (shape, loc, scale) = if parameters.len() == 2 {
        (None, parameters[0], parameters[1])
    } else {
        (Some(parameters[0]), parameters[1], parameters[2])
    };
    let Some(distribution) = FittedDistribution::from_kind_and_params(kind, shape, loc, scale)
    else {
        return f64::INFINITY;
    };
    let tail_start = truncation_atom.map(|atom| distribution.cdf(atom));
    if tail_start.is_some_and(|cdf| !cdf.is_finite() || 1.0 - cdf <= 1.0e-12) {
        return f64::INFINITY;
    }
    let weight_sum = weights.iter().sum::<f64>();
    let mut loss = 0.0;
    for ((&probability, &value), &weight) in probabilities.iter().zip(values).zip(weights) {
        if weight == 0.0 {
            continue;
        }
        let base_probability = tail_start.map_or(probability, |cdf| {
            (cdf + probability * (1.0 - cdf)).min(1.0 - 1.0e-15)
        });
        let fitted = distribution.ppf(base_probability);
        if !fitted.is_finite() {
            return f64::INFINITY;
        }
        loss += weight * ((fitted - value) / value_scale).powi(2);
    }
    loss / weight_sum
}

fn quantile_nelder_mead<F, C>(
    mut start: Vec<f64>,
    steps: Vec<f64>,
    objective: F,
    constrain: C,
    max_iterations: usize,
    parameter_tolerance: f64,
    objective_tolerance: f64,
) -> SimplexResult
where
    F: Fn(&[f64]) -> f64,
    C: Fn(&mut [f64]),
{
    constrain(&mut start);
    let dimensions = start.len();
    let mut simplex = Vec::with_capacity(dimensions + 1);
    simplex.push(start.clone());
    for dimension in 0..dimensions {
        let mut point = start.clone();
        point[dimension] += steps[dimension].max(1.0e-8);
        constrain(&mut point);
        simplex.push(point);
    }
    let mut values: Vec<f64> = simplex.iter().map(|point| objective(point)).collect();
    let mut evaluations = values.len();

    // Multi-start callers (e.g. GenExtreme's 5 shape starts) run this to
    // completion for every start even when a start is clearly not
    // converging. Nelder-Mead's improvement curve is front-loaded, so a
    // start that hasn't beaten its best objective in STALL_LIMIT iterations
    // is treated the same as one that exhausted max_iterations: it returns
    // early as not-converged, which the caller already ranks below any
    // converged candidate.
    const STALL_LIMIT: usize = 300;
    let mut best_seen = f64::INFINITY;
    let mut stall_count = 0usize;

    for iteration in 0..max_iterations {
        let mut order: Vec<usize> = (0..simplex.len()).collect();
        order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
        let best = order[0];
        let worst = order[dimensions];
        let second_worst = order[dimensions - 1];
        let parameter_spread = order[1..]
            .iter()
            .flat_map(|&index| {
                simplex[index]
                    .iter()
                    .zip(&simplex[best])
                    .map(|(left, right)| (left - right).abs())
            })
            .fold(0.0_f64, f64::max);
        let objective_spread = order[1..]
            .iter()
            .map(|&index| (values[index] - values[best]).abs())
            .fold(0.0_f64, f64::max);
        if parameter_spread <= parameter_tolerance && objective_spread <= objective_tolerance {
            return SimplexResult {
                parameters: simplex[best].clone(),
                objective: values[best],
                converged: values[best].is_finite(),
                iterations: iteration,
                evaluations,
            };
        }

        if values[best] < best_seen - objective_tolerance {
            best_seen = values[best];
            stall_count = 0;
        } else {
            stall_count += 1;
            if stall_count >= STALL_LIMIT {
                return SimplexResult {
                    parameters: simplex[best].clone(),
                    objective: values[best],
                    converged: false,
                    iterations: iteration,
                    evaluations,
                };
            }
        }

        let mut centroid = vec![0.0; dimensions];
        for &index in &order[..dimensions] {
            for (dimension, value) in centroid.iter_mut().enumerate() {
                *value += simplex[index][dimension] / dimensions as f64;
            }
        }
        let trial = |factor: f64, reference: &[f64]| {
            centroid
                .iter()
                .zip(reference)
                .map(|(center, point)| center + factor * (center - point))
                .collect::<Vec<_>>()
        };
        let mut reflected = trial(1.0, &simplex[worst]);
        constrain(&mut reflected);
        let reflected_value = objective(&reflected);
        evaluations += 1;

        if reflected_value < values[best] {
            let mut expanded: Vec<f64> = centroid
                .iter()
                .zip(&reflected)
                .map(|(center, point)| center + 2.0 * (point - center))
                .collect();
            constrain(&mut expanded);
            let expanded_value = objective(&expanded);
            evaluations += 1;
            if expanded_value < reflected_value {
                simplex[worst] = expanded;
                values[worst] = expanded_value;
            } else {
                simplex[worst] = reflected;
                values[worst] = reflected_value;
            }
        } else if reflected_value < values[second_worst] {
            simplex[worst] = reflected;
            values[worst] = reflected_value;
        } else {
            let outside = reflected_value < values[worst];
            let reference = if outside { &reflected } else { &simplex[worst] };
            let mut contracted: Vec<f64> = centroid
                .iter()
                .zip(reference)
                .map(|(center, point)| center + 0.5 * (point - center))
                .collect();
            constrain(&mut contracted);
            let contracted_value = objective(&contracted);
            evaluations += 1;
            if contracted_value < values[worst].min(reflected_value) {
                simplex[worst] = contracted;
                values[worst] = contracted_value;
            } else {
                let best_point = simplex[best].clone();
                for &index in &order[1..] {
                    for (value, best_value) in simplex[index].iter_mut().zip(&best_point) {
                        *value = best_value + 0.5 * (*value - best_value);
                    }
                    constrain(&mut simplex[index]);
                    values[index] = objective(&simplex[index]);
                    evaluations += 1;
                }
            }
        }
    }

    let best = values
        .iter()
        .enumerate()
        .min_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0);
    SimplexResult {
        parameters: simplex[best].clone(),
        objective: values[best],
        converged: false,
        iterations: max_iterations,
        evaluations,
    }
}

// Utility math helpers

fn euler_mascheroni() -> f64 {
    EULER
}

// Cephes gamma/lgamma coefficients
const GAMMA_P: [f64; 7] = [
    1.60119522476751861407E-4,
    1.19135147006586384913E-3,
    1.04213797561761569935E-2,
    4.76367800457137231464E-2,
    2.07448227648435975150E-1,
    4.94214826801497100753E-1,
    9.99999999999999996796E-1,
];

const GAMMA_Q: [f64; 8] = [
    -2.31581873324120129819E-5,
    5.39605580493303397842E-4,
    -4.45641913851797240494E-3,
    1.18139785222060435552E-2,
    3.58236398605498653373E-2,
    -2.34591795718243348568E-1,
    7.14304917030273074085E-2,
    1.00000000000000000320E0,
];

const GAMMA_STIR: [f64; 5] = [
    7.87311395793093628397E-4,
    -2.29549961613378126380E-4,
    -2.68132617805781232825E-3,
    3.47222221605458667310E-3,
    8.33333333333482257126E-2,
];

const MAXSTIR: f64 = 143.01608;

fn sinpi(x: f64) -> f64 {
    (std::f64::consts::PI * x).sin()
}

fn stirf(x: f64) -> f64 {
    if x >= MAXGAM {
        return f64::INFINITY;
    }
    let w = 1.0 + (1.0 / x) * polevl(1.0 / x, &GAMMA_STIR);
    let mut y = x.exp();
    let v;
    if x > MAXSTIR {
        v = x.powf(0.5 * x - 0.25);
        y = v * (v / y);
    } else {
        y = x.powf(x - 0.5) / y;
    }
    SQRTPI * y * w
}

fn polevl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

fn p1evl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = x + coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

fn erfc(a: f64) -> f64 {
    let p;
    let q;
    let x;
    let y;
    let mut z;

    if a.is_nan() {
        return f64::NAN;
    }

    if a < 0.0 {
        x = -a;
    } else {
        x = a;
    }

    if x < 1.0 {
        return 1.0 - erf(a);
    }

    z = -a * a;
    if z < -LOGXMAX {
        // underflow
        return if a < 0.0 { 2.0 } else { 0.0 };
    }
    z = z.exp();

    if x < 8.0 {
        p = polevl(x, &NDTR_P);
        q = p1evl(x, &NDTR_Q);
    } else {
        p = polevl(x, &NDTR_R);
        q = p1evl(x, &NDTR_S);
    }
    y = (z * p) / q;
    if a < 0.0 { 2.0 - y } else { y }
}

fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return -erf(-x);
    }
    if x.abs() > 1.0 {
        return 1.0 - erfc(x);
    }
    let z = x * x;
    x * polevl(z, &NDTR_T) / p1evl(z, &NDTR_U)
}

fn ndtr_cephes(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = a * (1.0 / SQRT_2);
    let z = x.abs();
    if z < 1.0 {
        0.5 + 0.5 * erf(x)
    } else {
        let mut y = 0.5 * erfc(z);
        if x > 0.0 {
            y = 1.0 - y;
        }
        y
    }
}

// Owen's T (Patefield–Tandy) constants and implementation
const OWENS_T_SELECT_METHOD: [i32; 120] = [
    0, 0, 1, 12, 12, 12, 12, 12, 12, 12, 12, 15, 15, 15, 8, 0, 1, 1, 2, 2, 4, 4, 13, 13, 14, 14,
    15, 15, 15, 8, 1, 1, 2, 2, 2, 4, 4, 14, 14, 14, 14, 15, 15, 15, 9, 1, 1, 2, 4, 4, 4, 4, 6, 6,
    15, 15, 15, 15, 15, 9, 1, 2, 2, 4, 4, 5, 5, 7, 7, 16, 16, 16, 11, 11, 10, 1, 2, 4, 4, 4, 5, 5,
    7, 7, 16, 16, 16, 11, 11, 11, 1, 2, 3, 3, 5, 5, 7, 7, 16, 16, 16, 16, 16, 11, 11, 1, 2, 3, 3,
    5, 5, 17, 17, 17, 17, 16, 16, 16, 11, 11,
];

const OWENS_T_HRANGE: [f64; 14] = [
    0.02, 0.06, 0.09, 0.125, 0.26, 0.4, 0.6, 1.6, 1.7, 2.33, 2.4, 3.36, 3.4, 4.8,
];
const OWENS_T_ARANGE: [f64; 7] = [0.025, 0.09, 0.15, 0.36, 0.5, 0.9, 0.99999];
const OWENS_T_ORD: [f64; 18] = [
    2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 18.0, 10.0, 20.0, 30.0, 0.0, 4.0, 7.0, 8.0, 20.0, 0.0, 0.0,
];
const OWENS_T_METHODS: [i32; 18] = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6];
const OWENS_T_C: [f64; 31] = [
    1.0,
    -1.0,
    1.0,
    -0.9999999999999998,
    0.9999999999999839,
    -0.9999999999993063,
    0.9999999999797337,
    -0.9999999995749584,
    0.9999999933226235,
    -0.9999999188923242,
    0.9999992195143483,
    -0.9999939351372067,
    0.9999613559769055,
    -0.9997955636651394,
    0.9990927896296171,
    -0.9965938374119182,
    0.9891001713838613,
    -0.9700785580406933,
    0.9291143868326319,
    -0.8542058695956156,
    0.737965260330301,
    -0.585234698828374,
    0.4159977761456763,
    -0.25882108752419436,
    0.13755358251638927,
    -0.060795276632595575,
    0.021633768329987153,
    -0.005934056934551867,
    0.0011743414818332946,
    -0.0001489155613350369,
    9.072354320794358e-06,
];

const OWENS_T_PTS: [f64; 13] = [
    0.0035082039676451715489,
    0.031279042338030753740,
    0.085266826283219451090,
    0.16245071730812277011,
    0.25851196049125434828,
    0.36807553840697533536,
    0.48501092905604697475,
    0.60277514152618576821,
    0.71477884217753226516,
    0.81475510988760098605,
    0.89711029755948965867,
    0.95723808085944261843,
    0.99178832974629703586,
];

const OWENS_T_WTS: [f64; 13] = [
    0.018831438115323502887,
    0.018567086243977649478,
    0.018042093461223385584,
    0.017263829606398753364,
    0.016243219975989856730,
    0.014994592034116704829,
    0.013535474469662088392,
    0.011886351605820165233,
    0.010070377242777431897,
    0.0081130545742299586629,
    0.0060419009528470238773,
    0.0038862217010742057883,
    0.0016793031084546090448,
];

fn owens_get_method(h: f64, a: f64) -> i32 {
    let mut ihint = 14usize;
    let mut iaint = 7usize;
    for i in 0..14 {
        if h <= OWENS_T_HRANGE[i] {
            ihint = i;
            break;
        }
    }
    for i in 0..7 {
        if a <= OWENS_T_ARANGE[i] {
            iaint = i;
            break;
        }
    }
    OWENS_T_SELECT_METHOD[iaint * 15 + ihint]
}

fn owens_norm1(x: f64) -> f64 {
    erf(x / SQRT_2) / 2.0
}
fn owens_norm2(x: f64) -> f64 {
    erfc(x / SQRT_2) / 2.0
}

fn owens_t1(h: f64, a: f64, m: f64) -> f64 {
    let mut j = 1.0f64;
    let mut jj = 1.0f64;
    let hs = -0.5 * h * h;
    let dhs = hs.exp();
    let asq = a * a;
    let mut aj = a / (2.0 * PI);
    let mut dj = expm1_stable(hs);
    let mut gj = hs * dhs;
    let mut val = a.atan() / (2.0 * PI);
    loop {
        val += dj * aj / jj;
        if m <= j {
            break;
        }
        j += 1.0;
        jj += 2.0;
        aj *= asq;
        dj = gj - dj;
        gj *= hs / j;
    }
    val
}

fn owens_t2(h: f64, a: f64, ah: f64, m: f64) -> f64 {
    let mut i = 1.0f64;
    let maxi = 2.0 * m + 1.0;
    let hs = h * h;
    let asq = -(a * a);
    let y = 1.0 / hs;
    let mut val = 0.0;
    let mut vi = a * (-0.5 * ah * ah).exp() / (2.0 * PI).sqrt();
    let mut z = (ndtr_cephes(ah) - 0.5) / h;
    loop {
        val += z;
        if maxi <= i {
            break;
        }
        z = y * (vi - i * z);
        vi *= asq;
        i += 2.0;
    }
    val * (-0.5 * hs).exp() / (2.0 * PI).sqrt()
}

fn owens_t3(h: f64, a: f64, ah: f64) -> f64 {
    let aa = a * a;
    let hh = h * h;
    let y = 1.0 / hh;
    let mut vi = a * (-(ah * ah) / 2.0).exp() / (2.0 * PI).sqrt();
    let mut zi = owens_norm1(ah) / h;
    let mut result = 0.0;
    for i in 0..31 {
        if i >= OWENS_T_C.len() {
            break;
        }
        result += zi * OWENS_T_C[i];
        zi = y * (((2 * i + 1) as f64) * zi - vi);
        vi *= aa;
    }
    result * (-(hh) / 2.0).exp() / (2.0 * PI).sqrt()
}

fn owens_t4(h: f64, a: f64, m: f64) -> f64 {
    let maxi = 2.0 * m + 1.0;
    let hh = h * h;
    let naa = -(a * a);
    let mut i = 1.0f64;
    let mut ai = a * (-(hh) * (1.0 - naa) / 2.0).exp() / (2.0 * PI);
    let mut yi = 1.0;
    let mut result = 0.0;
    loop {
        result += ai * yi;
        if maxi <= i {
            break;
        }
        i += 2.0;
        yi = (1.0 - hh * yi) / i;
        ai *= naa;
    }
    result
}

fn owens_t5(h: f64, a: f64) -> f64 {
    let mut result = 0.0;
    let aa = a * a;
    let nhh = -0.5 * h * h;
    for i in 0..13 {
        let r = 1.0 + aa * OWENS_T_PTS[i];
        result += OWENS_T_WTS[i] * (nhh * r).exp() / r;
    }
    result * a
}

fn owens_t6(h: f64, a: f64) -> f64 {
    let normh = owens_norm2(h);
    let y = 1.0 - a;
    let r = y.atan2(1.0 + a);
    let mut result = normh * (1.0 - normh) / 2.0;
    if r != 0.0 {
        result -= r * (-(y * h * h) / (2.0 * r)).exp() / (2.0 * PI);
    }
    result
}

fn owens_t_impl(h: f64, a: f64, ah: f64) -> f64 {
    if h == 0.0 {
        return a.atan() / (2.0 * PI);
    }
    if a == 0.0 {
        return 0.0;
    }
    if a == 1.0 {
        return owens_norm2(-h) * owens_norm2(h) / 2.0;
    }
    let index = owens_get_method(h, a) as usize;
    let m = OWENS_T_ORD[index];
    let meth = OWENS_T_METHODS[index];
    match meth {
        1 => owens_t1(h, a, m),
        2 => owens_t2(h, a, ah, m),
        3 => owens_t3(h, a, ah),
        4 => owens_t4(h, a, m),
        5 => owens_t5(h, a),
        6 => owens_t6(h, a),
        _ => f64::NAN,
    }
}

fn owens_t(h: f64, a: f64) -> f64 {
    if h.is_nan() || a.is_nan() {
        return f64::NAN;
    }
    let h = h.abs(); // T(-h,a) == T(h,a)
    let fabs_a = a.abs();
    let fabs_ah = fabs_a * h;
    let result = if fabs_a.is_infinite() {
        0.5 * owens_norm2(h)
    } else if h.is_infinite() {
        0.0
    } else if fabs_a <= 1.0 {
        owens_t_impl(h, fabs_a, fabs_ah)
    } else {
        if fabs_ah <= 0.67 {
            let normh = owens_norm1(h);
            let normah = owens_norm1(fabs_ah);
            0.25 - normh * normah - owens_t_impl(fabs_ah, 1.0 / fabs_a, h)
        } else {
            let normh = owens_norm2(h);
            let normah = owens_norm2(fabs_ah);
            (normh + normah) / 2.0 - normh * normah - owens_t_impl(fabs_ah, 1.0 / fabs_a, h)
        }
    };
    if a < 0.0 { -result } else { result }
}

pub fn gamma(x: f64) -> f64 {
    if !x.is_finite() {
        return if x > 0.0 { x } else { f64::NAN };
    }
    if x == 0.0 {
        return f64::INFINITY.copysign(x);
    }
    let q = x.abs();
    if q > 33.0 {
        if x < 0.0 {
            let mut p = q.floor();
            if p == q {
                return f64::NAN;
            }
            let mut sgngam = 1.0;
            if ((p as i64) & 1) == 0 {
                sgngam = -1.0;
            }
            let mut z = q - p;
            if z > 0.5 {
                p += 1.0;
                z = q - p;
            }
            z = q * sinpi(z);
            if z == 0.0 {
                return sgngam * f64::INFINITY;
            }
            let val = std::f64::consts::PI / (z.abs() * stirf(q));
            return sgngam * val;
        } else {
            return stirf(x);
        }
    }
    let mut z = 1.0;
    let mut xx = x;
    while xx >= 3.0 {
        xx -= 1.0;
        z *= xx;
    }
    while xx < 0.0 {
        if xx > -1.0e-9 {
            break;
        }
        z /= xx;
        xx += 1.0;
    }
    while xx < 2.0 {
        if xx < 1.0e-9 {
            break;
        }
        z /= xx;
        xx += 1.0;
    }
    if xx == 2.0 {
        return z;
    }
    xx -= 2.0;
    let p = polevl(xx, &GAMMA_P);
    let q2 = polevl(xx, &GAMMA_Q);
    z * (p / q2)
}

fn mean_var(data: &[f64]) -> Option<(f64, f64)> {
    if data.len() < 2 {
        return None;
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    Some((mean, var.max(0.0)))
}

fn mean_var_population(data: &[f64]) -> Option<(f64, f64)> {
    if data.is_empty() {
        return None;
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    Some((mean, var.max(0.0)))
}

fn moments_pop_m2_m3(data: &[f64], mean: f64) -> (f64, f64) {
    let n = data.len() as f64;
    let m2 = data
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let m3 = data
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d * d
        })
        .sum::<f64>()
        / n;
    (m2, m3)
}

fn skewness(data: &[f64]) -> Option<f64> {
    if data.len() < 3 {
        return None;
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let (m2, m3) = moments_pop_m2_m3(data, mean);
    if m2 <= 0.0 {
        return Some(0.0);
    }
    Some(m3 / m2.powf(1.5))
}

fn logsumexp(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut a_max = f64::NEG_INFINITY;
    for &v in a {
        if v.is_finite() && v > a_max {
            a_max = v;
        }
    }
    if !a_max.is_finite() {
        a_max = 0.0;
    }
    let mut sum = 0.0;
    for &v in a {
        sum += (v - a_max).exp();
    }
    sum.ln() + a_max
}

fn average_with_log_weights(x: &[f64], logweights: &[f64]) -> f64 {
    let n = x.len().min(logweights.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut maxlogw = f64::NEG_INFINITY;
    for &lw in &logweights[..n] {
        if lw.is_finite() && lw > maxlogw {
            maxlogw = lw;
        }
    }
    if !maxlogw.is_finite() {
        maxlogw = 0.0;
    }
    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    for i in 0..n {
        let w = (logweights[i] - maxlogw).exp();
        sum_w += w;
        sum_wx += w * x[i];
    }
    sum_wx / sum_w
}

// Distributions

// Generalized Extreme Value with shape `c`, location `loc`, scale `scale`.
// CDF(x) = exp(-(1 - c * z)^(1/c)), z = (x - loc)/scale, valid for 1 - c*z > 0.
#[derive(Debug, Clone, Copy)]
pub struct GEV {
    pub c: f64,
    pub loc: f64,
    pub scale: f64,
}

impl ContinuousDistribution for GEV {
    fn pdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 {
            return 0.0;
        }
        let z = (x - self.loc) / self.scale;
        let t = 1.0 - self.c * z;
        if t <= 0.0 {
            return 0.0;
        }
        let a = t.powf(1.0 / self.c);
        ((-a).exp() * t.powf(1.0 / self.c - 1.0)) / self.scale
    }
    fn cdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 {
            return 0.0;
        }
        let z = (x - self.loc) / self.scale;
        let t = 1.0 - self.c * z;
        if t <= 0.0 {
            if self.c > 0.0 {
                return 1.0; // beyond upper bound
            } else if self.c < 0.0 {
                return 0.0; // below lower bound
            } else {
                return 0.0;
            }
        }
        (-(t.powf(1.0 / self.c))).exp()
    }
    fn support_lower(&self) -> f64 {
        // For c < 0: lower bound = loc + scale/c; else -inf
        if self.c < 0.0 {
            self.loc + self.scale / self.c
        } else {
            f64::NEG_INFINITY
        }
    }
    fn support_upper(&self) -> f64 {
        // For c > 0: upper bound = loc + scale/c; else +inf
        if self.c > 0.0 {
            self.loc + self.scale / self.c
        } else {
            f64::INFINITY
        }
    }
    fn ppf(&self, p: f64) -> f64 {
        let q = p.clamp(0.0, 1.0);
        if q <= 0.0 {
            return self.support_lower();
        }
        if q >= 1.0 {
            return self.support_upper();
        }
        if self.c.abs() < 1e-12 {
            return self.loc - self.scale * (-(q.ln())).ln();
        }
        let a = (-(q.ln())).powf(self.c);
        self.loc + self.scale * (1.0 - a) / self.c
    }
}

impl FittableDistribution for GEV {
    fn fit(data: &[f64]) -> Option<Self> {
        if data.len() < 3 {
            return None;
        }

        // Check for NaN or infinite values
        let has_invalid = data.iter().any(|x| !x.is_finite());
        if has_invalid {
            return None;
        }

        // L-moments (Hosking, 1990)
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len() as f64;
        let b0 = sorted.iter().sum::<f64>() / n;
        // Unbiased PWM estimators (Hosking 1990):
        // b1 = (1/n) * sum_{i=0..n-1} [ i/(n-1) ] * x_(i)
        // b2 = (1/n) * sum_{i=0..n-1} [ i(i-1)/((n-1)(n-2)) ] * x_(i)
        let denom1 = (n - 1.0).max(1.0);
        let denom2 = ((n - 1.0) * (n - 2.0)).max(1.0);
        let mut b1 = 0.0;
        let mut b2 = 0.0;
        for (i, &xi) in sorted.iter().enumerate() {
            let ii = i as f64;
            b1 += (ii / denom1) * xi;
            b2 += ((ii * (ii - 1.0)) / denom2) * xi;
        }
        b1 /= n;
        b2 /= n;
        let l1 = b0;
        let l2 = 2.0 * b1 - b0;
        let l3 = 6.0 * b2 - 6.0 * b1 + b0;
        if l2 <= 0.0 {
            return None;
        }
        // L-skewness and Hosking's polynomial approximation for shape
        let tau3 = (l3 / l2).clamp(-0.359_999_999, 0.499_999_999);
        let a = 2.0 / (3.0 + tau3) - (2.0_f64).ln() / (3.0_f64).ln();
        let k_est = 7.8590 * a + 2.9554 * a * a;

        let gamma_k = gamma(1.0 + k_est);
        let denom = (1.0 - 2.0_f64.powf(-k_est)) * gamma_k;
        if denom.abs() < 1e-18 {
            return None;
        }
        let alpha = (l2 * k_est / denom).abs(); // ensure positive scale
        if !(alpha.is_finite() && alpha > 0.0) {
            return None;
        }
        let mu = l1 - alpha * (1.0 - gamma_k) / k_est;

        // Baseline (L-moment) penalized NLL
        let pnll0 = gev_penalized_nll(k_est, mu, alpha, &sorted);

        let g = skewness(&sorted).unwrap_or(0.0);
        let k0 = if g < 0.0 { 0.5 } else { -0.5 };
        let (m_mean, _v_var) = mean_var(&sorted)?;
        let mu0 = m_mean;
        let sigma0 = 1.0;

        let (best_params, _) = if let Some((k_mle2, mu_mle2, sigma_mle2)) =
            nelder_mead_gev_mle(&sorted, k0, mu0, sigma0, 1200, 1e-4, 1e-4)
        {
            let pnll2 = gev_penalized_nll(k_mle2, mu_mle2, sigma_mle2, &sorted);
            if pnll2.is_finite() && pnll2 < pnll0 {
                ((k_mle2, mu_mle2, sigma_mle2), pnll2)
            } else {
                ((k_est, mu, alpha), pnll0)
            }
        } else {
            ((k_est, mu, alpha), pnll0)
        };

        Some(Self {
            c: best_params.0,
            loc: best_params.1,
            scale: best_params.2,
        })
    }
}

// Generic Nelder-Mead optimizer for 3-parameter distributions
fn nelder_mead_3param<F>(
    nll_func: F,
    data: &[f64],
    start: [f64; 3],
    constraints: Option<fn(&mut [f64; 3])>,
    max_iter: usize,
    xatol: f64,
    fatol: f64,
) -> Option<[f64; 3]>
where
    F: Fn(f64, f64, f64, &[f64]) -> f64,
{
    let nonzdelt = 0.05;
    let zdelt = 0.00025;

    let mut x0 = start;
    if let Some(apply_constraints) = constraints {
        apply_constraints(&mut x0);
    }

    let mut simplex = [[0.0f64; 3]; 4];
    simplex[0] = x0;
    for j in 0..3 {
        let mut y = x0;
        if y[j] != 0.0 {
            y[j] = (1.0 + nonzdelt) * y[j];
        } else {
            y[j] = zdelt;
        }
        simplex[j + 1] = y;
    }

    let mut fvals = [0.0f64; 4];
    for i in 0..4 {
        if let Some(apply_constraints) = constraints {
            apply_constraints(&mut simplex[i]);
        }
        fvals[i] = nll_func(simplex[i][0], simplex[i][1], simplex[i][2], data);
    }

    let (alpha, gamma, rho, sigma_c) = (1.0, 2.0, 0.5, 0.5);
    let mut func_evals: usize = 4;
    let max_evals: usize = 3 * 5000;

    for _ in 0..max_iter {
        let mut idx = [0usize, 1, 2, 3];
        idx.sort_by(|&i, &j| fvals[i].partial_cmp(&fvals[j]).unwrap());
        let (best, second, third, worst) = (idx[0], idx[1], idx[2], idx[3]);

        // Convergence check
        let mut xdiff = 0.0f64;
        for &ii in &idx[1..] {
            for d in 0..3 {
                xdiff = xdiff.max((simplex[ii][d] - simplex[best][d]).abs());
            }
        }
        let mut fdiff = 0.0f64;
        for &ii in &idx[1..] {
            fdiff = fdiff.max((fvals[ii] - fvals[best]).abs());
        }
        if xdiff <= xatol && fdiff <= fatol {
            return Some(simplex[best]);
        }

        // Centroid of best three
        let mut centroid = [0.0f64; 3];
        for d in 0..3 {
            centroid[d] = (simplex[best][d] + simplex[second][d] + simplex[third][d]) / 3.0;
        }

        // Reflection
        let mut xr = [0.0f64; 3];
        for d in 0..3 {
            xr[d] = centroid[d] + alpha * (centroid[d] - simplex[worst][d]);
        }
        if let Some(apply_constraints) = constraints {
            apply_constraints(&mut xr);
        }
        let fr = nll_func(xr[0], xr[1], xr[2], data);
        func_evals += 1;

        if fr < fvals[best] {
            // Expansion
            let mut xe = [0.0f64; 3];
            for d in 0..3 {
                xe[d] = centroid[d] + gamma * (xr[d] - centroid[d]);
            }
            if let Some(apply_constraints) = constraints {
                apply_constraints(&mut xe);
            }
            let fe = nll_func(xe[0], xe[1], xe[2], data);
            func_evals += 1;
            if fe < fr {
                simplex[worst] = xe;
                fvals[worst] = fe;
            } else {
                simplex[worst] = xr;
                fvals[worst] = fr;
            }
            if func_evals >= max_evals {
                break;
            }
            let mut order = [0usize, 1, 2, 3];
            order.sort_by(|&i, &j| fvals[i].partial_cmp(&fvals[j]).unwrap());
            let old_simplex = simplex;
            let old_fvals = fvals;
            for (pos, &ii) in order.iter().enumerate() {
                simplex[pos] = old_simplex[ii];
                fvals[pos] = old_fvals[ii];
            }
            continue;
        }

        if fr < fvals[third] {
            simplex[worst] = xr;
            fvals[worst] = fr;
            if func_evals >= max_evals {
                break;
            }
            let mut order = [0usize, 1, 2, 3];
            order.sort_by(|&i, &j| fvals[i].partial_cmp(&fvals[j]).unwrap());
            let old_simplex = simplex;
            let old_fvals = fvals;
            for (pos, &ii) in order.iter().enumerate() {
                simplex[pos] = old_simplex[ii];
                fvals[pos] = old_fvals[ii];
            }
            continue;
        }

        // Contraction
        let mut xc = [0.0f64; 3];
        if fr < fvals[worst] {
            for d in 0..3 {
                xc[d] = centroid[d] + rho * (xr[d] - centroid[d]);
            }
        } else {
            for d in 0..3 {
                xc[d] = centroid[d] + rho * (simplex[worst][d] - centroid[d]);
            }
        }
        if let Some(apply_constraints) = constraints {
            apply_constraints(&mut xc);
        }
        let fc = nll_func(xc[0], xc[1], xc[2], data);
        func_evals += 1;
        if fc < fvals[worst] {
            simplex[worst] = xc;
            fvals[worst] = fc;
            if func_evals >= max_evals {
                break;
            }
            let mut order = [0usize, 1, 2, 3];
            order.sort_by(|&i, &j| fvals[i].partial_cmp(&fvals[j]).unwrap());
            let old_simplex = simplex;
            let old_fvals = fvals;
            for (pos, &ii) in order.iter().enumerate() {
                simplex[pos] = old_simplex[ii];
                fvals[pos] = old_fvals[ii];
            }
            continue;
        }

        // Shrink
        for i in [second, third, worst] {
            for d in 0..3 {
                simplex[i][d] = simplex[best][d] + sigma_c * (simplex[i][d] - simplex[best][d]);
            }
            if let Some(apply_constraints) = constraints {
                apply_constraints(&mut simplex[i]);
            }
            fvals[i] = nll_func(simplex[i][0], simplex[i][1], simplex[i][2], data);
            func_evals += 1;
            if func_evals >= max_evals {
                break;
            }
        }
        if func_evals >= max_evals {
            break;
        }

        let mut order = [0usize, 1, 2, 3];
        order.sort_by(|&i, &j| fvals[i].partial_cmp(&fvals[j]).unwrap());
        let old_simplex = simplex;
        let old_fvals = fvals;
        for (pos, &ii) in order.iter().enumerate() {
            simplex[pos] = old_simplex[ii];
            fvals[pos] = old_fvals[ii];
        }
    }

    let mut idx = [0usize, 1, 2, 3];
    idx.sort_by(|&i, &j| fvals[i].partial_cmp(&fvals[j]).unwrap());
    Some(simplex[idx[0]])
}

fn gev_penalized_nll(k: f64, mu: f64, sigma: f64, data: &[f64]) -> f64 {
    if !(sigma.is_finite() && sigma > 0.0 && k.is_finite() && mu.is_finite()) {
        return f64::INFINITY;
    }
    let inv_sigma = 1.0 / sigma;
    let mut sum_logpdf_std = 0.0f64;
    let mut n_bad: usize = 0;
    if k.abs() < 1e-10 {
        for &x in data {
            let z = (x - mu) * inv_sigma;
            let lp_std = -z - (-z).exp();
            if lp_std.is_finite() {
                sum_logpdf_std += lp_std;
            } else {
                n_bad += 1;
            }
        }
    } else {
        for &x in data {
            let z = (x - mu) * inv_sigma;
            let t = 1.0 - k * z;
            if !(t > 0.0) {
                n_bad += 1;
                continue;
            }
            let lex = log1p_stable(-k * z);
            let lpex = lex / k;
            let pex = lpex.exp();
            let lp_std = -pex + lpex - lex;
            if lp_std.is_finite() {
                sum_logpdf_std += lp_std;
            } else {
                n_bad += 1;
            }
        }
    }
    let base = -sum_logpdf_std + (data.len() as f64) * sigma.ln();
    if n_bad > 0 {
        base + (n_bad as f64) * LOGXMAX * 100.0
    } else {
        base
    }
}

// Constraint function for GEV: only scale >= 1e-18
fn gev_constraints(params: &mut [f64; 3]) {
    params[2] = params[2].max(1e-18);
}

fn nelder_mead_gev_mle(
    data: &[f64],
    start_k: f64,
    start_mu: f64,
    start_sigma: f64,
    max_iter: usize,
    xatol: f64,
    fatol: f64,
) -> Option<(f64, f64, f64)> {
    nelder_mead_3param(
        gev_penalized_nll,
        data,
        [start_k, start_mu, start_sigma],
        Some(gev_constraints),
        max_iter,
        xatol,
        fatol,
    )
    .map(|params| (params[0], params[1], params[2]))
}

// Gumbel for maxima. Special case of GEV with c=0.
#[derive(Debug, Clone, Copy)]
pub struct Gumbel {
    pub loc: f64,
    pub scale: f64,
}

impl ContinuousDistribution for Gumbel {
    fn pdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 {
            return 0.0;
        }
        let z = (x - self.loc) / self.scale;
        let e = (-z).exp();
        (e * (-e).exp()) / self.scale
    }
    fn cdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 {
            return 0.0;
        }
        let z = (x - self.loc) / self.scale;
        (-(-z).exp()).exp()
    }
    fn ppf(&self, p: f64) -> f64 {
        let q = p.clamp(0.0, 1.0);
        if q <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if q >= 1.0 {
            return f64::INFINITY;
        }
        self.loc - self.scale * (-(q.ln())).ln()
    }
}

impl FittableDistribution for Gumbel {
    fn fit(data: &[f64]) -> Option<Self> {
        if data.len() < 2 {
            return None;
        }
        // MLE: solve for scale from mean - wavg - scale = 0, then loc from scale
        let n = data.len() as f64;
        let mean_x = data.iter().sum::<f64>() / n;
        let (m, v) = mean_var(data)?; // reuse computed mean/var; m==mean_x
        let initial_scale = if v > 0.0 {
            v.sqrt() * (6.0_f64).sqrt() / PI
        } else {
            1.0
        };
        let func = |scale: f64| -> f64 {
            if !(scale > 0.0 && scale.is_finite()) {
                return f64::NAN;
            }
            let sdata: Vec<f64> = data.iter().map(|&d| -d / scale).collect();
            let wavg = average_with_log_weights(data, &sdata);
            mean_x - wavg - scale
        };
        let mut lbrack = initial_scale / 2.0_f64.max(2.0);
        let mut rbrack = initial_scale * 2.0;
        if lbrack <= 0.0 {
            lbrack = initial_scale * 0.5_f64.max(0.5);
        }
        // Expand until sign change or limits reached
        let mut tries = 0;
        let interval_contains_root = |lb: f64, rb: f64| -> bool {
            let fl = func(lb);
            let fr = func(rb);
            fl.is_finite() && fr.is_finite() && fl.signum() != fr.signum()
        };
        while !interval_contains_root(lbrack, rbrack) && tries < 60 {
            lbrack /= 2.0;
            if lbrack <= 1e-16 {
                lbrack = 1e-16;
            }
            rbrack *= 2.0;
            if !rbrack.is_finite() || rbrack > 1e16 {
                break;
            }
            tries += 1;
        }
        let scale_opt = bisection_root(&func, lbrack, rbrack, 1e-12, 200);
        if let Some(scale) = scale_opt.filter(|s| *s > 0.0 && s.is_finite()) {
            let sdata: Vec<f64> = data.iter().map(|&d| -d / scale).collect();
            let lse = logsumexp(&sdata);
            let loc = -scale * (lse - n.ln());
            return Some(Self { loc, scale });
        }
        // Fallback: method of moments
        if v <= 0.0 {
            return None;
        }
        let beta = v.sqrt() * (6.0_f64).sqrt() / PI; // scale
        let mu = m - euler_mascheroni() * beta; // loc
        Some(Self {
            loc: mu,
            scale: beta,
        })
    }
}

// Gumbel for minima. Distribution of -X when X ~ gumbel_r.
#[derive(Debug, Clone, Copy)]
pub struct GumbelL {
    pub loc: f64,
    pub scale: f64,
}

impl ContinuousDistribution for GumbelL {
    fn pdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 {
            return 0.0;
        }
        // gumbel_l cdf = 1 - exp(-exp((x-loc)/scale))
        let z = (x - self.loc) / self.scale;
        let e = z.exp();
        (e * (-(e)).exp()) / self.scale
    }
    fn cdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 {
            return 0.0;
        }
        let z = (x - self.loc) / self.scale;
        1.0 - (-z.exp()).exp()
    }
    fn ppf(&self, p: f64) -> f64 {
        let q = p.clamp(0.0, 1.0);
        if q <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if q >= 1.0 {
            return f64::INFINITY;
        }
        self.loc + self.scale * (-(1.0 - q).ln()).ln()
    }
}

impl FittableDistribution for GumbelL {
    fn fit(data: &[f64]) -> Option<Self> {
        if data.len() < 2 {
            return None;
        }
        // Fit -data as GumbelR via MLE (above), then map back
        let neg: Vec<f64> = data.iter().map(|x| -*x).collect();
        if let Some(g) = Gumbel::fit(&neg) {
            return Some(Self {
                loc: -g.loc,
                scale: g.scale,
            });
        }
        None
    }
}

fn weibull_min_penalized_nll(k: f64, loc: f64, scale: f64, data: &[f64]) -> f64 {
    if !(scale.is_finite() && scale > 0.0 && k.is_finite() && k > 0.0 && loc.is_finite()) {
        return f64::INFINITY;
    }
    let inv_scale = 1.0 / scale;
    let mut sum_logpdf_std = 0.0f64;
    let mut n_bad: usize = 0;

    for &x in data {
        let z = (x - loc) * inv_scale;
        if z < 0.0 {
            n_bad += 1;
            continue;
        }
        // log pdf = log(k) + (k-1)*log(z) - z^k - log(scale)
        // For standardized: log pdf_std = log(k) + (k-1)*log(z) - z^k
        let log_z = z.ln();
        let z_k = z.powf(k);
        let lp_std = k.ln() + (k - 1.0) * log_z - z_k;
        if lp_std.is_finite() {
            sum_logpdf_std += lp_std;
        } else {
            n_bad += 1;
        }
    }

    let base = -sum_logpdf_std + (data.len() as f64) * scale.ln();
    if n_bad > 0 {
        base + (n_bad as f64) * LOGXMAX * 100.0
    } else {
        base
    }
}

// Constraint function for WeibullMin: shape k >= 1e-6, scale >= 1e-18
fn weibull_min_constraints(params: &mut [f64; 3]) {
    params[0] = params[0].max(1e-6); // k must be positive
    params[2] = params[2].max(1e-18);
}

fn nelder_mead_weibull_min_mle(
    data: &[f64],
    start_k: f64,
    start_loc: f64,
    start_scale: f64,
    max_iter: usize,
    xatol: f64,
    fatol: f64,
) -> Option<(f64, f64, f64)> {
    nelder_mead_3param(
        weibull_min_penalized_nll,
        data,
        [start_k, start_loc, start_scale],
        Some(weibull_min_constraints),
        max_iter,
        xatol,
        fatol,
    )
    .map(|params| (params[0], params[1], params[2]))
}

// Weibull minimum with shape k and scale lambda and location loc.
#[derive(Debug, Clone, Copy)]
pub struct WeibullMin {
    pub k: f64,
    pub loc: f64,
    pub scale: f64,
}

impl ContinuousDistribution for WeibullMin {
    fn pdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 || self.k <= 0.0 {
            return 0.0;
        }
        let z = (x - self.loc) / self.scale;
        if z < 0.0 {
            return 0.0;
        }
        (self.k / self.scale) * z.powf(self.k - 1.0) * (-(z.powf(self.k))).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 || self.k <= 0.0 {
            return 0.0;
        }
        let z = (x - self.loc) / self.scale;
        if z < 0.0 {
            return 0.0;
        }
        1.0 - (-(z.powf(self.k))).exp()
    }
    fn support_lower(&self) -> f64 {
        self.loc
    }
    fn ppf(&self, p: f64) -> f64 {
        let q = p.clamp(0.0, 1.0);
        if q <= 0.0 {
            return self.loc;
        }
        if q >= 1.0 {
            return f64::INFINITY;
        }
        let t = (-log1p_stable(-q)).powf(1.0 / self.k);
        self.loc + self.scale * t
    }
}

impl FittableDistribution for WeibullMin {
    fn fit(data: &[f64]) -> Option<Self> {
        if data.len() < 3 {
            return None;
        }

        // Check for NaN or infinite values
        let has_invalid = data.iter().any(|x| !x.is_finite());
        if has_invalid {
            return None;
        }

        // Population skewness as function of c (shape)
        let skew_c = |c: f64| -> f64 {
            let g1 = gamma(1.0 + 1.0 / c);
            let g2 = gamma(1.0 + 2.0 / c);
            let g3 = gamma(1.0 + 3.0 / c);
            let num = 2.0 * g1.powi(3) - 3.0 * g1 * g2 + g3;
            let den = (g2 - g1 * g1).powf(1.5);
            if den <= 0.0 {
                return 0.0;
            }
            num / den
        };

        let (m, v) = mean_var_population(data)?;
        if v <= 0.0 {
            return None;
        }
        let s = skewness(data).unwrap_or(0.0);
        // Past c > 3e4, skewness varies wildly. Get out early if s < s_min.
        let max_c = 1.0e4;
        let s_min = skew_c(max_c);

        // Method of moments estimate for shape parameter
        let mut k_mom: Option<f64> = None;
        if s >= s_min && s.is_finite() {
            // Solve skew_c(c) = s in [0.02, max_c]
            let f = |c: f64| skew_c(c) - s;
            if let Some(root) = bisection_root(f, 0.02, max_c, 1e-8, 100) {
                k_mom = Some(root.max(1e-6));
            }
        }

        // If we couldn't solve for k via skewness (e.g., s < s_min), fall back to MLE
        let k_est = if let Some(k) = k_mom {
            k
        } else {
            // Use a simple initial guess based on coefficient of variation
            let cv = v.sqrt() / m.abs().max(1e-12);
            if cv > 0.0 { (1.0 / cv).max(0.5) } else { 1.0 }
        };

        // Scale from variance: Var = scale^2 * (Gamma(1+2/k) - Gamma(1+1/k)^2)
        let g1 = gamma(1.0 + 1.0 / k_est);
        let g2 = gamma(1.0 + 2.0 / k_est);
        let denom = (g2 - g1 * g1).max(1e-18);
        let scale_est = (v / denom).sqrt().max(1e-12);

        // Location from mean: mean = loc + scale * Gamma(1 + 1/k)
        let loc_est = m - scale_est * g1;

        // Baseline (moment-based) penalized NLL
        let pnll0 = weibull_min_penalized_nll(k_est, loc_est, scale_est, data);

        // Run MLE optimizer with moment-based estimate as starting point
        let (best_params, _) = if let Some((k_mle, loc_mle, scale_mle)) =
            nelder_mead_weibull_min_mle(data, k_est, loc_est, scale_est, 1200, 1e-4, 1e-4)
        {
            let pnll_mle = weibull_min_penalized_nll(k_mle, loc_mle, scale_mle, data);
            if pnll_mle.is_finite() && pnll_mle < pnll0 {
                ((k_mle, loc_mle, scale_mle), pnll_mle)
            } else {
                ((k_est, loc_est, scale_est), pnll0)
            }
        } else {
            ((k_est, loc_est, scale_est), pnll0)
        };

        Some(Self {
            k: best_params.0,
            loc: best_params.1,
            scale: best_params.2,
        })
    }
}

// Weibull maximum. If Y ~ weibull_min(k, loc, scale), then X = loc - Y.
#[derive(Debug, Clone, Copy)]
pub struct WeibullMax {
    pub k: f64,
    pub loc: f64,
    pub scale: f64,
}

impl ContinuousDistribution for WeibullMax {
    fn pdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 || self.k <= 0.0 {
            return 0.0;
        }
        let z = (self.loc - x) / self.scale;
        if z < 0.0 {
            return 0.0;
        }
        (self.k / self.scale) * z.powf(self.k - 1.0) * (-(z.powf(self.k))).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 || self.k <= 0.0 {
            return 0.0;
        }
        // CDF increases with x; for reversed Weibull: F(x) = exp(-((loc-x)/scale)^k) for x <= loc
        let z = (self.loc - x) / self.scale;
        if z <= 0.0 {
            return 1.0;
        }
        (-(z.powf(self.k))).exp()
    }
    fn support_upper(&self) -> f64 {
        self.loc
    }
    fn ppf(&self, p: f64) -> f64 {
        let q = p.clamp(0.0, 1.0);
        if q <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if q >= 1.0 {
            return self.loc;
        }
        let t = (-(q.ln())).powf(1.0 / self.k);
        self.loc - self.scale * t
    }
}

impl FittableDistribution for WeibullMax {
    fn fit(data: &[f64]) -> Option<Self> {
        if data.len() < 3 {
            return None;
        }
        let loc = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mirrored: Vec<f64> = data.iter().map(|x| loc - *x).filter(|v| *v > 0.0).collect();
        let wm = WeibullMin::fit(&mirrored)?;
        Some(Self {
            k: wm.k,
            loc,
            scale: wm.scale,
        })
    }
}

fn skewnorm_penalized_nll(alpha: f64, loc: f64, scale: f64, data: &[f64]) -> f64 {
    if !(scale.is_finite() && scale > 0.0 && alpha.is_finite() && loc.is_finite()) {
        return f64::INFINITY;
    }
    let inv_scale = 1.0 / scale;
    let mut sum_logpdf_std = 0.0f64;
    let mut n_bad: usize = 0;

    for &x in data {
        let z = (x - loc) * inv_scale;
        // Standardized log pdf: log(2) + log(phi(z)) + log(Phi(alpha*z))
        // where phi is standard normal PDF and Phi is standard normal CDF
        let log_phi = -0.5 * z * z - 0.5 * (2.0 * PI).ln();
        let alpha_z = alpha * z;

        // For log(Phi(alpha*z)), use ndtr_cephes which computes Phi
        let phi_alpha_z = ndtr_cephes(alpha_z);
        if phi_alpha_z <= 0.0 {
            n_bad += 1;
            continue;
        }
        let log_phi_alpha_z = phi_alpha_z.ln();

        let lp_std = (2.0_f64).ln() + log_phi + log_phi_alpha_z;
        if lp_std.is_finite() {
            sum_logpdf_std += lp_std;
        } else {
            n_bad += 1;
        }
    }

    let base = -sum_logpdf_std + (data.len() as f64) * scale.ln();
    if n_bad > 0 {
        base + (n_bad as f64) * LOGXMAX * 100.0
    } else {
        base
    }
}

// Constraint function for SkewNorm: only scale >= 1e-18
fn skewnorm_constraints(params: &mut [f64; 3]) {
    params[2] = params[2].max(1e-18);
}

fn nelder_mead_skewnorm_mle(
    data: &[f64],
    start_alpha: f64,
    start_loc: f64,
    start_scale: f64,
    max_iter: usize,
    xatol: f64,
    fatol: f64,
) -> Option<(f64, f64, f64)> {
    nelder_mead_3param(
        skewnorm_penalized_nll,
        data,
        [start_alpha, start_loc, start_scale],
        Some(skewnorm_constraints),
        max_iter,
        xatol,
        fatol,
    )
    .map(|params| (params[0], params[1], params[2]))
}

// Skew Normal with shape alpha, location xi, scale omega.
#[derive(Debug, Clone, Copy)]
pub struct SkewNormal {
    pub alpha: f64,
    pub loc: f64,
    pub scale: f64,
}

impl ContinuousDistribution for SkewNormal {
    fn pdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 {
            return 0.0;
        }
        let z = (x - self.loc) / self.scale;
        let phi = (-0.5 * z * z).exp() / (SQRT_2 * PI.sqrt());
        let cdf = 0.5 * (1.0 + erf(self.alpha * z / SQRT_2));
        2.0 * phi * cdf / self.scale
    }
    fn cdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 {
            return 0.0;
        }
        let z = (x - self.loc) / self.scale;
        let base = ndtr_cephes(z);
        let t = owens_t(z, self.alpha);
        (base - 2.0 * t).clamp(0.0, 1.0)
    }
}

impl FittableDistribution for SkewNormal {
    fn fit(data: &[f64]) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }

        // Check for NaN or infinite values
        let has_invalid = data.iter().any(|x| !x.is_finite());
        if has_invalid {
            return None;
        }

        let (m, v) = mean_var_population(data)?;
        if v <= 0.0 {
            return None;
        }
        let s = skewness(data).unwrap_or(0.0);

        // Max achievable skewness magnitude for skew-normal
        // See https://en.wikipedia.org/wiki/Skew_normal_distribution
        let skew_d = |d: f64| -> f64 {
            let num = (4.0 - PI) / 2.0 * (d * (2.0 / PI).sqrt()).powi(3);
            let den = (1.0 - 2.0 * d * d / PI).powf(1.5);
            num / den
        };
        let s_max = skew_d(1.0); // Maximum skewness ~0.9953
        let s_clipped = s.clamp(-s_max, s_max);

        // Solve for delta in [-1, 1]
        let d = if let Some(root) =
            bisection_root(|d| skew_d(d) - s_clipped, -0.999, 0.999, 1e-8, 100)
        {
            root
        } else {
            0.0
        };

        // Shape parameter alpha from delta: alpha = delta / sqrt(1 - delta^2)
        let alpha_est = if (1.0 - d * d) > 0.0 {
            d / (1.0 - d * d).sqrt()
        } else if d >= 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };

        // Scale from variance: Var = omega^2 * (1 - 2 delta^2 / pi)
        let omega_est = (v / (1.0 - 2.0 * d * d / PI).max(1e-18)).sqrt();
        // Location from mean: mean = xi + omega * delta * sqrt(2/pi)
        let xi_est = m - omega_est * d * (2.0 / PI).sqrt();

        // Baseline (moment-based) penalized NLL
        let pnll0 = skewnorm_penalized_nll(alpha_est, xi_est, omega_est, data);

        // Run MLE optimizer with moment-based estimate as starting point
        let (best_params, _) = if let Some((alpha_mle, loc_mle, scale_mle)) =
            nelder_mead_skewnorm_mle(data, alpha_est, xi_est, omega_est, 1200, 1e-4, 1e-4)
        {
            let pnll_mle = skewnorm_penalized_nll(alpha_mle, loc_mle, scale_mle, data);
            if pnll_mle.is_finite() && pnll_mle < pnll0 {
                ((alpha_mle, loc_mle, scale_mle), pnll_mle)
            } else {
                ((alpha_est, xi_est, omega_est), pnll0)
            }
        } else {
            ((alpha_est, xi_est, omega_est), pnll0)
        };

        Some(Self {
            alpha: best_params.0,
            loc: best_params.1,
            scale: best_params.2,
        })
    }
}

fn genpareto_penalized_nll(c: f64, loc: f64, scale: f64, data: &[f64]) -> f64 {
    if !(scale.is_finite() && scale > 0.0 && c.is_finite() && loc.is_finite()) {
        return f64::INFINITY;
    }
    let inv_scale = 1.0 / scale;
    let mut sum_logpdf_std = 0.0f64;
    let mut n_bad: usize = 0;

    for &x in data {
        let z = (x - loc) * inv_scale;
        if z < 0.0 {
            n_bad += 1;
            continue;
        }
        // Standardized log pdf
        // For c ≈ 0 (exponential): log pdf = -z
        // For c ≠ 0: log pdf = -(1 + 1/c) * log(1 + c*z)
        let lp_std = if c.abs() < 1e-12 {
            -z
        } else {
            let t = 1.0 + c * z;
            if t <= 0.0 {
                n_bad += 1;
                continue;
            }
            let log_t = log1p_stable(c * z);
            -(1.0 + 1.0 / c) * log_t
        };

        if lp_std.is_finite() {
            sum_logpdf_std += lp_std;
        } else {
            n_bad += 1;
        }
    }

    let base = -sum_logpdf_std + (data.len() as f64) * scale.ln();
    if n_bad > 0 {
        base + (n_bad as f64) * LOGXMAX * 100.0
    } else {
        base
    }
}

// Constraint function for GenPareto: only scale >= 1e-18
fn genpareto_constraints(params: &mut [f64; 3]) {
    params[2] = params[2].max(1e-18);
}

fn nelder_mead_genpareto_mle(
    data: &[f64],
    start_c: f64,
    start_loc: f64,
    start_scale: f64,
    max_iter: usize,
    xatol: f64,
    fatol: f64,
) -> Option<(f64, f64, f64)> {
    nelder_mead_3param(
        genpareto_penalized_nll,
        data,
        [start_c, start_loc, start_scale],
        Some(genpareto_constraints),
        max_iter,
        xatol,
        fatol,
    )
    .map(|params| (params[0], params[1], params[2]))
}

// Generalized Pareto with shape c, location loc, scale scale.
#[derive(Debug, Clone, Copy)]
pub struct GeneralizedPareto {
    pub c: f64,
    pub loc: f64,
    pub scale: f64,
}

impl ContinuousDistribution for GeneralizedPareto {
    fn pdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 {
            return 0.0;
        }
        let z = (x - self.loc) / self.scale;
        if z < 0.0 {
            return 0.0;
        }
        // Handle exponential case (c ≈ 0) separately for numerical stability
        if self.c.abs() < 1e-12 {
            return (-z).exp() / self.scale;
        }
        let t = 1.0 + self.c * z;
        if t <= 0.0 {
            return 0.0;
        }
        (1.0 / self.scale) * t.powf(-1.0 / self.c - 1.0)
    }
    fn cdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 {
            return 0.0;
        }
        let z = (x - self.loc) / self.scale;
        if z < 0.0 {
            return 0.0;
        }
        if self.c.abs() < 1e-12 {
            1.0 - (-z).exp()
        } else {
            let val = -inv_boxcox1p(-z, -self.c);
            val.clamp(0.0, 1.0)
        }
    }
    fn support_lower(&self) -> f64 {
        self.loc
    }
    fn support_upper(&self) -> f64 {
        if self.c < 0.0 {
            self.loc - self.scale / self.c
        } else {
            f64::INFINITY
        }
    }
    fn ppf(&self, p: f64) -> f64 {
        let q = p.clamp(0.0, 1.0);
        if q <= 0.0 {
            return self.support_lower();
        }
        if q >= 1.0 {
            return self.support_upper();
        }
        if self.c.abs() < 1e-12 {
            // Exponential case: use log1p for numerical stability near q=1
            return self.loc - self.scale * log1p_stable(-q);
        }
        self.loc - self.scale * boxcox1p(-q, -self.c)
    }
}

impl FittableDistribution for GeneralizedPareto {
    fn fit(data: &[f64]) -> Option<Self> {
        if data.len() < 3 {
            return None;
        }

        // Check for NaN or infinite values
        let has_invalid = data.iter().any(|x| !x.is_finite());
        if has_invalid {
            return None;
        }

        // Use method-of-moments on thresholded data with loc = min(data)
        let loc_est = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let y: Vec<f64> = data
            .iter()
            .map(|x| x - loc_est)
            .filter(|v| *v > 0.0)
            .collect();
        if y.len() < 3 {
            return None;
        }

        let (m, v) = mean_var_population(&y)?;
        if v <= 0.0 || !m.is_finite() {
            return None;
        }

        // Method of moments:
        // Mean = scale / (1 - c) for c < 1
        // Var = scale^2 / [(1 - c)^2 * (1 - 2c)] for c < 0.5
        // From these: r = m^2 / v = (1 - 2c), so c = (1 - r) / 2
        let r = (m * m) / v;
        let c_est = (1.0 - r) / 2.0; // shape

        // Check validity of moment estimates
        if !c_est.is_finite() || (1.0 - 2.0 * c_est) <= 0.0 || (1.0 - c_est) <= 0.0 {
            return None;
        }

        let scale_est = m * (1.0 - c_est);
        if !scale_est.is_finite() || scale_est <= 0.0 {
            return None;
        }

        // Baseline (moment-based) penalized NLL
        let pnll0 = genpareto_penalized_nll(c_est, loc_est, scale_est, data);

        // Run MLE optimizer with moment-based estimate as starting point
        let (best_params, _) = if let Some((c_mle, loc_mle, scale_mle)) =
            nelder_mead_genpareto_mle(data, c_est, loc_est, scale_est, 1200, 1e-4, 1e-4)
        {
            let pnll_mle = genpareto_penalized_nll(c_mle, loc_mle, scale_mle, data);
            if pnll_mle.is_finite() && pnll_mle < pnll0 {
                ((c_mle, loc_mle, scale_mle), pnll_mle)
            } else {
                ((c_est, loc_est, scale_est), pnll0)
            }
        } else {
            ((c_est, loc_est, scale_est), pnll0)
        };

        Some(Self {
            c: best_params.0,
            loc: best_params.1,
            scale: best_params.2,
        })
    }
}

// Error function, Owen's T, and integrators

// Cephes coeffs from ndtr.h
const NDTR_P: [f64; 9] = [
    2.46196981473530512524E-10,
    5.64189564831068821977E-1,
    7.46321056442269912687E0,
    4.86371970985681366614E1,
    1.96520832956077098242E2,
    5.26445194995477358631E2,
    9.34528527171957607540E2,
    1.02755188689515710272E3,
    5.57535335369399327526E2,
];

const NDTR_Q: [f64; 8] = [
    1.32281951154744992508E1,
    8.67072140885989742329E1,
    3.54937778887819891062E2,
    9.75708501743205489753E2,
    1.82390916687909736289E3,
    2.24633760818710981792E3,
    1.65666309194161350182E3,
    5.57535340817727675546E2,
];

const NDTR_R: [f64; 6] = [
    5.64189583547755073984E-1,
    1.27536670759978104416E0,
    5.01905042251180477414E0,
    6.16021097993053585195E0,
    7.40974269950448939160E0,
    2.97886665372100240670E0,
];

const NDTR_S: [f64; 6] = [
    2.26052863220117276590E0,
    9.39603524938001434673E0,
    1.20489539808096656605E1,
    1.70814450747565897222E1,
    9.60896809063285878198E0,
    3.36907645100081516050E0,
];

const NDTR_T: [f64; 5] = [
    9.60497373987051638749E0,
    9.00260197203842689217E1,
    2.23200534594684319226E3,
    7.00332514112805075473E3,
    5.55923013010394962768E4,
];

const NDTR_U: [f64; 5] = [
    3.35617141647503099647E1,
    5.21357949780152679795E2,
    4.59432382970980127987E3,
    2.26290000613890934246E4,
    4.92673942608635921086E4,
    // implicit leading 1.0 handled by p1evl
];

fn bisection_root<F: Fn(f64) -> f64>(
    f: F,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> Option<f64> {
    let mut fa = f(a);
    let fb = f(b);
    if !fa.is_finite() || !fb.is_finite() {
        return None;
    }
    if fa == 0.0 {
        return Some(a);
    }
    if fb == 0.0 {
        return Some(b);
    }
    if fa.signum() == fb.signum() {
        return None;
    }
    for _ in 0..max_iter {
        let m = 0.5 * (a + b);
        let fm = f(m);
        if !fm.is_finite() {
            return None;
        }
        if fm == 0.0 || (b - a).abs() < tol {
            return Some(m);
        }
        if fa.signum() != fm.signum() {
            b = m;
        } else {
            a = m;
            fa = fm;
        }
    }
    Some(0.5 * (a + b))
}

// Cephes unity polynomials for log1p/expm1
const UNITY_LP: [f64; 7] = [
    4.5270000862445199635215E-5,
    4.9854102823193375972212E-1,
    6.5787325942061044846969E0,
    2.9911919328553073277375E1,
    6.0949667980987787057556E1,
    5.7112963590585538103336E1,
    2.0039553499201281259648E1,
];
const UNITY_LQ: [f64; 6] = [
    1.5062909083469192043167E1,
    8.3047565967967209469434E1,
    2.2176239823732856465394E2,
    3.0909872225312059774938E2,
    2.1642788614495947685003E2,
    6.0118660497603843919306E1,
];
const UNITY_EP: [f64; 3] = [
    1.2617719307481059087798E-4,
    3.0299440770744196129956E-2,
    9.9999999999999999991025E-1,
];
const UNITY_EQ: [f64; 4] = [
    3.0019850513866445504159E-6,
    2.5244834034968410419224E-3,
    2.2726554820815502876593E-1,
    2.0000000000000000000897E0,
];

fn log1p_stable(x: f64) -> f64 {
    let z = 1.0 + x;
    if z < M_SQRT1_2 || z > M_SQRT2 {
        return z.ln();
    }
    let xx = x * x;
    let z2 = -0.5 * xx + x * (xx * polevl(x, &UNITY_LP) / p1evl(x, &UNITY_LQ));
    x + z2
}

fn expm1_stable(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() {
            return x;
        }
        if x > 0.0 {
            return x;
        } else {
            return -1.0;
        }
    }
    if x < -0.5 || x > 0.5 {
        return x.exp() - 1.0;
    }
    let xx = x * x;
    let mut r = x * polevl(xx, &UNITY_EP);
    r = r / (polevl(xx, &UNITY_EQ) - r);
    r + r
}

fn boxcox1p(x: f64, lambda: f64) -> f64 {
    if lambda.abs() < 1e-19 {
        return log1p_stable(x);
    }
    let t = lambda * log1p_stable(x);
    if t < LOGXMAX {
        expm1_stable(t) / lambda
    } else {
        (lambda.signum() * (lambda * log1p_stable(x)).exp() - 1.0) / lambda
    }
}

fn inv_boxcox1p(x: f64, lambda: f64) -> f64 {
    if lambda == 0.0 {
        return expm1_stable(x);
    }
    let t = lambda * x;
    if t < f64::MAX.log(EPS.recip()) {
        (log1p_stable(lambda * x) / lambda).exp() - 1.0
    } else {
        ((lambda.signum() * (x + 1.0 / lambda)).ln().abs() + lambda.abs().ln())
            .exp()
            .powf(1.0 / lambda)
            - 1.0
    }
}

// Generic fitting and KS selection
#[derive(Debug, Clone, Copy)]
pub struct CandidateFit {
    pub name: &'static str,
    pub dist: FittedDistribution,
    pub ks_stat: f64,
    pub p_value: f64,
    pub shape: Option<f64>,
    pub loc: f64,
    pub scale: f64,
}

// Focused diagnostics
#[derive(Debug, Clone)]
pub struct DiagnosticFit {
    pub name: &'static str,
    pub dist: FittedDistribution,

    // Basic goodness-of-fit (same as CandidateFit)
    pub ks_stat: f64,
    pub p_value: f64,

    // Parameters (same as CandidateFit)
    pub shape: Option<f64>,
    pub loc: f64,
    pub scale: f64,

    // Additional diagnostic metrics
    pub rmse: f64,      // Root mean square error between empirical and fitted CDFs
    pub r_squared: f64, // R-squared equivalent for distribution fitting
}

// Conversion from diagnostic to production structure if needed
impl From<&DiagnosticFit> for CandidateFit {
    fn from(diagnostic: &DiagnosticFit) -> Self {
        CandidateFit {
            name: diagnostic.name,
            dist: diagnostic.dist,
            ks_stat: diagnostic.ks_stat,
            p_value: diagnostic.p_value,
            shape: diagnostic.shape,
            loc: diagnostic.loc,
            scale: diagnostic.scale,
        }
    }
}

pub fn extract_params(d: &FittedDistribution) -> (Option<f64>, f64, f64) {
    match d {
        FittedDistribution::GenExtreme(gev) => (Some(gev.c), gev.loc, gev.scale),
        FittedDistribution::WeibullMin(w) => (Some(w.k), w.loc, w.scale),
        FittedDistribution::WeibullMax(w) => (Some(w.k), w.loc, w.scale),
        FittedDistribution::SkewNorm(s) => (Some(s.alpha), s.loc, s.scale),
        FittedDistribution::GumbelR(g) => (None, g.loc, g.scale),
        FittedDistribution::GumbelL(g) => (None, g.loc, g.scale),
        FittedDistribution::GenPareto(gp) => (Some(gp.c), gp.loc, gp.scale),
    }
}

pub fn ks_statistic(data: &[f64], dist: &dyn ContinuousDistribution) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len() as f64;
    let mut d: f64 = 0.0;
    for (i, &x) in sorted.iter().enumerate() {
        let fi = dist.cdf(x).clamp(0.0, 1.0);
        let emp_cdf = (i as f64 + 1.0) / n;
        let diff1 = (emp_cdf - fi).abs();
        let diff2 = ((i as f64) / n - fi).abs();
        d = d.max(diff1.max(diff2));
    }
    d
}

pub fn ks_pvalue_asymptotic(d: f64, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let z = (n as f64).sqrt() * d;
    if z <= 0.0 {
        return 1.0;
    }
    // Kolmogorov distribution Q_KS(z) ≈ 2 * sum_{k=1..∞} (-1)^{k-1} exp(-2 k^2 z^2)
    let mut sum = 0.0;
    let mut k = 1;
    loop {
        let term = (-2.0 * (k as f64).powi(2) * z * z).exp();
        let add = if k % 2 == 1 { term } else { -term };
        sum += add;
        if term < 1e-12 {
            break;
        }
        if k > 100000 {
            break;
        }
        k += 1;
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

pub fn candidate_kinds() -> [(&'static str, DistributionKind); 7] {
    [
        ("genextreme", DistributionKind::GenExtreme),
        ("weibull_min", DistributionKind::WeibullMin),
        ("weibull_max", DistributionKind::WeibullMax),
        ("skewnorm", DistributionKind::SkewNorm),
        ("gumbel_r", DistributionKind::GumbelR),
        ("gumbel_l", DistributionKind::GumbelL),
        ("genpareto", DistributionKind::GenPareto),
    ]
}

pub fn fit_candidates_and_select_best(data: &[f64], index_id: Option<i32>) -> Option<CandidateFit> {
    let mut best: Option<CandidateFit> = None;
    for (name, kind) in candidate_kinds() {
        if let Some(fd) = fit_distribution(kind, data) {
            let d = ks_statistic(data, &fd);
            let p = ks_pvalue_asymptotic(d, data.len());
            let (shape, loc, scale) = extract_params(&fd);
            let cand = CandidateFit {
                name,
                dist: fd,
                ks_stat: d,
                p_value: p,
                shape,
                loc,
                scale,
            };
            if let Some(ref b) = best {
                if let Some(index_id) = index_id {
                    let limit = get_hazard_limit(index_id);

                    // Test the RP1000 value to see if it is less than the hazard limit to validate the fit
                    let rp_value = FittedDistribution::from_name_and_params(
                        cand.name, cand.shape, cand.loc, cand.scale,
                    )
                    .and_then(|ff| Some(ff.ppf(0.999)))
                    .unwrap_or(0.0);

                    if cand.p_value > b.p_value && rp_value <= limit {
                        best = Some(cand);
                    }
                } else if cand.p_value > b.p_value {
                    best = Some(cand);
                }
            } else {
                if let Some(index_id) = index_id {
                    let limit = get_hazard_limit(index_id);

                    // Test the RP1000 value to see if it is less than the hazard limit to validate the fit
                    let rp_value = FittedDistribution::from_name_and_params(
                        cand.name, cand.shape, cand.loc, cand.scale,
                    )
                    .and_then(|ff| Some(ff.ppf(0.999)))
                    .unwrap_or(0.0);

                    if rp_value <= limit {
                        best = Some(cand);
                    }
                } else {
                    best = Some(cand);
                }
            }
        }
    }

    best
}

pub fn fit_all_candidates(data: &[f64]) -> Vec<CandidateFit> {
    let mut out = Vec::new();
    for (name, kind) in candidate_kinds() {
        if let Some(fd) = fit_distribution(kind, data) {
            let d = ks_statistic(data, &fd);
            let p = ks_pvalue_asymptotic(d, data.len());
            let (shape, loc, scale) = extract_params(&fd);
            out.push(CandidateFit {
                name,
                dist: fd,
                ks_stat: d,
                p_value: p,
                shape,
                loc,
                scale,
            });
        }
    }
    out
}

// Reconstruction API: build distributions and regenerate quantiles

impl FittedDistribution {
    // Construct a fitted distribution instance from a distribution kind and its parameters.
    // Some distributions require a shape parameter; for those, `shape` must be `Some`.
    pub fn from_kind_and_params(
        kind: DistributionKind,
        shape: Option<f64>,
        loc: f64,
        scale: f64,
    ) -> Option<Self> {
        if !loc.is_finite() || !(scale.is_finite() && scale > 0.0) {
            return None;
        }
        match kind {
            DistributionKind::GenExtreme => Some(Self::GenExtreme(GEV {
                c: shape?,
                loc,
                scale,
            })),
            DistributionKind::WeibullMin => Some(Self::WeibullMin(WeibullMin {
                k: shape?,
                loc,
                scale,
            })),
            DistributionKind::WeibullMax => Some(Self::WeibullMax(WeibullMax {
                k: shape?,
                loc,
                scale,
            })),
            DistributionKind::SkewNorm => Some(Self::SkewNorm(SkewNormal {
                alpha: shape?,
                loc,
                scale,
            })),
            DistributionKind::GumbelR => Some(Self::GumbelR(Gumbel { loc, scale })),
            DistributionKind::GumbelL => Some(Self::GumbelL(GumbelL { loc, scale })),
            DistributionKind::GenPareto => Some(Self::GenPareto(GeneralizedPareto {
                c: shape?,
                loc,
                scale,
            })),
        }
    }

    // Construct from a distribution name and parameters. Names follow `DistributionKind::from_name`.
    pub fn from_name_and_params(
        name: &str,
        shape: Option<f64>,
        loc: f64,
        scale: f64,
    ) -> Option<Self> {
        let kind = DistributionKind::from_name(name)?;
        Self::from_kind_and_params(kind, shape, loc, scale)
    }
}

// Convenience wrapper to recreate a distribution from persisted parameters.
pub fn recreate_distribution(
    kind: DistributionKind,
    shape: Option<f64>,
    loc: f64,
    scale: f64,
) -> Option<FittedDistribution> {
    FittedDistribution::from_kind_and_params(kind, shape, loc, scale)
}

// Convenience wrapper to recreate a distribution from a string name.
pub fn recreate_distribution_by_name(
    name: &str,
    shape: Option<f64>,
    loc: f64,
    scale: f64,
) -> Option<FittedDistribution> {
    FittedDistribution::from_name_and_params(name, shape, loc, scale)
}

// Generate an n-point quantile array using proper probability bounds to avoid infinite values.
// Maps to probability range [0.001, 0.999] (0.1% to 99.9%) to match Python reference implementation.
pub fn generate_quantiles(dist: &FittedDistribution, n: usize) -> Vec<f32> {
    let m = n.max(2);
    let mut out = Vec::with_capacity(m);

    // Use linspace from 0.1 to 99.9 (percentiles), then convert to probabilities
    for i in 0..m {
        let percentile = 0.1 + (99.8 * i as f32) / (m - 1) as f32; // 0.1 to 99.9
        let p = percentile / 100.0; // Convert to probability [0.001, 0.999]
        out.push(dist.ppf(p as f64) as f32);
    }
    out
}

// Generate 1001 quantiles matching the standard layout.
pub fn generate_standard_quantiles(dist: &FittedDistribution) -> Vec<f32> {
    generate_quantiles(dist, 1001)
}

// Generate quantiles with climate index-specific constraints (e.g., day-based indices capped at 365).
// Matches the Python reference implementation behavior exactly.
pub fn generate_quantiles_for_index(
    dist: &FittedDistribution,
    n: usize,
    index_name: &str,
) -> Vec<f64> {
    let m = n.max(2);
    let mut out = Vec::with_capacity(m);

    // Check if this index should be capped at 365 days
    let cap_at_365 = matches!(
        index_name,
        "frost_days" | "daily_freezethaw_cycles" | "hot_days"
    );

    // Use linspace from 0.1 to 99.9 (percentiles), then convert to probabilities
    for i in 0..m {
        let percentile = 0.1 + (99.8 * i as f64) / (m - 1) as f64; // 0.1 to 99.9
        let p = percentile / 100.0; // Convert to probability [0.001, 0.999]
        let mut val = dist.ppf(p);

        // Apply index-specific capping
        if cap_at_365 && val > 365.0 {
            val = 365.0;
        }

        out.push(val);
    }
    out
}

// Recreate a distribution from parameters and then generate its n-point quantile array.
pub fn generate_quantiles_from_params(
    name: &str,
    shape: Option<f64>,
    loc: f64,
    scale: f64,
    n: usize,
) -> Option<Vec<f32>> {
    let d = recreate_distribution_by_name(name, shape, loc, scale)?;
    Some(generate_quantiles(&d, n))
}

// Return level for a given return period. For upper-tail events use `upper_tail = true`.
pub fn return_level(dist: &FittedDistribution, return_period: f64, upper_tail: bool) -> f64 {
    if !(return_period.is_finite() && return_period > 1.0) {
        return f64::NAN;
    }
    let p = if upper_tail {
        (1.0 - 1.0 / return_period).clamp(0.0, 1.0)
    } else {
        (1.0 / return_period).clamp(0.0, 1.0)
    };
    dist.ppf(p)
}

// Calculate RMSE between empirical and fitted CDFs
fn calculate_rmse(data: &[f64], dist: &dyn ContinuousDistribution) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len() as f64;

    let mut sum_squared_errors = 0.0;
    for (i, &x) in sorted.iter().enumerate() {
        let empirical_cdf = (i as f64 + 1.0) / n;
        let fitted_cdf = dist.cdf(x).clamp(0.0, 1.0);
        let error = empirical_cdf - fitted_cdf;
        sum_squared_errors += error * error;
    }

    (sum_squared_errors / n).sqrt()
}

// Calculate R-squared equivalent for distribution fitting
fn calculate_r_squared(data: &[f64], dist: &dyn ContinuousDistribution) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len() as f64;

    let mut ss_res = 0.0; // Sum of squares of residuals
    let mut ss_tot = 0.0; // Total sum of squares

    for (i, &x) in sorted.iter().enumerate() {
        let empirical_cdf = (i as f64 + 1.0) / n;
        let fitted_cdf = dist.cdf(x).clamp(0.0, 1.0);

        // Residual from fitted model
        let residual = empirical_cdf - fitted_cdf;
        ss_res += residual * residual;

        // Total variation from mean (0.5 for uniform CDF)
        let deviation = empirical_cdf - 0.5;
        ss_tot += deviation * deviation;
    }

    if ss_tot == 0.0 {
        return 1.0; // Perfect fit case
    }

    1.0 - (ss_res / ss_tot)
}

pub fn fit_all_candidates_diagnostic(data: &[f64]) -> Vec<DiagnosticFit> {
    let mut out = Vec::new();
    for (name, kind) in candidate_kinds() {
        if let Some(fd) = fit_distribution(kind, data) {
            let d = ks_statistic(data, &fd);
            let p = ks_pvalue_asymptotic(d, data.len());
            let rmse = calculate_rmse(data, &fd);
            let r_squared = calculate_r_squared(data, &fd);
            let (shape, loc, scale) = extract_params(&fd);

            out.push(DiagnosticFit {
                name,
                dist: fd,
                ks_stat: d,
                p_value: p,
                shape,
                loc,
                scale,
                rmse,
                r_squared,
            });
        }
    }
    out
}

pub fn fit_candidates_and_select_best_diagnostic(data: &[f64]) -> Option<DiagnosticFit> {
    fit_all_candidates_diagnostic(data)
        .into_iter()
        .max_by(|a, b| {
            a.p_value
                .partial_cmp(&b.p_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

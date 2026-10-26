//! `E[f(X)]` for a validated impact definition.
//!
//! Piecewise-linear responses on piecewise-linear quantiles use the midpoint
//! rule, which is exact on each affine piece. Unclamped linear responses use a
//! closed mean when the source has one. Everything else uses `crc-gk15/1`
//! after an integrability check. Quadrature is never treated as proof that a
//! heavy tail is integrable.

use super::definition::{Clamp, Interpolation, Model, OutsideDomain};
use super::error::{ContractResult, fail};
use super::kernel;
use super::quadrature::{self, GaussKronrod, neumaier};
use super::source::{ImpactSource, Tail};

pub const EXACT_PIECEWISE_VERSION: &str = "crc-exact-piecewise/1";
pub const EXACT_MOMENT_VERSION: &str = "crc-exact-moment/1";

/// Caller request before the engine chooses an exact or numerical method.
#[derive(Clone, Debug)]
pub enum MethodRequest {
    Auto,
    Exact,
    GaussKronrod(GaussKronrodOptions),
}

/// Tolerances and budget for adaptive Gauss-Kronrod 15/7.
#[derive(Clone, Copy, Debug)]
pub struct GaussKronrodOptions {
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub max_evaluations: u64,
}

impl GaussKronrodOptions {
    pub fn new(
        absolute_tolerance: f64,
        relative_tolerance: f64,
        max_evaluations: u64,
    ) -> ContractResult<Self> {
        GaussKronrod::new(absolute_tolerance, relative_tolerance, max_evaluations)?;
        Ok(Self {
            absolute_tolerance,
            relative_tolerance,
            max_evaluations,
        })
    }

    pub fn defaults() -> Self {
        Self {
            absolute_tolerance: quadrature::DEFAULT_ABSOLUTE_TOLERANCE,
            relative_tolerance: quadrature::DEFAULT_RELATIVE_TOLERANCE,
            max_evaluations: quadrature::DEFAULT_MAX_EVALUATIONS,
        }
    }
}

/// Which method the caller asked for, without resolved defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestedMethod {
    Auto,
    Exact,
    GaussKronrod,
}

/// Method actually used, including every numerical default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResolvedMethod {
    Exact {
        version: &'static str,
    },
    GaussKronrod {
        version: &'static str,
        absolute_tolerance: f64,
        relative_tolerance: f64,
        max_evaluations: u64,
    },
}

impl ResolvedMethod {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Exact { .. } => "exact",
            Self::GaussKronrod { .. } => "gauss_kronrod",
        }
    }
}

/// Expectation and the diagnostics needed to record how it was obtained.
#[derive(Clone, Debug)]
pub struct Expectation {
    pub value: f64,
    pub requested: RequestedMethod,
    pub method: ResolvedMethod,
    pub evaluations: u64,
    pub error_estimate: f64,
    pub exact_component: f64,
    pub numerical_component: f64,
    /// Whether an exact (peeled-atom or closed-form) term actually ran, independent
    /// of whether it happened to evaluate to zero.
    pub exact_component_present: bool,
    /// Whether quadrature actually ran, independent of whether the integral
    /// happened to evaluate to zero.
    pub numerical_component_present: bool,
    pub original_mass_sum: Option<f64>,
    pub normalization_factor: Option<f64>,
}

pub(crate) fn expected_value(
    model: &Model,
    clamp: &Clamp,
    source: &ImpactSource,
    request: MethodRequest,
) -> ContractResult<Expectation> {
    let requested = match &request {
        MethodRequest::Auto => RequestedMethod::Auto,
        MethodRequest::Exact => RequestedMethod::Exact,
        MethodRequest::GaussKronrod(_) => RequestedMethod::GaussKronrod,
    };
    if !source.is_full_probability_support() {
        return Err(fail(
            "missing_probability_support",
            "",
            "expectation requires the full probability interval",
        ));
    }
    let (original_mass_sum, normalization_factor) = match source.histogram_diagnostics() {
        Some((sum, factor)) => (Some(sum), Some(factor)),
        None => (None, None),
    };
    if let Some((value, version)) = exact_expectation(model, clamp, source)? {
        return Ok(Expectation {
            value,
            requested,
            method: ResolvedMethod::Exact { version },
            evaluations: 0,
            error_estimate: 0.0,
            exact_component: value,
            numerical_component: 0.0,
            exact_component_present: true,
            numerical_component_present: false,
            original_mass_sum,
            normalization_factor,
        });
    }
    if matches!(request, MethodRequest::Exact) {
        return Err(fail(
            "exact_method_unavailable",
            "",
            "no exact expectation is implemented for this function and source",
        ));
    }
    ensure_integrable(model, clamp, source)?;
    let options = match request {
        MethodRequest::GaussKronrod(options) => GaussKronrod::new(
            options.absolute_tolerance,
            options.relative_tolerance,
            options.max_evaluations,
        )?,
        MethodRequest::Auto => GaussKronrod::defaults(),
        MethodRequest::Exact => unreachable!("exact requests already returned"),
    };
    let (exact_component, intervals) = peel_atoms(model, clamp, source)?;
    let intervals = split_at_breakpoints(model, clamp, source, &intervals);
    let integral = quadrature::integrate(&intervals, &options, |probability| {
        let exposure = source.quantile(probability)?;
        kernel::evaluate(model, clamp, exposure)
    })?;
    let value = exact_component + integral.value;
    if !value.is_finite() {
        return Err(fail("non_finite_result", "", "expectation is not finite"));
    }
    Ok(Expectation {
        value,
        requested,
        method: ResolvedMethod::GaussKronrod {
            version: quadrature::VERSION,
            absolute_tolerance: options.absolute_tolerance,
            relative_tolerance: options.relative_tolerance,
            max_evaluations: options.max_evaluations,
        },
        evaluations: integral.evaluations,
        error_estimate: integral.error,
        exact_component,
        numerical_component: integral.value,
        // Quadrature always runs on this path. The peeled-atom term only ran
        // when the source actually carries an atom to peel.
        exact_component_present: matches!(source, ImpactSource::Hurdle(_)),
        numerical_component_present: true,
        original_mass_sum,
        normalization_factor,
    })
}

fn exact_expectation(
    model: &Model,
    clamp: &Clamp,
    source: &ImpactSource,
) -> ContractResult<Option<(f64, &'static str)>> {
    // A piecewise-constant quantile makes `f(Q(p))` piecewise constant for every
    // finite response, so the midpoint rule is exact weighted summation.
    if source.has_piecewise_constant_quantile()
        || (piecewise_linear(model) && source.has_piecewise_linear_quantile())
    {
        return Ok(Some((
            piecewise_expectation(model, clamp, source)?,
            EXACT_PIECEWISE_VERSION,
        )));
    }
    if let Some(value) = moment_expectation(model, clamp, source) {
        return Ok(Some((value, EXACT_MOMENT_VERSION)));
    }
    Ok(None)
}

fn piecewise_linear(model: &Model) -> bool {
    matches!(
        model,
        Model::Linear { .. }
            | Model::Tabulated { .. }
            | Model::Binned { .. }
            | Model::Cooling { .. }
    )
}

/// Exact integral of a piecewise-linear `f(Q(p))` by the midpoint rule.
fn piecewise_expectation(
    model: &Model,
    clamp: &Clamp,
    source: &ImpactSource,
) -> ContractResult<f64> {
    let mut breaks = source.probability_knots();
    for exposure in response_breakpoints(model, clamp) {
        let probability = source.cdf(exposure);
        if probability > 0.0 && probability < 1.0 {
            breaks.push(probability);
        }
    }
    breaks.retain(|probability| probability.is_finite());
    breaks.sort_by(f64::total_cmp);
    breaks.dedup_by(|left, right| *left == *right);
    if breaks.first() != Some(&0.0) {
        breaks.insert(0, 0.0);
    }
    if breaks.last() != Some(&1.0) {
        breaks.push(1.0);
    }
    let mut contributions = Vec::new();
    for pair in breaks.windows(2) {
        let width = pair[1] - pair[0];
        if width == 0.0 {
            continue;
        }
        let probability = 0.5 * (pair[0] + pair[1]);
        let exposure = source.quantile(probability)?;
        let response = kernel::evaluate(model, clamp, exposure)?;
        contributions.push(width * response);
    }
    let value = neumaier(contributions);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(fail(
            "non_finite_result",
            "",
            "exact expectation is not finite",
        ))
    }
}

fn moment_expectation(model: &Model, clamp: &Clamp, source: &ImpactSource) -> Option<f64> {
    let Model::Linear { slope, intercept } = model else {
        return None;
    };
    if clamp.minimum.is_some() || clamp.maximum.is_some() {
        return None;
    }
    if *slope == 0.0 {
        return Some(*intercept);
    }
    source.mean().map(|mean| slope * mean + intercept)
}

fn split_at_breakpoints(
    model: &Model,
    clamp: &Clamp,
    source: &ImpactSource,
    intervals: &[(f64, f64)],
) -> Vec<(f64, f64)> {
    let breaks: Vec<f64> = response_breakpoints(model, clamp)
        .into_iter()
        .map(|exposure| source.cdf(exposure))
        .filter(|probability| probability.is_finite())
        .collect();
    let mut split = Vec::new();
    for &(left, right) in intervals {
        let mut points = vec![left, right];
        for probability in &breaks {
            if *probability > left && *probability < right {
                points.push(*probability);
            }
        }
        points.sort_by(f64::total_cmp);
        points.dedup_by(|left, right| *left == *right);
        for pair in points.windows(2) {
            if pair[1] > pair[0] {
                split.push((pair[0], pair[1]));
            }
        }
    }
    split
}

fn peel_atoms(
    model: &Model,
    clamp: &Clamp,
    source: &ImpactSource,
) -> ContractResult<(f64, Vec<(f64, f64)>)> {
    match source {
        ImpactSource::Hurdle(distribution) => {
            let response = kernel::evaluate(model, clamp, distribution.atom_location())?;
            Ok((
                distribution.atom_probability() * response,
                vec![(distribution.atom_probability(), 1.0)],
            ))
        }
        _ => Ok((0.0, vec![(0.0, 1.0)])),
    }
}

fn ensure_integrable(model: &Model, clamp: &Clamp, source: &ImpactSource) -> ContractResult<()> {
    if restricted_domain_exited(model, source) {
        return Err(fail(
            "out_of_domain",
            "",
            "the source support leaves a table domain whose outside policy is error",
        ));
    }
    if response_is_bounded(model, clamp, source)
        || (linear_growth(model) && source.mean_is_finite())
    {
        return Ok(());
    }
    Err(fail(
        "not_integrable",
        "",
        "the response is not known to be integrable over this source",
    ))
}

fn response_is_bounded(model: &Model, clamp: &Clamp, source: &ImpactSource) -> bool {
    if clamp.minimum.is_some() && clamp.maximum.is_some() {
        return raw_stays_finite(model, source);
    }
    match model {
        Model::Sigmoid { .. } | Model::Spei { .. } => true,
        Model::Tabulated {
            outside_domain: OutsideDomain::Clamp,
            ..
        }
        | Model::Binned {
            outside_domain: OutsideDomain::Clamp,
            ..
        } => true,
        Model::Tabulated {
            outside_domain: OutsideDomain::Error,
            ..
        }
        | Model::Binned {
            outside_domain: OutsideDomain::Error,
            ..
        } => !restricted_domain_exited(model, source) && support_is_bounded(source),
        Model::Linear { slope, .. } => *slope == 0.0 || !linear_escapes(*slope, clamp, source),
        Model::Cooling { cooling_cost, .. } => {
            *cooling_cost == 0.0
                || clamp.maximum.is_some()
                || !matches!(source.support().1, Tail::PositiveInfinity)
        }
        Model::Crop { .. } => support_is_bounded(source) || crop_is_constant(model),
    }
}

fn raw_stays_finite(model: &Model, source: &ImpactSource) -> bool {
    if support_is_bounded(source) {
        return !restricted_domain_exited(model, source);
    }
    !matches!(model, Model::Crop { .. })
}

fn crop_is_constant(model: &Model) -> bool {
    let Model::Crop { weights, index, .. } = model else {
        return false;
    };
    weights
        .iter()
        .zip(index)
        .all(|(weight, coefficient)| *weight == 0.0 || *coefficient == 0.0)
}

fn linear_growth(model: &Model) -> bool {
    matches!(model, Model::Linear { .. } | Model::Cooling { .. })
}

fn linear_escapes(slope: f64, clamp: &Clamp, source: &ImpactSource) -> bool {
    if slope == 0.0 || support_is_bounded(source) {
        return false;
    }
    let (lower, upper) = source.support();
    let unbounded_below = matches!(lower, Tail::NegativeInfinity);
    let unbounded_above = matches!(upper, Tail::PositiveInfinity);
    let escapes_above = (unbounded_above && slope > 0.0 && clamp.maximum.is_none())
        || (unbounded_below && slope < 0.0 && clamp.maximum.is_none());
    let escapes_below = (unbounded_above && slope < 0.0 && clamp.minimum.is_none())
        || (unbounded_below && slope > 0.0 && clamp.minimum.is_none());
    escapes_above || escapes_below
}

fn support_is_bounded(source: &ImpactSource) -> bool {
    let (lower, upper) = source.support();
    lower.is_finite() && upper.is_finite()
}

fn restricted_domain_exited(model: &Model, source: &ImpactSource) -> bool {
    let (outside, edges) = match model {
        Model::Tabulated {
            points,
            outside_domain,
            ..
        } => (
            *outside_domain,
            [points[0].0, points.last().expect("table").0],
        ),
        Model::Binned {
            edges,
            outside_domain,
            ..
        } => (*outside_domain, [edges[0], *edges.last().expect("edges")]),
        _ => return false,
    };
    if outside != OutsideDomain::Error {
        return false;
    }
    let (lower, upper) = source.support();
    let below = match lower {
        Tail::NegativeInfinity => true,
        Tail::Finite(value) if value < edges[0] => true,
        _ => false,
    };
    let above = match upper {
        Tail::PositiveInfinity => true,
        Tail::Finite(value) if value > edges[1] => true,
        _ => false,
    };
    below || above
}

pub(crate) fn response_breakpoints(model: &Model, clamp: &Clamp) -> Vec<f64> {
    let mut points = Vec::new();
    match model {
        Model::Linear { slope, intercept } => {
            if *slope != 0.0 {
                for limit in [clamp.minimum, clamp.maximum].into_iter().flatten() {
                    points.push((limit - intercept) / slope);
                }
            }
        }
        Model::Tabulated {
            points: knots,
            interpolation,
            ..
        } => {
            points.extend(knots.iter().map(|point| point.0));
            if *interpolation == Interpolation::Linear {
                for pair in knots.windows(2) {
                    push_crossings(
                        pair[0].0,
                        pair[0].1,
                        pair[1].0,
                        pair[1].1,
                        clamp,
                        &mut points,
                    );
                }
            }
        }
        Model::Binned { edges, .. } => points.extend(edges.iter().copied()),
        Model::Sigmoid { zero_below, .. } => {
            if let Some(threshold) = zero_below {
                points.push(*threshold);
            }
        }
        Model::Cooling {
            cooling_cost,
            historic_mean,
        } => {
            points.push(1.0);
            let slope = cooling_cost / (historic_mean * 10_000_000.0);
            if slope != 0.0 {
                for limit in [clamp.minimum, clamp.maximum].into_iter().flatten() {
                    let crossing = limit / slope;
                    if crossing > 1.0 {
                        points.push(crossing);
                    }
                }
            }
        }
        Model::Spei { .. } => points.push(0.0),
        Model::Crop { .. } => {}
    }
    points.retain(|point| point.is_finite());
    points.sort_by(f64::total_cmp);
    points.dedup_by(|left, right| *left == *right);
    points
}

fn push_crossings(x0: f64, y0: f64, x1: f64, y1: f64, clamp: &Clamp, points: &mut Vec<f64>) {
    if y0 == y1 {
        return;
    }
    for limit in [clamp.minimum, clamp.maximum].into_iter().flatten() {
        let low = y0.min(y1);
        let high = y0.max(y1);
        if limit > low && limit < high {
            points.push(x0 + (x1 - x0) * (limit - y0) / (y1 - y0));
        }
    }
}

//! Adaptive Gauss-Kronrod 15/7 on a union of intervals.
//!
//! `crc-gk15/1` freezes the QUADPACK `dqk15` nodes, weights, and error
//! estimate. Nodes lie in the open interval. The driver bisects the panel
//! with the largest error estimate and breaks ties by the lower endpoint.
//! Partial integrals are accumulated in ascending endpoint order.

use super::error::{ContractResult, fail};

pub(crate) const VERSION: &str = "crc-gk15/1";
pub(crate) const DEFAULT_ABSOLUTE_TOLERANCE: f64 = 1e-10;
pub(crate) const DEFAULT_RELATIVE_TOLERANCE: f64 = 1e-8;
pub(crate) const DEFAULT_MAX_EVALUATIONS: u64 = 100_000;
const PANEL_EVALUATIONS: u64 = 15;

// Positive Kronrod abscissae on [-1, 1], including the center. Gauss nodes are
// entries 1, 3, 5, and 7.
const XGK: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_4,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_5,
    0.0,
];
const WGK: [f64; 8] = [
    0.022_935_322_010_529_224,
    0.063_092_092_629_978_55,
    0.104_790_010_322_250_18,
    0.140_653_259_715_525_9,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_8,
];
// Gauss weights for nodes XGK[1], XGK[3], XGK[5], and the center XGK[7].
const WG: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_64,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

#[derive(Clone, Debug)]
pub(crate) struct GaussKronrod {
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub max_evaluations: u64,
}

impl GaussKronrod {
    pub(crate) fn new(
        absolute_tolerance: f64,
        relative_tolerance: f64,
        max_evaluations: u64,
    ) -> ContractResult<Self> {
        if !absolute_tolerance.is_finite() || absolute_tolerance <= 0.0 {
            return Err(fail(
                "invalid_value",
                "/absolute_tolerance",
                "absolute tolerance must be finite and positive",
            ));
        }
        if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
            return Err(fail(
                "invalid_value",
                "/relative_tolerance",
                "relative tolerance must be finite and positive",
            ));
        }
        if max_evaluations == 0 {
            return Err(fail(
                "invalid_value",
                "/max_evaluations",
                "maximum evaluations must be positive",
            ));
        }
        Ok(Self {
            absolute_tolerance,
            relative_tolerance,
            max_evaluations,
        })
    }

    pub(crate) fn defaults() -> Self {
        Self {
            absolute_tolerance: DEFAULT_ABSOLUTE_TOLERANCE,
            relative_tolerance: DEFAULT_RELATIVE_TOLERANCE,
            max_evaluations: DEFAULT_MAX_EVALUATIONS,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Panel {
    left: f64,
    right: f64,
    estimate: f64,
    error: f64,
}

pub(crate) struct Integral {
    pub value: f64,
    pub error: f64,
    pub evaluations: u64,
}

pub(crate) fn integrate<F>(
    intervals: &[(f64, f64)],
    options: &GaussKronrod,
    mut integrand: F,
) -> ContractResult<Integral>
where
    F: FnMut(f64) -> ContractResult<f64>,
{
    let mut evaluations = 0_u64;
    let mut panels = Vec::new();
    for &(left, right) in intervals {
        if !(left.is_finite() && right.is_finite()) || right < left {
            return Err(fail(
                "invalid_value",
                "",
                "quadrature intervals must be finite and ordered",
            ));
        }
        if right == left {
            continue;
        }
        panels.push(panel(
            left,
            right,
            options,
            &mut evaluations,
            &mut integrand,
        )?);
    }
    if panels.is_empty() {
        return Ok(Integral {
            value: 0.0,
            error: 0.0,
            evaluations,
        });
    }
    loop {
        panels.sort_by(|left, right| left.left.total_cmp(&right.left));
        let value = neumaier(panels.iter().map(|panel| panel.estimate));
        let error = neumaier(panels.iter().map(|panel| panel.error));
        let tolerance = options
            .absolute_tolerance
            .max(options.relative_tolerance * value.abs());
        if error <= tolerance && value.is_finite() {
            return Ok(Integral {
                value,
                error,
                evaluations,
            });
        }
        if !value.is_finite() {
            return Err(fail(
                "non_finite_result",
                "",
                "quadrature produced a non-finite estimate",
            ));
        }
        let worst = worst_panel(&panels);
        let current = panels[worst];
        let mid = (current.left + current.right) * 0.5;
        if evaluations.saturating_add(PANEL_EVALUATIONS * 2) > options.max_evaluations
            || mid <= current.left
            || mid >= current.right
        {
            return Err(fail(
                "convergence_failed",
                "",
                "Gauss-Kronrod integration did not meet its tolerance",
            ));
        }
        panels.remove(worst);
        let left = panel(current.left, mid, options, &mut evaluations, &mut integrand)?;
        let right = panel(
            mid,
            current.right,
            options,
            &mut evaluations,
            &mut integrand,
        )?;
        panels.push(left);
        panels.push(right);
    }
}

fn worst_panel(panels: &[Panel]) -> usize {
    let mut selected = 0;
    for index in 1..panels.len() {
        let candidate = &panels[index];
        let current = &panels[selected];
        if candidate.error > current.error
            || (candidate.error == current.error && candidate.left < current.left)
        {
            selected = index;
        }
    }
    selected
}

fn panel<F>(
    left: f64,
    right: f64,
    options: &GaussKronrod,
    evaluations: &mut u64,
    integrand: &mut F,
) -> ContractResult<Panel>
where
    F: FnMut(f64) -> ContractResult<f64>,
{
    if evaluations.saturating_add(PANEL_EVALUATIONS) > options.max_evaluations {
        return Err(fail(
            "convergence_failed",
            "",
            "Gauss-Kronrod integration did not meet its tolerance",
        ));
    }
    let half = 0.5 * (right - left);
    let center = 0.5 * (left + right);
    let mut left_values = [0.0; 7];
    let mut right_values = [0.0; 7];
    let center_value = sample(center, evaluations, integrand)?;
    let mut gauss = center_value * WG[3];
    let mut kronrod = center_value * WGK[7];
    let mut absolute = kronrod.abs();
    for index in 0..3 {
        let node = 2 * index + 1;
        let (sum, magnitude) = pair(
            center,
            half * XGK[node],
            &mut left_values,
            &mut right_values,
            node,
            evaluations,
            integrand,
        )?;
        gauss += WG[index] * sum;
        kronrod += WGK[node] * sum;
        absolute += WGK[node] * magnitude;
    }
    for index in 0..4 {
        let node = 2 * index;
        let (sum, magnitude) = pair(
            center,
            half * XGK[node],
            &mut left_values,
            &mut right_values,
            node,
            evaluations,
            integrand,
        )?;
        kronrod += WGK[node] * sum;
        absolute += WGK[node] * magnitude;
    }
    let half_kronrod = 0.5 * kronrod;
    let mut ascent = WGK[7] * (center_value - half_kronrod).abs();
    for node in 0..7 {
        ascent += WGK[node]
            * ((left_values[node] - half_kronrod).abs()
                + (right_values[node] - half_kronrod).abs());
    }
    let estimate = kronrod * half;
    let absolute = absolute * half.abs();
    let ascent = ascent * half.abs();
    let mut error = (kronrod - gauss).abs() * half.abs();
    if ascent != 0.0 && error != 0.0 {
        error = ascent * (200.0 * error / ascent).powf(1.5).min(1.0);
    }
    let machine = f64::EPSILON;
    if absolute > f64::MIN_POSITIVE / (50.0 * machine) {
        error = error.max(50.0 * machine * absolute);
    }
    if !estimate.is_finite() || !error.is_finite() {
        return Err(fail(
            "non_finite_result",
            "",
            "quadrature produced a non-finite estimate",
        ));
    }
    Ok(Panel {
        left,
        right,
        estimate,
        error,
    })
}

fn pair<F>(
    center: f64,
    offset: f64,
    left_values: &mut [f64; 7],
    right_values: &mut [f64; 7],
    node: usize,
    evaluations: &mut u64,
    integrand: &mut F,
) -> ContractResult<(f64, f64)>
where
    F: FnMut(f64) -> ContractResult<f64>,
{
    let left = sample(center - offset, evaluations, integrand)?;
    let right = sample(center + offset, evaluations, integrand)?;
    left_values[node] = left;
    right_values[node] = right;
    Ok((left + right, left.abs() + right.abs()))
}

fn sample<F>(x: f64, evaluations: &mut u64, integrand: &mut F) -> ContractResult<f64>
where
    F: FnMut(f64) -> ContractResult<f64>,
{
    *evaluations += 1;
    let value = integrand(x)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(fail(
            "non_finite_result",
            "",
            "integrand evaluation produced a non-finite response",
        ))
    }
}

pub(crate) fn neumaier(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut total = 0.0;
    let mut compensation = 0.0;
    for value in values {
        let candidate = total + value;
        if total.abs() >= value.abs() {
            compensation += (total - candidate) + value;
        } else {
            compensation += (value - candidate) + total;
        }
        total = candidate;
    }
    total + compensation
}

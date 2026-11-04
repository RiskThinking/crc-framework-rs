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
    // The frozen rule decides convergence from Neumaier sums taken in
    // ascending endpoint order. Recomputing those sums after every bisection
    // is quadratic, so exact running sums (Shewchuk expansions) of the
    // estimates and errors decide every iteration whose outcome is clear by
    // a rigorous margin, and only borderline iterations fall back to the
    // ordered sums. The returned value always comes from the ordered sums.
    let mut value_sum = ExactSum::default();
    let mut error_sum = ExactSum::default();
    let mut magnitude = 0.0_f64;
    let mut heap = std::collections::BinaryHeap::with_capacity(panels.len());
    for panel in panels {
        value_sum.add(panel.estimate);
        error_sum.add(panel.error);
        magnitude += panel.estimate.abs();
        heap.push(Worst(panel));
    }
    loop {
        let count = heap.len();
        match running_decision(
            value_sum.approximate(),
            error_sum.approximate(),
            magnitude,
            count,
            options,
        ) {
            Decision::Continue => {}
            Decision::Undecided => {
                let (value, error) = ordered_sums(heap.iter().map(|worst| worst.0));
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
            }
        }
        let current = heap.pop().expect("at least one panel remains").0;
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
        let left = panel(current.left, mid, options, &mut evaluations, &mut integrand)?;
        let right = panel(
            mid,
            current.right,
            options,
            &mut evaluations,
            &mut integrand,
        )?;
        value_sum.add(-current.estimate);
        error_sum.add(-current.error);
        // `magnitude` only bounds the ordered-sum rounding, so it may keep
        // the retired panel's contribution.
        for child in [left, right] {
            value_sum.add(child.estimate);
            error_sum.add(child.error);
            magnitude += child.estimate.abs();
            heap.push(Worst(child));
        }
    }
}

/// Neumaier sums of the estimates and errors in ascending endpoint order.
fn ordered_sums(panels: impl Iterator<Item = Panel>) -> (f64, f64) {
    let mut ordered: Vec<Panel> = panels.collect();
    ordered.sort_by(|left, right| left.left.total_cmp(&right.left));
    (
        neumaier(ordered.iter().map(|panel| panel.estimate)),
        neumaier(ordered.iter().map(|panel| panel.error)),
    )
}

enum Decision {
    /// The ordered sums certainly miss the tolerance.
    Continue,
    /// The running sums cannot decide; compute the ordered sums.
    Undecided,
}

/// Decides from exact running sums whether the ordered Neumaier sums could
/// meet the tolerance.
///
/// For `n` terms a Neumaier sum differs from the exact sum `S` by at most
/// `2 u |S| + 2 n u^2 sum |x_i|` (Higham, ASNA 2nd ed., section 4.3). The
/// margins below are several times that bound, and the running sums are
/// themselves within one rounding of exact. Whenever the tolerance test could
/// pass, the ordered sums are computed, so convergence happens at exactly the
/// same bisection as the sort-every-iteration driver and returns the same bits.
fn running_decision(
    value: f64,
    error: f64,
    magnitude: f64,
    count: usize,
    options: &GaussKronrod,
) -> Decision {
    const LIMIT: f64 = 1e300;
    if !value.is_finite()
        || !error.is_finite()
        || !magnitude.is_finite()
        || value.abs() > LIMIT
        || error > LIMIT
        || magnitude > LIMIT
    {
        return Decision::Undecided;
    }
    let unit = f64::EPSILON;
    let spread = 8.0 * (count as f64 + 2.0) * unit * unit;
    let value_slack = 8.0 * unit * value.abs() + spread * magnitude + f64::MIN_POSITIVE;
    // Errors are non-negative, so their absolute sum is the sum itself.
    let error_low = error - (8.0 * unit + spread) * error - f64::MIN_POSITIVE;
    let largest_value = value.abs() + value_slack;
    let tolerance_high = options
        .absolute_tolerance
        .max(options.relative_tolerance * largest_value)
        * (1.0 + 8.0 * unit);
    if error_low > tolerance_high {
        Decision::Continue
    } else {
        Decision::Undecided
    }
}

/// Exact floating-point sum kept as non-overlapping partials (Shewchuk).
#[derive(Default)]
struct ExactSum {
    partials: Vec<f64>,
}

impl ExactSum {
    fn add(&mut self, value: f64) {
        let mut x = value;
        let mut kept = 0;
        for index in 0..self.partials.len() {
            let mut y = self.partials[index];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let high = x + y;
            let low = y - (high - x);
            if low != 0.0 {
                self.partials[kept] = low;
                kept += 1;
            }
            x = high;
        }
        self.partials.truncate(kept);
        self.partials.push(x);
    }

    /// Sum of the partials from the largest down; within a few ulps of exact.
    fn approximate(&self) -> f64 {
        self.partials
            .iter()
            .rev()
            .fold(0.0, |total, partial| total + partial)
    }
}

/// Max-heap order: larger error first, then the lower endpoint.
struct Worst(Panel);

impl PartialEq for Worst {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for Worst {}

impl PartialOrd for Worst {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Worst {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .error
            .total_cmp(&other.0.error)
            .then_with(|| other.0.left.total_cmp(&self.0.left))
    }
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
    for (index, gauss_weight) in WG.iter().take(3).enumerate() {
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
        gauss += gauss_weight * sum;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The original driver: re-sort and re-sum every panel after each bisection.
    fn sorted_driver<F>(
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
            let mut worst = 0;
            for index in 1..panels.len() {
                let (candidate, current) = (&panels[index], &panels[worst]);
                if candidate.error > current.error
                    || (candidate.error == current.error && candidate.left < current.left)
                {
                    worst = index;
                }
            }
            let current = panels[worst];
            let mid = (current.left + current.right) * 0.5;
            if evaluations.saturating_add(PANEL_EVALUATIONS * 2) > options.max_evaluations
                || mid <= current.left
                || mid >= current.right
            {
                return Err(fail("convergence_failed", "", "did not converge"));
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

    #[test]
    fn heap_driver_matches_the_sorted_driver_bit_for_bit() {
        type Integrand = fn(f64) -> f64;
        let integrands: [(&str, Integrand); 8] = [
            ("cubic", |p| p * p * p - 0.3 * p),
            ("sqrt", |p: f64| p.sqrt()),
            ("log", |p: f64| (p + 1e-300).ln()),
            ("step", |p| if p < 1.0 / 3.0 { 1.0 } else { -2.0 }),
            ("sigmoid", |p: f64| 1.0 / (1.0 + (-40.0 * (p - 0.5)).exp())),
            ("tail", |p: f64| (1.0 - p).powf(-0.6)),
            ("oscillation", |p: f64| (60.0 * p).sin()),
            ("cancel", |p| if p < 0.5 { 1e8 } else { -1e8 + p }),
        ];
        let tolerances = [
            (1e-10, 1e-8),
            (1e-14, 1e-12),
            (1e-6, 1e-4),
            (1e-300, 1e-300),
        ];
        let intervals: [&[(f64, f64)]; 3] = [
            &[(0.0, 1.0)],
            &[(0.0, 0.25), (0.25, 0.7), (0.7, 1.0)],
            &[(0.2, 1.0)],
        ];
        for (name, function) in integrands {
            for (absolute, relative) in tolerances {
                for budget in [15_u64, 1_000, 20_000] {
                    for interval in intervals {
                        let options = GaussKronrod::new(absolute, relative, budget).unwrap();
                        let expected = sorted_driver(interval, &options, |p| Ok(function(p)));
                        let actual = integrate(interval, &options, |p| Ok(function(p)));
                        match (expected, actual) {
                            (Ok(expected), Ok(actual)) => {
                                assert_eq!(
                                    expected.value.to_bits(),
                                    actual.value.to_bits(),
                                    "{name} {absolute} {budget}"
                                );
                                assert_eq!(expected.error.to_bits(), actual.error.to_bits());
                                assert_eq!(expected.evaluations, actual.evaluations);
                            }
                            (Err(expected), Err(actual)) => {
                                assert_eq!(expected.code, actual.code, "{name}");
                            }
                            (expected, actual) => panic!(
                                "{name} {absolute} {budget}: {:?} vs {:?}",
                                expected.map(|integral| integral.value),
                                actual.map(|integral| integral.value)
                            ),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exact_sum_survives_cancellation() {
        let mut sum = ExactSum::default();
        for value in [1e100, 1.0, -1e100, 1e-20, 3.0] {
            sum.add(value);
        }
        assert_eq!(sum.approximate(), 4.0 + 1e-20);
    }
}

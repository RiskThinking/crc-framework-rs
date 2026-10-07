//! Unbiased sample L-moments (Hosking, 1990) for annual extreme families.
//! One O(n log n) sort, O(n) moment/diagnostic passes, no likelihood optimizer.

use super::{DistributionFamily, FittedDistribution};
use crate::error::{CrcError, Result};

const EULER: f64 = 0.577_215_664_901_532_9;

// Neumaier summation keeps the signed order-statistic sums accurate.
#[derive(Default)]
struct Sum(f64, f64);
impl Sum {
    fn add(&mut self, x: f64) {
        let next = self.0 + x;
        self.1 += if self.0.abs() >= x.abs() {
            (self.0 - next) + x
        } else {
            (x - next) + self.0
        };
        self.0 = next;
    }
    fn value(&self) -> f64 {
        self.0 + self.1
    }
}

pub(super) fn fit(
    samples: &[f64],
    family: DistributionFamily,
) -> Result<(FittedDistribution, Vec<f64>)> {
    if !matches!(
        family,
        DistributionFamily::GenExtreme
            | DistributionFamily::GumbelRight
            | DistributionFamily::GumbelLeft
    ) {
        return Err(CrcError::Unsupported(
            "L-moments supports genextreme, gumbel_r and gumbel_l only".into(),
        ));
    }
    if samples.len() < 4 || samples.iter().any(|x| !x.is_finite()) {
        return Err(CrcError::InvalidInput(
            "L-moments requires at least four finite samples".into(),
        ));
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    if sorted[0] == sorted[sorted.len() - 1] {
        return Err(CrcError::InvalidInput(
            "L-moments requires nonconstant samples".into(),
        ));
    }
    // Work in centered, bounded coordinates, avoiding overflow and cancellation
    // in records with a large offset or a very small physical scale.
    let magnitude = sorted[0].abs().max(sorted[sorted.len() - 1].abs());
    let offset = sorted[sorted.len() / 2];
    let n = sorted.len() as f64;
    let mut sums = [Sum::default(), Sum::default(), Sum::default()];
    for (i, &x) in sorted.iter().enumerate() {
        let i = i as f64;
        let delta = x - offset;
        let y = if delta.is_finite() {
            delta / magnitude
        } else {
            x / magnitude - offset / magnitude
        };
        let w1 = i / (n - 1.0);
        let w2 = i * (i - 1.0) / ((n - 1.0) * (n - 2.0));
        sums[0].add(y);
        sums[1].add((2.0 * w1 - 1.0) * y);
        sums[2].add((6.0 * w2 - 6.0 * w1 + 1.0) * y);
    }
    let l1 = sums[0].value() / n;
    let l2 = sums[1].value() / n;
    let l3 = sums[2].value() / n;
    if !l2.is_finite() || l2 <= 0.0 {
        return Err(CrcError::InvalidInput(
            "L-moments requires positive L-scale".into(),
        ));
    }
    let (shape, location, scale) = match family {
        DistributionFamily::GumbelRight => {
            let scale = l2 / std::f64::consts::LN_2;
            (None, l1 - EULER * scale, scale)
        }
        DistributionFamily::GumbelLeft => {
            let scale = l2 / std::f64::consts::LN_2;
            (None, l1 + EULER * scale, scale)
        }
        DistributionFamily::GenExtreme => gev_parameters(l1, l2, l3 / l2)?,
        _ => unreachable!(),
    };
    let distribution = FittedDistribution::from_parameters(
        family,
        shape,
        location * magnitude + offset,
        scale * magnitude,
    )?;
    Ok((distribution, sorted))
}

fn gev_skew(shape: f64) -> f64 {
    if shape.abs() < 1e-8 {
        // Continuous limit and first derivative at the Gumbel shape.
        let ratio = 3.0_f64.ln() / std::f64::consts::LN_2;
        return 2.0 * ratio - 3.0 - ratio * (3.0_f64.ln() - std::f64::consts::LN_2) * shape;
    }
    2.0 * (-shape * 3.0_f64.ln()).exp_m1() / (-shape * std::f64::consts::LN_2).exp_m1() - 3.0
}

fn gev_parameters(l1: f64, l2: f64, skew: f64) -> Result<(Option<f64>, f64, f64)> {
    if !skew.is_finite() || skew.abs() >= 1.0 {
        return Err(CrcError::InvalidInput(
            "GEV L-skewness must be strictly between -1 and 1".into(),
        ));
    }
    // The exact L-skewness equation is monotone for c > -1 (finite mean).
    // Bisection avoids approximation/clipping of strongly skewed samples.
    let mut low = -1.0;
    let mut high = 1.0;
    for _ in 0..7 {
        if gev_skew(high) <= skew {
            break;
        }
        high *= 2.0;
    }
    if gev_skew(high) > skew {
        return Err(CrcError::InvalidInput(
            "GEV L-skewness cannot be resolved numerically".into(),
        ));
    }
    for _ in 0..64 {
        let mid = (low + high) / 2.0;
        if mid == low || mid == high {
            break;
        }
        if gev_skew(mid) > skew {
            low = mid;
        } else {
            high = mid;
        }
    }
    let shape = (low + high) / 2.0;
    if shape.abs() < 1e-10 {
        let scale = l2 / std::f64::consts::LN_2;
        return Ok((Some(0.0), l1 - EULER * scale, scale));
    }
    let log_gamma = log_gamma_one_plus(shape);
    let denominator = -(-shape * std::f64::consts::LN_2).exp_m1();
    let scale = l2 * (shape / denominator) * (-log_gamma).exp();
    // Algebraically scale*(Gamma(1+c)-1)/c, without gamma overflow.
    let location = l1 + l2 / denominator * -(-log_gamma).exp_m1();
    Ok((Some(shape), location, scale))
}

fn log_gamma_one_plus(x: f64) -> f64 {
    if x.abs() < 0.01 {
        // log Gamma(1+x) = -gamma*x + sum (-1)^k zeta(k)*x^k/k.
        return x
            * (-EULER
                + x * (0.822_467_033_424_113_2
                    + x * (-0.400_685_634_386_531_4
                        + x * (0.270_580_808_427_784_5
                            + x * (-0.207_385_551_028_674
                                + x * (0.169_557_176_997_408_2 - x * 0.144_049_896_768_846_1))))));
    }
    // Lanczos in log space; recurrence keeps the argument >= 1, even c -> -1.
    let z = 1.0 + x;
    if z < 1.0 {
        return log_gamma_one_plus(x + 1.0) - z.ln();
    }
    let coefficients = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let mut a = 0.999_999_999_999_809_9;
    for (i, coefficient) in coefficients.iter().enumerate() {
        a += coefficient / (x + i as f64 + 1.0);
    }
    let t = x + 7.5;
    0.918_938_533_204_672_7 + (x + 0.5) * t.ln() - t + a.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_shape_equation_including_heavy_tails() {
        for shape in [-0.95, -0.7, -0.2, -1e-5, 0.0, 1e-5, 0.2, 1.0, 5.0] {
            let (actual, _, scale) = gev_parameters(2.0, 1.0, gev_skew(shape)).unwrap();
            assert!(
                (actual.unwrap() - shape).abs() < 1e-10,
                "shape {shape}: {actual:?}"
            );
            assert!(scale > 0.0 && scale.is_finite());
        }
    }
    #[test]
    fn small_scales_and_large_offsets() {
        for samples in [
            [1e-20, 2e-20, 3e-20, 4e-20],
            [1e15, 1e15 + 1.0, 1e15 + 2.0, 1e15 + 3.0],
            [-1e308, -5e307, 5e307, 1e308],
        ] {
            let (fit, _) = fit(&samples, DistributionFamily::GumbelRight).unwrap();
            assert!(fit.scale.is_finite() && fit.scale > 0.0);
        }
    }
}

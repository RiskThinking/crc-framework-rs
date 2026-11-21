//! Versioned random streams for the annual loss simulator.

use rand::SeedableRng;
use rand::distr::{Distribution, Open01};
use rand_chacha::ChaCha12Rng;
use rand_distr::{Gamma, StandardNormal};
use sha2::{Digest, Sha256};

use crate::reference::curve_fitting::ndtr_cephes;
use crate::{CrcError, Result};

pub const ANNUAL_LOSS_RNG_ALGORITHM: &str = "crc-chacha12-ziggurat-gamma-v1";
const VARIANCE_FRACTION_CAP: f64 = 1.0 - 1e-6;
// A unit-mean Gamma has a relative deviation of 1 / sqrt(shape).
// Beyond exp(80) its fluctuations are below binary64 resolution.
const CONCENTRATED_LOG_SHAPE: f64 = 80.0;

pub(super) trait Draws {
    fn standard_normal(&mut self) -> f64;
    fn damage_ratio(&mut self, mean: f64, cv: f64) -> Result<f64>;
}

pub(super) struct SeededDraws {
    primary: ChaCha12Rng,
    secondary: ChaCha12Rng,
}

impl SeededDraws {
    pub(super) fn new(seed: u64, secondary_stream: u64) -> Self {
        let primary = seeded_stream(seed, b"primary");
        let mut secondary = seeded_stream(seed, b"secondary");
        secondary.set_stream(secondary_stream);
        Self { primary, secondary }
    }
}

fn seeded_stream(seed: u64, domain: &[u8]) -> ChaCha12Rng {
    let mut key = Sha256::new();
    key.update(ANNUAL_LOSS_RNG_ALGORITHM.as_bytes());
    key.update(domain);
    key.update(seed.to_le_bytes());
    ChaCha12Rng::from_seed(key.finalize().into())
}

impl Draws for SeededDraws {
    fn standard_normal(&mut self) -> f64 {
        StandardNormal.sample(&mut self.primary)
    }

    fn damage_ratio(&mut self, mean: f64, cv: f64) -> Result<f64> {
        if !(0.0..=1.0).contains(&mean) {
            return Err(CrcError::InvalidInput(
                "mean damage ratio must be finite and in [0, 1]".into(),
            ));
        }
        if !cv.is_finite() || cv < 0.0 {
            return Err(CrcError::InvalidInput(
                "damage CV must be finite and nonnegative".into(),
            ));
        }
        if cv == 0.0 || mean == 0.0 || mean == 1.0 {
            return Ok(mean);
        }

        let log_mean = mean.ln();
        let log_complement = (-mean).ln_1p();
        // Express the requested variance as a fraction of m(1-m) in
        // logs, so neither the squared CV nor the variance can overflow.
        let log_fraction =
            (2.0 * cv.ln() + log_mean - log_complement).min(VARIANCE_FRACTION_CAP.ln());
        // k = (1 - fraction) / fraction. exp_m1 retains the small
        // difference from one when the variance approaches the cap.
        let log_concentration = (-log_fraction.exp_m1()).ln() - log_fraction;
        let log_alpha = log_mean + log_concentration;
        let log_beta = log_complement + log_concentration;
        if log_alpha >= CONCENTRATED_LOG_SHAPE && log_beta >= CONCENTRATED_LOG_SHAPE {
            return Ok(mean);
        }

        // A ratio of independent Gamma variates yields a Beta draw. Use
        // Gamma variates scaled to unit mean and retain their ratio in logs.
        let positive = log_mean + log_unit_mean_gamma(log_alpha, &mut self.secondary);
        let negative = log_complement + log_unit_mean_gamma(log_beta, &mut self.secondary);
        let log_odds = positive - negative;
        if log_odds >= 0.0 {
            Ok(1.0 / (1.0 + (-log_odds).exp()))
        } else {
            let odds = log_odds.exp();
            Ok(odds / (1.0 + odds))
        }
    }
}

fn log_unit_mean_gamma(log_shape: f64, rng: &mut ChaCha12Rng) -> f64 {
    if log_shape >= CONCENTRATED_LOG_SHAPE {
        return 0.0;
    }
    let shape = log_shape.exp();
    if shape == 0.0 {
        // A shape below binary64 resolution yields a rounded zero draw.
        // The variance cap guarantees the other Beta shape is representable.
        return f64::NEG_INFINITY;
    }
    if shape < 1.0 {
        let gamma: f64 = Gamma::new(shape + 1.0, 1.0).unwrap().sample(rng);
        let uniform: f64 = Open01.sample(rng);
        return gamma.ln() + uniform.ln() / shape - log_shape;
    }
    let gamma: f64 = Gamma::new(shape, shape.recip()).unwrap().sample(rng);
    gamma.ln()
}

pub(super) fn normal_rank(standard_normal: f64) -> f64 {
    ndtr_cephes(standard_normal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_draws_replay_independently_of_secondary_stream_and_consumption() {
        let mut baseline = SeededDraws::new(42, 1);
        let mut replay = SeededDraws::new(42, 1);
        let mut future = SeededDraws::new(42, u64::MAX);
        let mut different_seed = SeededDraws::new(43, 1);
        let normal = baseline.standard_normal();
        assert_eq!(normal, replay.standard_normal());
        assert_eq!(normal, future.standard_normal());
        assert_ne!(normal, different_seed.standard_normal());

        let damage = baseline.damage_ratio(0.5, 0.3).unwrap();
        assert_eq!(damage, replay.damage_ratio(0.5, 0.3).unwrap());
        assert_ne!(damage, future.damage_ratio(0.5, 0.3).unwrap());
        for _ in 0..100 {
            future.damage_ratio(0.8, 0.3).unwrap();
        }
        assert_eq!(baseline.standard_normal(), future.standard_normal());
    }

    #[test]
    fn damage_draws_preserve_mean_and_cap_the_variance() {
        let mut draws = SeededDraws::new(17, 0);
        for (mean, cv) in [(0.4_f64, 0.3_f64), (0.55, 100.0)] {
            let variance = (mean * cv)
                .powi(2)
                .min(mean * (1.0 - mean) * VARIANCE_FRACTION_CAP);
            let mut total = 0.0;
            let mut square = 0.0;
            let trials = 40_000;
            for _ in 0..trials {
                let damage = draws.damage_ratio(mean, cv).unwrap();
                assert!((0.0..=1.0).contains(&damage));
                total += damage;
                square += (damage - mean).powi(2);
            }
            let sampled_mean = total / f64::from(trials);
            let sampled_variance = square / f64::from(trials);
            assert!((sampled_mean - mean).abs() < 6.0 * (variance / f64::from(trials)).sqrt());
            assert!(
                (sampled_variance / variance - 1.0).abs() < 0.04,
                "sampled={sampled_variance}, expected={variance}"
            );
        }
    }

    #[test]
    fn damage_draws_handle_extreme_scales_and_deterministic_limits() {
        let mut draws = SeededDraws::new(42, 0);
        for (mean, cv) in [
            (0.0, f64::MAX),
            (1.0, f64::MAX),
            (0.3, 0.0),
            (0.3, f64::from_bits(1)),
        ] {
            assert_eq!(draws.damage_ratio(mean, cv).unwrap(), mean);
        }
        assert!(draws.damage_ratio(f64::NAN, 0.3).is_err());
        assert!(draws.damage_ratio(0.2, f64::INFINITY).is_err());
        assert!(draws.damage_ratio(0.2, -0.1).is_err());

        for mean in [f64::from_bits(1), 1e-300, 0.3, 1.0_f64.next_down()] {
            for cv in [1e-100, 0.3, 1e100, f64::MAX] {
                for _ in 0..100 {
                    assert!((0.0..=1.0).contains(&draws.damage_ratio(mean, cv).unwrap()));
                }
            }
        }

        // Treating an overflowing Beta shape as an error or returning the
        // mean would lose real relative variation in subnormal damage ratios.
        let mean = 1e-310;
        let mut scaled_total = 0.0;
        let mut scaled_square = 0.0;
        let trials = 20_000;
        for _ in 0..trials {
            let scaled = draws.damage_ratio(mean, 0.3).unwrap() / mean;
            scaled_total += scaled;
            scaled_square += (scaled - 1.0).powi(2);
        }
        assert!((scaled_total / f64::from(trials) - 1.0).abs() < 0.015);
        assert!((scaled_square / f64::from(trials) - 0.09).abs() < 0.006);
    }

    #[test]
    fn normal_ranks_keep_the_lower_tail() {
        assert_eq!(normal_rank(0.0), 0.5);
        assert!((normal_rank(1.0) - 0.841_344_746_068_542_9).abs() < 1e-15);
        assert!((normal_rank(-8.0) / 6.220_960_574_271_784e-16 - 1.0).abs() < 1e-13);
        assert_eq!(normal_rank(f64::NEG_INFINITY), 0.0);
        assert_eq!(normal_rank(f64::INFINITY), 1.0);
    }
}

//! Regression tests for distribution numerics (skew-normal cdf, KS p-value,
//! parameter validation, NaN handling and lower-tail precision).

use crc_framework_core::{
    Distribution, DistributionFamily, EmpiricalDistribution, FittedDistribution,
    HurdleDistribution, Interpolation, PointMassDistribution, TabulatedDistribution,
    distribution::calculate_diagnostics,
};

fn fitted(family: DistributionFamily, shape: Option<f64>) -> FittedDistribution {
    FittedDistribution::from_parameters(family, shape, 0.0, 1.0).unwrap()
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE)
}

#[test]
fn skew_normal_cdf_and_ppf_match_scipy() {
    // scipy.stats.skewnorm(a).cdf(x) and .ppf(0.01), SciPy 1.16.
    let cases = [
        (3.0, -1.0, 5.624443371196608e-05, -0.4368053115828171),
        (3.0, 0.5, 0.38929437512197634, -0.4368053115828171),
        (-4.0, -0.3, 0.7539094218598664, -2.575829303548901),
        (-4.0, -2.5, 0.012419330651552273, -2.575829303548901),
        (50.0, 0.01, 0.01113428752594725, 0.007886117724306137),
        (0.0, 1.2, 0.8849303297782917, -2.326347874040841),
    ];
    for (alpha, x, cdf, q01) in cases {
        let distribution = fitted(DistributionFamily::SkewNormal, Some(alpha));
        assert!(
            relative_error(distribution.cdf(x), cdf) < 1e-10,
            "alpha={alpha} x={x}: {} vs {cdf}",
            distribution.cdf(x)
        );
        let quantile = distribution.ppf(0.01).unwrap();
        assert!(
            relative_error(quantile, q01) < 1e-10,
            "alpha={alpha}: {quantile} vs {q01}"
        );
    }
}

#[test]
fn ks_pvalue_is_one_for_a_perfect_large_sample() {
    let distribution = fitted(DistributionFamily::GumbelRight, None);
    for count in [100usize, 10_000, 100_000] {
        let samples: Vec<f64> = (0..count)
            .map(|index| {
                distribution
                    .ppf((index as f64 + 0.5) / count as f64)
                    .unwrap()
            })
            .collect();
        let metrics = calculate_diagnostics(&samples, &distribution).unwrap();
        assert!(metrics.ks_statistic <= 0.5 / count as f64 + 1e-9);
        assert!(
            metrics.ks_pvalue > 0.999,
            "n={count}: p={}",
            metrics.ks_pvalue
        );
    }
}

#[test]
fn ks_pvalue_still_rejects_a_wrong_model() {
    let truth = fitted(DistributionFamily::GumbelRight, None);
    let wrong =
        FittedDistribution::from_parameters(DistributionFamily::GumbelRight, None, 0.5, 1.0)
            .unwrap();
    let samples: Vec<f64> = (0..2_000)
        .map(|index| truth.ppf((index as f64 + 0.5) / 2_000.0).unwrap())
        .collect();
    let metrics = calculate_diagnostics(&samples, &wrong).unwrap();
    assert!(metrics.ks_pvalue < 1e-6, "p={}", metrics.ks_pvalue);
}

#[test]
fn weibull_shape_must_be_positive() {
    for family in [
        DistributionFamily::WeibullMin,
        DistributionFamily::WeibullMax,
    ] {
        for shape in [0.0, -1.0, -0.0] {
            assert!(
                FittedDistribution::from_parameters(family, Some(shape), 0.0, 1.0).is_err(),
                "{family:?} accepted shape {shape}"
            );
        }
        assert!(FittedDistribution::from_parameters(family, Some(0.5), 0.0, 1.0).is_ok());
    }
}

#[test]
fn cdf_of_nan_is_nan_for_every_distribution() {
    let mut distributions: Vec<Box<dyn Distribution>> = vec![
        Box::new(
            TabulatedDistribution::new(
                vec![0.1, 0.5, 0.9],
                vec![1.0, 2.0, 3.0],
                Interpolation::LinearProbability,
                false,
            )
            .unwrap(),
        ),
        Box::new(EmpiricalDistribution::new(vec![1.0, 2.0, 3.0]).unwrap()),
        Box::new(PointMassDistribution::new(1.0).unwrap()),
        Box::new(
            HurdleDistribution::new(0.0, 0.3, fitted(DistributionFamily::WeibullMin, Some(1.5)))
                .unwrap(),
        ),
    ];
    for (family, shape) in [
        (DistributionFamily::GenExtreme, Some(0.2)),
        (DistributionFamily::WeibullMin, Some(1.5)),
        (DistributionFamily::WeibullMax, Some(1.5)),
        (DistributionFamily::SkewNormal, Some(2.0)),
        (DistributionFamily::GumbelRight, None),
        (DistributionFamily::GumbelLeft, None),
        (DistributionFamily::GenPareto, Some(0.25)),
    ] {
        distributions.push(Box::new(fitted(family, shape)));
    }
    for distribution in &distributions {
        assert!(distribution.cdf(f64::NAN).is_nan());
    }
}

#[test]
fn lower_tail_cdfs_keep_relative_precision() {
    // scipy.stats references.
    let cases = [
        (
            fitted(DistributionFamily::GumbelLeft, None),
            -40.0,
            4.248354255291589e-18,
        ),
        (
            fitted(DistributionFamily::WeibullMin, Some(1.5)),
            1e-12,
            9.999999999999999e-19,
        ),
        (
            fitted(DistributionFamily::GenPareto, Some(0.25)),
            1e-17,
            1e-17,
        ),
        (
            fitted(DistributionFamily::GenPareto, Some(0.0)),
            1e-17,
            1e-17,
        ),
    ];
    for (distribution, x, expected) in cases {
        let actual = distribution.cdf(x);
        assert!(
            relative_error(actual, expected) < 1e-12,
            "{:?}: cdf({x}) = {actual}, expected {expected}",
            distribution.family
        );
    }
}

#[test]
fn empirical_atoms_count_tied_samples() {
    let distribution =
        EmpiricalDistribution::new(vec![3.0, 1.0, 2.0, 2.0, 2.0, 5.0, 4.0, 2.0]).unwrap();
    assert_eq!(distribution.probability_atom(2.0), 0.5);
    assert_eq!(distribution.probability_atom(1.0), 0.125);
    assert_eq!(distribution.probability_atom(5.0), 0.125);
    assert_eq!(distribution.probability_atom(2.5), 0.0);
    assert_eq!(distribution.probability_atom(0.0), 0.0);
    assert_eq!(distribution.probability_atom(9.0), 0.0);
    assert_eq!(distribution.probability_atom(f64::NAN), 0.0);
    assert_eq!(distribution.cdf_below(2.0), 0.125);
}

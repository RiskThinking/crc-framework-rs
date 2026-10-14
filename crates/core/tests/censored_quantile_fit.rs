use crc_framework_core::{Distribution, DistributionFamily, fit_censored_quantiles};

const RETURN_PERIODS: [f64; 9] = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0];
const DEPTHS: [f64; 9] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.06, 0.10];

fn probabilities() -> Vec<f64> {
    RETURN_PERIODS
        .iter()
        .map(|period| 1.0 - 1.0 / period)
        .collect()
}

#[test]
fn ottawa_sparse_return_period_curve_is_a_censored_wet_fit() {
    let probabilities = probabilities();
    let result = fit_censored_quantiles(
        &probabilities,
        &DEPTHS,
        None,
        DistributionFamily::GenExtreme,
        0.0,
        1.0e-3,
    )
    .expect("the censored Ottawa curve should fit");

    let diagnostics = result.diagnostics;
    assert_eq!(diagnostics.censored_point_count, 6);
    assert_eq!(diagnostics.uncensored_point_count, 3);
    assert!(diagnostics.residual_scale >= diagnostics.residual_scale_floor);
    assert!(diagnostics.negative_log_likelihood.is_finite());

    let latent = result
        .distribution
        .quantiles(&probabilities)
        .expect("interior probabilities are valid");
    let observed: Vec<f64> = latent.iter().map(|value| value.max(0.0)).collect();
    assert!(observed[..6].iter().all(|value| *value == 0.0));
    assert!(observed[6..].windows(2).all(|pair| pair[0] < pair[1]));
    for (actual, expected) in observed[6..].iter().zip(&DEPTHS[6..]) {
        assert!(
            (actual - expected).abs() < 0.002,
            "expected {expected}, got {actual}"
        );
    }
}

#[test]
fn weibull_and_skew_normal_use_a_censored_fit() {
    let probabilities = probabilities();
    for family in [
        DistributionFamily::WeibullMin,
        DistributionFamily::SkewNormal,
    ] {
        let result = fit_censored_quantiles(&probabilities, &DEPTHS, None, family, 0.0, 1.0e-3)
            .unwrap_or_else(|error| panic!("{family:?} should fit the censored curve: {error}"));
        assert!(result.diagnostics.residual_scale >= result.diagnostics.residual_scale_floor);
        let latent = result
            .distribution
            .quantiles(&probabilities)
            .unwrap_or_else(|error| panic!("{family:?} quantiles: {error}"));
        assert!(latent[..6].iter().all(|value| *value <= 1.0e-9));
        assert!(latent[6..].iter().all(|value| *value > 0.0));
    }
}

#[test]
fn censored_fit_requires_both_sides_of_the_censor() {
    let probabilities = probabilities();
    let error = fit_censored_quantiles(
        &probabilities,
        &[0.0; 9],
        None,
        DistributionFamily::GenExtreme,
        0.0,
        1.0e-3,
    )
    .expect_err("an all-censored row is a point mass, not a censored fit");
    assert!(
        error.to_string().contains("non-zero range") || error.to_string().contains("uncensored")
    );
}

#[test]
fn zero_weight_endpoints_do_not_change_the_censored_fit() {
    let baseline_probabilities = probabilities();
    let baseline = fit_censored_quantiles(
        &baseline_probabilities,
        &DEPTHS,
        None,
        DistributionFamily::GenExtreme,
        0.0,
        1.0e-3,
    )
    .expect("the unstretched Ottawa curve should fit");

    let mut probabilities = vec![0.1];
    probabilities.extend(baseline_probabilities);
    probabilities.push(0.9995);
    let mut values = vec![-1.0e20];
    values.extend(DEPTHS);
    values.push(1.0e20);
    let mut weights = vec![1.0; DEPTHS.len()];
    weights.insert(0, 0.0);
    weights.push(0.0);

    let stretched = fit_censored_quantiles(
        &probabilities,
        &values,
        Some(&weights),
        DistributionFamily::GenExtreme,
        0.0,
        1.0e-3,
    )
    .expect("zero-weight endpoints must not reject the fit");

    assert_eq!(stretched.diagnostics.censored_point_count, 6);
    assert_eq!(stretched.diagnostics.uncensored_point_count, 3);
    assert_eq!(
        stretched.diagnostics.observed.point_count,
        baseline.diagnostics.observed.point_count
    );
    assert_eq!(
        stretched
            .diagnostics
            .observed
            .maximum_absolute_residual
            .to_bits(),
        baseline
            .diagnostics
            .observed
            .maximum_absolute_residual
            .to_bits()
    );
    assert_eq!(
        stretched.diagnostics.residual_scale_floor.to_bits(),
        baseline.diagnostics.residual_scale_floor.to_bits()
    );
    assert_eq!(
        stretched.diagnostics.residual_scale.to_bits(),
        baseline.diagnostics.residual_scale.to_bits()
    );
    assert_eq!(
        stretched.distribution.shape.map(f64::to_bits),
        baseline.distribution.shape.map(f64::to_bits)
    );
    assert_eq!(
        stretched.distribution.location.to_bits(),
        baseline.distribution.location.to_bits()
    );
    assert_eq!(
        stretched.distribution.scale.to_bits(),
        baseline.distribution.scale.to_bits()
    );
}

#[test]
fn below_censor_placeholder_does_not_change_the_censored_fit() {
    let probabilities = probabilities();
    let baseline = fit_censored_quantiles(
        &probabilities,
        &DEPTHS,
        None,
        DistributionFamily::GenExtreme,
        0.0,
        1.0e-3,
    )
    .expect("the Ottawa curve should fit");
    let mut buried = DEPTHS;
    buried[0] = -1.0e20;
    let result = fit_censored_quantiles(
        &probabilities,
        &buried,
        None,
        DistributionFamily::GenExtreme,
        0.0,
        1.0e-3,
    )
    .expect("a below-censor placeholder must not reject the fit");
    assert_eq!(
        result.diagnostics.residual_scale_floor.to_bits(),
        baseline.diagnostics.residual_scale_floor.to_bits()
    );
    assert_eq!(
        result.diagnostics.residual_scale.to_bits(),
        baseline.diagnostics.residual_scale.to_bits()
    );
    assert_eq!(
        result.distribution.shape.map(f64::to_bits),
        baseline.distribution.shape.map(f64::to_bits)
    );
    assert_eq!(
        result.distribution.location.to_bits(),
        baseline.distribution.location.to_bits()
    );
    assert_eq!(
        result.distribution.scale.to_bits(),
        baseline.distribution.scale.to_bits()
    );
    assert_eq!(
        result.diagnostics.observed.rmse.to_bits(),
        baseline.diagnostics.observed.rmse.to_bits()
    );
    assert_eq!(
        result.diagnostics.observed.weighted_r_squared.to_bits(),
        baseline.diagnostics.observed.weighted_r_squared.to_bits()
    );
    assert_eq!(
        result
            .diagnostics
            .observed
            .maximum_absolute_residual
            .to_bits(),
        baseline
            .diagnostics
            .observed
            .maximum_absolute_residual
            .to_bits()
    );
}

#[test]
fn excessive_residual_floor_is_rejected_without_panicking() {
    let probabilities = probabilities();
    let ratio_error = fit_censored_quantiles(
        &probabilities,
        &DEPTHS,
        None,
        DistributionFamily::GenExtreme,
        0.0,
        1.0e30,
    )
    .expect_err("a floor above 1e12 is invalid input");
    assert!(ratio_error.to_string().contains("no greater than 1e12"));

    let huge = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e20, 2.0e20, 3.0e20];
    let scale_error = fit_censored_quantiles(
        &probabilities,
        &huge,
        None,
        DistributionFamily::GenExtreme,
        0.0,
        1.0e-3,
    )
    .expect_err("a huge curve must not make the floor clamp panic");
    assert!(scale_error.to_string().contains("no greater than 1e12"));
}

#[test]
#[allow(clippy::excessive_precision)]
fn residual_scale_never_undercuts_a_non_round_tripping_floor() {
    let result = fit_censored_quantiles(
        &probabilities(),
        &DEPTHS,
        None,
        DistributionFamily::GenExtreme,
        0.0,
        1.00000000000000002e-3,
    )
    .expect("a floor that ln/exp cannot round-trip should still fit");
    let floor = result.diagnostics.residual_scale_floor;
    assert!(
        floor.ln().exp() < floor,
        "this ratio must expose the ln/exp gap"
    );
    assert!(result.diagnostics.residual_scale >= floor);
}

#[test]
fn ottawa_fit_is_stable_across_nearby_residual_floors() {
    let probabilities = probabilities();
    let tails: Vec<Vec<f64>> = [1.0e-4, 1.0e-3, 1.0e-2]
        .iter()
        .map(|&floor_ratio| {
            let result = fit_censored_quantiles(
                &probabilities,
                &DEPTHS,
                None,
                DistributionFamily::GenExtreme,
                0.0,
                floor_ratio,
            )
            .expect("nearby safeguards should all produce a fit");
            result
                .distribution
                .quantiles(&probabilities[6..])
                .expect("tail probabilities are valid")
        })
        .collect();
    for pair in tails.windows(2) {
        for (left, right) in pair[0].iter().zip(&pair[1]) {
            assert!((left - right).abs() < 0.002);
        }
    }
}

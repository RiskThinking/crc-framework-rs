//! Regression tests for expectation and pushforward defects found before v1.
use crc_framework_core::{
    CurveParameters, DiscreteDistribution, EmpiricalDistribution, GaussKronrodOptions,
    HistogramExposure, ImpactDefinition, ImpactSource, MethodRequest, ResolvedMethod,
};

fn definition(model: &str, extra: &str) -> ImpactDefinition {
    ImpactDefinition::from_json(&format!(
        r#"{{
            "schema_version": "1.0",
            "semantics_version": "crc-impact/1",
            "calibration": {{
                "exposure_variable": "value",
                "exposure_unit": "1",
                "impact_metric": "response",
                "impact_unit": "1"
            }},
            "model": {model},
            {extra}
            "provenance": {{"origin": {{"kind": "user_defined"}}, "sources": [], "derived_from": []}}
        }}"#
    ))
    .expect("valid definition")
}

fn fitted(family: &str, shape: Option<f64>, location: f64, scale: f64) -> ImpactSource {
    ImpactSource::from_curve(
        CurveParameters {
            curve_kind: "fitted".into(),
            curve_type: family.into(),
            curve_shape: shape,
            curve_location: Some(location),
            curve_scale: Some(scale),
            ..CurveParameters::default()
        }
        .reconstruct()
        .expect("curve reconstructs"),
    )
    .expect("full distribution")
}

fn empirical(samples: &[f64]) -> ImpactSource {
    ImpactSource::Empirical(EmpiricalDistribution::new(samples.to_vec()).unwrap())
}

fn discrete(support: &[f64], probabilities: &[f64]) -> ImpactSource {
    ImpactSource::Discrete(
        DiscreteDistribution::new(support.to_vec(), probabilities.to_vec()).unwrap(),
    )
}

fn atom_total(impact: &ImpactDefinition, source: &ImpactSource) -> f64 {
    impact
        .transform_distribution(source)
        .unwrap()
        .atoms()
        .iter()
        .map(|atom| atom.probability)
        .sum()
}

// F1: the impact source uses the law of the empirical `ppf` (type-7
// interpolation) for its cdf and atoms too, not the step ECDF.
#[test]
fn empirical_breakpoints_use_the_interpolated_quantile_law() {
    let binned = definition(
        r#"{"kind": "binned", "bin_edges": [-10.0, 0.5, 10.0], "values": [0.0, 1.0]}"#,
        "",
    );
    let source = empirical(&[0.0, 1.0, 2.0]);
    // Q(p) = 2p, so P(Q(U) >= 0.5) = 0.75.
    let result = binned.expected_value(&source, MethodRequest::Auto).unwrap();
    assert!(matches!(result.method, ResolvedMethod::Exact { .. }));
    assert!((result.value - 0.75).abs() < 1e-15);
    let distribution = binned.transform_distribution(&source).unwrap();
    let atoms = distribution.atoms();
    assert_eq!(atoms.len(), 2);
    assert!((atoms[0].probability - 0.25).abs() < 1e-15);
    assert!((atoms[1].probability - 0.75).abs() < 1e-15);

    let capped = definition(
        r#"{"kind": "linear", "slope": 1.0, "intercept": 0.0}"#,
        r#""clamp": {"maximum": 0.5},"#,
    );
    let result = capped.expected_value(&source, MethodRequest::Auto).unwrap();
    // E[min(2U, 0.5)] = 0.25 * 0.25 + 0.75 * 0.5.
    assert!((result.value - 0.4375).abs() < 1e-15);
    let atoms = capped
        .transform_distribution(&source)
        .unwrap()
        .atoms()
        .to_vec();
    assert_eq!(atoms.len(), 1);
    assert!((atoms[0].probability - 0.75).abs() < 1e-15);
}

#[test]
fn empirical_ties_form_atoms_spanning_their_quantile_flat() {
    let identity = definition(r#"{"kind": "linear", "slope": 1.0, "intercept": 0.0}"#, "");
    // Order statistics at p = 0, 1/3, 2/3, 1: the tie at 1 spans [0, 1/3].
    let source = empirical(&[1.0, 1.0, 3.0, 5.0]);
    let distribution = identity.transform_distribution(&source).unwrap();
    let atoms = distribution.atoms();
    assert_eq!(atoms.len(), 1);
    assert_eq!(atoms[0].value, 1.0);
    assert!((atoms[0].probability - 1.0 / 3.0).abs() < 1e-15);
    let step = definition(
        r#"{"kind": "binned", "bin_edges": [0.0, 1.0, 2.0, 6.0], "values": [0.0, 1.0, 2.0]}"#,
        "",
    );
    // P(X < 2) = 1/3 + (1/3) * (1/2) = 1/2 under the interpolated law.
    let result = step.expected_value(&source, MethodRequest::Auto).unwrap();
    assert!((result.value - (0.5 * 1.0 + 0.5 * 2.0)).abs() < 1e-15);
}

// F2: two flat pieces meeting at a knot must not both claim its mass.
#[test]
fn adjacent_flat_segments_do_not_double_count_the_shared_knot() {
    let impact = definition(
        r#"{"kind": "tabulated", "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 2.0, "y": 0.0}, {"x": 3.0, "y": 1.0}], "interpolation": "linear", "outside_domain": "clamp"}"#,
        "",
    );
    let source = discrete(&[1.0, 2.5], &[0.5, 0.5]);
    let distribution = impact.transform_distribution(&source).unwrap();
    let atoms = distribution.atoms();
    assert_eq!(atoms.len(), 2);
    assert_eq!((atoms[0].value, atoms[0].probability), (0.0, 0.5));
    assert_eq!((atoms[1].value, atoms[1].probability), (0.5, 0.5));
    assert_eq!(distribution.quantile(0.7).unwrap(), 0.5);

    let plateau = definition(
        r#"{"kind": "tabulated", "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 2.0}], "interpolation": "linear", "outside_domain": "clamp"}"#,
        r#""clamp": {"maximum": 1.0},"#,
    );
    let source = discrete(&[1.0, 0.1], &[0.5, 0.5]);
    assert!((atom_total(&plateau, &source) - 1.0).abs() < 1e-15);
    let histogram = ImpactSource::Histogram(
        HistogramExposure::new(vec![0.0, 1.0, 2.0, 3.0], vec![0.25, 0.5, 0.25]).unwrap(),
    );
    assert!(atom_total(&impact, &histogram) <= 1.0 + 1e-12);
}

// F4: an error-policy table rejects transformation exactly as it rejects expectation.
#[test]
fn transform_honours_outside_domain_error() {
    for model in [
        r#"{"kind": "tabulated", "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}], "interpolation": "linear", "outside_domain": "error"}"#,
        r#"{"kind": "tabulated", "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}], "interpolation": "step_left", "outside_domain": "error"}"#,
        r#"{"kind": "binned", "bin_edges": [0.0, 1.0], "values": [5.0], "outside_domain": "error"}"#,
    ] {
        let impact = definition(model, "");
        for source in [
            discrete(&[-5.0, 0.5], &[0.5, 0.5]),
            fitted("gumbel_r", None, 0.5, 0.05),
        ] {
            let expectation = impact
                .expected_value(&source, MethodRequest::Auto)
                .unwrap_err();
            let transform = impact.transform_distribution(&source).unwrap_err();
            assert_eq!(expectation.code, "out_of_domain", "{model}");
            assert_eq!(transform.code, "out_of_domain", "{model}");
        }
        // Support inside the domain, including its closed endpoints, still works.
        let inside = discrete(&[0.0, 1.0], &[0.5, 0.5]);
        assert!((atom_total(&impact, &inside) - 1.0).abs() < 1e-15);
    }
}

// F5: the zero_below floor is exact even where `threshold - 1` rounds back.
#[test]
fn sigmoid_floor_is_exact_for_large_thresholds() {
    let impact = definition(
        r#"{"kind": "sigmoid", "midpoint": 1e17, "steepness": 1e-17, "offset": 0.3, "amplitude": 1.0, "zero_below": 1e17}"#,
        "",
    );
    assert_eq!(impact.monotonicity(), "non_decreasing");
    let source = fitted("gumbel_r", None, 1e17, 1e15);
    let distribution = impact.transform_distribution(&source).unwrap();
    let atoms = distribution.atoms();
    assert_eq!(atoms.len(), 1);
    assert_eq!(atoms[0].value, 0.0);
    assert!((atoms[0].probability - (-1.0_f64).exp()).abs() < 1e-12);
    assert_eq!(distribution.quantile(0.1).unwrap(), 0.0);
}

// F3: heavy but integrable tails have closed means; others fail honestly.
#[test]
fn heavy_tail_means_are_closed_form() {
    let identity = definition(r#"{"kind": "linear", "slope": 1.0, "intercept": 0.0}"#, "");
    for shape in [0.1, 0.45, 0.5, 0.9, 0.95] {
        let result = identity
            .expected_value(
                &fitted("genpareto", Some(shape), 0.0, 1.0),
                MethodRequest::Auto,
            )
            .unwrap();
        assert!(matches!(result.method, ResolvedMethod::Exact { .. }));
        let expected = 1.0 / (1.0 - shape);
        assert!(
            (result.value - expected).abs() <= 1e-14 * expected,
            "{shape}"
        );
    }
    // (1 - Gamma(1 + c)) / c, from a 40-digit reference.
    for (shape, expected) in [
        (-0.95, 19.442_195_064_479_49),
        (-0.5, 1.544_907_701_811_032),
        (-0.09, 0.674_367_839_696_559_4),
        (-0.05, 0.629_066_342_580_644),
        (1e-6, 0.577_214_675_846_445),
        (0.05, 0.529_914_688_744_487_1),
        (0.09, 0.494_894_575_835_541_75),
        (0.3, 0.341_764_345_645_742_7),
    ] {
        let result = identity
            .expected_value(
                &fitted("genextreme", Some(shape), 0.0, 1.0),
                MethodRequest::Auto,
            )
            .unwrap();
        assert!(
            (result.value - expected).abs() <= 1e-14 * expected,
            "{shape}: {} vs {expected}",
            result.value
        );
    }
    let floored = definition(
        r#"{"kind": "linear", "slope": 1.0, "intercept": 0.0}"#,
        r#""clamp": {"minimum": 0.5},"#,
    );
    let error = floored
        .expected_value(
            &fitted("genpareto", Some(0.6), 0.0, 1.0),
            MethodRequest::Auto,
        )
        .unwrap_err();
    assert_eq!(error.code, "convergence_failed");
}

// F6: a decreasing map over a zero-mass gap uses the generalized inverse.
#[test]
fn decreasing_map_over_a_gap_returns_the_lower_value() {
    let negate = definition(r#"{"kind": "linear", "slope": -1.0, "intercept": 0.0}"#, "");
    let source = ImpactSource::Histogram(
        HistogramExposure::new(vec![0.0, 1.0, 2.0, 3.0], vec![0.5, 0.0, 0.5]).unwrap(),
    );
    let distribution = negate.transform_distribution(&source).unwrap();
    // P(Y <= -2) = P(X >= 2) = 0.5, so Q_Y(0.5) = -2.
    assert_eq!(distribution.quantile(0.5).unwrap(), -2.0);
    assert_eq!(distribution.quantile(0.25).unwrap(), -2.5);
    assert_eq!(distribution.quantile(0.75).unwrap(), -0.5);
}

// FP3: binary-searched histogram and discrete laws agree with direct scans.
#[test]
fn indexed_histogram_and_discrete_laws_match_direct_scans() {
    let identity = definition(r#"{"kind": "linear", "slope": 1.0, "intercept": 0.0}"#, "");
    let edges = vec![0.0, 0.5, 1.0, 1.5, 4.0, 4.25, 10.0];
    let weights = vec![0.1, 0.0, 0.3, 0.2, 0.0, 0.4];
    let histogram =
        ImpactSource::Histogram(HistogramExposure::new(edges.clone(), weights.clone()).unwrap());
    let pushforward = identity.transform_distribution(&histogram).unwrap();
    let support = vec![3.0, -1.0, 2.0, 2.0, 7.5];
    let masses = vec![0.2, 0.1, 0.3, 0.1, 0.3];
    let atoms = identity
        .transform_distribution(&discrete(&support, &masses))
        .unwrap();
    for step in 0..=200 {
        let probability = step as f64 / 200.0;
        // Lower quantile of the uniform-within-bin histogram, scanning bins.
        let mut expected = *edges.last().unwrap();
        if probability <= 0.0 {
            expected = edges[0];
        } else {
            let mut cumulative = 0.0;
            for (index, weight) in weights.iter().enumerate() {
                let next = cumulative + weight;
                if *weight > 0.0 && probability <= next {
                    expected = edges[index]
                        + (probability - cumulative) / weight * (edges[index + 1] - edges[index]);
                    break;
                }
                cumulative = next;
            }
        }
        if probability > 0.0 && probability < 1.0 {
            assert_eq!(
                pushforward.quantile(probability).unwrap(),
                expected,
                "{probability}"
            );
        }
        let discrete_expected = if probability <= 0.1 {
            -1.0
        } else if probability <= 0.5 {
            2.0
        } else if probability <= 0.7 {
            3.0
        } else {
            7.5
        };
        assert_eq!(atoms.quantile(probability).unwrap(), discrete_expected);
    }
    let binned = definition(
        r#"{"kind": "binned", "bin_edges": [-2.0, 0.0, 2.0, 2.5, 8.0], "values": [1.0, 2.0, 3.0, 4.0]}"#,
        "",
    );
    let result = binned
        .expected_value(&discrete(&support, &masses), MethodRequest::Auto)
        .unwrap();
    assert!((result.value - (0.1 * 1.0 + 0.4 * 3.0 + 0.2 * 4.0 + 0.3 * 4.0)).abs() < 1e-15);
}

// FP1: a large bisection budget is no longer quadratic.
#[test]
fn large_quadrature_budget_fails_quickly() {
    let impact = definition(
        r#"{"kind": "sigmoid", "midpoint": 0.5, "steepness": 10.0}"#,
        "",
    );
    let options = GaussKronrodOptions::new(1e-300, 1e-300, 400_000).unwrap();
    let started = std::time::Instant::now();
    let error = impact
        .expected_value(
            &fitted("gumbel_r", None, 0.5, 0.1),
            MethodRequest::GaussKronrod(options),
        )
        .unwrap_err();
    assert_eq!(error.code, "convergence_failed");
    assert!(started.elapsed().as_secs_f64() < 30.0);
}

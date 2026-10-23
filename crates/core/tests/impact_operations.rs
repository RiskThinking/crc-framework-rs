//! Expectation and pushforward conformance.
//!
//! Reference values come from the published operation fixtures. This runner
//! does not call the Python reference implementation.
use crc_framework_core::{
    CurveParameters, DiscreteDistribution, GaussKronrodOptions, HistogramExposure,
    ImpactDefinition, ImpactSource, Interpolation, MethodRequest, ResolvedMethod,
};
use serde_json::Value;

const OPERATIONS: &str =
    include_str!("../../../python/crc_framework/contracts/impact/fixtures/operations.json");

fn document() -> Value {
    serde_json::from_str(OPERATIONS).expect("operations fixture is valid JSON")
}

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("numeric array")
        .iter()
        .map(|item| item.as_f64().expect("finite number"))
        .collect()
}

fn definition(value: &Value) -> ImpactDefinition {
    let text = serde_json::to_string(value).expect("definition serializes");
    ImpactDefinition::from_json(&text).expect("fixture definition is valid")
}

fn source(value: &Value) -> Result<ImpactSource, crc_framework_core::ImpactContractError> {
    match value["kind"].as_str() {
        Some("histogram") => {
            if value["within_bin"] != "uniform" {
                return Err(crc_framework_core::ImpactContractError {
                    code: "invalid_histogram",
                    path: "/within_bin".into(),
                    message: "only uniform-within-bin histograms are supported".into(),
                });
            }
            Ok(ImpactSource::Histogram(HistogramExposure::new(
                numbers(&value["bin_edges"]),
                numbers(&value["weights"]),
            )?))
        }
        Some("discrete") => Ok(ImpactSource::Discrete(DiscreteDistribution::new(
            numbers(&value["support"]),
            numbers(&value["probabilities"]),
        )?)),
        Some("canonical_distribution") => {
            let parameters: CurveParameters =
                serde_json::from_value(value["parameters"].clone()).expect("curve parameters");
            let curve = parameters
                .reconstruct()
                .expect("canonical curve reconstructs");
            ImpactSource::from_curve(curve)
        }
        other => panic!("unsupported fixture source {other:?}"),
    }
}

fn method(value: &Value) -> MethodRequest {
    match value {
        Value::String(name) if name == "auto" => MethodRequest::Auto,
        Value::String(name) if name == "exact" => MethodRequest::Exact,
        Value::Object(_) => MethodRequest::GaussKronrod(
            GaussKronrodOptions::new(
                value["absolute_tolerance"]
                    .as_f64()
                    .expect("absolute tolerance"),
                value["relative_tolerance"]
                    .as_f64()
                    .expect("relative tolerance"),
                value["max_evaluations"]
                    .as_u64()
                    .expect("evaluation budget"),
            )
            .expect("quadrature options"),
        ),
        other => panic!("unsupported method {other}"),
    }
}

fn close(actual: f64, expected: f64, tolerance: &Value) -> bool {
    let absolute = tolerance["absolute"].as_f64().expect("absolute tolerance");
    let relative = tolerance["relative"].as_f64().expect("relative tolerance");
    (actual - expected).abs() <= absolute + relative * expected.abs()
}

#[test]
fn expectation_fixtures_match_reference_values_and_errors() {
    let operations = document();
    for case in operations["expected_value"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap();
        let impact = definition(&case["definition"]);
        assert_eq!(
            impact.definition_hash(),
            case["definition_hash"].as_str().unwrap(),
            "{name}"
        );
        let outcome = source(&case["source"])
            .and_then(|exposure| impact.expected_value(&exposure, method(&case["method"])));
        match case.get("error").and_then(Value::as_str) {
            Some(code) => {
                let error = outcome.expect_err(name);
                assert_eq!(error.code, code, "{name}: {}", error.message);
            }
            None => {
                let result = outcome.unwrap_or_else(|error| panic!("{name}: {error}"));
                assert!(
                    close(
                        result.value,
                        case["expected"].as_f64().unwrap(),
                        &case["tolerance"]
                    ),
                    "{name}: got {}, expected {}, error estimate {}",
                    result.value,
                    case["expected"],
                    result.error_estimate
                );
                match case["resolved_method"].as_str().unwrap() {
                    "exact" => assert!(
                        matches!(result.method, ResolvedMethod::Exact { .. }),
                        "{name}"
                    ),
                    "gauss_kronrod" => {
                        assert!(
                            matches!(result.method, ResolvedMethod::GaussKronrod { .. }),
                            "{name}"
                        )
                    }
                    "any" => {}
                    other => panic!("unknown resolved method {other}"),
                }
                if let Some(diagnostics) = case.get("source_diagnostics") {
                    assert_eq!(
                        result.original_mass_sum,
                        diagnostics["original_mass_sum"].as_f64()
                    );
                    assert_eq!(
                        result.normalization_factor,
                        diagnostics["normalization_factor"].as_f64()
                    );
                }
            }
        }
    }
}

#[test]
fn pushforward_fixtures_match_quantiles_atoms_and_errors() {
    let operations = document();
    for case in operations["transform_distribution"]
        .as_array()
        .expect("cases")
    {
        let name = case["name"].as_str().unwrap();
        let impact = definition(&case["definition"]);
        let outcome =
            source(&case["source"]).and_then(|exposure| impact.transform_distribution(&exposure));
        match case.get("error").and_then(Value::as_str) {
            Some(code) => {
                let error = outcome.expect_err(name);
                assert_eq!(error.code, code, "{name}: {}", error.message);
            }
            None => {
                let distribution = outcome.unwrap_or_else(|error| panic!("{name}: {error}"));
                assert_eq!(distribution.algorithm(), "crc-pushforward/1");
                let probabilities = numbers(&case["probabilities"]);
                let actual = distribution
                    .quantiles(&probabilities)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                for (got, expected) in actual.iter().zip(numbers(&case["quantiles"])) {
                    assert!(
                        close(*got, expected, &case["tolerance"]),
                        "{name}: quantile {got} != {expected}"
                    );
                }
                if let Some(expected_atoms) = case.get("atoms").and_then(Value::as_array) {
                    let mut actual_atoms: Vec<(f64, f64)> = distribution
                        .atoms()
                        .iter()
                        .map(|atom| (atom.value, atom.probability))
                        .collect();
                    actual_atoms.sort_by(|left, right| left.0.total_cmp(&right.0));
                    assert_eq!(
                        actual_atoms.len(),
                        expected_atoms.len(),
                        "{name}: {actual_atoms:?}"
                    );
                    for (actual, expected) in actual_atoms.iter().zip(expected_atoms) {
                        assert!(
                            close(
                                actual.0,
                                expected["value"].as_f64().unwrap(),
                                &case["tolerance"]
                            ),
                            "{name}: atom value"
                        );
                        assert!(
                            close(
                                actual.1,
                                expected["probability"].as_f64().unwrap(),
                                &case["tolerance"]
                            ),
                            "{name}: atom probability {} != {}",
                            actual.1,
                            expected["probability"]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn decreasing_transform_uses_each_probability_not_array_reversal() {
    let operations = document();
    let case = operations["transform_distribution"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "decreasing-uniform")
        .unwrap();
    let impact = definition(&case["definition"]);
    let exposure = source(&case["source"]).unwrap();
    let distribution = impact.transform_distribution(&exposure).unwrap();
    let quantiles = distribution.quantiles(&[0.1, 0.5, 0.8]).unwrap();
    assert_eq!(quantiles, vec![1.0, 5.0, 8.0]);
    let reflected_wrongly = {
        let forward = distribution.quantiles(&[0.9, 0.5, 0.2]).unwrap();
        // Reversing an asymmetric forward image is not the decreasing quantile map.
        let mut reversed = forward.clone();
        reversed.reverse();
        reversed
    };
    assert_ne!(quantiles, reflected_wrongly);
}

#[test]
fn threshold_tie_uses_the_lower_response() {
    let impact = ImpactDefinition::from_json(
        r#"{
            "schema_version": "1.0",
            "semantics_version": "crc-impact/1",
            "calibration": {
                "exposure_variable": "value",
                "exposure_unit": "1",
                "impact_metric": "response",
                "impact_unit": "1"
            },
            "model": {
                "kind": "sigmoid",
                "midpoint": 5.0,
                "steepness": 20.0,
                "zero_below": 5.0
            },
            "provenance": {"origin": {"kind": "user_defined"}, "sources": [], "derived_from": []}
        }"#,
    )
    .unwrap();
    assert_eq!(impact.monotonicity(), "non_decreasing");
    let exposure = ImpactSource::from_curve(
        CurveParameters {
            curve_kind: "tabulated".into(),
            curve_type: "linear_probability".into(),
            curve_probabilities: Some(vec![0.0, 1.0]),
            curve_values: Some(vec![0.0, 10.0]),
            ..CurveParameters::default()
        }
        .reconstruct()
        .unwrap(),
    )
    .unwrap();
    let distribution = impact.transform_distribution(&exposure).unwrap();
    assert!((distribution.quantile(0.5).unwrap() - 0.0).abs() < 1e-12);
    assert!(distribution.quantile(0.75).unwrap() > 0.9);
    let atoms = distribution.atoms();
    assert_eq!(atoms.len(), 1);
    assert!((atoms[0].value - 0.0).abs() < 1e-12);
    assert!((atoms[0].probability - 0.5).abs() < 1e-12);
    let table = distribution
        .tabulate(&[0.25, 0.5, 0.75], Interpolation::LinearProbability)
        .unwrap();
    assert_eq!(table.probabilities(), &[0.25, 0.5, 0.75]);
}

#[test]
fn quadrature_panel_is_exact_for_low_degree_polynomials() {
    let options = GaussKronrodOptions::defaults();
    let impact = ImpactDefinition::from_json(
        r#"{
            "schema_version": "1.0",
            "semantics_version": "crc-impact/1",
            "calibration": {
                "exposure_variable": "value",
                "exposure_unit": "1",
                "impact_metric": "response",
                "impact_unit": "1"
            },
            "model": {"kind": "linear", "slope": 1.0, "intercept": 0.0},
            "clamp": {"maximum": 1.0},
            "provenance": {"origin": {"kind": "user_defined"}, "sources": [], "derived_from": []}
        }"#,
    )
    .unwrap();
    let exposure = ImpactSource::from_curve(
        CurveParameters {
            curve_kind: "fitted".into(),
            curve_type: "genpareto".into(),
            curve_shape: Some(1.0),
            curve_location: Some(0.0),
            curve_scale: Some(1.0),
            ..CurveParameters::default()
        }
        .reconstruct()
        .unwrap(),
    )
    .unwrap();
    let result = impact
        .expected_value(&exposure, MethodRequest::GaussKronrod(options))
        .unwrap();
    let expected = std::f64::consts::LN_2;
    assert!((result.value - expected).abs() <= 1e-10 + 1e-8 * expected);
    assert!(result.evaluations >= 15);
}

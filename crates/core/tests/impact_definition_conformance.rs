//! Native conformance for impact definitions, point kernels, and monotonicity.
//!
//! Expectations come only from the published artifact. This runner does not
//! call the Python reference implementation.
use std::collections::BTreeSet;

use crc_framework_core::{ImpactDefinition, impact};
use serde_json::Value;

const DEFINITIONS: &str =
    include_str!("../../../python/crc_framework/contracts/impact/fixtures/definitions.json");
const POINTS: &str =
    include_str!("../../../python/crc_framework/contracts/impact/fixtures/point-evaluations.json");
const MONOTONICITY: &str =
    include_str!("../../../python/crc_framework/contracts/impact/fixtures/monotonicity.json");
const CAPABILITIES: &str = include_str!(
    "../../../python/crc_framework/contracts/impact/catalogs/native-capabilities.json"
);
const OVERRIDES: &str =
    include_str!("../../../python/crc_framework/contracts/impact/catalogs/registry-overrides.json");

fn document(text: &str) -> Value {
    serde_json::from_str(text).expect("contract fixture is valid JSON")
}

fn exposure(case: &Value) -> f64 {
    match case.get("x_special").and_then(Value::as_str) {
        Some("NaN") => f64::NAN,
        Some("Infinity") => f64::INFINITY,
        Some("-Infinity") => f64::NEG_INFINITY,
        Some(other) => panic!("unknown special exposure {other}"),
        None => case["x"]
            .as_f64()
            .expect("exposure is a finite JSON number"),
    }
}

fn definition_from_fixture(value: &Value, name: &str) -> ImpactDefinition {
    let text = serde_json::to_string(value).expect("fixture definition serializes");
    ImpactDefinition::from_json(&text)
        .unwrap_or_else(|error| panic!("fixture {name} definition was rejected: {error}"))
}

#[test]
fn definitions_match_canonical_bytes_hashes_and_errors() {
    let definitions = document(DEFINITIONS);
    let capabilities = document(CAPABILITIES);
    assert!(
        capabilities["supported"]
            .as_array()
            .unwrap()
            .iter()
            .any(|family| family == "impact_definition")
    );
    for case in definitions["accepted"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let definition = ImpactDefinition::from_json(case["input_json"].as_str().unwrap())
            .unwrap_or_else(|error| panic!("{name}: accepted definition was rejected: {error}"));
        assert_eq!(
            definition.canonical_json(),
            case["canonical_json"].as_str().unwrap(),
            "{name}: canonical bytes differ"
        );
        assert_eq!(
            definition.definition_hash(),
            case["content_hash"].as_str().unwrap(),
            "{name}: content hash differs"
        );
        let reloaded = ImpactDefinition::from_json(definition.canonical_json())
            .unwrap_or_else(|error| panic!("{name}: canonical form did not reload: {error}"));
        assert_eq!(
            reloaded.canonical_json(),
            definition.canonical_json(),
            "{name}"
        );
        assert_eq!(
            reloaded.definition_hash(),
            definition.definition_hash(),
            "{name}"
        );
    }
    for case in definitions["rejected"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let error = ImpactDefinition::from_json(case["input_json"].as_str().unwrap())
            .err()
            .unwrap_or_else(|| panic!("{name}: rejected definition was accepted"));
        assert_eq!(
            (error.code, error.path.as_str()),
            (
                case["error"]["code"].as_str().unwrap(),
                case["error"]["path"].as_str().unwrap(),
            ),
            "{name}: expected {} at {}, got {} at {}",
            case["error"]["code"],
            case["error"]["path"],
            error.code,
            error.path
        );
    }
}

#[test]
fn every_model_kind_passes_its_point_fixtures() {
    let points = document(POINTS);
    let capabilities = document(CAPABILITIES);
    assert!(
        capabilities["supported"]
            .as_array()
            .unwrap()
            .iter()
            .any(|family| family == "point_evaluations")
    );
    let required = capabilities["model_kinds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|kind| kind.as_str().unwrap().to_string())
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    for fixture in points["fixtures"].as_array().unwrap() {
        let name = fixture["name"].as_str().unwrap();
        let definition = definition_from_fixture(&fixture["definition"], name);
        seen.insert(definition.model_kind().to_string());
        assert_eq!(
            definition.definition_hash(),
            fixture["definition_hash"].as_str().unwrap(),
            "{name}: definition hash differs"
        );
        let absolute = fixture["tolerance"]["absolute"].as_f64().unwrap();
        let relative = fixture["tolerance"]["relative"].as_f64().unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let x = exposure(case);
            match case.get("error").and_then(Value::as_str) {
                Some(expected) => {
                    let error = definition.evaluate(x).expect_err(&format!(
                        "{name}: expected {expected} at {x}, but evaluation succeeded"
                    ));
                    assert_eq!(error.code, expected, "{name} at {x}");
                }
                None => {
                    let expected = case["expected"].as_f64().unwrap();
                    let actual = definition.evaluate(x).unwrap_or_else(|error| {
                        panic!("{name}: evaluation at {x} failed: {error}")
                    });
                    let tolerance = absolute + relative * expected.abs();
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{name} at {x}: expected {expected}, actual {actual}, tolerance {tolerance}"
                    );
                }
            }
        }
    }
    assert_eq!(
        seen, required,
        "point fixtures do not cover every native model"
    );
}

#[test]
fn monotonicity_matches_the_certified_class() {
    let monotonicity = document(MONOTONICITY);
    let capabilities = document(CAPABILITIES);
    assert!(
        capabilities["supported"]
            .as_array()
            .unwrap()
            .iter()
            .any(|family| family == "monotonicity")
    );
    for fixture in monotonicity["fixtures"].as_array().unwrap() {
        let name = fixture["name"].as_str().unwrap();
        let definition = definition_from_fixture(&fixture["definition"], name);
        let actual = definition.monotonicity();
        let acceptable = fixture["acceptable"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect::<Vec<_>>();
        assert!(
            acceptable.contains(&actual),
            "{name}: {actual} is outside the acceptable set {acceptable:?}"
        );
        assert_eq!(
            actual,
            fixture["reference"].as_str().unwrap(),
            "{name}: certified class differs"
        );
    }
}

#[test]
fn registry_override_catalog_is_available_to_definition_validation() {
    let catalog = document(OVERRIDES);
    let declared = catalog["factors"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    let known = impact::registry_factors()
        .iter()
        .map(|factor| (*factor).to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(declared, known);
    for (factor, body) in catalog["factors"].as_object().unwrap() {
        for (key, specification) in body["overrides"].as_object().unwrap() {
            assert_eq!(
                impact::registry_override_nullable(factor, key),
                Some(specification["nullable"].as_bool().unwrap()),
                "{factor}.{key}"
            );
        }
    }
}

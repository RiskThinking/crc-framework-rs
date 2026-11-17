//! Current numerical documents have one shared admission and query contract.
use crc_framework_core::{CrcError, Numerical};
use serde_json::Value;
const FIXTURES: &str =
    include_str!("../../../python/crc_framework/contracts/numerical/fixtures-1.0.json");

#[test]
fn numerical_admission_and_queries_agree_with_python() {
    let cases: Value = serde_json::from_str(FIXTURES).unwrap();
    for case in cases["accepted"].as_array().unwrap() {
        let numerical = Numerical::from_value(case["document"].clone())
            .unwrap_or_else(|e| panic!("{}: {e}", case["name"]));
        for query in case["queries"].as_array().unwrap() {
            let actual = numerical.quantile(query["probability"].as_f64().unwrap());
            match query["outcome"].as_str() {
                Some("out_of_support") => assert!(
                    matches!(actual, Err(CrcError::OutOfSupport { .. })),
                    "{}",
                    case["name"]
                ),
                Some("positive_infinity") => assert_eq!(actual.unwrap(), f64::INFINITY),
                Some("negative_infinity") => assert_eq!(actual.unwrap(), f64::NEG_INFINITY),
                _ => {
                    let expected = query["value"].as_f64().unwrap();
                    let actual = actual.unwrap();
                    assert!(
                        (actual - expected).abs() <= 2e-10 * expected.abs().max(1.0),
                        "{}: {query}: {actual}",
                        case["name"]
                    );
                }
            }
        }
    }
    for case in cases["rejected"].as_array().unwrap() {
        assert!(
            Numerical::from_value(case["document"].clone()).is_err(),
            "{}",
            case["name"]
        );
    }
    for case in cases["rejected_text"].as_array().unwrap() {
        assert!(Numerical::from_json(case["text"].as_str().unwrap()).is_err());
    }
}

#[test]
fn scientific_admission_agrees_with_python() {
    let cases: Value = serde_json::from_str(include_str!(
        "../../../python/crc_framework/contracts/scientific/fixtures-1.0.json"
    ))
    .unwrap();
    for case in cases["cases"].as_array().unwrap() {
        let result = crc_framework_core::canonical::validate_document(
            case["schema"].as_str().unwrap(),
            &case["document"].to_string(),
        );
        assert_eq!(
            result.is_ok(),
            case["valid"].as_bool().unwrap(),
            "{}: {result:?}",
            case["name"]
        );
    }
}

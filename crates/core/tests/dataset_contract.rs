//! Scientific dataset conformance uses the same fixtures as the Python consumer.
use crc_framework_core::canonical::parse_document;
use crc_framework_core::dataset::validate_dataset;
use serde_json::{Value, json};

const FIXTURES: &str =
    include_str!("../../../python/crc_framework/contracts/dataset/fixtures.json");

fn fixture() -> Value {
    serde_json::from_str::<Value>(FIXTURES).unwrap()["accepted"][0]["document"].clone()
}

#[test]
fn dataset_consumes_all_shared_valid_and_invalid_records() {
    let fixtures: Value = serde_json::from_str(FIXTURES).unwrap();
    for case in fixtures["accepted"].as_array().unwrap() {
        validate_dataset(&case["document"])
            .unwrap_or_else(|error| panic!("{}: {error}", case["name"]));
    }
    for case in fixtures["rejected"].as_array().unwrap() {
        assert!(
            validate_dataset(&case["document"]).is_err(),
            "{} unexpectedly accepted",
            case["name"]
        );
    }
}

#[test]
fn dataset_json_rejects_duplicate_fields_before_validation() {
    assert!(parse_document(r#"{"metadata":{},"metadata":{},"rows":[]}"#).is_err());
}

fn encode(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn point(x: f64, y: f64, little: bool) -> String {
    let mut bytes = vec![u8::from(little)];
    bytes.extend(if little {
        1u32.to_le_bytes()
    } else {
        1u32.to_be_bytes()
    });
    bytes.extend(if little {
        x.to_le_bytes()
    } else {
        x.to_be_bytes()
    });
    bytes.extend(if little {
        y.to_le_bytes()
    } else {
        y.to_be_bytes()
    });
    encode(&bytes)
}

fn polygon(rings: &[&[(f64, f64)]]) -> String {
    let mut bytes = vec![1];
    bytes.extend(3u32.to_le_bytes());
    bytes.extend((rings.len() as u32).to_le_bytes());
    for ring in rings {
        bytes.extend((ring.len() as u32).to_le_bytes());
        for (x, y) in *ring {
            bytes.extend(x.to_le_bytes());
            bytes.extend(y.to_le_bytes());
        }
    }
    encode(&bytes)
}

fn support(geometry: &str, kind: &str) -> Value {
    let mut value = fixture();
    value["rows"][0]["geometry"] = json!(geometry);
    value["metadata"]["spatial"]["supports"] = json!([kind]);
    value
}

#[test]
fn wkb_support_is_exact_not_expanded_to_the_delivery_cell() {
    let little = point(24.0, 45.0, true);
    let big = point(24.0, 45.0, false);
    let mut value = support(&little, "point");
    validate_dataset(&value).unwrap();
    value["metadata"]["spec"]["variables"]
        .as_array_mut()
        .unwrap()
        .push(json!({"id":"second"}));
    value["metadata"]["spec"]["series"]
        .as_array_mut()
        .unwrap()
        .push(json!({"variable_id":"second","scenario_id":null,"period_id":null}));
    let mut row = value["rows"][0].clone();
    row["variable_id"] = json!("second");
    row["geometry"] = json!(big);
    value["rows"].as_array_mut().unwrap().push(row);
    value["metadata"]["report"]["counts"]["produced"] = json!(2);
    validate_dataset(&value).unwrap();
    value["rows"][1]["geometry"] = json!(point(24.1, 45.0, true));
    assert!(
        validate_dataset(&value)
            .unwrap_err()
            .to_string()
            .contains("source supports")
    );
    assert!(validate_dataset(&support(&point(181.0, 45.0, true), "point")).is_err());
    assert!(validate_dataset(&support(&(little + "00"), "point")).is_err());
}

#[test]
fn native_polygon_topology_includes_connected_interiors() {
    let shell = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)];
    let touching = [(0.0, 2.0), (1.0, 3.0), (1.0, 1.0), (0.0, 2.0)];
    validate_dataset(&support(&polygon(&[&shell, &touching]), "polygon")).unwrap();
    let barrier = [(0.0, 2.0), (2.0, 3.0), (4.0, 2.0), (2.0, 1.0), (0.0, 2.0)];
    assert!(
        validate_dataset(&support(&polygon(&[&shell, &barrier]), "polygon"))
            .unwrap_err()
            .to_string()
            .contains("disconnected")
    );
    let outside = [(8.0, 8.0), (9.0, 8.0), (9.0, 9.0), (8.0, 8.0)];
    assert!(validate_dataset(&support(&polygon(&[&shell, &outside]), "polygon")).is_err());
}

#[test]
fn optional_dimensions_are_part_of_one_exact_row_key() {
    let mut value = fixture();
    let row = value["rows"][0].clone();
    value["rows"].as_array_mut().unwrap().push(row);
    assert!(validate_dataset(&value).is_err());
}

#[test]
fn fit_records_replay_only_declared_attempts_and_actual_counts() {
    let fixtures: Value = serde_json::from_str(FIXTURES).unwrap();
    let base = fixtures["accepted"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "authoritative-fit-record")
        .unwrap()["document"]
        .clone();
    let mut value = base.clone();
    value["rows"][0]["fit"]["specification"]["family"] = json!("gumbel_l");
    assert!(validate_dataset(&value).is_err());
    value = base.clone();
    let duplicate = value["rows"][0]["fit"]["attempts"][0].clone();
    value["rows"][0]["fit"]["attempts"]
        .as_array_mut()
        .unwrap()
        .push(duplicate);
    assert!(validate_dataset(&value).is_err());
    value = base.clone();
    value["rows"][0]["fit"]["attempts"][0]["reason"] = json!("not a success");
    assert!(validate_dataset(&value).is_err());
    value = base;
    value["rows"][0]["fit"]["used_count"] = json!(2);
    assert!(validate_dataset(&value).is_err());
}

#[test]
fn native_taxonomy_consumes_the_same_current_catalogs() {
    let fixtures: Value = serde_json::from_str(include_str!(
        "../../../python/crc_framework/contracts/taxonomy/fixtures.json"
    ))
    .unwrap();
    for case in fixtures["accepted"].as_array().unwrap() {
        crc_framework_core::dataset::validate_catalog(&case["document"])
            .unwrap_or_else(|error| panic!("{}: {error}", case["name"]));
    }
    for case in fixtures["rejected"].as_array().unwrap() {
        assert!(
            crc_framework_core::dataset::validate_catalog(&case["document"]).is_err(),
            "{}",
            case["name"]
        );
    }
}

#[test]
fn standalone_fitting_validation_uses_the_exact_declared_family() {
    use crc_framework_core::dataset::validate_specification;
    validate_specification(&json!({"kind":"samples","family":"gumbel_r"})).unwrap();
    assert!(
        validate_specification(
            &json!({"kind":"samples","family":"weibull_min","estimator":"lmoments"})
        )
        .is_err()
    );
    assert!(validate_specification(&json!({"kind":"model_selection","candidates":[{"kind":"samples","family":"gumbel_r"},{"kind":"quantiles","family":"gumbel_r"}]})).is_err());
    assert!(
        validate_specification(&json!({"kind":"quantiles","family":"gumbel_r","weights":[0,0]}))
            .is_err()
    );
    assert!(validate_specification(&json!({"kind":"quantiles","family":"gumbel_r","constraints":{"minimum_value":2,"maximum_value":1}})).is_err());
}

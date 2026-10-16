use crc_framework_core::CurveParameters;
use serde_json::{Value, json};

fn reconstruct(row: Value) -> crc_framework_core::Result<crc_framework_core::CanonicalCurve> {
    serde_json::from_value::<CurveParameters>(row)
        .unwrap()
        .reconstruct()
}

#[test]
fn native_python_reference_quantiles() {
    let cases: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/canonical-quantiles.json"
    ))
    .unwrap();
    for case in cases.as_array().unwrap() {
        let curve = reconstruct(case["properties"].clone()).unwrap();
        for (q, expected) in case["probabilities"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["expected"].as_array().unwrap())
        {
            let actual = curve.quantile(q.as_f64().unwrap()).unwrap();
            if expected.is_null() {
                assert_eq!(actual, None);
            } else {
                let expected = expected.as_f64().unwrap();
                assert!(
                    (actual.unwrap() - expected).abs() <= 2e-10 * expected.abs().max(1.0),
                    "{} q={q}",
                    case["name"]
                );
            }
        }
    }
}

#[test]
fn canonical_schema_rejects_malformed_records_even_outside_support() {
    let valid =
        json!({"curve_kind":"fitted","curve_type":"gumbel_r","curve_location":0,"curve_scale":1});
    let mut bad_rows = vec![
        json!({"curve_kind":"other","curve_type":"gumbel_r"}),
        json!({"curve_kind":"no_data","curve_type":"unknown"}),
        json!({"curve_kind":"point_mass","curve_type":"point_mass","curve_location":0}),
        json!({"curve_kind":"tabulated","curve_type":"linear_probability","curve_probabilities":[0.1,0.1],"curve_values":[1,2]}),
    ];
    for (field, value) in [
        ("curve_scale", json!(0)),
        ("curve_type", json!("unknown")),
        ("curve_atom_probability", json!(0.2)),
        ("curve_probabilities", json!([0, 1])),
    ] {
        let mut row = valid.clone();
        row[field] = value;
        bad_rows.push(row);
    }
    for row in bad_rows {
        assert!(reconstruct(row.clone()).is_err(), "{row}");
    }
    let curve = reconstruct(valid).unwrap();
    for q in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert!(curve.quantile(q).is_err());
    }
}

#[test]
fn gumbel_shape_normalization_and_signed_zero() {
    let curve = CurveParameters {
        curve_kind: "fitted".into(),
        curve_type: "gumbel_r".into(),
        curve_shape: Some(f64::NAN),
        curve_location: Some(0.0),
        curve_scale: Some(1.0),
        ..Default::default()
    }
    .reconstruct()
    .unwrap();
    assert!(curve.quantile(0.5).unwrap().is_some());
    let curve = reconstruct(json!({"curve_kind":"point_mass","curve_type":"point_mass",
        "curve_location":-0.0,"curve_scale":0,"curve_atom_probability":1,"curve_atom_location":0}))
    .unwrap();
    assert!(curve.quantile(0.5).unwrap().unwrap().is_sign_negative());
}

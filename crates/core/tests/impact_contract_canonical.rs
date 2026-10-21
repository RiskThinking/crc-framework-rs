//! Confirms that the native JSON serializer reproduces the canonical profile
//! pinned by the independently generated impact contract fixtures.
use serde_json::Value;

const CANONICAL: &str =
    include_str!("../../../python/crc_framework/contracts/impact/fixtures/canonical-json.json");

const NATIVE_CAPABILITIES: &str = include_str!(
    "../../../python/crc_framework/contracts/impact/catalogs/native-capabilities.json"
);

const CONTRACT_FIXTURES: [&str; 5] = [
    include_str!("../../../python/crc_framework/contracts/impact/fixtures/definitions.json"),
    include_str!(
        "../../../python/crc_framework/contracts/impact/fixtures/registry-references.json"
    ),
    include_str!("../../../python/crc_framework/contracts/impact/fixtures/evaluation-records.json"),
    include_str!("../../../python/crc_framework/contracts/impact/fixtures/spatial-policies.json"),
    include_str!("../../../python/crc_framework/contracts/impact/fixtures/spatial-records.json"),
];

fn fixtures(text: &str) -> Value {
    serde_json::from_str(text).expect("fixture is valid JSON")
}

#[test]
fn numbers_parse_to_pinned_bits_and_format_canonically() {
    let canonical = fixtures(CANONICAL);
    for case in canonical["numbers"].as_array().unwrap() {
        let input = case["input"].as_str().unwrap();
        let bits = u64::from_str_radix(case["bits"].as_str().unwrap(), 16).unwrap();
        let parsed: f64 = serde_json::from_str(input).unwrap();
        let parsed = if parsed == 0.0 { 0.0 } else { parsed };
        assert_eq!(parsed.to_bits(), bits, "parsing {input}");
        assert_eq!(
            serde_json::to_string(&f64::from_bits(bits)).unwrap(),
            case["canonical"].as_str().unwrap(),
            "formatting {input}"
        );
    }
}

#[test]
fn strings_use_the_canonical_escapes() {
    let canonical = fixtures(CANONICAL);
    for case in canonical["strings"].as_array().unwrap() {
        let encoded = serde_json::to_string(case["value"].as_str().unwrap()).unwrap();
        assert_eq!(
            encoded,
            case["canonical"].as_str().unwrap(),
            "{}",
            case["name"]
        );
    }
}

#[test]
fn accepted_canonical_documents_round_trip_byte_for_byte() {
    for text in CONTRACT_FIXTURES {
        let document = fixtures(text);
        for case in document["accepted"].as_array().unwrap() {
            let canonical = case["canonical_json"].as_str().unwrap();
            let value: Value = serde_json::from_str(canonical).unwrap();
            assert_eq!(
                serde_json::to_string(&value).unwrap(),
                canonical,
                "{} {}",
                document["contract"],
                case["name"]
            );
        }
    }
}

#[test]
fn exponent_inputs_are_normalized_to_exact_canonical_bytes() {
    let canonical = fixtures(CANONICAL);
    for input in ["1e+16", "1e+16 "] {
        let case = canonical["numbers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["input"] == input)
            .expect("exponent regression vector is present");
        let value: f64 = serde_json::from_str(input).unwrap();
        assert_eq!(
            serde_json::to_string(&value).unwrap(),
            case["canonical"].as_str().unwrap(),
            "{input:?} must serialize to the exact canonical spelling"
        );
        assert_eq!(case["canonical"], "1e+16");
    }
}

#[test]
fn native_capability_is_only_canonical_json() {
    let capabilities = fixtures(NATIVE_CAPABILITIES);
    assert_eq!(
        capabilities["supported"],
        serde_json::json!(["canonical_json"])
    );
    assert_eq!(
        capabilities["required_fixture_families"]["canonical_json"],
        serde_json::json!(["numbers", "strings"])
    );
    assert!(
        capabilities["future_families"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("strict_duplicate_key_parser"))
    );
}

#[test]
fn native_json_consumer_rejects_non_json_numeric_vectors() {
    let canonical = fixtures(CANONICAL);
    let capabilities = fixtures(NATIVE_CAPABILITIES);
    for name in capabilities["native_parser_rejections"].as_array().unwrap() {
        let name = name.as_str().unwrap();
        let case = canonical["rejected"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == name)
            .expect("native parser fixture exists");
        assert!(
            serde_json::from_str::<Value>(case["input"].as_str().unwrap()).is_err(),
            "{name} must not be accepted by serde_json"
        );
    }
}

#[test]
fn accepted_distribution_source_snapshots_reconstruct_in_the_native_core() {
    let document = fixtures(CONTRACT_FIXTURES[2]);
    for case in document["accepted"].as_array().unwrap() {
        let record: Value = serde_json::from_str(case["canonical_json"].as_str().unwrap()).unwrap();
        if record["source"]["kind"] != "canonical_distribution" {
            continue;
        }
        let parameters: crc_framework_core::CurveParameters =
            serde_json::from_value(record["source"]["parameters"].clone()).unwrap();
        let curve = parameters.reconstruct().unwrap_or_else(|error| {
            panic!(
                "source snapshot in {} cannot reconstruct: {error}",
                case["name"]
            )
        });
        assert!(!matches!(curve, crc_framework_core::CanonicalCurve::NoData));
    }
}

#[test]
fn integer_metadata_uses_native_u64_bounds() {
    assert_eq!(
        serde_json::from_str::<u64>("18446744073709551615").unwrap(),
        u64::MAX
    );
    assert!(serde_json::from_str::<u64>("18446744073709551616").is_err());
}

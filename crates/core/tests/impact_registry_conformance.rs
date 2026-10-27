//! Registry reference documents and frozen resolution.
//!
//! Fixture expectations come from the published artifact. Resolution uses the
//! installed climate tables and does not call the Python oracle.
use std::collections::HashMap;
use std::str::FromStr;

use crc_framework_core::{
    ImpactContext, ImpactDefinition, ImpactRegistry,
    impact::{
        RegistryImpactReference, ResolutionContext, manifest_json, manifest_sha256,
        registry_identity,
    },
};
use h3o::{CellIndex, Resolution};
use serde_json::{Value, json};

const REFERENCES: &str = include_str!(
    "../../../python/crc_framework/contracts/impact/fixtures/registry-references.json"
);

fn document(text: &str) -> Value {
    serde_json::from_str(text).expect("registry fixture is valid JSON")
}

fn reference(
    factor: &str,
    fallback: &str,
    overrides: Value,
    defaults: Value,
) -> RegistryImpactReference {
    let text = json!({
        "schema_version": "1.0",
        "registry_id": "crc-climate-impacts",
        "registry_revision": registry_identity().1,
        "registry_manifest_sha256": manifest_sha256(),
        "resolver_semantics_version": "crc-impact-registry/1",
        "factor": factor,
        "overrides": overrides,
        "context_defaults": defaults,
        "fallback_policy": fallback,
    })
    .to_string();
    RegistryImpactReference::from_json(&text)
        .unwrap_or_else(|error| panic!("{factor} reference was rejected: {error}"))
}

fn sample_cell() -> String {
    "8400001ffffffff".to_string()
}

fn context_for(factor: &str, agricultural: bool) -> ResolutionContext {
    let building = if agricultural {
        "Agricultural buildings"
    } else {
        "Commercial buildings"
    };
    match factor {
        "daily_freezethaw_cycles" | "frost_days" | "rx1day" => ResolutionContext {
            cell: Some(sample_cell()),
            ..ResolutionContext::default()
        },
        "cyclone" | "wind_max_daily_max" => ResolutionContext {
            country: Some("USA".to_string()),
            ..ResolutionContext::default()
        },
        "cflood" | "rflood" | "inundation" => ResolutionContext {
            continent: Some("Europe".to_string()),
            building_type: Some("Residential buildings".to_string()),
            ..ResolutionContext::default()
        },
        "spei" => ResolutionContext {
            country: Some("USA".to_string()),
            building_type: Some(building.to_string()),
            ..ResolutionContext::default()
        },
        "cooling_degree_days" | "maximum_consecutive_dry_days" => ResolutionContext {
            country: Some("USA".to_string()),
            building_type: Some(building.to_string()),
            historic_mean: Some(410.5),
            ..ResolutionContext::default()
        },
        _ => ResolutionContext::default(),
    }
}

fn assert_reloads(definition: &ImpactDefinition, label: &str) {
    let response = definition
        .evaluate(3.0)
        .unwrap_or_else(|error| panic!("{label} did not evaluate: {error}"));
    assert!(response.is_finite(), "{label}");
    let reloaded = ImpactDefinition::from_json(definition.canonical_json())
        .unwrap_or_else(|error| panic!("{label} snapshot did not reload: {error}"));
    assert_eq!(
        reloaded.canonical_json(),
        definition.canonical_json(),
        "{label}"
    );
    assert_eq!(
        reloaded.evaluate(3.0).unwrap(),
        response,
        "{label} replay changed the response"
    );
    assert_eq!(
        reloaded.monotonicity(),
        definition.monotonicity(),
        "{label}"
    );
}

#[test]
fn registry_reference_fixtures_match_canonical_bytes_and_errors() {
    let references = document(REFERENCES);
    for case in references["accepted"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let reference = RegistryImpactReference::from_json(case["input_json"].as_str().unwrap())
            .unwrap_or_else(|error| panic!("{name}: accepted reference was rejected: {error}"));
        assert_eq!(
            reference.canonical_json(),
            case["canonical_json"].as_str().unwrap(),
            "{name}: canonical bytes differ"
        );
        assert_eq!(
            reference.reference_hash(),
            case["content_hash"].as_str().unwrap(),
            "{name}: content hash differs"
        );
        let reloaded = RegistryImpactReference::from_json(reference.canonical_json())
            .unwrap_or_else(|error| panic!("{name}: canonical form did not reload: {error}"));
        assert_eq!(reloaded.canonical_json(), reference.canonical_json());
        assert_eq!(reloaded.reference_hash(), reference.reference_hash());
    }
    for case in references["rejected"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let error = RegistryImpactReference::from_json(case["input_json"].as_str().unwrap())
            .err()
            .unwrap_or_else(|| panic!("{name}: rejected reference was accepted"));
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
fn installed_manifest_is_stable() {
    assert_eq!(manifest_sha256().len(), 64);
    assert!(
        manifest_sha256()
            .chars()
            .all(|char| char.is_ascii_hexdigit() && !char.is_ascii_uppercase())
    );
    assert_eq!(manifest_sha256(), manifest_sha256());
    assert!(manifest_json().contains("\"registry_id\":\"crc-climate-impacts\""));
    assert!(manifest_json().contains(&format!(
        "\"registry_revision\":\"{}\"",
        registry_identity().1
    )));
    assert!(!manifest_json().contains(' '));
    let manifest: Value = serde_json::from_str(manifest_json()).unwrap();
    let cooling = manifest["factors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|factor| factor["factor"] == "cooling_degree_days")
        .unwrap();
    assert_eq!(
        cooling["branch_calibrations"]["agricultural"]["impact_metric"],
        "crop_yield_change"
    );
    assert_eq!(
        cooling["branch_calibrations"]["non_agricultural"]["impact_metric"],
        "normalized_cooling_cost"
    );
    let cooling_cost = cooling["overrides"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["key"] == "cooling_cost")
        .unwrap();
    assert_eq!(cooling_cost["branches"], json!(["non_agricultural"]));
    let tables = manifest["tables"].as_array().unwrap();
    let precipitation = tables
        .iter()
        .find(|table| table["id"] == "precipitation.rx1day")
        .unwrap();
    assert_eq!(precipitation["key_normalization"], "h3_parent_resolution_4");
    let wind = tables
        .iter()
        .find(|table| table["id"] == "wind.cyclone")
        .unwrap();
    assert!(wind.get("key_normalization").is_none());
}

#[test]
fn every_factor_and_branch_resolves_to_a_replayable_snapshot() {
    for factor in [
        "daily_freezethaw_cycles",
        "frost_days",
        "rx1day",
        "cyclone",
        "wind_max_daily_max",
        "fwi",
        "hot_days",
        "cflood",
        "rflood",
        "inundation",
        "spei",
        "cooling_degree_days",
        "maximum_consecutive_dry_days",
    ] {
        let agricultural = factor == "maximum_consecutive_dry_days";
        let definition = reference(factor, "error", json!({}), json!({}))
            .resolve(&context_for(factor, agricultural))
            .unwrap_or_else(|error| panic!("{factor} did not resolve: {error}"));
        assert_reloads(&definition, factor);
        let origin: Value = serde_json::from_str(definition.canonical_json()).unwrap();
        assert_eq!(origin["provenance"]["origin"]["kind"], "registry");
        assert_eq!(
            origin["provenance"]["origin"]["registry_manifest_sha256"],
            manifest_sha256()
        );
        if factor != "fwi" && factor != "hot_days" {
            assert!(
                !origin["provenance"]["origin"]["lookups"]
                    .as_array()
                    .unwrap()
                    .is_empty(),
                "{factor} consulted no tables"
            );
        }
    }

    let cooling = reference("cooling_degree_days", "error", json!({}), json!({}))
        .resolve(&context_for("cooling_degree_days", true))
        .unwrap_or_else(|error| panic!("agricultural cooling did not resolve: {error}"));
    assert_eq!(cooling.model_kind(), "crop");
    let cooling_document: Value = serde_json::from_str(cooling.canonical_json()).unwrap();
    assert_eq!(
        cooling_document["calibration"]["impact_metric"],
        "crop_yield_change"
    );
    assert_reloads(&cooling, "agricultural cooling");

    let rejected = reference(
        "maximum_consecutive_dry_days",
        "error",
        json!({}),
        json!({}),
    )
    .resolve(&context_for("maximum_consecutive_dry_days", false))
    .expect_err("non-agricultural dry days must fail");
    assert_eq!(
        (rejected.code, rejected.path.as_str()),
        ("unsupported_value", "/context/building_type")
    );
}

#[test]
fn resolution_checks_identity_context_overrides_and_fallbacks() {
    let mut wrong_revision = json!({
        "schema_version": "1.0",
        "registry_id": "crc-climate-impacts",
        "registry_revision": "1999.1.0",
        "registry_manifest_sha256": manifest_sha256(),
        "resolver_semantics_version": "crc-impact-registry/1",
        "factor": "fwi",
    });
    let error = RegistryImpactReference::from_json(&wrong_revision.to_string())
        .unwrap()
        .resolve(&ResolutionContext::default())
        .expect_err("wrong revision");
    assert_eq!(error.code, "registry_unavailable");
    assert_eq!(error.path, "/registry_revision");

    wrong_revision["registry_revision"] = json!(registry_identity().1);
    wrong_revision["registry_id"] = json!("other-registry");
    let error = RegistryImpactReference::from_json(&wrong_revision.to_string())
        .unwrap()
        .resolve(&ResolutionContext::default())
        .expect_err("wrong registry");
    assert_eq!(
        (error.code, error.path.as_str()),
        ("registry_unavailable", "/registry_id")
    );

    wrong_revision["registry_id"] = json!("crc-climate-impacts");
    let mut digest = manifest_sha256().to_string();
    digest.replace_range(0..1, "0");
    if digest == manifest_sha256() {
        digest.replace_range(0..1, "1");
    }
    wrong_revision["registry_manifest_sha256"] = json!(digest);
    let error = RegistryImpactReference::from_json(&wrong_revision.to_string())
        .unwrap()
        .resolve(&ResolutionContext::default())
        .expect_err("wrong manifest");
    assert_eq!(
        (error.code, error.path.as_str()),
        ("registry_manifest_mismatch", "/registry_manifest_sha256")
    );

    let missing = reference("cyclone", "error", json!({}), json!({}))
        .resolve(&ResolutionContext::default())
        .expect_err("country is required");
    assert_eq!(
        (missing.code, missing.path.as_str()),
        ("missing_context", "/context/country")
    );
    let from_default = reference("cyclone", "error", json!({}), json!({"country": "USA"}))
        .resolve(&ResolutionContext::default())
        .unwrap_or_else(|error| panic!("context default was not applied: {error}"));
    assert_reloads(&from_default, "cyclone default country");

    let unknown = RegistryImpactReference::from_json(
        &json!({
            "schema_version": "1.0",
            "registry_id": "crc-climate-impacts",
            "registry_revision": registry_identity().1,
            "registry_manifest_sha256": manifest_sha256(),
            "resolver_semantics_version": "crc-impact-registry/1",
            "factor": "spei",
            "overrides": {"midpiont": 1.0},
        })
        .to_string(),
    )
    .expect_err("unknown override");
    assert_eq!(
        (unknown.code, unknown.path.as_str()),
        ("unknown_override", "/overrides/midpiont")
    );

    let branch = reference(
        "cooling_degree_days",
        "error",
        json!({"cooling_cost": 12.0}),
        json!({}),
    )
    .resolve(&context_for("cooling_degree_days", true))
    .expect_err("agricultural cooling rejects cooling_cost");
    assert_eq!(
        (branch.code, branch.path.as_str()),
        ("inapplicable_override", "/overrides/cooling_cost")
    );

    let flood = reference("cflood", "error", json!({}), json!({}))
        .resolve(&ResolutionContext {
            continent: Some("Europe".to_string()),
            building_type: Some("Unknown buildings".to_string()),
            ..ResolutionContext::default()
        })
        .expect_err("unknown flood building");
    assert_eq!(flood.code, "fallback_required");

    let fallback = reference("cyclone", "error", json!({}), json!({}))
        .resolve(&ResolutionContext {
            country: Some("ZZZ".to_string()),
            ..ResolutionContext::default()
        })
        .expect_err("missing cyclone country");
    assert_eq!(fallback.code, "fallback_required");
    let legacy = reference("cyclone", "legacy_defaults", json!({}), json!({}))
        .resolve(&ResolutionContext {
            country: Some("ZZZ".to_string()),
            ..ResolutionContext::default()
        })
        .unwrap_or_else(|error| panic!("legacy cyclone fallback failed: {error}"));
    let legacy_document: Value = serde_json::from_str(legacy.canonical_json()).unwrap();
    let lookup = &legacy_document["provenance"]["origin"]["lookups"][0];
    assert_eq!(lookup["status"], "fallback");
    assert_eq!(lookup["resolved_key"], "global_default");
    assert_reloads(&legacy, "legacy cyclone");

    let precip = reference(
        "daily_freezethaw_cycles",
        "legacy_defaults",
        json!({}),
        json!({}),
    )
    .resolve(&ResolutionContext {
        cell: Some("8406121ffffffff".to_string()),
        ..ResolutionContext::default()
    });
    match precip {
        Err(error) => assert_eq!(
            (error.code, error.path.as_str()),
            ("fallback_required", "/context/cell")
        ),
        Ok(_) => panic!("precipitation has no global default"),
    }
}

#[test]
fn frozen_origin_keeps_effective_context_and_normalized_keys() {
    let plain = reference("fwi", "error", json!({}), json!({}))
        .resolve(&ResolutionContext::default())
        .unwrap();
    let with_unused = reference(
        "fwi",
        "error",
        json!({}),
        json!({"country": "USA", "historic_mean": 12.0}),
    )
    .resolve(&ResolutionContext {
        cell: Some(sample_cell()),
        continent: Some("Europe".to_string()),
        building_type: Some("Commercial buildings".to_string()),
        ..ResolutionContext::default()
    })
    .unwrap();
    assert_eq!(with_unused.definition_hash(), plain.definition_hash());
    assert_eq!(with_unused.canonical_json(), plain.canonical_json());
    let origin: Value = serde_json::from_str(plain.canonical_json()).unwrap();
    for field in [
        "cell",
        "country",
        "continent",
        "building_type",
        "historic_mean",
    ] {
        assert!(
            origin["provenance"]["origin"]["context"][field].is_null(),
            "{field}"
        );
    }

    let input = CellIndex::from_str(&sample_cell()).unwrap();
    assert!(input.resolution() >= Resolution::Four);
    let parent = input.parent(Resolution::Four).unwrap();
    let child = parent.center_child(Resolution::Seven).unwrap();
    let child_text = format!("{:015x}", u64::from(child));
    let parent_text = format!("{:015x}", u64::from(parent));
    let from_child = reference("rx1day", "error", json!({}), json!({"country": "USA"}))
        .resolve(&ResolutionContext {
            cell: Some(child_text.clone()),
            ..ResolutionContext::default()
        })
        .unwrap();
    let from_parent = reference("rx1day", "error", json!({}), json!({}))
        .resolve(&ResolutionContext {
            cell: Some(parent_text.clone()),
            ..ResolutionContext::default()
        })
        .unwrap();
    let child_doc: Value = serde_json::from_str(from_child.canonical_json()).unwrap();
    let parent_doc: Value = serde_json::from_str(from_parent.canonical_json()).unwrap();
    assert_eq!(
        child_doc["provenance"]["origin"]["context"]["cell"],
        child_text
    );
    assert!(child_doc["provenance"]["origin"]["context"]["country"].is_null());
    let lookup = &child_doc["provenance"]["origin"]["lookups"][0];
    assert_eq!(lookup["status"], "exact");
    assert_eq!(lookup["requested_key"], lookup["resolved_key"]);
    assert_eq!(lookup["requested_key"]["h3_resolution"], "4");
    assert_eq!(lookup["requested_key"]["cell"], parent_text);
    assert_eq!(
        child_doc["model"], parent_doc["model"],
        "cells that share an H3 parent must resolve the same curve"
    );
    assert_ne!(
        from_child.definition_hash(),
        from_parent.definition_hash(),
        "the requested cell remains part of the resolution trace"
    );
    assert_reloads(&from_child, "normalized rx1day");

    let invalid = reference("daily_freezethaw_cycles", "error", json!({}), json!({}))
        .resolve(&ResolutionContext {
            cell: Some("fffffffffffffff".to_string()),
            ..ResolutionContext::default()
        })
        .expect_err("15 hex characters are not necessarily an H3 cell");
    assert_eq!(
        (invalid.code, invalid.path.as_str()),
        ("invalid_value", "/context/cell")
    );
    let coarse = parent.parent(Resolution::Three).unwrap();
    let coarse =
        reference("frost_days", "error", json!({}), json!({})).resolve(&ResolutionContext {
            cell: Some(format!("{:015x}", u64::from(coarse))),
            ..ResolutionContext::default()
        });
    let coarse = coarse.expect_err("resolution 3 cannot supply a resolution 4 threshold");
    assert_eq!(
        (coarse.code, coarse.path.as_str()),
        ("invalid_value", "/context/cell")
    );

    let flood = reference("cflood", "legacy_defaults", json!({}), json!({}))
        .resolve(&ResolutionContext {
            continent: Some("Not A Continent".to_string()),
            building_type: Some("Commercial buildings".to_string()),
            cell: Some(sample_cell()),
            ..ResolutionContext::default()
        })
        .unwrap();
    let flood_doc: Value = serde_json::from_str(flood.canonical_json()).unwrap();
    assert!(flood_doc["provenance"]["origin"]["context"]["cell"].is_null());
    let flood_lookup = &flood_doc["provenance"]["origin"]["lookups"][0];
    assert_eq!(flood_lookup["status"], "fallback");
    assert_eq!(
        flood_lookup["resolved_key"],
        json!({"building_type": "Commercial buildings", "continent": "Global"})
    );
    let exposures = [0.0, 0.5, 1.0, 3.0];
    let legacy = ImpactRegistry::evaluate(
        &ImpactContext {
            factor: "cflood".to_string(),
            continent: Some("Not A Continent".to_string()),
            building_type: Some("Commercial buildings".to_string()),
            ..ImpactContext::default()
        },
        &HashMap::new(),
        &exposures,
    )
    .unwrap();
    for (exposure, expected) in exposures.iter().zip(legacy) {
        let actual = flood.evaluate(*exposure).unwrap();
        assert!(
            (actual - expected).abs() <= 1e-12,
            "global flood fallback at {exposure}: snapshot {actual}, legacy {expected}"
        );
    }
    assert!(flood.evaluate(3.0).unwrap() > 0.0);

    let zeros = reference("rflood", "legacy_defaults", json!({}), json!({}))
        .resolve(&ResolutionContext {
            continent: Some("Europe".to_string()),
            building_type: Some("Unknown buildings".to_string()),
            ..ResolutionContext::default()
        })
        .unwrap();
    let zeros_doc: Value = serde_json::from_str(zeros.canonical_json()).unwrap();
    assert_eq!(
        zeros_doc["provenance"]["origin"]["lookups"][0]["resolved_key"],
        "global_default"
    );
    assert_eq!(zeros.evaluate(2.0).unwrap(), 0.0);

    let strict = reference("inundation", "error", json!({}), json!({}))
        .resolve(&ResolutionContext {
            continent: Some("Not A Continent".to_string()),
            building_type: Some("Residential buildings".to_string()),
            ..ResolutionContext::default()
        })
        .expect_err("unknown continent");
    assert_eq!(
        (strict.code, strict.path.as_str()),
        ("fallback_required", "/context/continent")
    );
}

#[test]
fn coefficient_overflow_is_an_invalid_value() {
    let error = reference(
        "fwi",
        "error",
        json!({"minimum_damage": -1.0e308, "maximum_damage": 1.0e308}),
        json!({}),
    )
    .resolve(&ResolutionContext::default())
    .expect_err("damage levels overflow");
    assert_eq!(
        (error.code, error.path.as_str()),
        ("invalid_value", "/overrides/maximum_damage")
    );
}

#[test]
fn frozen_snapshots_match_the_legacy_registry() {
    let exposures = [0.0, 1.0, 10.0, 35.65, 100.0, 365.0];
    for (factor, context, call) in [
        (
            "fwi",
            ImpactContext {
                factor: "fwi".to_string(),
                ..ImpactContext::default()
            },
            ResolutionContext::default(),
        ),
        (
            "hot_days",
            ImpactContext {
                factor: "hot_days".to_string(),
                ..ImpactContext::default()
            },
            ResolutionContext::default(),
        ),
        (
            "cflood",
            ImpactContext {
                factor: "cflood".to_string(),
                continent: Some("Europe".to_string()),
                building_type: Some("Residential buildings".to_string()),
                ..ImpactContext::default()
            },
            context_for("cflood", false),
        ),
        (
            "cyclone",
            ImpactContext {
                factor: "cyclone".to_string(),
                country: Some("USA".to_string()),
                ..ImpactContext::default()
            },
            context_for("cyclone", false),
        ),
        (
            "spei",
            ImpactContext {
                factor: "spei".to_string(),
                country: Some("USA".to_string()),
                building_type: Some("Commercial buildings".to_string()),
                ..ImpactContext::default()
            },
            context_for("spei", false),
        ),
        (
            "cooling_degree_days",
            ImpactContext {
                factor: "cooling_degree_days".to_string(),
                country: Some("USA".to_string()),
                building_type: Some("Commercial buildings".to_string()),
                historic_mean: Some(410.5),
                ..ImpactContext::default()
            },
            context_for("cooling_degree_days", false),
        ),
    ] {
        let frozen = reference(factor, "error", json!({}), json!({}))
            .resolve(&call)
            .unwrap_or_else(|error| panic!("{factor}: {error}"));
        let legacy = ImpactRegistry::evaluate(&context, &HashMap::new(), &exposures)
            .unwrap_or_else(|error| panic!("{factor} legacy: {error}"));
        for (exposure, expected) in exposures.iter().zip(legacy) {
            let actual = frozen.evaluate(*exposure).unwrap();
            assert!(
                (actual - expected).abs() <= 1e-12,
                "{factor} at {exposure}: snapshot {actual}, legacy {expected}"
            );
        }
    }
}

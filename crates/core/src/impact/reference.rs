use super::definition::{
    choice, context, factor, optional_number, overrides, read_fields, sha256_text, version,
};
use super::error::ContractResult;
use super::json::{Json, canonical_json, content_hash, parse_json};

#[derive(Clone, Debug)]
pub(crate) struct CompiledReference {
    pub canonical: String,
    pub hash: String,
    pub registry_id: String,
    pub revision: String,
    pub manifest_sha256: String,
    pub factor: String,
    pub overrides: Vec<(String, Option<f64>)>,
    pub context: ContextValues,
    pub fallback: String,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ContextValues {
    pub cell: Option<String>,
    pub country: Option<String>,
    pub continent: Option<String>,
    pub building_type: Option<String>,
    pub historic_mean: Option<f64>,
}

pub(crate) fn compile_reference(text: &str) -> ContractResult<CompiledReference> {
    let value = parse_json(text)?;
    version(
        &value,
        "",
        "schema_version",
        "1.0",
        "unsupported_schema_version",
    )?;
    let fields = read_fields(
        &value,
        "",
        &[
            "schema_version",
            "registry_id",
            "registry_revision",
            "registry_manifest_sha256",
            "resolver_semantics_version",
            "factor",
        ],
        &["overrides", "context_defaults", "fallback_policy"],
    )?;
    let registry_id = identifier_owned(fields.require("registry_id")?, "/registry_id")?;
    let revision = identifier_owned(fields.require("registry_revision")?, "/registry_revision")?;
    let manifest_sha256 = sha256_text(
        fields.require("registry_manifest_sha256")?,
        "/registry_manifest_sha256",
    )?;
    let resolver = super::definition::string(
        fields.require("resolver_semantics_version")?,
        "/resolver_semantics_version",
    )?;
    if resolver != "crc-impact-registry/1" {
        return Err(super::error::fail(
            "unsupported_semantics_version",
            "/resolver_semantics_version",
            "unsupported resolver semantics",
        ));
    }
    let factor_name = factor(fields.require("factor")?, "/factor")?;
    let override_json = match fields.optional("overrides") {
        Some(value) => overrides(value, "/overrides", &factor_name)?,
        None => Json::Object(Vec::new()),
    };
    let context_json = match fields.optional("context_defaults") {
        Some(value) => context(value, "/context_defaults")?,
        None => context(&Json::Object(Vec::new()), "/context_defaults")?,
    };
    let fallback = match fields.optional("fallback_policy") {
        Some(value) => choice(value, "/fallback_policy", &["error", "legacy_defaults"])?,
        None => "error".to_string(),
    };
    let normalized = Json::object([
        ("schema_version", Json::String("1.0".to_string())),
        ("registry_id", Json::String(registry_id.clone())),
        ("registry_revision", Json::String(revision.clone())),
        (
            "registry_manifest_sha256",
            Json::String(manifest_sha256.clone()),
        ),
        (
            "resolver_semantics_version",
            Json::String("crc-impact-registry/1".to_string()),
        ),
        ("factor", Json::String(factor_name.clone())),
        ("overrides", override_json.clone()),
        ("context_defaults", context_json.clone()),
        ("fallback_policy", Json::String(fallback.clone())),
    ]);
    Ok(CompiledReference {
        canonical: canonical_json(&normalized),
        hash: content_hash(&normalized),
        registry_id,
        revision,
        manifest_sha256,
        factor: factor_name,
        overrides: override_pairs(&override_json),
        context: context_values(&context_json),
        fallback,
    })
}

fn identifier_owned(value: &Json, path: &str) -> ContractResult<String> {
    super::definition::identifier(value, path)
}

fn override_pairs(value: &Json) -> Vec<(String, Option<f64>)> {
    let Json::Object(entries) = value else {
        return Vec::new();
    };
    entries
        .iter()
        .map(|(key, item)| {
            let number = match item {
                Json::Number(number) => Some(if *number == 0.0 { 0.0 } else { *number }),
                _ => None,
            };
            (key.clone(), number)
        })
        .collect()
}

fn context_values(value: &Json) -> ContextValues {
    let Json::Object(entries) = value else {
        return ContextValues::default();
    };
    let text = |name: &str| -> Option<String> {
        entries
            .iter()
            .find_map(|(key, item)| match (key.as_str(), item) {
                (key, Json::String(text)) if key == name => Some(text.clone()),
                _ => None,
            })
    };
    let historic = entries
        .iter()
        .find(|(key, _)| key == "historic_mean")
        .map(|(_, item)| item);
    ContextValues {
        cell: text("cell"),
        country: text("country"),
        continent: text("continent"),
        building_type: text("building_type"),
        historic_mean: optional_number(historic, "/context_defaults/historic_mean")
            .ok()
            .flatten(),
    }
}

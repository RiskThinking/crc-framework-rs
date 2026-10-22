use super::error::{ContractResult, fail};
use super::json::{Json, pointer, pointer_index};

const SCHEMA_VERSION: &str = "1.0";
const SEMANTICS_VERSION: &str = "crc-impact/1";
const RESOLVER_VERSION: &str = "crc-impact-registry/1";
const CROPS: [&str; 3] = ["maize", "rice", "wheat"];

const MODEL_KINDS: [&str; 7] = [
    "linear",
    "sigmoid",
    "tabulated",
    "binned",
    "spei",
    "cooling",
    "crop",
];
const ORIGIN_KINDS: [&str; 5] = [
    "user_defined",
    "published",
    "imported",
    "derived",
    "registry",
];
const SOURCE_KINDS: [&str; 4] = ["publication", "dataset", "code", "definition"];
const CONTEXT_FIELDS: [&str; 5] = [
    "cell",
    "country",
    "continent",
    "building_type",
    "historic_mean",
];
const FALLBACK_POLICIES: [&str; 2] = ["error", "legacy_defaults"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    Linear,
    Sigmoid,
    Tabulated,
    Binned,
    Spei,
    Cooling,
    Crop,
}

impl ModelKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Sigmoid => "sigmoid",
            Self::Tabulated => "tabulated",
            Self::Binned => "binned",
            Self::Spei => "spei",
            Self::Cooling => "cooling",
            Self::Crop => "crop",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Linear,
    StepLeft,
    StepRight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutsideDomain {
    Clamp,
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OriginAdjustment {
    None,
    SubtractAtZero,
}

#[derive(Clone, Debug)]
pub struct Clamp {
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
}

#[derive(Clone, Debug)]
pub enum Model {
    Linear {
        slope: f64,
        intercept: f64,
    },
    Sigmoid {
        midpoint: f64,
        steepness: f64,
        offset: f64,
        amplitude: f64,
        origin_adjustment: OriginAdjustment,
        zero_below: Option<f64>,
    },
    Tabulated {
        points: Vec<(f64, f64)>,
        interpolation: Interpolation,
        outside_domain: OutsideDomain,
    },
    Binned {
        edges: Vec<f64>,
        values: Vec<f64>,
        outside_domain: OutsideDomain,
    },
    Spei {
        water_consumption_asset: f64,
        original_water_unit_cost: f64,
    },
    Cooling {
        cooling_cost: f64,
        historic_mean: f64,
    },
    Crop {
        weights: [f64; 3],
        country: [f64; 3],
        yearly: [f64; 3],
        index: [f64; 3],
        historic_mean: f64,
    },
}

pub(crate) struct CompiledDefinition {
    pub normalized: Json,
    pub model: Model,
    pub clamp: Clamp,
    pub kind: ModelKind,
}

pub(crate) fn compile(value: Json) -> ContractResult<CompiledDefinition> {
    version(
        &value,
        "",
        "schema_version",
        SCHEMA_VERSION,
        "unsupported_schema_version",
    )?;
    version(
        &value,
        "",
        "semantics_version",
        SEMANTICS_VERSION,
        "unsupported_semantics_version",
    )?;
    let fields = read_fields(
        &value,
        "",
        &[
            "schema_version",
            "semantics_version",
            "calibration",
            "model",
            "provenance",
        ],
        &["clamp"],
    )?;
    let calibration = calibration(fields.require("calibration")?, "/calibration")?;
    let (kind, model, model_json) = model(fields.require("model")?, "/model")?;
    let clamp = clamp(fields.optional("clamp"), "/clamp")?;
    let provenance = provenance(fields.require("provenance")?, "/provenance")?;
    let normalized = Json::object([
        ("schema_version", Json::String(SCHEMA_VERSION.to_string())),
        (
            "semantics_version",
            Json::String(SEMANTICS_VERSION.to_string()),
        ),
        ("calibration", calibration),
        ("model", model_json),
        ("clamp", clamp_json(&clamp)),
        ("provenance", provenance),
    ]);
    Ok(CompiledDefinition {
        normalized,
        model,
        clamp,
        kind,
    })
}

fn calibration(value: &Json, path: &str) -> ContractResult<Json> {
    let names = [
        "exposure_variable",
        "exposure_unit",
        "impact_metric",
        "impact_unit",
    ];
    let fields = read_fields(value, path, &names, &[])?;
    let mut entries = Vec::new();
    for name in names {
        entries.push((
            name.to_string(),
            Json::String(identifier(fields.require(name)?, &pointer(path, name))?),
        ));
    }
    Ok(Json::Object(entries))
}

fn model(value: &Json, path: &str) -> ContractResult<(ModelKind, Model, Json)> {
    let kind = kind(value, path, &MODEL_KINDS)?;
    let at = |name: &str| pointer(path, name);
    match kind.as_str() {
        "linear" => {
            let fields = read_fields(value, path, &["kind", "slope"], &["intercept"])?;
            let slope = number(fields.require("slope")?, &at("slope"), None, false, None)?;
            let intercept = match fields.optional("intercept") {
                Some(value) => number(value, &at("intercept"), None, false, None)?,
                None => 0.0,
            };
            let model = Model::Linear { slope, intercept };
            let json = Json::object([
                ("kind", Json::String("linear".to_string())),
                ("slope", Json::number(slope)),
                ("intercept", Json::number(intercept)),
            ]);
            Ok((ModelKind::Linear, model, json))
        }
        "sigmoid" => {
            let fields = read_fields(
                value,
                path,
                &["kind", "midpoint", "steepness"],
                &["offset", "amplitude", "origin_adjustment", "zero_below"],
            )?;
            let midpoint = number(
                fields.require("midpoint")?,
                &at("midpoint"),
                None,
                false,
                None,
            )?;
            let steepness = number(
                fields.require("steepness")?,
                &at("steepness"),
                None,
                false,
                None,
            )?;
            let offset =
                optional_present_number(fields.optional("offset"), &at("offset"))?.unwrap_or(0.0);
            let amplitude = match fields.optional("amplitude") {
                Some(value) => number(value, &at("amplitude"), Some(0.0), false, None)?,
                None => 1.0,
            };
            let adjustment = match fields.optional("origin_adjustment") {
                Some(value) => choice(
                    value,
                    &at("origin_adjustment"),
                    &["none", "subtract_at_zero"],
                )?,
                None => "none".to_string(),
            };
            let zero_below = optional_number(fields.optional("zero_below"), &at("zero_below"))?;
            let origin_adjustment = if adjustment == "subtract_at_zero" {
                OriginAdjustment::SubtractAtZero
            } else {
                OriginAdjustment::None
            };
            let model = Model::Sigmoid {
                midpoint,
                steepness,
                offset,
                amplitude,
                origin_adjustment,
                zero_below,
            };
            let json = Json::object([
                ("kind", Json::String("sigmoid".to_string())),
                ("midpoint", Json::number(midpoint)),
                ("steepness", Json::number(steepness)),
                ("offset", Json::number(offset)),
                ("amplitude", Json::number(amplitude)),
                ("origin_adjustment", Json::String(adjustment)),
                ("zero_below", Json::number_opt(zero_below)),
            ]);
            Ok((ModelKind::Sigmoid, model, json))
        }
        "tabulated" => {
            let fields = read_fields(
                value,
                path,
                &["kind", "points"],
                &["interpolation", "outside_domain"],
            )?;
            let raw = array(fields.require("points")?, &at("points"), 2)?;
            let mut points = Vec::with_capacity(raw.len());
            let mut point_json = Vec::with_capacity(raw.len());
            for (index, item) in raw.iter().enumerate() {
                let location = pointer_index(&at("points"), index);
                let point = read_fields(item, &location, &["x", "y"], &[])?;
                let x = number(
                    point.require("x")?,
                    &pointer(&location, "x"),
                    None,
                    false,
                    None,
                )?;
                let y = number(
                    point.require("y")?,
                    &pointer(&location, "y"),
                    None,
                    false,
                    None,
                )?;
                points.push((x, y));
                point_json.push(Json::object([
                    ("x", Json::number(x)),
                    ("y", Json::number(y)),
                ]));
            }
            increasing(
                &points.iter().map(|point| point.0).collect::<Vec<_>>(),
                &at("points"),
                Some("x"),
            )?;
            let interpolation = match fields.optional("interpolation") {
                Some(value) => choice(
                    value,
                    &at("interpolation"),
                    &["linear", "step_left", "step_right"],
                )?,
                None => "linear".to_string(),
            };
            let outside = outside_domain(fields.optional("outside_domain"), &at("outside_domain"))?;
            let model = Model::Tabulated {
                points,
                interpolation: match interpolation.as_str() {
                    "step_left" => Interpolation::StepLeft,
                    "step_right" => Interpolation::StepRight,
                    _ => Interpolation::Linear,
                },
                outside_domain: outside,
            };
            let json = Json::object([
                ("kind", Json::String("tabulated".to_string())),
                ("points", Json::Array(point_json)),
                ("interpolation", Json::String(interpolation)),
                ("outside_domain", Json::String(outside.as_str().to_string())),
            ]);
            Ok((ModelKind::Tabulated, model, json))
        }
        "binned" => {
            let fields = read_fields(
                value,
                path,
                &["kind", "bin_edges", "values"],
                &["outside_domain"],
            )?;
            let raw_edges = array(fields.require("bin_edges")?, &at("bin_edges"), 2)?;
            let mut edges = Vec::with_capacity(raw_edges.len());
            for (index, item) in raw_edges.iter().enumerate() {
                edges.push(number(
                    item,
                    &pointer_index(&at("bin_edges"), index),
                    None,
                    false,
                    None,
                )?);
            }
            increasing(&edges, &at("bin_edges"), None)?;
            let raw_values = array(fields.require("values")?, &at("values"), 1)?;
            let mut values = Vec::with_capacity(raw_values.len());
            for (index, item) in raw_values.iter().enumerate() {
                values.push(number(
                    item,
                    &pointer_index(&at("values"), index),
                    None,
                    false,
                    None,
                )?);
            }
            if values.len() != edges.len() - 1 {
                return Err(fail(
                    "invalid_value",
                    at("values"),
                    "requires exactly one value per interval",
                ));
            }
            let outside = outside_domain(fields.optional("outside_domain"), &at("outside_domain"))?;
            let model = Model::Binned {
                edges: edges.clone(),
                values: values.clone(),
                outside_domain: outside,
            };
            let json = Json::object([
                ("kind", Json::String("binned".to_string())),
                (
                    "bin_edges",
                    Json::Array(edges.into_iter().map(Json::number).collect()),
                ),
                (
                    "values",
                    Json::Array(values.into_iter().map(Json::number).collect()),
                ),
                ("outside_domain", Json::String(outside.as_str().to_string())),
            ]);
            Ok((ModelKind::Binned, model, json))
        }
        "spei" => {
            let names = ["water_consumption_asset", "original_water_unit_cost"];
            let fields = read_fields(
                value,
                path,
                &[
                    "kind",
                    "water_consumption_asset",
                    "original_water_unit_cost",
                ],
                &[],
            )?;
            let mut numbers = [0.0; 2];
            for (index, name) in names.iter().enumerate() {
                numbers[index] = number(fields.require(name)?, &at(name), Some(0.0), false, None)?;
            }
            let model = Model::Spei {
                water_consumption_asset: numbers[0],
                original_water_unit_cost: numbers[1],
            };
            let json = Json::object([
                ("kind", Json::String("spei".to_string())),
                ("water_consumption_asset", Json::number(numbers[0])),
                ("original_water_unit_cost", Json::number(numbers[1])),
            ]);
            Ok((ModelKind::Spei, model, json))
        }
        "cooling" => {
            let fields = read_fields(value, path, &["kind", "cooling_cost", "historic_mean"], &[])?;
            let cooling_cost = number(
                fields.require("cooling_cost")?,
                &at("cooling_cost"),
                Some(0.0),
                false,
                None,
            )?;
            let historic_mean = number(
                fields.require("historic_mean")?,
                &at("historic_mean"),
                None,
                true,
                None,
            )?;
            let model = Model::Cooling {
                cooling_cost,
                historic_mean,
            };
            let json = Json::object([
                ("kind", Json::String("cooling".to_string())),
                ("cooling_cost", Json::number(cooling_cost)),
                ("historic_mean", Json::number(historic_mean)),
            ]);
            Ok((ModelKind::Cooling, model, json))
        }
        _ => {
            let fields = read_fields(
                value,
                path,
                &[
                    "kind",
                    "weights",
                    "country",
                    "yearly",
                    "index",
                    "historic_mean",
                ],
                &[],
            )?;
            let weights = crop_vector(fields.require("weights")?, &at("weights"), Some(0.0))?;
            if weights.iter().sum::<f64>() <= 0.0 {
                return Err(fail(
                    "invalid_value",
                    at("weights"),
                    "crop weights require a positive total",
                ));
            }
            let country = crop_vector(fields.require("country")?, &at("country"), None)?;
            let yearly = crop_vector(fields.require("yearly")?, &at("yearly"), None)?;
            let index = crop_vector(fields.require("index")?, &at("index"), None)?;
            let historic_mean = number(
                fields.require("historic_mean")?,
                &at("historic_mean"),
                None,
                false,
                None,
            )?;
            let model = Model::Crop {
                weights,
                country,
                yearly,
                index,
                historic_mean,
            };
            let json = Json::object([
                ("kind", Json::String("crop".to_string())),
                ("weights", crop_json(weights)),
                ("country", crop_json(country)),
                ("yearly", crop_json(yearly)),
                ("index", crop_json(index)),
                ("historic_mean", Json::number(historic_mean)),
            ]);
            Ok((ModelKind::Crop, model, json))
        }
    }
}

fn clamp(value: Option<&Json>, path: &str) -> ContractResult<Clamp> {
    let Some(value) = value else {
        return Ok(Clamp {
            minimum: None,
            maximum: None,
        });
    };
    if matches!(value, Json::Null) {
        return Ok(Clamp {
            minimum: None,
            maximum: None,
        });
    }
    let fields = read_fields(value, path, &[], &["minimum", "maximum"])?;
    let minimum = optional_number(fields.optional("minimum"), &pointer(path, "minimum"))?;
    let maximum = optional_number(fields.optional("maximum"), &pointer(path, "maximum"))?;
    if minimum.is_none() && maximum.is_none() {
        return Ok(Clamp {
            minimum: None,
            maximum: None,
        });
    }
    if minimum.is_some_and(|minimum| maximum.is_some_and(|maximum| minimum > maximum)) {
        return Err(fail(
            "invalid_value",
            path,
            "clamp minimum must not exceed maximum",
        ));
    }
    Ok(Clamp { minimum, maximum })
}

fn clamp_json(clamp: &Clamp) -> Json {
    if clamp.minimum.is_none() && clamp.maximum.is_none() {
        Json::Null
    } else {
        Json::object([
            ("minimum", Json::number_opt(clamp.minimum)),
            ("maximum", Json::number_opt(clamp.maximum)),
        ])
    }
}

fn provenance(value: &Json, path: &str) -> ContractResult<Json> {
    let fields = read_fields(value, path, &["origin"], &["sources", "derived_from"])?;
    let sources_path = pointer(path, "sources");
    let mut sources = Vec::new();
    let mut source_json = Vec::new();
    let mut seen = Vec::new();
    let source_values = match fields.optional("sources") {
        Some(value) => array(value, &sources_path, 0)?,
        None => &[],
    };
    for (index, item) in source_values.iter().enumerate() {
        let location = pointer_index(&sources_path, index);
        let source = source(item, &location)?;
        if seen.iter().any(|id| id == &source.id) {
            return Err(fail(
                "duplicate_identifier",
                pointer(&location, "id"),
                "duplicate source id",
            ));
        }
        seen.push(source.id.clone());
        source_json.push(source.json);
        sources.push(source.id);
    }
    let parents_path = pointer(path, "derived_from");
    let mut parents = Vec::new();
    let parent_values = match fields.optional("derived_from") {
        Some(value) => array(value, &parents_path, 0)?,
        None => &[],
    };
    for (index, item) in parent_values.iter().enumerate() {
        let parent = content_hash_text(item, &pointer_index(&parents_path, index))?;
        if parents.iter().any(|existing| existing == &parent) {
            return Err(fail(
                "duplicate_identifier",
                pointer_index(&parents_path, index),
                "duplicate parent hash",
            ));
        }
        parents.push(parent);
    }
    let origin = origin(
        fields.require("origin")?,
        &pointer(path, "origin"),
        &sources,
        &parents,
        path,
    )?;
    parents.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    Ok(Json::object([
        ("origin", origin),
        ("sources", Json::Array(source_json)),
        (
            "derived_from",
            Json::Array(parents.into_iter().map(Json::String).collect()),
        ),
    ]))
}

struct Source {
    id: String,
    json: Json,
}

fn source(value: &Json, path: &str) -> ContractResult<Source> {
    let fields = read_fields(
        value,
        path,
        &["id", "kind", "uri"],
        &["revision", "content_sha256"],
    )?;
    let id = identifier(fields.require("id")?, &pointer(path, "id"))?;
    let kind = choice(
        fields.require("kind")?,
        &pointer(path, "kind"),
        &SOURCE_KINDS,
    )?;
    let uri = identifier(fields.require("uri")?, &pointer(path, "uri"))?;
    let revision = optional_identifier(fields.optional("revision"), &pointer(path, "revision"))?;
    let digest = optional_sha256(
        fields.optional("content_sha256"),
        &pointer(path, "content_sha256"),
    )?;
    let json = Json::object([
        ("id", Json::String(id.clone())),
        ("kind", Json::String(kind)),
        ("uri", Json::String(uri)),
        ("revision", revision.map(Json::String).unwrap_or(Json::Null)),
        (
            "content_sha256",
            digest.map(Json::String).unwrap_or(Json::Null),
        ),
    ]);
    Ok(Source { id, json })
}

fn origin(
    value: &Json,
    path: &str,
    source_ids: &[String],
    parents: &[String],
    provenance: &str,
) -> ContractResult<Json> {
    let kind = kind(value, path, &ORIGIN_KINDS)?;
    match kind.as_str() {
        "user_defined" => {
            read_fields(value, path, &["kind"], &[])?;
            Ok(Json::object([(
                "kind",
                Json::String("user_defined".to_string()),
            )]))
        }
        "published" => {
            let fields = read_fields(value, path, &["kind", "source_id"], &[])?;
            let source_id = source_reference(&fields, path, source_ids)?;
            Ok(Json::object([
                ("kind", Json::String("published".to_string())),
                ("source_id", Json::String(source_id)),
            ]))
        }
        "imported" => {
            let fields = read_fields(
                value,
                path,
                &["kind", "source_id", "importer_name", "importer_version"],
                &["importer_build_sha256"],
            )?;
            let source_id = source_reference(&fields, path, source_ids)?;
            let name = identifier(
                fields.require("importer_name")?,
                &pointer(path, "importer_name"),
            )?;
            let version = identifier(
                fields.require("importer_version")?,
                &pointer(path, "importer_version"),
            )?;
            let build = optional_sha256(
                fields.optional("importer_build_sha256"),
                &pointer(path, "importer_build_sha256"),
            )?;
            Ok(Json::object([
                ("kind", Json::String("imported".to_string())),
                ("source_id", Json::String(source_id)),
                ("importer_name", Json::String(name)),
                ("importer_version", Json::String(version)),
                (
                    "importer_build_sha256",
                    build.map(Json::String).unwrap_or(Json::Null),
                ),
            ]))
        }
        "derived" => {
            let fields = read_fields(value, path, &["kind", "description"], &[])?;
            let description = text(
                fields.require("description")?,
                &pointer(path, "description"),
            )?;
            if parents.is_empty() {
                return Err(fail(
                    "invalid_reference",
                    pointer(provenance, "derived_from"),
                    "a derived origin requires at least one parent definition",
                ));
            }
            Ok(Json::object([
                ("kind", Json::String("derived".to_string())),
                ("description", Json::String(description)),
            ]))
        }
        _ => registry_origin(value, path),
    }
}

fn registry_origin(value: &Json, path: &str) -> ContractResult<Json> {
    let names = [
        "kind",
        "registry_id",
        "registry_revision",
        "registry_manifest_sha256",
        "resolver_semantics_version",
        "factor",
        "context",
        "overrides",
        "fallback_policy",
        "lookups",
    ];
    let fields = read_fields(value, path, &names, &[])?;
    let identity = registry_identity(&fields, path)?;
    let factor = factor(fields.require("factor")?, &pointer(path, "factor"))?;
    let policy = choice(
        fields.require("fallback_policy")?,
        &pointer(path, "fallback_policy"),
        &FALLBACK_POLICIES,
    )?;
    let lookups_path = pointer(path, "lookups");
    let raw_lookups = array(fields.require("lookups")?, &lookups_path, 0)?;
    let mut lookups = Vec::with_capacity(raw_lookups.len());
    for (index, item) in raw_lookups.iter().enumerate() {
        lookups.push(lookup(item, &pointer_index(&lookups_path, index))?);
    }
    if policy == "error" {
        for (index, lookup) in lookups.iter().enumerate() {
            if lookup.status == "fallback" {
                return Err(fail(
                    "invalid_value",
                    pointer(&pointer_index(&lookups_path, index), "status"),
                    "fallback lookups require fallback_policy legacy_defaults",
                ));
            }
        }
    }
    let context = context(fields.require("context")?, &pointer(path, "context"))?;
    let overrides = overrides(
        fields.require("overrides")?,
        &pointer(path, "overrides"),
        &factor,
    )?;
    Ok(Json::Object(vec![
        ("kind".to_string(), Json::String("registry".to_string())),
        ("registry_id".to_string(), identity[0].clone()),
        ("registry_revision".to_string(), identity[1].clone()),
        ("registry_manifest_sha256".to_string(), identity[2].clone()),
        (
            "resolver_semantics_version".to_string(),
            identity[3].clone(),
        ),
        ("factor".to_string(), Json::String(factor)),
        ("context".to_string(), context),
        ("overrides".to_string(), overrides),
        ("fallback_policy".to_string(), Json::String(policy)),
        (
            "lookups".to_string(),
            Json::Array(lookups.into_iter().map(|item| item.json).collect()),
        ),
    ]))
}

struct Lookup {
    status: String,
    json: Json,
}

fn lookup(value: &Json, path: &str) -> ContractResult<Lookup> {
    let names = [
        "table_id",
        "table_sha256",
        "requested_key",
        "resolved_key",
        "status",
    ];
    let fields = read_fields(value, path, &names, &[])?;
    let table_id = identifier(fields.require("table_id")?, &pointer(path, "table_id"))?;
    let digest = sha256_text(
        fields.require("table_sha256")?,
        &pointer(path, "table_sha256"),
    )?;
    let requested = lookup_key(
        fields.require("requested_key")?,
        &pointer(path, "requested_key"),
    )?;
    let resolved = lookup_key(
        fields.require("resolved_key")?,
        &pointer(path, "resolved_key"),
    )?;
    let status = choice(
        fields.require("status")?,
        &pointer(path, "status"),
        &["exact", "fallback"],
    )?;
    let json = Json::object([
        ("table_id", Json::String(table_id)),
        ("table_sha256", Json::String(digest)),
        ("requested_key", requested),
        ("resolved_key", resolved),
        ("status", Json::String(status.clone())),
    ]);
    Ok(Lookup { status, json })
}

fn lookup_key(value: &Json, path: &str) -> ContractResult<Json> {
    if let Json::String(_) = value {
        return Ok(Json::String(identifier(value, path)?));
    }
    let object = object(value, path)?;
    if object.is_empty() {
        return Err(fail(
            "invalid_value",
            path,
            "composite keys require at least one field",
        ));
    }
    let mut names = object
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    names.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    let mut entries = Vec::new();
    for name in names {
        let location = pointer(path, name);
        identifier_text(name, &location)?;
        let item = object
            .iter()
            .find(|(key, _)| key == name)
            .expect("key exists")
            .1
            .clone();
        entries.push((
            name.to_string(),
            Json::String(identifier(&item, &location)?),
        ));
    }
    Ok(Json::Object(entries))
}

pub(crate) fn context(value: &Json, path: &str) -> ContractResult<Json> {
    let fields = read_fields(value, path, &[], &CONTEXT_FIELDS)?;
    let cell = match fields.optional("cell") {
        Some(Json::Null) | None => None,
        Some(value) => {
            let cell = string(value, &pointer(path, "cell"))?;
            if !is_h3_cell(&cell) {
                return Err(fail(
                    "invalid_value",
                    pointer(path, "cell"),
                    "must be 15 lowercase hex characters",
                ));
            }
            Some(cell)
        }
    };
    Ok(Json::object([
        ("cell", cell.map(Json::String).unwrap_or(Json::Null)),
        (
            "country",
            optional_identifier(fields.optional("country"), &pointer(path, "country"))?
                .map(Json::String)
                .unwrap_or(Json::Null),
        ),
        (
            "continent",
            optional_identifier(fields.optional("continent"), &pointer(path, "continent"))?
                .map(Json::String)
                .unwrap_or(Json::Null),
        ),
        (
            "building_type",
            optional_identifier(
                fields.optional("building_type"),
                &pointer(path, "building_type"),
            )?
            .map(Json::String)
            .unwrap_or(Json::Null),
        ),
        (
            "historic_mean",
            Json::number_opt(optional_number(
                fields.optional("historic_mean"),
                &pointer(path, "historic_mean"),
            )?),
        ),
    ]))
}

pub(crate) fn overrides(value: &Json, path: &str, factor: &str) -> ContractResult<Json> {
    let object = object(value, path)?;
    let mut names = object
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    names.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    let mut entries = Vec::new();
    for name in names {
        let location = pointer(path, &name);
        let Some(nullable) = override_rule(factor, &name) else {
            if registry_override_key_exists(&name) {
                return Err(fail(
                    "inapplicable_override",
                    location,
                    format!("not accepted for factor {factor}"),
                ));
            }
            return Err(fail("unknown_override", location, "unknown override key"));
        };
        let item = object
            .iter()
            .find(|(key, _)| key == &name)
            .expect("key exists")
            .1
            .clone();
        let normalized = if matches!(item, Json::Null) {
            if !nullable {
                return Err(fail(
                    "invalid_value",
                    location,
                    "null is accepted only to clear a threshold or clamp",
                ));
            }
            Json::Null
        } else {
            Json::number(number(&item, &location, None, false, None)?)
        };
        entries.push((name, normalized));
    }
    Ok(Json::Object(entries))
}

pub(crate) fn registry_override_key_exists(key: &str) -> bool {
    [
        "midpoint",
        "steepness",
        "minimum_damage",
        "maximum_damage",
        "zero_below",
        "slope",
        "intercept",
        "water_unit_cost",
        "cooling_cost",
    ]
    .contains(&key)
}

pub(crate) fn override_rule(factor: &str, key: &str) -> Option<bool> {
    let sigmoid = |key: &str| match key {
        "midpoint" | "steepness" | "minimum_damage" | "maximum_damage" => Some(false),
        "zero_below" => Some(true),
        _ => None,
    };
    match factor {
        "daily_freezethaw_cycles"
        | "frost_days"
        | "rx1day"
        | "cyclone"
        | "wind_max_daily_max"
        | "fwi" => sigmoid(key),
        "hot_days" => match key {
            "slope" | "intercept" => Some(false),
            "minimum_damage" | "maximum_damage" => Some(true),
            _ => None,
        },
        "cflood" | "rflood" | "inundation" if key == "maximum_damage" => Some(false),
        "spei" if key == "water_unit_cost" => Some(false),
        "cooling_degree_days" if key == "cooling_cost" => Some(false),
        "maximum_consecutive_dry_days" => None,
        _ => None,
    }
}

pub(crate) fn known_registry_factor(factor: &str) -> bool {
    matches!(
        factor,
        "daily_freezethaw_cycles"
            | "frost_days"
            | "rx1day"
            | "cyclone"
            | "wind_max_daily_max"
            | "fwi"
            | "hot_days"
            | "cflood"
            | "rflood"
            | "inundation"
            | "spei"
            | "cooling_degree_days"
            | "maximum_consecutive_dry_days"
    )
}

fn registry_identity(fields: &Fields<'_>, path: &str) -> ContractResult<[Json; 4]> {
    let location = pointer(path, "resolver_semantics_version");
    if string(fields.require("resolver_semantics_version")?, &location)? != RESOLVER_VERSION {
        return Err(fail(
            "unsupported_semantics_version",
            location,
            "unsupported resolver semantics",
        ));
    }
    Ok([
        Json::String(identifier(
            fields.require("registry_id")?,
            &pointer(path, "registry_id"),
        )?),
        Json::String(identifier(
            fields.require("registry_revision")?,
            &pointer(path, "registry_revision"),
        )?),
        Json::String(sha256_text(
            fields.require("registry_manifest_sha256")?,
            &pointer(path, "registry_manifest_sha256"),
        )?),
        Json::String(RESOLVER_VERSION.to_string()),
    ])
}

fn source_reference(
    fields: &Fields<'_>,
    path: &str,
    source_ids: &[String],
) -> ContractResult<String> {
    let location = pointer(path, "source_id");
    let source_id = identifier(fields.require("source_id")?, &location)?;
    if !source_ids.iter().any(|id| id == &source_id) {
        return Err(fail(
            "invalid_reference",
            location,
            "source_id does not name a source record",
        ));
    }
    Ok(source_id)
}

fn crop_vector(value: &Json, path: &str, minimum: Option<f64>) -> ContractResult<[f64; 3]> {
    let fields = read_fields(value, path, &CROPS, &[])?;
    let mut result = [0.0; 3];
    for (index, crop) in CROPS.iter().enumerate() {
        result[index] = number(
            fields.require(crop)?,
            &pointer(path, crop),
            minimum,
            false,
            None,
        )?;
    }
    Ok(result)
}

fn crop_json(values: [f64; 3]) -> Json {
    Json::object([
        ("maize", Json::number(values[0])),
        ("rice", Json::number(values[1])),
        ("wheat", Json::number(values[2])),
    ])
}

fn outside_domain(value: Option<&Json>, path: &str) -> ContractResult<OutsideDomain> {
    let text = match value {
        Some(value) => choice(value, path, &["clamp", "error"])?,
        None => "clamp".to_string(),
    };
    Ok(if text == "error" {
        OutsideDomain::Error
    } else {
        OutsideDomain::Clamp
    })
}

impl OutsideDomain {
    fn as_str(self) -> &'static str {
        match self {
            Self::Clamp => "clamp",
            Self::Error => "error",
        }
    }
}

fn increasing(values: &[f64], path: &str, suffix: Option<&str>) -> ContractResult<()> {
    for index in 1..values.len() {
        if values[index] <= values[index - 1] {
            let mut location = pointer_index(path, index);
            if let Some(suffix) = suffix {
                location = pointer(&location, suffix);
            }
            return Err(fail(
                "invalid_value",
                location,
                "values must strictly increase",
            ));
        }
    }
    Ok(())
}

pub(crate) struct Fields<'a> {
    entries: Vec<(&'a str, &'a Json)>,
}

impl<'a> Fields<'a> {
    pub(crate) fn optional(&self, name: &str) -> Option<&'a Json> {
        self.entries
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| *value)
    }

    pub(crate) fn require(&self, name: &str) -> ContractResult<&'a Json> {
        self.optional(name)
            .ok_or_else(|| fail("missing_field", "", "required field is missing"))
    }
}

pub(crate) fn read_fields<'a>(
    value: &'a Json,
    path: &str,
    required: &[&str],
    optional: &[&str],
) -> ContractResult<Fields<'a>> {
    let entries = object(value, path)?;
    let mut names = entries
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    names.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    for name in names {
        if !required.contains(&name) && !optional.contains(&name) {
            return Err(fail("unknown_field", pointer(path, name), "unknown field"));
        }
    }
    for name in required {
        if !entries.iter().any(|(key, _)| key == name) {
            return Err(fail(
                "missing_field",
                pointer(path, name),
                "required field is missing",
            ));
        }
    }
    Ok(Fields {
        entries: entries
            .iter()
            .map(|(key, value)| (key.as_str(), value))
            .collect(),
    })
}

fn object<'a>(value: &'a Json, path: &str) -> ContractResult<&'a [(String, Json)]> {
    match value {
        Json::Object(entries) => Ok(entries),
        _ => Err(fail("invalid_type", path, "expected an object")),
    }
}

fn array<'a>(value: &'a Json, path: &str, min_items: usize) -> ContractResult<&'a [Json]> {
    match value {
        Json::Array(items) if items.len() >= min_items => Ok(items),
        Json::Array(_) => Err(fail(
            "invalid_value",
            path,
            format!("requires at least {min_items} items"),
        )),
        _ => Err(fail("invalid_type", path, "expected an array")),
    }
}

pub(crate) fn number(
    value: &Json,
    path: &str,
    minimum: Option<f64>,
    positive: bool,
    maximum: Option<f64>,
) -> ContractResult<f64> {
    let Json::Number(value) = value else {
        return Err(fail("invalid_type", path, "expected a JSON number"));
    };
    let result = if *value == 0.0 { 0.0 } else { *value };
    if !result.is_finite() {
        return Err(fail(
            "non_finite_number",
            path,
            "numbers must be finite binary64 values",
        ));
    }
    if minimum.is_some_and(|minimum| result < minimum) {
        return Err(fail("invalid_value", path, "value is below its minimum"));
    }
    if maximum.is_some_and(|maximum| result > maximum) {
        return Err(fail("invalid_value", path, "value is above its maximum"));
    }
    if positive && result <= 0.0 {
        return Err(fail("invalid_value", path, "must be positive"));
    }
    Ok(result)
}

pub(crate) fn optional_number(value: Option<&Json>, path: &str) -> ContractResult<Option<f64>> {
    match value {
        None | Some(Json::Null) => Ok(None),
        Some(value) => number(value, path, None, false, None).map(Some),
    }
}

fn optional_present_number(value: Option<&Json>, path: &str) -> ContractResult<Option<f64>> {
    match value {
        None => Ok(None),
        Some(Json::Null) => Err(fail("invalid_type", path, "expected a JSON number")),
        Some(value) => number(value, path, None, false, None).map(Some),
    }
}

pub(crate) fn string(value: &Json, path: &str) -> ContractResult<String> {
    match value {
        Json::String(text) => Ok(text.clone()),
        _ => Err(fail("invalid_type", path, "expected a string")),
    }
}

fn identifier_text(text: &str, path: &str) -> ContractResult<()> {
    let trimmed = text.trim();
    if text.is_empty() || trimmed != text {
        return Err(fail(
            "invalid_value",
            path,
            "must be non-empty without surrounding whitespace",
        ));
    }
    Ok(())
}

pub(crate) fn identifier(value: &Json, path: &str) -> ContractResult<String> {
    let text = string(value, path)?;
    identifier_text(&text, path)?;
    Ok(text)
}

pub(crate) fn optional_identifier(
    value: Option<&Json>,
    path: &str,
) -> ContractResult<Option<String>> {
    match value {
        None | Some(Json::Null) => Ok(None),
        Some(value) => identifier(value, path).map(Some),
    }
}

fn text(value: &Json, path: &str) -> ContractResult<String> {
    let text = string(value, path)?;
    if text.is_empty() {
        return Err(fail("invalid_value", path, "must be non-empty"));
    }
    Ok(text)
}

pub(crate) fn choice(value: &Json, path: &str, options: &[&str]) -> ContractResult<String> {
    let text = string(value, path)?;
    if options.contains(&text.as_str()) {
        Ok(text)
    } else {
        Err(fail(
            "unsupported_value",
            path,
            "unsupported enumerated value",
        ))
    }
}

fn kind(value: &Json, path: &str, options: &[&str]) -> ContractResult<String> {
    let entries = object(value, path)?;
    let Some(kind) = entries
        .iter()
        .find(|(key, _)| key == "kind")
        .map(|(_, value)| value)
    else {
        return Err(fail(
            "missing_field",
            pointer(path, "kind"),
            "required field is missing",
        ));
    };
    let text = string(kind, &pointer(path, "kind"))?;
    if options.contains(&text.as_str()) {
        Ok(text)
    } else {
        Err(fail(
            "unsupported_kind",
            pointer(path, "kind"),
            "unsupported kind",
        ))
    }
}

pub(crate) fn version(
    value: &Json,
    path: &str,
    field: &str,
    expected: &str,
    code: &'static str,
) -> ContractResult<()> {
    let entries = object(value, path)?;
    if let Some((_, found)) = entries.iter().find(|(key, _)| key == field) {
        let text = string(found, &pointer(path, field))?;
        if text != expected {
            return Err(fail(code, pointer(path, field), "unsupported version"));
        }
    }
    Ok(())
}

pub(crate) fn factor(value: &Json, path: &str) -> ContractResult<String> {
    let text = identifier(value, path)?;
    if known_registry_factor(&text) {
        Ok(text)
    } else {
        Err(fail(
            "unsupported_value",
            path,
            "unsupported registry factor",
        ))
    }
}

fn is_h3_cell(text: &str) -> bool {
    text.len() == 15
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_hex64(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn sha256_text(value: &Json, path: &str) -> ContractResult<String> {
    let text = string(value, path)?;
    if is_hex64(&text) {
        Ok(text)
    } else {
        Err(fail(
            "invalid_value",
            path,
            "must be 64 lowercase hexadecimal characters",
        ))
    }
}

fn optional_sha256(value: Option<&Json>, path: &str) -> ContractResult<Option<String>> {
    match value {
        None | Some(Json::Null) => Ok(None),
        Some(value) => sha256_text(value, path).map(Some),
    }
}

fn content_hash_text(value: &Json, path: &str) -> ContractResult<String> {
    let text = string(value, path)?;
    if let Some(digest) = text.strip_prefix("sha256:") {
        if is_hex64(digest) {
            return Ok(text);
        }
    }
    Err(fail(
        "invalid_value",
        path,
        "must be sha256: followed by 64 lowercase hex",
    ))
}

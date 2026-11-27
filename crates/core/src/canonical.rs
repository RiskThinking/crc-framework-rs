//! Current numerical interchange shared by Python, Arrow consumers, and WASM.
use serde_json::{Value, json};
use std::sync::LazyLock;

use crate::error::{CrcError, Result};
use crate::impact::{
    ImpactContractError,
    quadrature::{GaussKronrod, integrate},
};
use crate::numerical::{
    IntegralMoment, NumericalLaw, ProbabilitySegment, compensated, condition_segments,
    linear_interpolate, segment_cvar, segment_integral, segment_inverse_survival,
    segment_probability, segment_quantile,
};
use crate::{
    DiscreteDistribution, HistogramExposure, HurdleDistribution, ImpactDefinition,
    ImpactDistribution, ImpactSource, Interpolation, PointMassDistribution, TabulatedDistribution,
};

const NUMERICAL_URI: &str = "https://climateriskcommons.org/schemas/numerical/1.0";
const RESOURCES: &[(&str, &str)] = &[
    (
        "https://climateriskcommons.org/schemas/taxonomy/1.0",
        include_str!("../../../python/crc_framework/contracts/taxonomy/taxonomy-1.0.schema.json"),
    ),
    (
        "https://climateriskcommons.org/schemas/dataset/2.0",
        include_str!("../../../python/crc_framework/contracts/dataset/dataset-2.0.schema.json"),
    ),
    (
        NUMERICAL_URI,
        include_str!("../../../python/crc_framework/contracts/numerical/numerical-1.0.schema.json"),
    ),
    (
        "https://climateriskcommons.org/schemas/fitting/1.1",
        include_str!("../../../python/crc_framework/contracts/numerical/fitting-1.1.schema.json"),
    ),
    (
        "https://climateriskcommons.org/schemas/measurement-1.0.schema.json",
        include_str!(
            "../../../python/crc_framework/contracts/scientific/measurement-1.0.schema.json"
        ),
    ),
    (
        "https://climateriskcommons.org/schemas/impact-context-1.1.schema.json",
        include_str!(
            "../../../python/crc_framework/contracts/scientific/impact-context-1.1.schema.json"
        ),
    ),
    (
        "https://climateriskcommons.org/schemas/impact-2.0.schema.json",
        include_str!("../../../python/crc_framework/contracts/scientific/impact-2.0.schema.json"),
    ),
    (
        "https://climateriskcommons.org/schemas/impact-reference-2.0.schema.json",
        include_str!(
            "../../../python/crc_framework/contracts/scientific/impact-reference-2.0.schema.json"
        ),
    ),
];

fn invalid(message: impl ToString) -> CrcError {
    CrcError::InvalidInput(message.to_string())
}

static VALIDATORS: LazyLock<std::result::Result<Vec<(String, jsonschema::Validator)>, String>> =
    LazyLock::new(|| {
        let mut registry = jsonschema::Registry::new();
        for (uri, text) in RESOURCES {
            registry = registry
                .add(
                    *uri,
                    serde_json::from_str::<Value>(text).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
        }
        let registry = registry.prepare().map_err(|e| e.to_string())?;
        let mut validators: Vec<(String, jsonschema::Validator)> = RESOURCES
            .iter()
            .map(|(uri, text)| {
                let schema: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
                Ok((
                    uri.to_string(),
                    jsonschema::options()
                        .with_registry(&registry)
                        .should_validate_formats(true)
                        .build(&schema)
                        .map_err(|e| e.to_string())?,
                ))
            })
            .collect::<std::result::Result<_, String>>()?;
        let uri = "https://climateriskcommons.org/schemas/fitting/1.1#/$defs/specification";
        validators.push((
            uri.to_string(),
            jsonschema::options()
                .with_registry(&registry)
                .build(&json!({"$ref":uri}))
                .map_err(|e| e.to_string())?,
        ));
        Ok(validators)
    });

/// Strict parsing also rejects duplicate keys before serde preserves integer values.
pub fn parse_document(text: &str) -> Result<Value> {
    struct Checked;
    impl<'de> serde::Deserialize<'de> for Checked {
        fn deserialize<D: serde::Deserializer<'de>>(
            deserializer: D,
        ) -> std::result::Result<Self, D::Error> {
            struct Visitor;
            impl<'de> serde::de::Visitor<'de> for Visitor {
                type Value = Checked;
                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("a JSON value without duplicate fields")
                }
                fn visit_map<M: serde::de::MapAccess<'de>>(
                    self,
                    mut map: M,
                ) -> std::result::Result<Checked, M::Error> {
                    let mut seen = std::collections::HashSet::new();
                    while let Some(key) = map.next_key::<String>()? {
                        if !seen.insert(key) {
                            return Err(serde::de::Error::custom("duplicate JSON field"));
                        }
                        map.next_value::<Checked>()?;
                    }
                    Ok(Checked)
                }
                fn visit_seq<S: serde::de::SeqAccess<'de>>(
                    self,
                    mut seq: S,
                ) -> std::result::Result<Checked, S::Error> {
                    while seq.next_element::<Checked>()?.is_some() {}
                    Ok(Checked)
                }
                fn visit_bool<E: serde::de::Error>(
                    self,
                    _: bool,
                ) -> std::result::Result<Checked, E> {
                    Ok(Checked)
                }
                fn visit_i64<E: serde::de::Error>(self, _: i64) -> std::result::Result<Checked, E> {
                    Ok(Checked)
                }
                fn visit_u64<E: serde::de::Error>(self, _: u64) -> std::result::Result<Checked, E> {
                    Ok(Checked)
                }
                fn visit_f64<E: serde::de::Error>(self, _: f64) -> std::result::Result<Checked, E> {
                    Ok(Checked)
                }
                fn visit_str<E: serde::de::Error>(
                    self,
                    _: &str,
                ) -> std::result::Result<Checked, E> {
                    Ok(Checked)
                }
                fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Checked, E> {
                    Ok(Checked)
                }
            }
            deserializer.deserialize_any(Visitor)
        }
    }
    serde_json::from_str::<Checked>(text).map_err(invalid)?;
    let value: Value = serde_json::from_str(text).map_err(invalid)?;
    fn finite_floats(value: &Value) -> Result<()> {
        match value {
            Value::Number(number) => {
                let spelling = number.to_string();
                if spelling.contains(['.', 'e', 'E'])
                    && !number.as_f64().is_some_and(f64::is_finite)
                {
                    return Err(invalid(
                        "floating scientific values must be finite binary64 numbers",
                    ));
                }
            }
            Value::Array(items) => {
                for v in items {
                    finite_floats(v)?;
                }
            }
            Value::Object(map) => {
                for v in map.values() {
                    finite_floats(v)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    finite_floats(&value)?;
    Ok(value)
}

pub fn validate_structure(uri: &str, value: &Value) -> Result<()> {
    let validators = VALIDATORS.as_ref().map_err(invalid)?;
    let validator = &validators
        .iter()
        .find(|(id, _)| id == uri)
        .ok_or_else(|| invalid("unknown current schema URI"))?
        .1;
    validator.validate(value).map_err(invalid)?;
    match uri {
        "https://climateriskcommons.org/schemas/measurement-1.0.schema.json"
        | "https://climateriskcommons.org/schemas/impact-context-1.1.schema.json" => {
            validate_custom_science(value)
        }
        "https://climateriskcommons.org/schemas/impact-2.0.schema.json" => {
            validate_structure(
                "https://climateriskcommons.org/schemas/measurement-1.0.schema.json",
                &value["input"],
            )?;
            validate_structure(
                "https://climateriskcommons.org/schemas/measurement-1.0.schema.json",
                &value["output"],
            )?;
            if let Some(proof) = value.get("provenance") {
                validate_structure(
                    "https://climateriskcommons.org/schemas/impact-context-1.1.schema.json",
                    &proof["context"],
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Validate a complete current document, including its scientific invariants.
pub fn validate_document(uri: &str, text: &str) -> Result<()> {
    let value = parse_document(text)?;
    match uri {
        NUMERICAL_URI => {
            Numerical::from_value(value)?;
            Ok(())
        }
        crate::dataset::DATASET_URI => crate::dataset::validate_dataset(&value),
        "https://climateriskcommons.org/schemas/taxonomy/1.0" => {
            crate::dataset::validate_catalog(&value)
        }
        "https://climateriskcommons.org/schemas/fitting/1.1" => {
            crate::dataset::validate_fit_record(&value)
        }
        "https://climateriskcommons.org/schemas/fitting/1.1#/$defs/specification" => {
            crate::dataset::validate_specification(&value)
        }
        "https://climateriskcommons.org/schemas/impact-2.0.schema.json" => {
            compile_impact(&value)?;
            Ok(())
        }
        "https://climateriskcommons.org/schemas/impact-reference-2.0.schema.json" => {
            validate_structure(uri, &value)?;
            validate_reference(&value)
        }
        _ => validate_structure(uri, &value),
    }
}

fn validate_reference(value: &Value) -> Result<()> {
    let (_, version, _, digest) = crate::impact::registry_identity();
    if value["version"] != version || value["manifest_sha256"] != format!("sha256:{digest}") {
        return Err(invalid("the pinned model manifest is not installed"));
    }
    Ok(())
}

pub(crate) fn scientific_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(left), Value::Number(right)) => {
            // Python treats decimals as binary64 and keeps integer tokens exact.
            // Compare integral values without collapsing adjacent large integers.
            left == right
                || left.as_f64().zip(right.as_f64()).is_some_and(|(x, y)| {
                    x == y
                        && (x.fract() != 0.0
                            || crate::dataset::integer(a)
                                .ok()
                                .zip(crate::dataset::integer(b).ok())
                                .is_some_and(|(a, b)| a == b))
                })
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| scientific_equal(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, a)| b.get(k).is_some_and(|b| scientific_equal(a, b)))
        }
        _ => a == b,
    }
}

fn validate_custom_science(value: &Value) -> Result<()> {
    if let Value::Object(map) = value {
        if map.get("kind").and_then(Value::as_str) == Some("measurement") {
            if let Some(schema) = map.get("schema") {
                validate_extension_schema(schema)?;
                let validator = jsonschema::options()
                    .should_validate_formats(false)
                    .build(schema)
                    .map_err(invalid)?;
                validator
                    .validate(map.get("fields").unwrap_or(&json!({})))
                    .map_err(invalid)?;
            }
        }
        if map.get("kind").and_then(Value::as_str) == Some("climate_index") {
            if let Some(period) = map.get("reference_period") {
                if crate::dataset::greater(
                    &crate::dataset::integer(&period["start"])?,
                    &crate::dataset::integer(&period["end"])?,
                ) {
                    return Err(invalid("reference period boundaries must be ordered"));
                }
            }
        }
        if map.get("kind").and_then(Value::as_str) == Some("cell_context")
            && map.get("cell").is_some_and(Value::is_number)
        {
            let cell = map["cell"]
                .as_u64()
                .ok_or_else(|| invalid("invalid integer H3 cell"))?;
            crate::spatial::cell_from_u64(cell)?;
        }
    }
    Ok(())
}

pub fn validate_extension_schema(schema: &Value) -> Result<()> {
    fn fragment(value: &str) -> Result<String> {
        let bytes = value.as_bytes();
        let mut result = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' {
                if i + 2 >= bytes.len() {
                    return Err(invalid("invalid local schema reference"));
                }
                let n = std::str::from_utf8(&bytes[i + 1..i + 3]).map_err(invalid)?;
                result.push(u8::from_str_radix(n, 16).map_err(invalid)?);
                i += 3;
            } else {
                result.push(bytes[i]);
                i += 1;
            }
        }
        String::from_utf8(result).map_err(invalid)
    }
    let mut pending = vec![schema];
    let mut visited = std::collections::HashSet::new();
    while let Some(node) = pending.pop() {
        if !visited.insert(node as *const Value as usize) {
            continue;
        }
        let Some(map) = node.as_object() else {
            continue;
        };
        if map
            .get("$schema")
            .is_some_and(|v| v != "https://json-schema.org/draft/2020-12/schema")
        {
            return Err(invalid("custom schemas use Draft 2020-12"));
        }
        for key in ["$ref", "$dynamicRef"] {
            if let Some(value) = map.get(key) {
                let reference = value
                    .as_str()
                    .filter(|s| s.starts_with('#'))
                    .ok_or_else(|| invalid("custom schema references must be local"))?;
                let pointer = fragment(&reference[1..])?;
                if pointer.is_empty() {
                    pending.push(schema);
                } else if pointer.starts_with('/') {
                    if let Some(target) = schema.pointer(&pointer) {
                        pending.push(target);
                    }
                }
            }
        }
        for key in [
            "else",
            "then",
            "contentSchema",
            "if",
            "contains",
            "unevaluatedItems",
            "items",
            "additionalProperties",
            "unevaluatedProperties",
            "propertyNames",
            "not",
        ] {
            if let Some(child) = map.get(key) {
                pending.push(child);
            }
        }
        for key in [
            "$defs",
            "properties",
            "patternProperties",
            "definitions",
            "dependentSchemas",
        ] {
            if let Some(children) = map.get(key).and_then(Value::as_object) {
                pending.extend(children.values());
            }
        }
        for key in ["prefixItems", "oneOf", "allOf", "anyOf"] {
            if let Some(children) = map.get(key).and_then(Value::as_array) {
                pending.extend(children);
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
enum Compiled {
    Law(NumericalLaw),
    Curve {
        probabilities: Vec<f64>,
        values: Vec<f64>,
    },
    Hurdle {
        base: Box<Numerical>,
        probability: f64,
        location: f64,
        log_surviving_mass: f64,
        segments: Option<Vec<ProbabilitySegment>>,
    },
    Transformed {
        source: Box<Numerical>,
        definition: Box<ImpactDefinition>,
        image: Box<ImpactDistribution>,
    },
}

#[derive(Clone, Debug)]
pub struct Numerical {
    document: Value,
    compiled: Compiled,
}

fn number(value: &Value, key: &str) -> Result<f64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .ok_or_else(|| invalid(format!("{key} must be finite")))
}
fn numbers(value: &Value, key: &str) -> Result<Vec<f64>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("missing {key}")))?
        .iter()
        .map(|v| {
            v.as_f64()
                .filter(|n| n.is_finite())
                .ok_or_else(|| invalid("expected finite coordinates"))
        })
        .collect()
}

fn bare(document: &Value) -> Value {
    let mut data = document.clone();
    let map = data.as_object_mut().expect("validated numerical object");
    map.remove("schema_version");
    map.remove("object");
    map.remove("evidence");
    data
}
fn envelope(data: &Value) -> Value {
    let mut map = data.as_object().expect("validated nested data").clone();
    map.insert("schema_version".into(), json!("1.0"));
    map.insert("object".into(), json!("distribution"));
    Value::Object(map)
}

impl Numerical {
    pub fn from_json(text: &str) -> Result<Self> {
        Self::from_value(parse_document(text)?)
    }
    pub fn from_value(document: Value) -> Result<Self> {
        validate_structure(NUMERICAL_URI, &document)?;
        Self::compile(document)
    }
    fn compile(document: Value) -> Result<Self> {
        let compiled = if document["object"] == "quantile_curve" {
            let probabilities = numbers(&document, "probabilities")?;
            let values = numbers(&document, "values")?;
            validate_knots(&probabilities, &values)?;
            Compiled::Curve {
                probabilities,
                values,
            }
        } else if document["kind"] == "transformed" {
            let source = Box::new(Self::from_value(document["source"].clone())?);
            let definition = compile_impact(&document["impact"])?;
            let image = source.transform(&definition, &document["impact"])?;
            Compiled::Transformed {
                source,
                definition: Box::new(definition),
                image: Box::new(image),
            }
        } else if document["kind"] == "hurdle" {
            let base = Box::new(Self::from_value(envelope(&document["base"]))?);
            let probability = number(&document, "atom_probability")?;
            let location = number(&document, "atom_location")?;
            match base.compiled {
                Compiled::Law(law) => {
                    Compiled::Law(NumericalLaw::hurdle(law, probability, location)?)
                }
                _ => {
                    let log_surviving_mass = if probability < 1.0 {
                        base.log_exceedance(location, false)?
                    } else {
                        0.0
                    };
                    // Conditioning an affine parametric image can use the
                    // same logarithmic survival coordinates as its base law.
                    let affine =
                        if log_surviving_mass < (2.0 * f64::MIN_POSITIVE / f64::EPSILON).ln() {
                            base.affine_parametric()
                        } else {
                            None
                        };
                    if let Some(affine) = affine {
                        Compiled::Law(NumericalLaw::hurdle(
                            NumericalLaw::Parametric(affine),
                            probability,
                            location,
                        )?)
                    } else {
                        if !base.is_distribution() || !log_surviving_mass.is_finite() {
                            return Err(invalid(
                                "hurdle requires a complete nonempty conditional tail",
                            ));
                        }
                        let segments = if probability == 1.0 {
                            Some(vec![(location, location, 1.0)])
                        } else {
                            base.finite_segments()?.map(|segments| {
                                condition_segments(
                                    segments,
                                    probability,
                                    location,
                                    log_surviving_mass.exp(),
                                )
                            })
                        };
                        if segments.as_ref().is_some_and(Vec::is_empty) {
                            return Err(invalid("hurdle requires a nonempty strict upper tail"));
                        }
                        Compiled::Hurdle {
                            base,
                            probability,
                            location,
                            log_surviving_mass,
                            segments,
                        }
                    }
                }
            }
        } else {
            Compiled::Law(NumericalLaw::from_json(&bare(&document).to_string())?)
        };
        let result = Self { document, compiled };
        result.validate_evidence()?;
        Ok(result)
    }
    pub fn is_distribution(&self) -> bool {
        self.document["object"] == "distribution"
    }

    fn affine_parametric(&self) -> Option<crate::FittedDistribution> {
        use crate::DistributionFamily as Family;
        match &self.compiled {
            Compiled::Law(NumericalLaw::Parametric(law)) => Some(law.clone()),
            Compiled::Transformed { source, .. } => {
                let impact = &self.document["impact"];
                let model = &impact["model"];
                if model["kind"] != "linear" || impact.get("clamp").is_some() {
                    return None;
                }
                let slope = model["slope"].as_f64()?;
                let intercept = model["intercept"].as_f64().unwrap_or(0.0);
                let source = source.affine_parametric()?;
                let (family, shape) = if slope > 0.0 {
                    (source.family, source.shape)
                } else {
                    match source.family {
                        Family::GumbelRight => (Family::GumbelLeft, None),
                        Family::GumbelLeft => (Family::GumbelRight, None),
                        Family::WeibullMin => (Family::WeibullMax, source.shape),
                        Family::WeibullMax => (Family::WeibullMin, source.shape),
                        Family::SkewNormal => (source.family, source.shape.map(|a| -a)),
                        _ => return None,
                    }
                };
                crate::FittedDistribution::from_parameters(
                    family,
                    shape,
                    slope * source.location + intercept,
                    slope.abs() * source.scale,
                )
                .ok()
            }
            _ => None,
        }
    }

    pub fn hurdle_atoms(&self) -> Result<Vec<(f64, f64)>> {
        if let Compiled::Law(law @ NumericalLaw::Hurdle { .. }) = &self.compiled {
            return Ok(law.atoms());
        }
        let Compiled::Hurdle {
            base,
            probability,
            location,
            segments,
            ..
        } = &self.compiled
        else {
            return Err(invalid("requires a hurdle"));
        };
        if let Some(segments) = segments {
            return Ok(segments
                .iter()
                .filter_map(|&(a, b, p)| (a == b).then_some((a, p)))
                .collect());
        }
        let mut locations: Vec<_> = match &base.compiled {
            Compiled::Law(law) => law.atoms().into_iter().map(|(x, _)| x).collect(),
            Compiled::Transformed { image, .. } => {
                image.atoms().iter().map(|atom| atom.value).collect()
            }
            Compiled::Hurdle { .. } => base.hurdle_atoms()?.into_iter().map(|(x, _)| x).collect(),
            Compiled::Curve { .. } => unreachable!("hurdles require complete distributions"),
        };
        // An endpoint clamp can carry a positive atom too small to appear in
        // the unconditioned list, but significant after normalizing.
        let upper = base.quantile(1.0)?;
        if upper.is_finite() {
            locations.push(upper);
        }
        locations.retain(|x| *x > *location);
        locations.sort_by(f64::total_cmp);
        locations.dedup();
        let mut atoms = Vec::new();
        if *probability > 0.0 {
            atoms.push((*location, *probability));
        }
        for value in locations {
            let mass = self.hurdle_probability(value, value, true, true)?;
            if mass > 0.0 {
                atoms.push((value, mass));
            }
        }
        Ok(atoms)
    }
    /// Whether a compound hurdle has exact, prepared interval and atom masses.
    pub fn finite_hurdle_segments(&self) -> Option<&[(f64, f64, f64)]> {
        if let Compiled::Hurdle { segments, .. } = &self.compiled {
            segments.as_deref()
        } else {
            None
        }
    }
    pub fn hurdle_probability(
        &self,
        lower: f64,
        upper: f64,
        include_lower: bool,
        include_upper: bool,
    ) -> Result<f64> {
        if let Compiled::Law(law @ NumericalLaw::Hurdle { .. }) = &self.compiled {
            return Ok(if lower == f64::NEG_INFINITY && !include_lower {
                law.cdf(upper, include_upper)
            } else if lower == upper && include_lower && include_upper {
                law.mass(lower)
            } else if upper == f64::INFINITY && !include_upper {
                law.exceedance(lower, include_lower)
            } else {
                return Err(invalid("invalid hurdle probability bounds"));
            });
        }
        if let Some(segments) = self.finite_hurdle_segments() {
            return Ok(segment_probability(
                segments,
                lower,
                upper,
                include_lower,
                include_upper,
            ));
        }
        let Compiled::Hurdle {
            probability,
            location,
            ..
        } = &self.compiled
        else {
            return Err(invalid("requires a hurdle"));
        };
        if lower == f64::NEG_INFINITY && !include_lower {
            return Ok(
                if upper < *location || upper == *location && !include_upper {
                    0.0
                } else {
                    -self.log_exceedance(upper, !include_upper)?.exp_m1()
                },
            );
        }
        if lower == upper && include_lower && include_upper {
            return Ok(if lower == *location {
                *probability
            } else {
                let inclusive = self.log_exceedance(lower, true)?;
                let strict = self.log_exceedance(lower, false)?;
                if inclusive.is_finite() {
                    inclusive.exp() * -(strict - inclusive).min(0.0).exp_m1()
                } else {
                    0.0
                }
            });
        }
        if upper == f64::INFINITY && !include_upper {
            return Ok(self.log_exceedance(lower, include_lower)?.exp());
        }
        Err(invalid("invalid hurdle probability bounds"))
    }
    pub fn hurdle_integral(
        &self,
        lo: f64,
        hi: f64,
        center: f64,
        square: bool,
        survival: bool,
    ) -> Result<f64> {
        self.scaled_hurdle_integral(lo, hi, center, square, survival, 1.0)
    }
    pub fn scaled_hurdle_integral(
        &self,
        lo: f64,
        hi: f64,
        center: f64,
        square: bool,
        survival: bool,
        scale: f64,
    ) -> Result<f64> {
        if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi > 1.0 || hi < lo {
            return Err(invalid("invalid integration probability interval"));
        }
        if let Compiled::Law(law @ NumericalLaw::Hurdle { .. }) = &self.compiled {
            return law.scaled_integral(lo, hi, center, square, survival, scale);
        }
        let moment = IntegralMoment::new(center, square, scale);
        if let Some(segments) = self.finite_hurdle_segments() {
            return Ok(segment_integral(segments, lo, hi, moment, survival));
        }
        let (lo, hi) = if survival {
            (lo, hi)
        } else {
            (1.0 - hi, 1.0 - lo)
        };
        if lo == hi {
            return Ok(0.0);
        }
        let mut contributions = Vec::new();
        let mut start = lo;
        for (value, mass) in self.hurdle_atoms()?.into_iter().rev() {
            let left = self.log_exceedance(value, false)?.exp();
            let right = left + mass;
            if left > hi || right < lo {
                continue;
            }
            if start < left {
                contributions.push(self.integrate_survival(start, left, moment)?);
            }
            let width = if lo <= left && hi >= right {
                mass
            } else {
                (hi.min(right) - lo.max(left)).max(0.0)
            };
            contributions.push(moment.segment(width, value, value));
            start = start.max(right);
        }
        if start < hi {
            contributions.push(self.integrate_survival(start, hi, moment)?);
        }
        Ok(compensated(contributions.into_iter()))
    }

    fn integrate_survival(&self, lo: f64, hi: f64, moment: IntegralMoment) -> Result<f64> {
        let width = hi - lo;
        let result = integrate(&[(0.0, 1.0)], &GaussKronrod::defaults(), |t| {
            // Integrating after conditioning avoids dividing by the (possibly
            // unrepresentable) probability of the original tail.
            let (logarithm, weight) = if lo == 0.0 {
                (hi.ln() + 4.0 * t.ln(), width * (4.0 * t.powi(3)))
            } else {
                ((lo + width * t).ln(), width)
            };
            let value = self.inverse_log_survival(logarithm).map_err(|error| {
                ImpactContractError::new("numerical_failure", "", error.to_string())
            })?;
            Ok(moment.segment(weight, value, value))
        })
        .map_err(invalid)?;
        Ok(result.value)
    }
    pub fn canonical_json(&self) -> String {
        self.document.to_string()
    }
    pub fn quantile(&self, q: f64) -> Result<f64> {
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return Err(invalid("probability must be finite and in [0,1]"));
        }
        match &self.compiled {
            Compiled::Law(law) => law.quantile(q),
            Compiled::Curve {
                probabilities,
                values,
            } => interpolate(probabilities, values, q),
            Compiled::Hurdle {
                probability: p,
                location: a,
                segments,
                ..
            } => {
                if *p == 1.0 || q <= *p && *p > 0.0 {
                    Ok(*a)
                } else if let Some(segments) = segments {
                    Ok(segment_quantile(segments, q, false))
                } else {
                    self.inverse_survival(1.0 - q)
                }
            }
            Compiled::Transformed { image, .. } => image.quantile(q).map_err(invalid),
        }
    }
    /// Evaluate a complete law at a direct upper-tail probability.
    pub fn inverse_survival(&self, s: f64) -> Result<f64> {
        if !s.is_finite() || !(0.0..=1.0).contains(&s) {
            return Err(invalid("survival probability must be finite and in [0,1]"));
        }
        match &self.compiled {
            Compiled::Law(law) => law.inverse_survival(s),
            Compiled::Transformed { image, .. } => image.inverse_survival(s).map_err(invalid),
            Compiled::Hurdle {
                base,
                probability: p,
                location: a,
                log_surviving_mass,
                segments,
            } => {
                if *p == 1.0 || *p > 0.0 && s >= 1.0 - p {
                    Ok(*a)
                } else if let Some(segments) = segments {
                    Ok(segment_inverse_survival(segments, s, false))
                } else {
                    base.inverse_log_survival(s.ln() - (-p).ln_1p() + log_surviving_mass)
                        .map(|value| value.max(*a))
                }
            }
            Compiled::Curve { .. } => Err(invalid("partial quantile evidence is not a law")),
        }
    }
    fn inverse_log_survival(&self, logarithm: f64) -> Result<f64> {
        if logarithm >= f64::MIN_POSITIVE.ln() || logarithm == f64::NEG_INFINITY {
            return self.inverse_survival(logarithm.exp());
        }
        match &self.compiled {
            Compiled::Law(law) => law.inverse_log_survival(logarithm),
            Compiled::Transformed { image, .. } => {
                image.inverse_log_survival(logarithm).map_err(invalid)
            }
            Compiled::Hurdle {
                base,
                probability,
                location,
                log_surviving_mass,
                segments: None,
            } => {
                let log_weight = (-probability).ln_1p();
                if logarithm >= log_weight {
                    Ok(*location)
                } else {
                    base.inverse_log_survival(logarithm - log_weight + log_surviving_mass)
                        .map(|value| value.max(*location))
                }
            }
            _ => self.inverse_survival(logarithm.exp()),
        }
    }

    fn log_exceedance(&self, x: f64, inclusive: bool) -> Result<f64> {
        match &self.compiled {
            Compiled::Law(law) => Ok(law.log_exceedance(x, inclusive)),
            Compiled::Hurdle {
                base,
                probability: p,
                location: a,
                log_surviving_mass,
                segments,
            } => {
                if let Some(segments) = segments {
                    return Ok(
                        segment_probability(segments, x, f64::INFINITY, inclusive, false).ln(),
                    );
                }
                if x < *a || inclusive && x == *a {
                    Ok(0.0)
                } else if *p == 1.0 {
                    Ok(f64::NEG_INFINITY)
                } else {
                    Ok((-p).ln_1p()
                        + (base.log_exceedance(x, inclusive)? - log_surviving_mass).min(0.0))
                }
            }
            Compiled::Transformed { image, .. } => {
                if let Some(segments) = self.finite_segments()? {
                    return Ok(
                        segment_probability(&segments, x, f64::INFINITY, inclusive, false).ln(),
                    );
                }
                image.log_exceedance(x, inclusive).map_err(invalid)
            }
            Compiled::Curve { .. } => Err(invalid(
                "quantile evidence has no whole-law threshold probability",
            )),
        }
    }
    fn transform(
        &self,
        function: &ImpactDefinition,
        document: &Value,
    ) -> Result<ImpactDistribution> {
        if self
            .output_measurement()
            .is_some_and(|output| !scientific_equal(output, &document["input"]))
        {
            return Err(invalid(
                "source impact output does not match the next impact input",
            ));
        }
        match self.impact_source() {
            Ok(source) => function.pushforward(&source).map_err(invalid),
            Err(error @ CrcError::Unsupported(_)) => {
                if let Compiled::Transformed { source, .. } = &self.compiled {
                    if let Some(composed) = compose_affine(document, &self.document["impact"]) {
                        let function = compile_impact(&composed)?;
                        return source.transform(&function, &composed);
                    }
                }
                Err(error)
            }
            Err(error) => Err(error),
        }
    }

    fn output_measurement(&self) -> Option<&Value> {
        let mut document = &self.document;
        while document["kind"] == "hurdle" {
            document = &document["base"];
        }
        if document["kind"] == "transformed" {
            return document["impact"].get("output");
        }
        None
    }

    pub fn impact_source(&self) -> Result<ImpactSource> {
        if !self.is_distribution() {
            return Err(invalid(
                "quantile evidence must be explicitly completed or conditioned",
            ));
        }
        match &self.compiled {
            Compiled::Law(law) => law_source(law),
            Compiled::Hurdle { .. } | Compiled::Transformed { .. } => {
                let segments = self.finite_segments()?.ok_or_else(|| {
                    CrcError::Unsupported("no exact source for this compound law".into())
                })?;
                ImpactSource::from_segments(segments).map_err(invalid)
            }
            Compiled::Curve { .. } => unreachable!(),
        }
    }

    /// Compute a trimmed mean for a law with finite probability-linear segments.
    pub fn finite_cvar(&self, level: f64, upper: bool) -> Result<Option<f64>> {
        Ok(self
            .finite_segments()?
            .and_then(|segments| segment_cvar(&segments, level, upper)))
    }

    fn finite_segments(&self) -> Result<Option<Vec<ProbabilitySegment>>> {
        match &self.compiled {
            Compiled::Law(law) => Ok(law.finite_segments()),
            Compiled::Hurdle { segments, .. } => Ok(segments.clone()),
            Compiled::Transformed { image, .. } if image.is_discrete() => Ok(Some(
                image
                    .atoms()
                    .iter()
                    .map(|atom| (atom.value, atom.value, atom.probability))
                    .collect(),
            )),
            Compiled::Transformed {
                source, definition, ..
            } => {
                let Some(segments) = source.finite_segments()? else {
                    return Ok(None);
                };
                definition.image_segments(&segments).map_err(invalid)
            }
            Compiled::Curve { .. } => Ok(None),
        }
    }
    fn validate_evidence(&self) -> Result<()> {
        let Some(evidence) = self.document.get("evidence") else {
            return Ok(());
        };
        let source = Self::from_value(evidence["source"].clone())?;
        let Compiled::Curve {
            probabilities,
            values,
        } = source.compiled
        else {
            return Err(invalid("modelling source must be quantile evidence"));
        };
        let (expected_q, expected_x) = if evidence["operation"] == "condition" {
            let lo = number(evidence, "q_min")?;
            let hi = number(evidence, "q_max")?;
            if lo >= hi || lo < probabilities[0] || hi > *probabilities.last().unwrap() {
                return Err(invalid("invalid conditioning interval"));
            }
            let q: Vec<_> = std::iter::once(lo)
                .chain(probabilities.iter().copied().filter(|q| *q > lo && *q < hi))
                .chain(std::iter::once(hi))
                .collect();
            let x = q
                .iter()
                .map(|q| interpolate(&probabilities, &values, *q))
                .collect::<Result<Vec<_>>>()?;
            (
                q.iter().map(|q| (q - lo) / (hi - lo)).collect::<Vec<_>>(),
                x,
            )
        } else {
            let lower = number(&evidence["policy"], "lower")?;
            let upper = number(&evidence["policy"], "upper")?;
            if lower > values[0]
                || upper < *values.last().unwrap()
                || probabilities[0] == 0.0 && lower != values[0]
                || *probabilities.last().unwrap() == 1.0 && upper != *values.last().unwrap()
            {
                return Err(invalid("completion contradicts supplied endpoints"));
            }
            let (mut q, mut x) = (probabilities, values);
            if q[0] > 0.0 {
                q.insert(0, 0.0);
                x.insert(0, lower);
            }
            if *q.last().unwrap() < 1.0 {
                q.push(1.0);
                x.push(upper);
            }
            (q, x)
        };
        if numbers(&self.document, "probabilities")? != expected_q
            || numbers(&self.document, "values")? != expected_x
        {
            return Err(invalid(
                "numerical payload contradicts its modelling record",
            ));
        }
        Ok(())
    }
}

fn validate_knots(q: &[f64], x: &[f64]) -> Result<()> {
    if q.len() < 2
        || q.len() != x.len()
        || q.windows(2).any(|a| a[0] >= a[1])
        || x.windows(2).any(|a| a[0] > a[1])
    {
        Err(invalid(
            "quantile knots must have matching ordered coordinates",
        ))
    } else {
        Ok(())
    }
}
fn interpolate(q: &[f64], x: &[f64], p: f64) -> Result<f64> {
    if p < q[0] || p > *q.last().unwrap() {
        return Err(CrcError::OutOfSupport {
            probability: p,
            minimum: q[0],
            maximum: *q.last().unwrap(),
        });
    }
    let upper = q.partition_point(|a| *a < p);
    if upper == 0 {
        Ok(x[0])
    } else if q[upper] == p {
        Ok(x[upper])
    } else {
        let i = upper - 1;
        Ok(linear_interpolate(
            x[i],
            x[upper],
            (p - q[i]) / (q[upper] - q[i]),
        ))
    }
}
fn law_source(law: &NumericalLaw) -> Result<ImpactSource> {
    match law {
        NumericalLaw::Parametric(d) => Ok(ImpactSource::Fitted(d.clone())),
        NumericalLaw::PointMass(x) => PointMassDistribution::new(*x).map(ImpactSource::PointMass),
        NumericalLaw::Discrete {
            values,
            counts: Some(ranks),
            ..
        } => DiscreteDistribution::from_counts(values.clone(), ranks)
            .map(ImpactSource::Discrete)
            .map_err(invalid),
        NumericalLaw::Discrete {
            values,
            probabilities,
            ..
        } => DiscreteDistribution::new(values.clone(), probabilities.clone())
            .map(ImpactSource::Discrete)
            .map_err(invalid),
        NumericalLaw::Histogram {
            edges,
            probabilities,
            ..
        } => HistogramExposure::new(edges.clone(), probabilities.clone())
            .map(ImpactSource::Histogram)
            .map_err(invalid),
        NumericalLaw::Quantile {
            probabilities,
            values,
        } => TabulatedDistribution::new(
            probabilities.clone(),
            values.clone(),
            Interpolation::LinearProbability,
            false,
        )
        .map(ImpactSource::Tabulated),
        NumericalLaw::Hurdle {
            base,
            atom_probability,
            atom_location,
        } => match (*atom_probability, base.as_ref()) {
            (1.0, _) => PointMassDistribution::new(*atom_location).map(ImpactSource::PointMass),
            (p, NumericalLaw::Parametric(d)) if p > 0.0 => {
                HurdleDistribution::new(*atom_location, *atom_probability, d.clone())
                    .map(ImpactSource::Hurdle)
            }
            _ => {
                let segments = law.finite_segments().ok_or_else(|| {
                    CrcError::Unsupported("no exact source for this hurdle law".into())
                })?;
                ImpactSource::from_segments(segments).map_err(invalid)
            }
        },
    }
}

fn compose_affine(outer: &Value, inner: &Value) -> Option<Value> {
    let outer_model = &outer["model"];
    let inner_model = &inner["model"];
    if outer_model["kind"] != "linear"
        || inner_model["kind"] != "linear"
        || outer.get("clamp").is_some()
    {
        return None;
    }
    let slope = outer_model["slope"].as_f64()?;
    let intercept = outer_model["intercept"].as_f64().unwrap_or(0.0);
    let mut composed = json!({
        "schema_version": "2.0",
        "input": inner["input"],
        "output": outer["output"],
        "model": {
            "kind": "linear",
            "slope": slope * inner_model["slope"].as_f64()?,
            "intercept": slope * inner_model["intercept"].as_f64().unwrap_or(0.0) + intercept,
        },
    });
    if let Some(clamp) = inner.get("clamp") {
        let minimum = clamp["minimum"].as_f64().map(|x| slope * x + intercept);
        let maximum = clamp["maximum"].as_f64().map(|x| slope * x + intercept);
        let (minimum, maximum) = if slope < 0.0 {
            (maximum, minimum)
        } else {
            (minimum, maximum)
        };
        let mut mapped = serde_json::Map::new();
        for (bound, value) in [("minimum", minimum), ("maximum", maximum)] {
            if let Some(value) = value {
                mapped.insert(bound.into(), json!(value));
            }
        }
        composed["clamp"] = Value::Object(mapped);
    }
    Some(composed)
}
fn compile_impact(document: &Value) -> Result<ImpactDefinition> {
    validate_structure(
        "https://climateriskcommons.org/schemas/impact-2.0.schema.json",
        document,
    )?;
    fn calibration(m: &Value) -> Result<(String, String)> {
        let kind = m["kind"]
            .as_str()
            .ok_or_else(|| invalid("missing measurement kind"))?;
        let schema: Value = serde_json::from_str(
            RESOURCES
                .iter()
                .find(|(uri, _)| uri.ends_with("measurement-1.0.schema.json"))
                .expect("embedded measurement schema")
                .1,
        )
        .map_err(invalid)?;
        let branch = schema["oneOf"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["properties"]["kind"]["const"] == kind)
            .ok_or_else(|| invalid("unknown measurement kind"))?;
        let rules = &branch["x-crc-calibration"];
        let quantity = rules
            .get("quantity_field")
            .and_then(Value::as_str)
            .and_then(|field| m.get(field))
            .and_then(Value::as_str)
            .or_else(|| rules.get("quantity").and_then(Value::as_str))
            .ok_or_else(|| invalid("calibrated measurement requires quantity"))?
            .to_string();
        let unit = rules
            .get("unit_field")
            .and_then(Value::as_str)
            .and_then(|field| m.get(field))
            .and_then(Value::as_str)
            .or_else(|| rules.get("fixed_unit").and_then(Value::as_str))
            .ok_or_else(|| invalid("calibrated measurement requires unit"))?
            .to_string();
        Ok((quantity, unit))
    }
    let (input, iu) = calibration(&document["input"])?;
    let (output, ou) = calibration(&document["output"])?;
    let mut internal = json!({"schema_version":"1.0","semantics_version":"crc-impact/1","calibration":{"exposure_variable":input,"exposure_unit":iu,"impact_metric":output,"impact_unit":ou},"model":document["model"],"provenance":{"origin":{"kind":"user_defined"}}});
    if let Some(clamp) = document.get("clamp") {
        internal["clamp"] = clamp.clone();
    }
    let actual = ImpactDefinition::from_json(&internal.to_string()).map_err(invalid)?;
    if let Some(proof) = document.get("provenance") {
        let reference = &proof["model"];
        validate_reference(reference)?;
        let bindings: Value = serde_json::from_str(include_str!(
            "../../../python/crc_framework/contracts/scientific/kernel-bindings-1.0.json"
        ))
        .map_err(invalid)?;
        let profiles: Value = serde_json::from_str(include_str!(
            "../../../python/crc_framework/contracts/scientific/models-1.0.json"
        ))
        .map_err(invalid)?;
        let id = reference["id"]
            .as_str()
            .ok_or_else(|| invalid("invalid model identity"))?;
        let profile = profiles["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["id"] == id)
            .ok_or_else(|| invalid("unknown model identity"))?;
        if !scientific_equal(&document["input"], &profile["input"])
            || !scientific_equal(&document["output"], &profile["output"])
        {
            return Err(invalid(
                "impact measurements contradict pinned model provenance",
            ));
        }
        let context = &proof["context"];
        jsonschema::options()
            .should_validate_formats(true)
            .build(&profile["context_schema"])
            .map_err(invalid)?
            .validate(context)
            .map_err(invalid)?;
        let mut values = context
            .as_object()
            .ok_or_else(|| invalid("invalid model context"))?
            .clone();
        if let Some(fixed) = bindings["fixed_context"].get(id).and_then(Value::as_object) {
            values.extend(fixed.clone());
        }
        let mapped = |key: &str, table: &str| -> Result<Option<String>> {
            match values.get(key) {
                None => Ok(None),
                Some(value) => value
                    .as_str()
                    .and_then(|v| bindings[table][v].as_str())
                    .map(|v| Some(v.to_string()))
                    .ok_or_else(|| invalid("unsupported model context value")),
            }
        };
        let cell = values
            .get("cell")
            .map(|v| {
                v.as_u64()
                    .map(|v| format!("{v:x}"))
                    .ok_or_else(|| invalid("unresolved model cell binding"))
            })
            .transpose()?;
        let country = values
            .get("country")
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| invalid("unresolved country binding"))
            })
            .transpose()?;
        let historic_mean = values
            .get("historic_mean")
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| invalid("unresolved historical mean binding"))
            })
            .transpose()?;
        let (registry, version, resolver, digest) = crate::impact::registry_identity();
        let reference=crate::RegistryImpactReference::from_json(&json!({"schema_version":"1.0","registry_id":registry,"registry_revision":version,"registry_manifest_sha256":digest,"resolver_semantics_version":resolver,"factor":bindings["models"][id]}).to_string()).map_err(invalid)?;
        let expected = reference
            .resolve(&crate::ResolutionContext {
                cell,
                country,
                continent: mapped("region", "regions")?,
                building_type: mapped("asset_type", "asset_types")?,
                historic_mean,
            })
            .map_err(invalid)?;
        let expected: Value = serde_json::from_str(expected.canonical_json()).map_err(invalid)?;
        let actual_doc: Value = serde_json::from_str(actual.canonical_json()).map_err(invalid)?;
        if !scientific_equal(&actual_doc["model"], &expected["model"])
            || !scientific_equal(&actual_doc["clamp"], &expected["clamp"])
        {
            return Err(invalid(
                "impact numerical definition contradicts pinned model provenance",
            ));
        }
    }
    Ok(actual)
}

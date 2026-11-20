//! Strict scientific dataset records shared with Python and WASM consumers.

use std::collections::{BTreeMap, BTreeSet};

use geo::line_intersection::{LineIntersection, line_intersection};
use geo::{Coord, Geometry, LineString, MultiPolygon, Point, Polygon, Relate, Validation};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::canonical::{Numerical, scientific_equal, validate_structure};
use crate::error::{CrcError, Result};
use crate::numerical::compensated;

pub const DATASET_URI: &str = "https://climateriskcommons.org/schemas/dataset/2.0";

fn invalid(message: impl ToString) -> CrcError {
    CrcError::InvalidInput(message.to_string())
}

fn array(value: &Value) -> Result<&Vec<Value>> {
    value.as_array().ok_or_else(|| invalid("expected an array"))
}

fn text(value: &Value) -> Result<&str> {
    identifier(
        value
            .as_str()
            .ok_or_else(|| invalid("expected an identifier"))?,
    )
}

fn identifier(value: &str) -> Result<&str> {
    let boundary = |c: char| c.is_whitespace() || ('\u{001c}'..='\u{001f}').contains(&c);
    if value.is_empty()
        || value.chars().next().is_some_and(boundary)
        || value.chars().next_back().is_some_and(boundary)
    {
        return Err(invalid(
            "identifiers are exact nonempty strings without boundary whitespace",
        ));
    }
    Ok(value)
}

// Coefficients to a common reference for each physical dimension. Calendar
// durations are separate from durations measured in seconds. Precipitation
// mass flux is the standard water-equivalent depth rate (1 kg/m² = 1 mm).
fn unit_basis(unit: &str) -> Option<(&'static str, f64, f64)> {
    Some(match unit {
        "m" => ("length", 1.0, 0.0),
        "cm" => ("length", 0.01, 0.0),
        "mm" => ("length", 0.001, 0.0),
        "ft" => ("length", 0.3048, 0.0),
        "s" => ("duration", 1.0, 0.0),
        "h" => ("duration", 3600.0, 0.0),
        "d" => ("duration", 86400.0, 0.0),
        "month" => ("calendar_duration", 1.0, 0.0),
        "yr" => ("calendar_duration", 12.0, 0.0),
        "1" => ("ratio", 1.0, 0.0),
        "percent" => ("ratio", 0.01, 0.0),
        "degC" => ("temperature", 1.0, 273.15),
        "K" => ("temperature", 1.0, 0.0),
        "degF" => ("temperature", 5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
        "m/s" => ("speed", 1.0, 0.0),
        "km/h" => ("speed", 1.0 / 3.6, 0.0),
        "mm/day" => ("precipitation_rate", 1.0 / 86400.0, 0.0),
        "mm/h" => ("precipitation_rate", 1.0 / 3600.0, 0.0),
        "kg m-2 s-1" => ("precipitation_rate", 1.0, 0.0),
        "K.d" | "degC.d" => ("degree_days", 1.0, 0.0),
        _ => return None,
    })
}

/// Return the affine coefficients for a supported conversion of units.
#[doc(hidden)]
pub fn unit_conversion(source: &str, target: &str) -> Result<(f64, f64)> {
    match (unit_basis(source), unit_basis(target)) {
        (Some((from, from_scale, from_offset)), Some((to, to_scale, to_offset))) if from == to => {
            Ok((from_scale / to_scale, (from_offset - to_offset) / to_scale))
        }
        _ => Err(invalid(format!("cannot convert {source:?} to {target:?}"))),
    }
}

/// Validate a recorded transformation and recover the input measurement.
#[doc(hidden)]
pub fn reverse_unit_conversion(
    measurement: &Value,
    source: &str,
    target: &str,
    scale: f64,
    offset: f64,
) -> Result<Value> {
    if source == target || !scale.is_finite() || scale == 0.0 || !offset.is_finite() {
        return Err(invalid(
            "unit conversion requires changed units and finite coefficients with nonzero scale",
        ));
    }
    if measurement["unit"].as_str() != Some(target) {
        return Err(invalid(
            "conversion target unit contradicts the variable measurement",
        ));
    }
    let mut input = measurement.clone();
    input["unit"] = json!(source);
    validate_structure(
        "https://climateriskcommons.org/schemas/measurement-1.0.schema.json",
        &input,
    )
    .map_err(|error| {
        invalid(format!(
            "source unit {source:?} contradicts the variable measurement: {error}"
        ))
    })?;
    if unit_basis(source).is_some() || unit_basis(target).is_some() {
        let (expected_scale, expected_offset) = unit_conversion(source, target)?;
        let same = |actual: f64, expected: f64| {
            (actual - expected).abs() <= 32.0 * f64::EPSILON * expected.abs()
        };
        if !same(scale, expected_scale) || !same(offset, expected_offset) {
            return Err(invalid(format!(
                "unit conversion coefficients for {source:?} to {target:?} must be scale {expected_scale} and offset {expected_offset}"
            )));
        }
    }
    Ok(input)
}

pub(crate) fn integer(value: &Value) -> Result<Value> {
    let number = value
        .as_number()
        .ok_or_else(|| invalid("expected an integer JSON number"))?;
    let text = number.to_string();
    let normalized = if text.contains(['.', 'e', 'E']) {
        let value = number
            .as_f64()
            .filter(|n| n.is_finite() && n.fract() == 0.0)
            .ok_or_else(|| invalid("expected an integral JSON number"))?;
        format!("{value:.0}")
    } else {
        text
    };
    let normalized = if normalized == "-0" { "0" } else { &normalized };
    serde_json::from_str(normalized).map_err(invalid)
}

fn natural(value: &Value) -> Result<String> {
    let text = integer(value)?.to_string();
    if text.starts_with('-') {
        return Err(invalid("expected a nonnegative integer"));
    }
    Ok(text)
}

fn unsigned(value: &Value) -> Result<u64> {
    natural(value)?.parse().map_err(invalid)
}

pub(crate) fn greater(a: &Value, b: &Value) -> bool {
    let a = a.to_string();
    let b = b.to_string();
    let negative_a = a.starts_with('-');
    let negative_b = b.starts_with('-');
    if negative_a != negative_b {
        return negative_b;
    }
    let left = a.trim_start_matches('-');
    let right = b.trim_start_matches('-');
    let order = left.len().cmp(&right.len()).then(left.cmp(right));
    if negative_a {
        order.is_lt()
    } else {
        order.is_gt()
    }
}

fn natural_sum(a: &str, b: &str) -> String {
    let mut a = a.bytes().rev();
    let mut b = b.bytes().rev();
    let mut digits = Vec::new();
    let mut carry = 0;
    loop {
        let left = a.next();
        let right = b.next();
        if left.is_none() && right.is_none() && carry == 0 {
            break;
        }
        let total = left.map_or(0, |x| x - b'0') + right.map_or(0, |x| x - b'0') + carry;
        digits.push(b'0' + total % 10);
        carry = total / 10;
    }
    digits.into_iter().rev().map(char::from).collect()
}

fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), canonical(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        _ => value.clone(),
    }
}

fn period_identity(value: &Value) -> Result<String> {
    let mut definition = Map::new();
    let kind = text(&value["kind"])?;
    definition.insert("kind".into(), json!(kind));
    if kind == "window" {
        let start = integer(&value["start"])?;
        let end = integer(&value["end"])?;
        if greater(&start, &end) {
            return Err(invalid("inclusive period start must not exceed end"));
        }
        definition.insert("start".into(), json!(start));
        definition.insert("end".into(), json!(end));
        if let Some(calendar) = value.get("calendar") {
            definition.insert("calendar".into(), calendar.clone());
        }
    }
    let bytes = canonical(&Value::Object(definition))
        .to_string()
        .into_bytes();
    Ok(format!("period:{:x}", Sha256::digest(bytes)))
}

fn descriptors(values: &Value, periods: bool) -> Result<BTreeMap<String, &Value>> {
    let mut result = BTreeMap::new();
    for value in array(values)? {
        let id = text(&value["id"])?;
        if result.insert(id.into(), value).is_some() {
            return Err(invalid("duplicate scientific descriptor identity"));
        }
        if periods && period_identity(value)? != id {
            return Err(invalid("period id contradicts its scientific definition"));
        }
        if let Some(measurement) = value.get("measurement") {
            validate_structure(
                "https://climateriskcommons.org/schemas/measurement-1.0.schema.json",
                measurement,
            )?;
        }
    }
    Ok(result)
}

fn scenario_schema(value: &Value) -> Result<()> {
    if let Some(schema) = value.get("schema") {
        crate::canonical::validate_extension_schema(schema)?;
        if schema
            .get("$schema")
            .and_then(Value::as_str)
            .is_some_and(|id| id != "https://json-schema.org/draft/2020-12/schema")
        {
            return Err(invalid("custom schemas use JSON Schema draft2020-12"));
        }
        let validator = jsonschema::validator_for(schema).map_err(invalid)?;
        validator.validate(&value["definition"]).map_err(invalid)?;
    }
    Ok(())
}

fn series_key(value: &Value) -> Result<(String, Option<String>, Option<String>)> {
    let optional = |value: &Value| -> Result<Option<String>> {
        if value.is_null() {
            Ok(None)
        } else {
            Ok(Some(text(value)?.into()))
        }
    };
    Ok((
        text(&value["variable_id"])?.into(),
        optional(&value["scenario_id"])?,
        optional(&value["period_id"])?,
    ))
}

fn fit_specification(value: &Value) -> Result<Value> {
    let mut result = value
        .as_object()
        .ok_or_else(|| invalid("fit specification must be an object"))?
        .clone();
    let kind = text(&value["kind"])?;
    if kind == "model_selection" {
        let candidates = array(&value["candidates"])?
            .iter()
            .map(fit_specification)
            .collect::<Result<Vec<_>>>()?;
        if candidates.is_empty() || candidates.iter().any(|c| c["kind"] == "model_selection") {
            return Err(invalid("selection needs concrete candidate specifications"));
        }
        let samples = candidates[0]["kind"] == "samples";
        if candidates
            .iter()
            .any(|c| (c["kind"] == "samples") != samples)
        {
            return Err(invalid("selection cannot mix sample and quantile evidence"));
        }
        if candidates
            .iter()
            .find_map(|candidate| candidate["censor_location"].as_f64())
            .is_some_and(|censor| {
                candidates
                    .iter()
                    .any(|candidate| candidate["censor_location"].as_f64() != Some(censor))
            })
        {
            return Err(invalid(
                "censored model selection requires every candidate to use the same censor_location",
            ));
        }
        let criterion = value
            .get("criterion")
            .and_then(Value::as_str)
            .unwrap_or("ks");
        if criterion != if samples { "ks" } else { "normalized_rmse" } {
            return Err(invalid("selection criterion contradicts evidence kind"));
        }
        let identities = candidates
            .iter()
            .map(|v| canonical(v).to_string())
            .collect::<BTreeSet<_>>();
        if identities.len() != candidates.len() {
            return Err(invalid("candidate specifications must be distinct"));
        }
        result.insert("candidates".into(), Value::Array(candidates));
        result.insert("criterion".into(), json!(criterion));
        return Ok(Value::Object(result));
    }
    let constraints = if value["constraints"].is_null() {
        Value::Null
    } else {
        let min = value["constraints"]["minimum_value"].as_f64();
        let max = value["constraints"]["maximum_value"].as_f64();
        if min.zip(max).is_some_and(|(a, b)| a > b) {
            return Err(invalid("fit constraint minimum exceeds maximum"));
        }
        json!({"minimum_value":min,"maximum_value":max,"probability":value["constraints"]["probability"].as_f64(),"require_sample_support":value["constraints"].get("require_sample_support").and_then(Value::as_bool).unwrap_or(false)})
    };
    result.insert("constraints".into(), constraints);
    if kind == "samples" {
        let estimator = value
            .get("estimator")
            .and_then(Value::as_str)
            .unwrap_or("mle");
        let family = text(&value["family"])?;
        if estimator == "lmoments" && !["genextreme", "gumbel_r", "gumbel_l"].contains(&family) {
            return Err(invalid("family does not support L-moments"));
        }
        result.insert("estimator".into(), json!(estimator));
        result.insert(
            "minimum_samples".into(),
            integer(value.get("minimum_samples").unwrap_or(&json!(3)))?,
        );
        let fallbacks = if let Some(values) = value.get("fallbacks") {
            array(values)?
                .iter()
                .map(fit_specification)
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![]
        };
        result.insert("fallbacks".into(), Value::Array(fallbacks));
    } else {
        result.insert(
            "minimum_knots".into(),
            integer(value.get("minimum_knots").unwrap_or(&json!(4)))?,
        );
        if kind == "quantiles" {
            let weights = if let Some(weights) = value["weights"].as_array() {
                let values = weights
                    .iter()
                    .map(|v| v.as_f64().ok_or_else(|| invalid("invalid quantile weight")))
                    .collect::<Result<Vec<_>>>()?;
                if !values.iter().any(|w| *w > 0.0) {
                    return Err(invalid("at least one fitting weight must be positive"));
                }
                json!(values)
            } else {
                Value::Null
            };
            result.insert("weights".into(), weights);
        } else if kind == "censored_quantiles" {
            result.insert(
                "censor_location".into(),
                json!(value["censor_location"].as_f64()),
            );
            result.insert(
                "residual_scale_floor_ratio".into(),
                json!(
                    value
                        .get("residual_scale_floor_ratio")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.001)
                ),
            );
        } else if kind == "hurdle_quantiles" {
            result.insert(
                "atom_probability".into(),
                json!(value["atom_probability"].as_f64()),
            );
            result.insert(
                "atom_location".into(),
                json!(
                    value
                        .get("atom_location")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                ),
            );
        }
    }
    Ok(Value::Object(result))
}

fn replay_attempts(requested: &Value, attempts: &[Value], cursor: &mut usize) -> Result<bool> {
    let attempt = attempts
        .get(*cursor)
        .ok_or_else(|| invalid("attempt trace is missing a requested candidate"))?;
    if canonical(&fit_specification(&attempt["specification"])?) != canonical(requested) {
        return Err(invalid(
            "attempt trace does not execute requested candidates in order",
        ));
    }
    *cursor += 1;
    if attempt["status"] == "converged" {
        return Ok(true);
    }
    if requested["kind"] == "samples" {
        for alternative in array(&requested["fallbacks"])? {
            if replay_attempts(alternative, attempts, cursor)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Validate the one shared scientific fitting specification family.
pub fn validate_specification(value: &Value) -> Result<()> {
    validate_structure(
        "https://climateriskcommons.org/schemas/fitting/1.0#/$defs/specification",
        value,
    )?;
    fit_specification(value).map(|_| ())
}

/// Validate an exact immutable catalog without registering user-owned IDs.
pub fn validate_catalog(value: &Value) -> Result<()> {
    validate_structure("https://climateriskcommons.org/schemas/taxonomy/1.0", value)?;
    text(&value["namespace"])?;
    text(&value["revision"])?;
    descriptors(&value["variables"], false)?;
    let scenarios = descriptors(&value["scenarios"], false)?;
    for scenario in scenarios.values() {
        scenario_schema(scenario)?;
    }
    for (id, specification) in value["presets"]
        .as_object()
        .ok_or_else(|| invalid("presets must be an object"))?
    {
        text(&json!(id))?;
        fit_specification(specification)?;
    }
    Ok(())
}

/// Validate a persisted authoritative fitting record independently of row data.
pub fn validate_fit_record(record: &Value) -> Result<()> {
    validate_record(record).map(|_| ())
}

/// Check that a row carries the law produced by its selected fitting operation.
pub fn validate_fitted_curve(
    record: &Value,
    curve: &Value,
    variable: &str,
    provenance: &[Value],
) -> Result<()> {
    let selected = validate_record(record)?
        .ok_or_else(|| invalid("fit record has no converged selected attempt"))?;
    let mut fitted = curve;
    let mut variable = variable;
    for step in provenance.iter().rev() {
        if fitted["kind"] != "transformed" {
            break;
        }
        if step["kind"] == "rename" && step["dimension"] == "variable" {
            if let Some((old, _)) = step["mapping"]
                .as_object()
                .and_then(|mapping| mapping.iter().find(|(_, new)| *new == variable))
            {
                variable = old;
            }
        } else if step["kind"] == "unit_conversion" && step["variable_id"] == variable {
            let impact = &fitted["impact"];
            let model = &impact["model"];
            let mut input = impact["input"].clone();
            let output = &impact["output"];
            if input["unit"] != step["source_unit"]
                || output["unit"] != step["target_unit"]
                || impact.get("clamp").is_some()
                || impact.get("provenance").is_some()
                || model["kind"] != "linear"
                || !scientific_equal(&model["slope"], &step["scale"])
                || !scientific_equal(&model["intercept"], &step["offset"])
            {
                return Err(invalid("fitted curve contradicts its unit conversion"));
            }
            input["unit"] = output["unit"].clone();
            if !scientific_equal(&input, output) {
                return Err(invalid("unit conversion changes other measurement science"));
            }
            fitted = &fitted["source"];
        }
    }
    let base = if selected["kind"] == "hurdle_quantiles" {
        if fitted["kind"] != "hurdle"
            || !scientific_equal(&fitted["atom_probability"], &selected["atom_probability"])
            || fitted["atom_location"].as_f64()
                != Some(selected["atom_location"].as_f64().unwrap_or(0.0))
        {
            return Err(invalid("curve atom contradicts the selected hurdle fit"));
        }
        &fitted["base"]
    } else {
        fitted
    };
    if fitted["object"] != "distribution"
        || base["kind"] != "parametric"
        || base["family"] != selected["family"]
    {
        return Err(invalid("curve representation contradicts the selected fit"));
    }
    if let Some(constraints) = selected.get("constraints").filter(|v| !v.is_null()) {
        let law = Numerical::from_value(fitted.clone())?;
        let probability = constraints["probability"].as_f64();
        if let Some(minimum) = constraints["minimum_value"].as_f64() {
            if law.quantile(probability.unwrap_or(0.0))? < minimum {
                return Err(invalid("curve violates the selected fit minimum_value"));
            }
        }
        if let Some(maximum) = constraints["maximum_value"].as_f64() {
            if law.quantile(probability.unwrap_or(1.0))? > maximum {
                return Err(invalid("curve violates the selected fit maximum_value"));
            }
        }
        if constraints["require_sample_support"] == true {
            let (first, last) = fit_interval(&record["value_interval"])?
                .ok_or_else(|| invalid("fit requires its observed value interval"))?;
            if first < law.quantile(0.0)? || last > law.quantile(1.0)? {
                return Err(invalid(
                    "curve excludes the recorded observed value interval",
                ));
            }
        }
    }
    Ok(())
}

fn requested_weights(specification: &Value, count: usize) -> Option<Vec<f64>> {
    let Some(requested) = specification["weights"].as_array() else {
        return Some(vec![1.0 / count as f64; count]);
    };
    if requested.len() != count {
        return None;
    }
    let mut normalized = requested
        .iter()
        .map(Value::as_f64)
        .collect::<Option<Vec<_>>>()?;
    let maximum = normalized.iter().copied().fold(0.0, f64::max);
    let sum = compensated(normalized.iter().map(|weight| weight / maximum));
    for weight in &mut normalized {
        let effective = (*weight / maximum) / sum;
        if (*weight > 0.0 && effective == 0.0) || !effective.is_finite() {
            return None;
        }
        *weight = effective;
    }
    Some(normalized)
}

fn eligible_quantile_count(
    spec: &Value,
    evidence_count: &Value,
    (first, last): (f64, f64),
) -> Result<Option<Value>> {
    let effective_count = if let Some(requested) = spec["weights"].as_array() {
        if json!(requested.len()) != *evidence_count {
            return Ok(None);
        }
        let Some(weights) = requested_weights(spec, requested.len()) else {
            return Ok(None);
        };
        if (first == 0.0 && weights.first().is_some_and(|&w| w > 0.0))
            || (last == 1.0 && weights.last().is_some_and(|&w| w > 0.0))
        {
            return Ok(None);
        }
        json!(weights.iter().filter(|&&w| w > 0.0).count())
    } else {
        if first == 0.0 || last == 1.0 {
            return Ok(None);
        }
        evidence_count.clone()
    };
    let minimum = integer(spec.get("minimum_knots").unwrap_or(&json!(4)))?;
    Ok((!greater(&minimum, &effective_count)).then_some(effective_count))
}

fn converged_score(attempt: &Value, effective_count: &Value) -> Result<f64> {
    let spec = &attempt["specification"];
    let details = &attempt["diagnostic"];
    let partition = match spec["kind"].as_str() {
        Some("censored_quantiles") => Some(("censored_count", "uncensored_count")),
        Some("hurdle_quantiles") => Some(("atom_count", "tail_count")),
        _ => None,
    };
    if let Some((a, b)) = partition {
        let first = natural(&details[a])?;
        let second = natural(&details[b])?;
        if natural_sum(&first, &second) != natural(effective_count)?
            || (spec["kind"] == "censored_quantiles" && (first == "0" || second == "0"))
            || (spec["kind"] == "hurdle_quantiles" && greater(&json!(4), &integer(&details[b])?))
        {
            return Err(invalid(
                "fit diagnostic counts contradict effective evidence",
            ));
        }
    }
    details[if spec["kind"] == "samples" {
        "ks_statistic"
    } else {
        "normalized_rmse"
    }]
    .as_f64()
    .ok_or_else(|| invalid("a converged attempt requires its selection score"))
}

fn fit_interval(value: &Value) -> Result<Option<(f64, f64)>> {
    if value.is_null() {
        return Ok(None);
    }
    value[0]
        .as_f64()
        .zip(value[1].as_f64())
        .filter(|(first, last)| first.is_finite() && last.is_finite() && first <= last)
        .map(Some)
        .ok_or_else(|| invalid("invalid fitting interval"))
}

fn validate_record(record: &Value) -> Result<Option<&Value>> {
    validate_structure("https://climateriskcommons.org/schemas/fitting/1.0", record)?;
    let evidence_count = integer(&record["evidence_count"])?;
    let used_count = integer(&record["used_count"])?;
    let count = natural(&evidence_count)?;
    let used = natural(&used_count)?;
    let excluded = natural(&record["excluded_count"])?;
    if natural_sum(&used, &excluded) != count {
        return Err(invalid(
            "used and excluded counts must sum to evidence_count",
        ));
    }
    let requested = fit_specification(&record["specification"])?;
    let sample_evidence = requested["kind"] == "samples"
        || (requested["kind"] == "model_selection"
            && requested["candidates"][0]["kind"] == "samples");
    let values = fit_interval(&record["value_interval"])?;
    let probabilities = fit_interval(&record["probability_interval"])?;
    if sample_evidence {
        if probabilities.is_some()
            || values.is_some() != (count != "0")
            || (count == "1" && values.is_some_and(|(first, last)| first != last))
        {
            return Err(invalid(
                "sample evidence requires its actual observed value interval and no probability interval",
            ));
        }
        if !record["weights"].is_null() {
            return Err(invalid("sample fits do not have quantile weights"));
        }
    } else if greater(&json!(2), &evidence_count)
        || probabilities.is_none_or(|(a, b)| a < 0.0 || a >= b || b > 1.0)
    {
        return Err(invalid(
            "quantile fits require at least two knots and their actual probability interval",
        ));
    }
    let attempts = array(&record["attempts"])?;
    text(&record["engine_version"])?;
    let mut cursor = 0;
    if requested["kind"] == "model_selection" {
        for candidate in array(&requested["candidates"])? {
            replay_attempts(candidate, attempts, &mut cursor)?;
        }
    } else {
        replay_attempts(&requested, attempts, &mut cursor)?;
    }
    if cursor != attempts.len() {
        return Err(invalid("attempt trace contains unexecuted alternatives"));
    }
    let mut winner = None;
    let mut comparison_span = None;
    let mut last_eligible_spec = None;
    for (index, attempt) in attempts.iter().enumerate() {
        let spec = &attempt["specification"];
        let effective_count = if sample_evidence {
            let minimum = integer(spec.get("minimum_samples").unwrap_or(&json!(3)))?;
            (!greater(&minimum, &evidence_count)).then_some(evidence_count.clone())
        } else {
            eligible_quantile_count(spec, &evidence_count, probabilities.expect("validated"))?
        };
        if effective_count.is_some() {
            last_eligible_spec = Some(spec);
        }
        if attempt["status"] == "converged" {
            let effective_count = effective_count.ok_or_else(|| {
                invalid("converged attempt contradicts the recorded evidence support or count")
            })?;
            let score = converged_score(attempt, &effective_count)?;
            if let Some(span) = attempt["diagnostic"]["normalization_scale"].as_f64() {
                if comparison_span
                    .replace(span)
                    .is_some_and(|previous| previous != span)
                {
                    return Err(invalid(
                        "quantile candidates must share one comparison span",
                    ));
                }
            }
            if winner.is_none_or(|(_, best)| score < best) {
                winner = Some((index, score));
            }
        }
    }
    let selected_index = if record["selected_attempt"].is_null() {
        None
    } else {
        let index = unsigned(&record["selected_attempt"])?;
        Some(usize::try_from(index).map_err(invalid)?)
    };
    if selected_index != winner.map(|(index, _)| index) {
        return Err(invalid(
            "selected attempt must match the recorded selection criterion",
        ));
    }
    let selected = selected_index.map(|index| &attempts[index]["specification"]);
    if record["converged"].as_bool() != Some(selected.is_some()) {
        return Err(invalid("fit convergence contradicts actual attempts"));
    }
    let estimator = selected
        .filter(|s| s["kind"] == "samples")
        .map(|s| s.get("estimator").cloned().unwrap_or(json!("mle")))
        .unwrap_or(Value::Null);
    if record["estimator"] != estimator {
        return Err(invalid("actual estimator contradicts selected attempt"));
    }
    if sample_evidence {
        let expected = if last_eligible_spec.is_some() {
            count.as_str()
        } else {
            "0"
        };
        if used != expected {
            return Err(invalid("sample counts contradict the executed attempts"));
        }
        return Ok(selected);
    }
    // If no fit converged, Python records the last attempt whose evidence
    // passed the preflight checks. Attempts rejected at preflight use no knots.
    let expected_spec = selected.or(last_eligible_spec);
    if let Some(specification) = expected_spec {
        if values.is_none() {
            return Err(invalid(
                "quantile fits using knots require their observed value interval",
            ));
        }
        let weights = record["weights"]
            .as_array()
            .ok_or_else(|| invalid("quantile fits using knots require effective weights"))?;
        let weights = weights
            .iter()
            .map(|weight| {
                weight
                    .as_f64()
                    .filter(|w| (0.0..=1.0).contains(w))
                    .ok_or_else(|| invalid("invalid effective weight"))
            })
            .collect::<Result<Vec<_>>>()?;
        let sum = compensated(weights.iter().copied());
        let positive = weights.iter().filter(|&&w| w > 0.0).count();
        if weights.len().to_string() != count
            || (sum - 1.0).abs() > 1e-12
            || positive.to_string() != used
        {
            return Err(invalid("effective weights contradict evidence counts"));
        }
        if requested_weights(specification, weights.len()).is_none_or(|expected| {
            weights.iter().zip(expected).any(|(&weight, expected)| {
                let tolerance = (1e-12 * weight.max(expected)).max(f64::from_bits(1));
                (weight > 0.0) != (expected > 0.0) || (weight - expected).abs() > tolerance
            })
        }) {
            return Err(invalid(
                "effective weights contradict the executed specification",
            ));
        }
    } else if used != "0" || !record["weights"].is_null() || values.is_some() {
        return Err(invalid(
            "no executed quantile attempt used the recorded evidence",
        ));
    }
    Ok(selected)
}

struct Wkb<'a> {
    data: &'a [u8],
    position: usize,
}

impl Wkb<'_> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self
            .position
            .checked_add(N)
            .ok_or_else(|| invalid("WKB overflow"))?;
        let result = self
            .data
            .get(self.position..end)
            .ok_or_else(|| invalid("truncated WKB"))?
            .try_into()
            .map_err(invalid)?;
        self.position = end;
        Ok(result)
    }
    fn u32(&mut self, little: bool) -> Result<u32> {
        let bytes = self.bytes()?;
        Ok(if little {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        })
    }
    fn coord(&mut self, little: bool) -> Result<Coord> {
        let bytes = self.bytes()?;
        let x = if little {
            f64::from_le_bytes(bytes)
        } else {
            f64::from_be_bytes(bytes)
        };
        let bytes = self.bytes()?;
        let y = if little {
            f64::from_le_bytes(bytes)
        } else {
            f64::from_be_bytes(bytes)
        };
        if !x.is_finite()
            || !y.is_finite()
            || !(-180.0..=180.0).contains(&x)
            || !(-90.0..=90.0).contains(&y)
        {
            return Err(invalid("support requires finite WGS84 coordinates"));
        }
        Ok(Coord { x, y })
    }
    fn polygon(&mut self, little: bool) -> Result<Polygon> {
        let count = self.u32(little)? as usize;
        if count == 0 || count > (self.data.len() - self.position) / 4 {
            return Err(invalid("invalid polygon ring count"));
        }
        let mut rings = Vec::new();
        for _ in 0..count {
            let count = self.u32(little)? as usize;
            if count < 4 || count > (self.data.len() - self.position) / 16 {
                return Err(invalid("invalid polygon vertex count"));
            }
            let coords = (0..count)
                .map(|_| self.coord(little))
                .collect::<Result<Vec<_>>>()?;
            if coords.first() != coords.last() {
                return Err(invalid("polygon rings must be closed"));
            }
            let origin = coords[0];
            let area = coords
                .windows(2)
                .map(|p| {
                    (p[0].x - origin.x) * (p[1].y - origin.y)
                        - (p[1].x - origin.x) * (p[0].y - origin.y)
                })
                .sum::<f64>();
            if area == 0.0 {
                return Err(invalid("polygon ring has zero area"));
            }
            rings.push(LineString::new(coords));
        }
        let exterior = rings.remove(0);
        let polygon = Polygon::new(exterior, rings);
        polygon.check_validation().map_err(invalid)?;
        connected_interior(&polygon)?;
        Ok(polygon)
    }
    fn geometry(&mut self, polygon_only: bool) -> Result<Geometry> {
        let little = match self.bytes::<1>()?[0] {
            0 => false,
            1 => true,
            _ => return Err(invalid("invalid WKB byte order")),
        };
        let kind = self.u32(little)?;
        if polygon_only && kind != 3 {
            return Err(invalid("multipolygon members must be polygons"));
        }
        match kind {
            1 => Ok(Geometry::Point(Point::from(self.coord(little)?))),
            3 => Ok(Geometry::Polygon(self.polygon(little)?)),
            6 => {
                let count = self.u32(little)? as usize;
                if count == 0 || count > (self.data.len() - self.position) / 9 {
                    return Err(invalid("invalid multipolygon member count"));
                }
                let mut polygons = Vec::new();
                for _ in 0..count {
                    let Geometry::Polygon(polygon) = self.geometry(true)? else {
                        return Err(invalid("multipolygon members must be polygons"));
                    };
                    polygons.push(polygon);
                }
                let polygons = MultiPolygon::new(polygons);
                polygons.check_validation().map_err(invalid)?;
                Ok(Geometry::MultiPolygon(polygons))
            }
            _ => Err(invalid(
                "support must be canonical2D Point/Polygon/MultiPolygon WKB",
            )),
        }
    }
}

/// geo validates OGC topology except connected polygon interiors. A cycle in
/// the bipartite graph of boundary rings and their contact points disconnects
/// the interior. Multiple rings touching at the same point form a star, not a
/// cycle; they are not rejected merely for sharing that point.
fn connected_interior(polygon: &Polygon) -> Result<()> {
    let rings = std::iter::once(polygon.exterior())
        .chain(polygon.interiors())
        .collect::<Vec<_>>();
    let mut parents = (0..rings.len()).collect::<Vec<_>>();
    let mut contacts = BTreeMap::new();
    let mut edges = BTreeSet::new();
    fn root(parents: &mut [usize], mut i: usize) -> usize {
        while parents[i] != i {
            parents[i] = parents[parents[i]];
            i = parents[i];
        }
        i
    }
    for i in 0..rings.len() {
        for j in i + 1..rings.len() {
            for first in rings[i].lines() {
                for second in rings[j].lines() {
                    if let Some(LineIntersection::SinglePoint { intersection, .. }) =
                        line_intersection(first, second)
                    {
                        let bits = |value: f64| if value == 0.0 { 0 } else { value.to_bits() };
                        let key = (bits(intersection.x), bits(intersection.y));
                        let point = *contacts.entry(key).or_insert_with(|| {
                            let point = parents.len();
                            parents.push(point);
                            point
                        });
                        for ring in [i, j] {
                            if edges.insert((ring, point)) {
                                let a = root(&mut parents, ring);
                                let b = root(&mut parents, point);
                                if a == b {
                                    return Err(invalid("polygon interior is disconnected"));
                                }
                                parents[a] = b;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn geometry(value: &Value) -> Result<Geometry> {
    let hex = value
        .as_str()
        .ok_or_else(|| invalid("geometry must be WKB hex"))?;
    let data = hex
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digit = |byte: u8| match byte {
                b'0'..=b'9' => Ok(byte - b'0'),
                b'a'..=b'f' => Ok(byte - b'a' + 10),
                _ => Err(invalid("geometry must use canonical lowercase hex")),
            };
            Ok(digit(pair[0])? * 16 + digit(pair[1])?)
        })
        .collect::<Result<Vec<_>>>()?;
    if hex.len() % 2 != 0 {
        return Err(invalid("invalid WKB hex length"));
    }
    let mut reader = Wkb {
        data: &data,
        position: 0,
    };
    let value = reader.geometry(false)?;
    if reader.position != data.len() {
        return Err(invalid("trailing WKB data"));
    }
    Ok(value)
}

fn source_support(row: &Value) -> Result<Option<Geometry>> {
    row.get("geometry").map(geometry).transpose()
}

fn validate_source(source: &Value) -> Result<()> {
    let mut uris = BTreeSet::new();
    for resource in array(&source["resources"])? {
        if !uris.insert(text(&resource["uri"])?) {
            return Err(invalid(
                "resource URIs must be unique within a source identity",
            ));
        }
    }
    Ok(())
}

fn validate_report(report: &Value) -> Result<()> {
    for source in array(&report["source_identities"])? {
        validate_source(source)?;
    }
    for diagnostic in array(&report["diagnostics"])? {
        if let Some(record) = diagnostic.get("record") {
            if diagnostic["stage"] != "fitting" {
                return Err(invalid(
                    "fitting records require a fitting-stage diagnostic",
                ));
            }
            validate_fit_record(record)?;
        }
    }
    if report["valid"] == true
        && array(&report["diagnostics"])?
            .iter()
            .any(|d| d["severity"] == "error")
    {
        return Err(invalid("a valid report cannot contain error diagnostics"));
    }
    Ok(())
}

fn sampling(value: &Value) -> Result<(f64, &str)> {
    if !matches!(
        value["calendar"].as_str(),
        Some("gregorian" | "proleptic_gregorian" | "no_leap" | "all_leap" | "360_day" | "julian")
    ) || !matches!(value["timestamp"].as_str(), Some("start" | "end"))
    {
        return Err(invalid(
            "sampling requires a known calendar and timestamp position",
        ));
    }
    let step = &value["step"];
    let duration = step["value"]
        .as_f64()
        .filter(|v| v.is_finite() && *v > 0.0)
        .ok_or_else(|| invalid("sampling step must be a positive finite duration"))?;
    let unit = text(&step["unit"])?;
    if !matches!(unit, "s" | "h" | "d" | "month" | "yr") {
        return Err(invalid("sampling step uses an unsupported time unit"));
    }
    Ok((duration, unit))
}

fn validate_reduction(value: &Value) -> Result<()> {
    let (input_value, input_unit) = sampling(&value["input_sampling"])?;
    let (output_value, output_unit) = sampling(&value["output_sampling"])?;
    if value["input_sampling"]["calendar"] != value["output_sampling"]["calendar"] {
        return Err(invalid(
            "temporal reduction cannot change the source calendar",
        ));
    }
    let operation = text(&value["operation"])?;
    let annual = matches!(operation, "annual_maximum" | "annual_minimum");
    let rolling = operation == "rolling_total";
    if !annual
        && !rolling
        && !matches!(operation, "daily_total" | "daily_maximum" | "daily_minimum")
    {
        return Err(invalid("unknown temporal reduction operation"));
    }
    if output_value != 1.0
        || output_unit != if annual { "yr" } else { "d" }
        || value["output_sampling"]["timestamp"] != "start"
    {
        return Err(invalid(
            "temporal reduction output cadence contradicts its operation",
        ));
    }
    if rolling != value.get("window").is_some() {
        return Err(invalid("only rolling totals require a window"));
    }
    if rolling {
        let window = &value["window"];
        let width = window["value"].as_f64().unwrap_or(f64::NAN);
        if window["unit"] != "d" || !width.is_finite() || width < 1.0 || width.fract() != 0.0 {
            return Err(invalid(
                "rolling totals require a positive whole-day window",
            ));
        }
        if input_value != 1.0 || input_unit != "d" {
            return Err(invalid("rolling totals require one-day input sampling"));
        }
    } else if annual && input_unit == "month" {
        if input_value.fract() != 0.0 || 12.0 % input_value != 0.0 {
            return Err(invalid(
                "annual monthly sampling must divide twelve calendar months",
            ));
        }
    } else if annual && input_unit == "yr" {
        if input_value != 1.0 {
            return Err(invalid(
                "annual sampling requires one calendar-year observations",
            ));
        }
    } else {
        let factor = match input_unit {
            "s" => 1.0,
            "h" => 3600.0,
            "d" => 86400.0,
            _ => return Err(invalid("daily sampling requires a fixed time step")),
        };
        let per_day = 86400.0 / (input_value * factor);
        if !per_day.is_finite()
            || per_day < 1.0
            || (per_day - per_day.round()).abs() > 1e-12 * per_day.abs()
        {
            return Err(invalid("sampling step must divide a native calendar day"));
        }
    }
    if !value["coverage"]
        .as_f64()
        .is_some_and(|v| v.is_finite() && v > 0.0 && v <= 1.0)
    {
        return Err(invalid("coverage fraction must be finite and in (0, 1]"));
    }
    let mut variables = BTreeSet::new();
    if let Some(changes) = value.get("measurements") {
        for change in array(changes)? {
            if !variables.insert(text(&change["variable_id"])?) {
                return Err(invalid(
                    "measurement changes must refer to distinct variables",
                ));
            }
            for name in ["input", "output"] {
                validate_structure(
                    "https://climateriskcommons.org/schemas/measurement-1.0.schema.json",
                    &change[name],
                )?;
            }
            if scientific_equal(&change["input"], &change["output"]) {
                return Err(invalid("measurement change must change scientific meaning"));
            }
        }
    }
    Ok(())
}

/// Validate the current dataset envelope, complete scientific references, and
/// exact source support without assigning scientific meaning to delivery cells.
pub fn validate_dataset(value: &Value) -> Result<()> {
    validate_structure(DATASET_URI, value)?;
    let metadata = &value["metadata"];
    let spec = &metadata["spec"];
    let variables = descriptors(&spec["variables"], false)?;
    let scenarios = descriptors(&spec["scenarios"], false)?;
    let periods = descriptors(&spec["periods"], true)?;
    for scenario in scenarios.values() {
        scenario_schema(scenario)?;
    }
    let mut series = BTreeSet::new();
    for item in array(&spec["series"])? {
        let key = series_key(item)?;
        if !variables.contains_key(&key.0)
            || key.1.as_ref().is_some_and(|id| !scenarios.contains_key(id))
            || key.2.as_ref().is_some_and(|id| !periods.contains_key(id))
        {
            return Err(invalid("dangling scientific series descriptor reference"));
        }
        if !series.insert(key) {
            return Err(invalid("duplicate scientific series identity"));
        }
    }
    validate_report(&metadata["report"])?;
    if let Some(source) = metadata.get("source") {
        validate_source(source)?;
    }
    let mut scenario_ids: BTreeSet<_> = scenarios.keys().cloned().collect();
    let mut measurements = variables
        .iter()
        .map(|(id, v)| {
            (
                id.clone(),
                v.get("measurement").cloned().unwrap_or(Value::Null),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let provenance = array(&metadata["provenance"])?;
    for transform in provenance.iter().rev() {
        if transform["kind"] == "unit_conversion" {
            let id = text(&transform["variable_id"])?;
            let measurement = measurements
                .get_mut(id)
                .ok_or_else(|| invalid("conversion provenance refers to an unknown variable"))?;
            *measurement = reverse_unit_conversion(
                measurement,
                text(&transform["source_unit"])?,
                text(&transform["target_unit"])?,
                transform["scale"]
                    .as_f64()
                    .ok_or_else(|| invalid("invalid unit conversion scale"))?,
                transform["offset"]
                    .as_f64()
                    .ok_or_else(|| invalid("invalid unit conversion offset"))?,
            )?;
        } else if transform["kind"] == "temporal_reduction" {
            validate_reduction(transform)?;
            if let Some(changes) = transform.get("measurements") {
                for change in array(changes)? {
                    let id = text(&change["variable_id"])?;
                    let measurement = measurements.get_mut(id).ok_or_else(|| {
                        invalid("temporal reduction refers to an unknown variable")
                    })?;
                    if !scientific_equal(measurement, &change["output"]) {
                        return Err(invalid(
                            "temporal reduction output contradicts current variable measurement",
                        ));
                    }
                    *measurement = change["input"].clone();
                }
            }
        } else if transform["kind"] == "observation_time_selection" {
            let period = &transform["time"];
            if period["kind"] != "window" || period_identity(period)? != text(&period["id"])? {
                return Err(invalid(
                    "observation time selection has an invalid window identity",
                ));
            }
        } else if transform["kind"] == "rename" {
            let mapping = transform["mapping"]
                .as_object()
                .ok_or_else(|| invalid("rename mapping must be an object"))?;
            let inverse = mapping
                .iter()
                .map(|(old, new)| Ok((text(new)?, identifier(old)?.to_owned())))
                .collect::<Result<BTreeMap<_, _>>>()?;
            if inverse.len() != mapping.len() {
                return Err(invalid("rename mappings must be injective"));
            }
            if transform["dimension"] == "variable" {
                let before = measurements
                    .iter()
                    .map(|(id, unit)| {
                        (inverse.get(id.as_str()).unwrap_or(id).clone(), unit.clone())
                    })
                    .collect::<BTreeMap<_, _>>();
                if before.len() != measurements.len() {
                    return Err(invalid(
                        "rename provenance collides with participating identities",
                    ));
                }
                measurements = before;
            } else {
                let before = scenario_ids
                    .iter()
                    .map(|id| inverse.get(id.as_str()).unwrap_or(id).clone())
                    .collect::<BTreeSet<_>>();
                if before.len() != scenario_ids.len() {
                    return Err(invalid(
                        "rename provenance collides with participating scenarios",
                    ));
                }
                scenario_ids = before;
            }
        }
    }
    let spatial = &metadata["spatial"];
    let supported = array(&spatial["supports"])?
        .iter()
        .filter_map(Value::as_str)
        .collect::<BTreeSet<_>>();
    if supported.is_empty() || supported.len() != array(&spatial["supports"])?.len() {
        return Err(invalid("spatial supports must be nonempty and unique"));
    }
    let resolution = spatial.get("index_resolution").map(unsigned).transpose()?;
    let mut rows = BTreeSet::new();
    let mut supports: BTreeMap<String, (Option<Geometry>, Option<u64>)> = BTreeMap::new();
    let mut failed = 0;
    for row in array(&value["rows"])? {
        let key = series_key(row)?;
        if !series.contains(&key) {
            return Err(invalid("row refers to an undeclared scientific series"));
        }
        let support_id = text(&row["support_id"])?;
        let cell_id = unsigned(&row["cell_id"])?;
        let cell = h3o::CellIndex::try_from(cell_id).map_err(invalid)?;
        if resolution.is_some_and(|r| u64::from(u8::from(cell.resolution())) != r) {
            return Err(invalid("delivery index contradicts spatial resolution"));
        }
        if !rows.insert((key, support_id.to_string(), cell_id)) {
            return Err(invalid(
                "duplicate dataset row key, including missing dimensions",
            ));
        }
        let shape = source_support(row)?;
        let kind = match &shape {
            None => "cell",
            Some(Geometry::Point(_)) => "point",
            _ => "polygon",
        };
        if !supported.contains(kind) {
            return Err(invalid(
                "source geometry contradicts spatial support contract",
            ));
        }
        let support_cell = shape.is_none().then_some(cell_id);
        if let Some((previous, previous_cell)) = supports.get(support_id) {
            let equal = match (previous, &shape) {
                (Some(a), Some(b)) => a.relate(b).is_equal_topo(),
                (None, None) => true,
                _ => false,
            };
            if !equal || *previous_cell != support_cell {
                return Err(invalid(
                    "one support_id identifies different source supports",
                ));
            }
        }
        supports.insert(support_id.into(), (shape, support_cell));
        if let Some(curve) = row.get("curve") {
            Numerical::from_value(curve.clone())?;
            if let Some(record) = row.get("fit") {
                validate_fitted_curve(record, curve, text(&row["variable_id"])?, provenance)?;
            }
        }
        if let Some(failure) = row.get("failure") {
            failed += 1;
            if validate_record(&failure["record"])?.is_some() {
                return Err(invalid("retained failure cannot contain a converged fit"));
            }
        }
    }
    if metadata["report"]["complete"] == true
        && (natural(&metadata["report"]["counts"]["produced"])? != rows.len().to_string()
            || natural(&metadata["report"]["counts"]["no_data"])? != failed.to_string())
    {
        return Err(invalid("complete report counts contradict dataset rows"));
    }
    Ok(())
}

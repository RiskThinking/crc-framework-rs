use crc_framework_core::{CanonicalCurve, CurveParameters};
use wasm_bindgen::prelude::*;

fn type_error(error: impl ToString) -> JsValue {
    js_sys::TypeError::new(&error.to_string()).into()
}

fn check_probability(q: f64) -> Result<(), JsValue> {
    if !q.is_finite() || !(0.0..=1.0).contains(&q) {
        Err(js_sys::RangeError::new("probability must be finite and in [0, 1]").into())
    } else {
        Ok(())
    }
}

/// An explicitly owned, validated curve; call free() when finished.
#[wasm_bindgen]
pub struct PreparedCurve {
    inner: CanonicalCurve,
}

#[wasm_bindgen]
impl PreparedCurve {
    pub fn evaluate(&self, q: f64) -> Result<JsValue, JsValue> {
        check_probability(q)?;
        Ok(self
            .inner
            .quantile(q)
            .map_err(type_error)?
            .map(JsValue::from_f64)
            .unwrap_or(JsValue::NULL))
    }

    /// Multiple queries without reparsing the same record.
    pub fn quantiles(&self, probabilities: &[f64]) -> Result<js_sys::Array, JsValue> {
        probabilities.iter().map(|&q| self.evaluate(q)).collect()
    }
}

/// A tile-sized collection reconstructed once and evaluated in one WASM call.
#[wasm_bindgen]
pub struct PreparedCurves {
    inner: Vec<CanonicalCurve>,
}

#[wasm_bindgen]
impl PreparedCurves {
    // wasm-bindgen copies one contiguous f64 buffer, avoiding a JS callback and
    // JS Array insertion per curve. NaN is an internal missing-data sentinel;
    // the public JS adapter converts it back to null. Core outputs are finite.
    pub fn evaluate(&self, q: f64) -> Result<Vec<f64>, JsValue> {
        check_probability(q)?;
        self.inner
            .iter()
            .map(|curve| {
                curve
                    .quantile(q)
                    .map(|value| value.unwrap_or(f64::NAN))
                    .map_err(type_error)
            })
            .collect()
    }
}

// Private binding wire format: twelve f64 columns per record. Names/codes are
// exported from Rust once during initialization, not duplicated in JavaScript.
const KIND_NAMES: &[&str] = &["fitted", "hurdle", "point_mass", "tabulated", "no_data"];
const TYPE_NAMES: &[&str] = &[
    "genextreme",
    "weibull_min",
    "weibull_max",
    "skewnorm",
    "gumbel_r",
    "gumbel_l",
    "genpareto",
    "point_mass",
    "linear_probability",
    "below_effective_resolution",
    "insufficient_informative_support",
    "degenerate_effective_range",
    "scientific_exclusion",
];
const ROW_WIDTH: usize = 12;

#[wasm_bindgen(js_name = curveKindNames)]
pub fn curve_kind_names() -> js_sys::Array {
    KIND_NAMES
        .iter()
        .map(|name| JsValue::from_str(name))
        .collect()
}
#[wasm_bindgen(js_name = curveTypeNames)]
pub fn curve_type_names() -> js_sys::Array {
    TYPE_NAMES
        .iter()
        .map(|name| JsValue::from_str(name))
        .collect()
}

fn wire_index(value: f64) -> Result<usize, JsValue> {
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= u32::MAX as f64 {
        Ok(value as usize)
    } else {
        Err(type_error("invalid packed curve index"))
    }
}

fn wire_knots(offset: f64, length: f64, knots: &[f64]) -> Result<Option<Vec<f64>>, JsValue> {
    if length == -1.0 {
        return Ok(None);
    }
    let offset = wire_index(offset)?;
    let length = wire_index(length)?;
    let end = offset
        .checked_add(length)
        .ok_or_else(|| type_error("invalid packed knot range"))?;
    Ok(Some(
        knots
            .get(offset..end)
            .ok_or_else(|| type_error("invalid packed knot range"))?
            .to_vec(),
    ))
}

fn unpack_curve(row: &[f64], knots: &[f64]) -> Result<CanonicalCurve, JsValue> {
    if row.len() != ROW_WIDTH {
        return Err(type_error("invalid packed curve width"));
    }
    let kind = KIND_NAMES
        .get(wire_index(row[0])?)
        .ok_or_else(|| type_error("unsupported curve_kind"))?;
    let family = TYPE_NAMES
        .get(wire_index(row[1])?)
        .ok_or_else(|| type_error("unsupported curve_type"))?;
    let mask = wire_index(row[2])?;
    if mask > 31 {
        return Err(type_error("invalid packed curve presence mask"));
    }
    let optional = |index: usize| {
        if mask & (1usize << index) != 0 {
            Some(row[3 + index])
        } else {
            None
        }
    };
    CurveParameters {
        curve_kind: (*kind).into(),
        curve_type: (*family).into(),
        curve_shape: optional(0),
        curve_location: optional(1),
        curve_scale: optional(2),
        curve_atom_probability: optional(3),
        curve_atom_location: optional(4),
        curve_probabilities: wire_knots(row[8], row[9], knots)?,
        curve_values: wire_knots(row[10], row[11], knots)?,
    }
    .reconstruct()
    .map_err(type_error)
}

#[wasm_bindgen]
impl PreparedCurves {
    #[wasm_bindgen(js_name = fromPacked)]
    pub fn from_packed(rows: &[f64], knots: &[f64]) -> Result<PreparedCurves, JsValue> {
        if rows.len() % ROW_WIDTH != 0 {
            return Err(type_error("invalid packed curve width"));
        }
        Ok(Self {
            inner: rows
                .chunks_exact(ROW_WIDTH)
                .map(|row| unpack_curve(row, knots))
                .collect::<Result<_, _>>()?,
        })
    }
}

#[wasm_bindgen]
impl PreparedCurve {
    #[wasm_bindgen(js_name = fromPacked)]
    pub fn from_packed(row: &[f64], knots: &[f64]) -> Result<PreparedCurve, JsValue> {
        Ok(Self {
            inner: unpack_curve(row, knots)?,
        })
    }
}

#[wasm_bindgen(js_name = evaluateCurvePacked)]
pub fn evaluate_curve_packed(row: &[f64], knots: &[f64], q: f64) -> Result<JsValue, JsValue> {
    check_probability(q)?;
    let curve = unpack_curve(row, knots)?;
    Ok(curve
        .quantile(q)
        .map_err(type_error)?
        .map(JsValue::from_f64)
        .unwrap_or(JsValue::NULL))
}

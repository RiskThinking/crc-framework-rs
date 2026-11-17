//! Owned current-format numerical documents and shared scientific validation.
use crc_framework_core::{CrcError, Numerical};
use wasm_bindgen::prelude::*;

fn error(error: CrcError) -> JsValue {
    match error {
        CrcError::OutOfSupport { .. } => js_sys::RangeError::new(&error.to_string()).into(),
        _ => js_sys::TypeError::new(&error.to_string()).into(),
    }
}
fn probability(q: f64) -> Result<(), JsValue> {
    if !q.is_finite() || !(0.0..=1.0).contains(&q) {
        Err(js_sys::RangeError::new("probability must be finite and in [0,1]").into())
    } else {
        Ok(())
    }
}

#[wasm_bindgen]
pub struct PreparedNumerical {
    inner: Numerical,
}

#[wasm_bindgen]
impl PreparedNumerical {
    #[wasm_bindgen(constructor)]
    pub fn new(text: &str) -> Result<PreparedNumerical, JsValue> {
        Ok(Self {
            inner: Numerical::from_json(text).map_err(error)?,
        })
    }
    pub fn quantile(&self, q: f64) -> Result<f64, JsValue> {
        probability(q)?;
        self.inner.quantile(q).map_err(error)
    }
    #[wasm_bindgen(js_name=quantileBatch)]
    pub fn quantile_batch(&self, probabilities: &[f64]) -> Result<Vec<f64>, JsValue> {
        probabilities.iter().map(|q| self.quantile(*q)).collect()
    }
}

#[wasm_bindgen]
pub struct PreparedNumericals {
    inner: Vec<Numerical>,
}

#[wasm_bindgen]
impl PreparedNumericals {
    #[wasm_bindgen(constructor)]
    pub fn new(text: &str) -> Result<PreparedNumericals, JsValue> {
        let data = crc_framework_core::canonical::parse_document(text).map_err(error)?;
        let array = data.as_array().ok_or_else(|| {
            js_sys::TypeError::new("expected an array of current numerical documents")
        })?;
        let inner = array
            .iter()
            .map(|v| Numerical::from_value(v.clone()).map_err(error))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { inner })
    }
    pub fn quantile(&self, q: f64) -> Result<Vec<f64>, JsValue> {
        probability(q)?;
        self.inner
            .iter()
            .map(|n| n.quantile(q).map_err(error))
            .collect()
    }
}

#[wasm_bindgen(js_name=validateDocument)]
pub fn validate_document(uri: &str, text: &str) -> Result<(), JsValue> {
    crc_framework_core::canonical::validate_document(uri, text).map_err(error)
}

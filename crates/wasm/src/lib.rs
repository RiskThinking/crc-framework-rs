//! Owned current-format numerical documents and shared scientific validation.
use crc_framework_core::{CrcError, Numerical};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

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

/// Per-document outcome of a batched query; mirrored by `QuantileOutcome` in JS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Outcome {
    Value = 0,
    BelowSupport = 1,
    AboveSupport = 2,
    InvalidDocument = 3,
    Unavailable = 4,
}

/// A value, or NaN with the reason there is none. Framework quantiles are never
/// NaN, so NaN only marks a missing value.
fn outcome(result: Result<f64, CrcError>) -> (f64, Outcome) {
    match result {
        Ok(value) => (value, Outcome::Value),
        Err(CrcError::OutOfSupport {
            probability,
            minimum,
            ..
        }) if probability < minimum => (f64::NAN, Outcome::BelowSupport),
        Err(CrcError::OutOfSupport { .. }) => (f64::NAN, Outcome::AboveSupport),
        Err(_) => (f64::NAN, Outcome::Unavailable),
    }
}

#[wasm_bindgen]
pub struct PreparedNumericals {
    inner: Vec<Result<Numerical, String>>,
}

#[wasm_bindgen]
impl PreparedNumericals {
    /// Parse a JSON array of numerical documents. An invalid document is kept as
    /// a per-item failure; text that is not a JSON array is rejected.
    #[wasm_bindgen(constructor)]
    pub fn new(text: &str) -> Result<PreparedNumericals, JsValue> {
        let data = crc_framework_core::canonical::parse_document(text).map_err(error)?;
        let array = data.as_array().ok_or_else(|| {
            js_sys::TypeError::new("expected an array of current numerical documents")
        })?;
        let inner = array
            .iter()
            .map(|v| Numerical::from_value(v.clone()).map_err(|e| e.to_string()))
            .collect();
        Ok(Self { inner })
    }
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }
    /// Indices of documents that failed validation, in order.
    #[wasm_bindgen(js_name=invalidIndices)]
    pub fn invalid_indices(&self) -> Vec<u32> {
        self.inner
            .iter()
            .enumerate()
            .filter(|(_, item)| item.is_err())
            .map(|(index, _)| index as u32)
            .collect()
    }
    /// Validation messages for `invalidIndices`, in the same order.
    #[wasm_bindgen(js_name=invalidMessages)]
    pub fn invalid_messages(&self) -> Vec<String> {
        self.inner
            .iter()
            .filter_map(|item| item.as_ref().err().cloned())
            .collect()
    }
    /// One value and outcome per document. Only an invalid probability throws.
    pub fn quantile(&self, q: f64) -> Result<QuantileBatch, JsValue> {
        probability(q)?;
        let mut values = Vec::with_capacity(self.inner.len());
        let mut outcomes = Vec::with_capacity(self.inner.len());
        for item in &self.inner {
            let (value, outcome) = match item {
                Ok(numerical) => outcome(numerical.quantile(q)),
                Err(_) => (f64::NAN, Outcome::InvalidDocument),
            };
            values.push(value);
            outcomes.push(outcome as u8);
        }
        Ok(QuantileBatch { values, outcomes })
    }
}

#[wasm_bindgen]
pub struct QuantileBatch {
    values: Vec<f64>,
    outcomes: Vec<u8>,
}

#[wasm_bindgen]
impl QuantileBatch {
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f64> {
        self.values.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn outcomes(&self) -> Vec<u8> {
        self.outcomes.clone()
    }
}

#[wasm_bindgen(js_name=validateDocument)]
pub fn validate_document(uri: &str, text: &str) -> Result<(), JsValue> {
    crc_framework_core::canonical::validate_document(uri, text).map_err(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcomes_distinguish_values_and_missing_support() {
        assert_eq!(outcome(Ok(2.5)), (2.5, Outcome::Value));
        let support = |probability| CrcError::OutOfSupport {
            probability,
            minimum: 0.01,
            maximum: 0.99,
        };
        let (value, below) = outcome(Err(support(0.001)));
        assert!(value.is_nan());
        assert_eq!(below, Outcome::BelowSupport);
        assert_eq!(outcome(Err(support(0.998))).1, Outcome::AboveSupport);
        let unavailable = outcome(Err(CrcError::Unsupported("no inverse".into())));
        assert!(unavailable.0.is_nan());
        assert_eq!(unavailable.1, Outcome::Unavailable);
    }

    #[test]
    fn batches_keep_invalid_documents_as_items() {
        let valid = r#"{"family":"gumbel_r","kind":"parametric","location":1,"object":"distribution","scale":1,"schema_version":"1.0"}"#;
        let invalid = valid.replace(r#""scale":1"#, r#""scale":-1"#);
        let partial = r#"{"object":"quantile_curve","probabilities":[0.01,0.99],"schema_version":"1.0","values":[0,5]}"#;
        let batch = PreparedNumericals::new(&format!("[{valid},{invalid},{partial}]")).unwrap();
        assert_eq!(batch.length(), 3);
        assert_eq!(batch.invalid_indices(), vec![1]);
        assert_eq!(batch.invalid_messages().len(), 1);
        let result = batch.quantile(0.998).unwrap();
        assert_eq!(result.outcomes, vec![0, 3, 2]);
        assert!(result.values[0].is_finite());
        assert!(result.values[1].is_nan() && result.values[2].is_nan());
        let result = batch.quantile(0.5).unwrap();
        assert_eq!(result.outcomes, vec![0, 3, 0]);
        assert_eq!(result.values[2], 2.5);
    }
}

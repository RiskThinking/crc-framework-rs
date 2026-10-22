mod definition;
mod error;
mod json;
mod kernel;
mod reference;
mod registry;

pub use registry::{RegistryImpactReference, ResolutionContext, manifest_json, manifest_sha256};

pub use error::ImpactContractError;

use definition::{ModelKind, compile, known_registry_factor, override_rule};
use json::{canonical_json, content_hash, parse_json};

/// A validated impact definition and its compiled point kernel.
#[derive(Clone, Debug)]
pub struct ImpactDefinition {
    canonical: String,
    hash: String,
    kind: ModelKind,
    model: definition::Model,
    clamp: definition::Clamp,
}

impl ImpactDefinition {
    /// Parse and validate a definition, materializing its canonical form.
    pub fn from_json(text: &str) -> Result<Self, ImpactContractError> {
        let compiled = compile(parse_json(text)?)?;
        let canonical = canonical_json(&compiled.normalized);
        let hash = content_hash(&compiled.normalized);
        Ok(Self {
            canonical,
            hash,
            kind: compiled.kind,
            model: compiled.model,
            clamp: compiled.clamp,
        })
    }

    pub fn canonical_json(&self) -> &str {
        &self.canonical
    }

    pub fn definition_hash(&self) -> &str {
        &self.hash
    }

    pub fn model_kind(&self) -> &'static str {
        self.kind.as_str()
    }

    pub fn evaluate(&self, exposure: f64) -> Result<f64, ImpactContractError> {
        kernel::evaluate(&self.model, &self.clamp, exposure)
    }

    pub fn monotonicity(&self) -> &'static str {
        kernel::monotonicity(&self.model, &self.clamp)
    }
}

pub use definition::ModelKind as ImpactModelKind;

pub fn registry_factors() -> &'static [&'static str] {
    &[
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
    ]
}

pub fn registry_override_nullable(factor: &str, key: &str) -> Option<bool> {
    if known_registry_factor(factor) {
        override_rule(factor, key)
    } else {
        None
    }
}

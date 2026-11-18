mod definition;
mod error;
mod expectation;
pub(crate) mod json;
mod kernel;
mod kernel_curve;
#[doc(hidden)]
pub use kernel_curve::{KernelCurve, KernelCurveParameters};
mod pushforward;
pub(crate) mod quadrature;
mod reference;
mod registry;
mod source;

pub use registry::{
    RegistryImpactReference, ResolutionContext, manifest_json, manifest_sha256, registry_identity,
};

pub use error::ImpactContractError;

use crate::numerical::ProbabilitySegment;
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

    pub fn expected_value(
        &self,
        source: &ImpactSource,
        method: MethodRequest,
    ) -> Result<Expectation, ImpactContractError> {
        expectation::expected_value(&self.model, &self.clamp, source, method)
    }

    pub fn transform_distribution(
        &self,
        source: &ImpactSource,
    ) -> Result<ImpactDistribution, ImpactContractError> {
        pushforward::transform(&self.model, &self.clamp, &self.hash, source)
    }

    /// Exact image of finite uniform intervals and atoms through a piecewise
    /// affine impact. Boundaries include all response knots and clamp crossings.
    pub(crate) fn image_segments(
        &self,
        source: &[ProbabilitySegment],
    ) -> Result<Option<Vec<ProbabilitySegment>>, ImpactContractError> {
        if !matches!(
            self.model,
            definition::Model::Linear { .. }
                | definition::Model::Tabulated {
                    interpolation: definition::Interpolation::Linear,
                    ..
                }
        ) {
            return Ok(None);
        }
        let cuts = expectation::response_breakpoints(&self.model, &self.clamp);
        let mut segments = Vec::new();
        for &(lower, upper, mass) in source {
            if lower == upper {
                let value = self.evaluate(lower)?;
                segments.push((value, value, mass));
                continue;
            }
            let mut left = lower;
            for right in cuts
                .iter()
                .copied()
                .filter(|cut| lower < *cut && *cut < upper)
                .chain(std::iter::once(upper))
            {
                let portion = mass * ((right - left) / (upper - lower));
                if portion > 0.0 {
                    let (a, b) = (self.evaluate(left)?, self.evaluate(right)?);
                    segments.push((a.min(b), a.max(b), portion));
                }
                left = right;
            }
        }
        segments.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        Ok(Some(segments))
    }
}

pub use expectation::{
    Expectation, GaussKronrodOptions, MethodRequest, RequestedMethod, ResolvedMethod,
};
pub use pushforward::{ImpactAtom, ImpactDistribution};
pub use source::{DiscreteDistribution, HistogramExposure, ImpactSource};

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

//! Transport-independent consumption of canonical CRC curve records.
use serde::Deserialize;

use crate::{
    CrcError, Distribution, DistributionFamily, FittedDistribution, HurdleDistribution,
    Interpolation, PointMassDistribution, Result, TabulatedDistribution,
};

/// Canonical curve fields. Missing optional fields are equivalent to null.
/// Unknown feature properties are ignored. Transport string decoding belongs to bindings.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CurveParameters {
    pub curve_kind: String,
    pub curve_type: String,
    pub curve_shape: Option<f64>,
    pub curve_location: Option<f64>,
    pub curve_scale: Option<f64>,
    pub curve_atom_probability: Option<f64>,
    pub curve_atom_location: Option<f64>,
    pub curve_probabilities: Option<Vec<f64>>,
    pub curve_values: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub enum CanonicalCurve {
    Fitted(FittedDistribution),
    Hurdle(HurdleDistribution),
    PointMass(PointMassDistribution),
    Tabulated(TabulatedDistribution),
    NoData,
}

fn invalid(message: &str) -> CrcError {
    CrcError::InvalidInput(message.into())
}

fn required(value: Option<f64>, field: &str) -> Result<f64> {
    value.ok_or_else(|| CrcError::MissingParameter(field.into()))
}

impl CurveParameters {
    /// Validate the canonical schema and reconstruct once for repeated queries.
    /// Gumbel shape is ignored, as in CRC-SDK's canonical normalization.
    pub fn reconstruct(&self) -> Result<CanonicalCurve> {
        let shape = if matches!(self.curve_type.as_str(), "gumbel_r" | "gumbel_l") {
            None
        } else {
            self.curve_shape
        };
        let scalars = [
            shape,
            self.curve_location,
            self.curve_scale,
            self.curve_atom_probability,
            self.curve_atom_location,
        ];
        let has_knots = self.curve_probabilities.is_some() || self.curve_values.is_some();
        if self.curve_kind == "no_data" {
            if !matches!(
                self.curve_type.as_str(),
                "below_effective_resolution"
                    | "insufficient_informative_support"
                    | "degenerate_effective_range"
                    | "scientific_exclusion"
            ) {
                return Err(invalid(
                    "no-data curves require a no-data reason as curve_type",
                ));
            }
            if scalars.iter().any(Option::is_some) || has_knots {
                return Err(invalid("no-data curves must not define curve parameters"));
            }
            return Ok(CanonicalCurve::NoData);
        }
        if self.curve_kind == "tabulated" {
            if self.curve_type != "linear_probability" || scalars.iter().any(Option::is_some) {
                return Err(invalid(
                    "tabulated curves require linear_probability and no scalar parameters",
                ));
            }
            let probabilities = self
                .curve_probabilities
                .clone()
                .ok_or_else(|| CrcError::MissingParameter("curve_probabilities".into()))?;
            let values = self
                .curve_values
                .clone()
                .ok_or_else(|| CrcError::MissingParameter("curve_values".into()))?;
            return Ok(CanonicalCurve::Tabulated(TabulatedDistribution::new(
                probabilities,
                values,
                Interpolation::LinearProbability,
                false,
            )?));
        }
        if has_knots {
            return Err(invalid(
                "non-tabulated curves must not define tabulated knots",
            ));
        }
        if self.curve_kind == "point_mass" {
            let location = required(self.curve_location, "curve_location")?;
            if self.curve_type != "point_mass"
                || shape.is_some()
                || self.curve_scale != Some(0.0)
                || self.curve_atom_probability != Some(1.0)
                || self.curve_atom_location != Some(location)
            {
                return Err(invalid(
                    "point-mass curves require no shape, zero scale and probability one at curve_location",
                ));
            }
            return Ok(CanonicalCurve::PointMass(PointMassDistribution::new(
                location,
            )?));
        }
        if !matches!(self.curve_kind.as_str(), "fitted" | "hurdle") {
            return Err(invalid("unsupported curve_kind"));
        }
        let family = DistributionFamily::from_name(&self.curve_type)
            .ok_or_else(|| invalid("fitted and hurdle curves require a supported fitted family"))?;
        let base = FittedDistribution::from_parameters(
            family,
            shape,
            required(self.curve_location, "curve_location")?,
            required(self.curve_scale, "curve_scale")?,
        )?;
        if self.curve_kind == "fitted" {
            if self.curve_atom_probability.is_some() || self.curve_atom_location.is_some() {
                return Err(invalid("fitted curves must not define hurdle atom fields"));
            }
            Ok(CanonicalCurve::Fitted(base))
        } else {
            Ok(CanonicalCurve::Hurdle(HurdleDistribution::new(
                required(self.curve_atom_location, "curve_atom_location")?,
                required(self.curve_atom_probability, "curve_atom_probability")?,
                base,
            )?))
        }
    }
}

impl CanonicalCurve {
    /// None means no data or a valid query outside the curve's quantile support.
    /// Invalid probabilities and malformed records are errors, never missing data.
    pub fn quantile(&self, q: f64) -> Result<Option<f64>> {
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return Err(invalid("probability must be finite and in [0, 1]"));
        }
        let value = match self {
            Self::NoData => return Ok(None),
            Self::Fitted(_) if q == 0.0 || q == 1.0 => return Ok(None),
            Self::Hurdle(_) if q == 1.0 => return Ok(None),
            Self::Fitted(curve) => curve.ppf(q),
            Self::Hurdle(curve) => curve.ppf(q),
            Self::PointMass(curve) => curve.ppf(q),
            Self::Tabulated(curve) => curve.ppf(q),
        };
        match value {
            Ok(value) if value.is_finite() => Ok(Some(value)),
            Ok(_) => Err(invalid("quantile evaluation produced a non-finite result")),
            Err(CrcError::OutOfSupport { .. }) => Ok(None),
            Err(error) => Err(error),
        }
    }
}

use std::collections::HashMap;

use crate::{
    distribution::{Distribution, Interpolation, TabulatedDistribution},
    error::{CrcError, Result},
    reference::climate::{
        ag_params, energy_params, flood_params, precipitation_thresholds, spei_params, wind_params,
    },
};

pub trait Transform: Send + Sync {
    fn transform_value(&self, value: f64) -> f64;

    fn is_decreasing(&self) -> bool {
        false
    }

    fn transform_values(&self, values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .map(|&value| self.transform_value(value))
            .collect()
    }

    fn apply(
        &self,
        distribution: &dyn Distribution,
        probabilities: &[f64],
    ) -> Result<TabulatedDistribution> {
        let values = distribution.quantiles(probabilities)?;
        let mut transformed = self.transform_values(&values);
        if self.is_decreasing() {
            transformed.reverse();
        }
        TabulatedDistribution::new(
            probabilities.to_vec(),
            transformed,
            Interpolation::LinearProbability,
            false,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SigmoidImpact {
    pub midpoint: f64,
    pub steepness: f64,
    pub minimum: f64,
    pub maximum: f64,
    pub zero_below: Option<f64>,
    pub adjust_at_zero: bool,
}

impl Transform for SigmoidImpact {
    fn transform_value(&self, value: f64) -> f64 {
        if self.zero_below.is_some_and(|threshold| value < threshold) {
            return 0.0;
        }
        let mut unit = 1.0 / (1.0 + (-self.steepness * (value - self.midpoint)).exp());
        if self.adjust_at_zero {
            unit -= 1.0 / (1.0 + (self.steepness * self.midpoint).exp());
        }
        self.minimum + unit * (self.maximum - self.minimum)
    }

    fn is_decreasing(&self) -> bool {
        self.steepness < 0.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LinearImpact {
    pub slope: f64,
    pub intercept: f64,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
}

impl Transform for LinearImpact {
    fn transform_value(&self, value: f64) -> f64 {
        let mut result = self.slope.mul_add(value, self.intercept);
        if let Some(minimum) = self.minimum {
            result = result.max(minimum);
        }
        if let Some(maximum) = self.maximum {
            result = result.min(maximum);
        }
        result
    }

    fn is_decreasing(&self) -> bool {
        self.slope < 0.0
    }
}

#[derive(Debug, Clone)]
pub struct PiecewiseLinearImpact {
    pub exposure: Vec<f64>,
    pub impact: Vec<f64>,
}

impl PiecewiseLinearImpact {
    pub fn new(exposure: Vec<f64>, impact: Vec<f64>) -> Result<Self> {
        if exposure.len() != impact.len() || exposure.len() < 2 {
            return Err(CrcError::InvalidInput(
                "piecewise exposure and impact arrays must have equal length of at least two"
                    .into(),
            ));
        }
        if exposure.windows(2).any(|pair| pair[0] >= pair[1])
            || impact.windows(2).any(|pair| pair[0] > pair[1])
            || exposure
                .iter()
                .chain(&impact)
                .any(|value| !value.is_finite())
        {
            return Err(CrcError::InvalidInput(
                "piecewise exposure must increase and impact must not decrease".into(),
            ));
        }
        Ok(Self { exposure, impact })
    }
}

impl Transform for PiecewiseLinearImpact {
    fn transform_value(&self, value: f64) -> f64 {
        if value <= self.exposure[0] {
            return self.impact[0];
        }
        if value >= *self.exposure.last().expect("validated") {
            return *self.impact.last().expect("validated");
        }
        let upper = self
            .exposure
            .partition_point(|candidate| *candidate < value);
        let lower = upper - 1;
        let weight = (value - self.exposure[lower]) / (self.exposure[upper] - self.exposure[lower]);
        self.impact[lower] + weight * (self.impact[upper] - self.impact[lower])
    }
}

#[derive(Debug, Clone)]
pub struct SpeiImpact {
    water_consumption_asset: f64,
    original_water_unit_cost: f64,
}

impl Transform for SpeiImpact {
    fn transform_value(&self, value: f64) -> f64 {
        let midpoint = -1.0 + (199.0_f64).ln() / -3.0;
        let drought_scalar = 1.0 / (1.0 + (3.0 * (value - midpoint)).exp());
        let water_unit_cost = if value < 0.0 {
            self.original_water_unit_cost * (1.0 + drought_scalar * 10.0)
        } else {
            self.original_water_unit_cost
        };
        water_unit_cost * self.water_consumption_asset / 10_000_000.0
    }

    fn is_decreasing(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
pub struct CoolingImpact {
    cooling_cost: f64,
    historic_mean: f64,
}

impl Transform for CoolingImpact {
    fn transform_value(&self, value: f64) -> f64 {
        if value <= 1.0 {
            0.0
        } else {
            self.cooling_cost * value / (self.historic_mean * 10_000_000.0)
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CropValues {
    maize: f64,
    rice: f64,
    wheat: f64,
}

#[derive(Debug, Clone)]
pub struct CropImpact {
    weights: CropValues,
    country: CropValues,
    yearly: CropValues,
    index: CropValues,
    historic_mean: f64,
}

impl Transform for CropImpact {
    fn transform_value(&self, value: f64) -> f64 {
        let impact = |weight: f64, country: f64, yearly: f64, index: f64| {
            let historic = (country + yearly * 2010.0 + index * self.historic_mean).exp();
            let future = (country + yearly * 2010.0 + index * value).exp();
            if historic > 0.0 {
                weight * -(future - historic) / historic
            } else {
                0.0
            }
        };
        impact(
            self.weights.maize,
            self.country.maize,
            self.yearly.maize,
            self.index.maize,
        ) + impact(
            self.weights.rice,
            self.country.rice,
            self.yearly.rice,
            self.index.rice,
        ) + impact(
            self.weights.wheat,
            self.country.wheat,
            self.yearly.wheat,
            self.index.wheat,
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct ImpactContext {
    pub factor: String,
    pub cell: Option<u64>,
    pub country: Option<String>,
    pub continent: Option<String>,
    pub building_type: Option<String>,
    pub historic_mean: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum BuiltinImpact {
    Sigmoid(SigmoidImpact),
    Linear(LinearImpact),
    Piecewise(PiecewiseLinearImpact),
    Spei(SpeiImpact),
    Cooling(CoolingImpact),
    Crop(CropImpact),
}

impl Transform for BuiltinImpact {
    fn transform_value(&self, value: f64) -> f64 {
        match self {
            Self::Sigmoid(transform) => transform.transform_value(value),
            Self::Linear(transform) => transform.transform_value(value),
            Self::Piecewise(transform) => transform.transform_value(value),
            Self::Spei(transform) => transform.transform_value(value),
            Self::Cooling(transform) => transform.transform_value(value),
            Self::Crop(transform) => transform.transform_value(value),
        }
    }

    fn is_decreasing(&self) -> bool {
        match self {
            Self::Sigmoid(transform) => transform.is_decreasing(),
            Self::Linear(transform) => transform.is_decreasing(),
            Self::Piecewise(transform) => transform.is_decreasing(),
            Self::Spei(transform) => transform.is_decreasing(),
            Self::Cooling(transform) => transform.is_decreasing(),
            Self::Crop(transform) => transform.is_decreasing(),
        }
    }
}

pub struct ImpactRegistry;

impl ImpactRegistry {
    pub fn resolve(
        context: &ImpactContext,
        overrides: &HashMap<String, f64>,
    ) -> Result<BuiltinImpact> {
        let number = |name: &str, default: f64| overrides.get(name).copied().unwrap_or(default);
        let factor = context.factor.as_str();
        match factor {
            "daily_freezethaw_cycles" | "frost_days" | "rx1day" => {
                let cell = context.cell.ok_or_else(|| {
                    CrcError::MissingParameter(format!(
                        "cell is required for the built-in {factor} impact"
                    ))
                })?;
                let cell = h3o::CellIndex::try_from(cell)
                    .map_err(|_| CrcError::InvalidInput(format!("invalid H3 cell {cell:x}")))?;
                let parent = cell.parent(h3o::Resolution::Four).ok_or_else(|| {
                    CrcError::InvalidInput("cell cannot be normalized to H3 resolution 4".into())
                })?;
                let parent = u64::from(parent);
                let lower = match factor {
                    "daily_freezethaw_cycles" => {
                        precipitation_thresholds::dlyfrzthw_threshold(parent)
                    }
                    "frost_days" => precipitation_thresholds::frost_days_threshold(parent),
                    _ => precipitation_thresholds::rx1day_threshold(parent),
                }
                .ok_or_else(|| {
                    CrcError::MissingParameter(format!(
                        "no geographic damage threshold exists for {factor} at {parent:x}"
                    ))
                })?;
                let upper: f32 = if factor == "rx1day" { 1000.0 } else { 365.0 };
                let lower_damage = if factor == "rx1day" { 0.5 } else { 5.0 };
                let midpoint = (f64::from(upper) + f64::from(lower)) / 2.0;
                let steepness =
                    (1.0_f64 / (lower_damage / 100.0) - 1.0).ln() / (midpoint - f64::from(lower));
                Ok(BuiltinImpact::Sigmoid(SigmoidImpact {
                    midpoint: number("midpoint", midpoint),
                    steepness: number("steepness", steepness),
                    minimum: number("minimum_damage", 0.0),
                    maximum: number(
                        "maximum_damage",
                        if factor == "rx1day" { 0.3737 } else { 1.0 },
                    ),
                    zero_below: overrides.get("zero_below").copied(),
                    adjust_at_zero: true,
                }))
            }
            "cyclone" | "wind_max_daily_max" => {
                let country = context.country.as_deref().ok_or_else(|| {
                    CrcError::MissingParameter(format!(
                        "country is required for the built-in {factor} impact"
                    ))
                })?;
                let params = if factor == "cyclone" {
                    wind_params::CYCLONE_PARAMS.get(country)
                } else {
                    wind_params::WIND_MAX_DAILY_MAX_PARAMS.get(country)
                };
                let (midpoint, steepness, threshold) = params
                    .map(|params| {
                        let _upper_damage_threshold = params.t2;
                        (
                            f64::from(params.x0),
                            f64::from(params.k),
                            f64::from(params.t1),
                        )
                    })
                    .unwrap_or((74.7, 0.11457369750485913, 28.5));
                Ok(BuiltinImpact::Sigmoid(SigmoidImpact {
                    midpoint: number("midpoint", midpoint),
                    steepness: number("steepness", steepness),
                    minimum: number("minimum_damage", 0.0),
                    maximum: number("maximum_damage", 1.0),
                    zero_below: Some(number("zero_below", threshold)),
                    adjust_at_zero: true,
                }))
            }
            "fwi" => Ok(BuiltinImpact::Sigmoid(SigmoidImpact {
                midpoint: number("midpoint", 35.65),
                steepness: number("steepness", 0.2051873853077659),
                minimum: number("minimum_damage", 0.0),
                maximum: number("maximum_damage", 1.0),
                zero_below: overrides.get("zero_below").copied(),
                adjust_at_zero: true,
            })),
            "hot_days" => Ok(BuiltinImpact::Linear(LinearImpact {
                slope: number("slope", 0.0813 / 365.0),
                intercept: number("intercept", 0.0),
                minimum: Some(number("minimum_damage", 0.0)),
                maximum: overrides.get("maximum_damage").copied(),
            })),
            "cflood" | "rflood" | "inundation" => {
                let maximum = number("maximum_damage", 1.0);
                let continent = context.continent.as_deref().ok_or_else(|| {
                    CrcError::MissingParameter(
                        "continent is required for a built-in flood impact".into(),
                    )
                })?;
                let building_type = context.building_type.as_deref().ok_or_else(|| {
                    CrcError::MissingParameter(
                        "building_type is required for a built-in flood impact".into(),
                    )
                })?;
                let params = flood_params::continent_to_flood_params(continent);
                let curve = flood_params::building_type_to_damage(building_type, params);
                Ok(BuiltinImpact::Piecewise(PiecewiseLinearImpact::new(
                    flood_params::FLOOD_DAMAGE_THRESHOLDS
                        .into_iter()
                        .map(f64::from)
                        .collect(),
                    curve
                        .into_iter()
                        .map(|value| f64::from(value) * maximum)
                        .collect(),
                )?))
            }
            "spei" => {
                let country = context.country.as_deref().ok_or_else(|| {
                    CrcError::MissingParameter(
                        "country is required for the built-in spei impact".into(),
                    )
                })?;
                let building_type = context.building_type.as_deref().ok_or_else(|| {
                    CrcError::MissingParameter(
                        "building_type is required for the built-in spei impact".into(),
                    )
                })?;
                let ratio = f64::from(spei_params::get_national_energy_consumption(country))
                    / f64::from(spei_params::get_national_water_consumption(country));
                if !ratio.is_finite() || ratio <= 0.0 {
                    return Err(CrcError::MissingParameter(format!(
                        "energy/water ratio is unavailable for country {country}"
                    )));
                }
                Ok(BuiltinImpact::Spei(SpeiImpact {
                    water_consumption_asset: f64::from(
                        spei_params::get_asset_type_energy_consumption(building_type),
                    ) / ratio,
                    original_water_unit_cost: number("water_unit_cost", 2.9809),
                }))
            }
            "cooling_degree_days" | "maximum_consecutive_dry_days" => {
                let baseline = context.historic_mean.ok_or_else(|| {
                    CrcError::MissingParameter(
                        "historic_mean is required for baseline-relative impacts".into(),
                    )
                })?;
                let country = context.country.as_deref().ok_or_else(|| {
                    CrcError::MissingParameter(format!(
                        "country is required for the built-in {factor} impact"
                    ))
                })?;
                let building_type = context.building_type.as_deref().ok_or_else(|| {
                    CrcError::MissingParameter(format!(
                        "building_type is required for the built-in {factor} impact"
                    ))
                })?;
                if building_type == "Agricultural buildings" {
                    let source = if factor == "cooling_degree_days" {
                        ag_params::get_cdd_crop_values(country)
                    } else {
                        ag_params::get_cxdd_crop_values(country)
                    };
                    let yearly = if factor == "cooling_degree_days" {
                        &ag_params::CDD_YEARLY_CROP_ESTIMATES
                    } else {
                        &ag_params::CXDD_YEARLY_CROP_ESTIMATES
                    };
                    let index = if factor == "cooling_degree_days" {
                        &ag_params::CDD_INDEX_CROP_ESTIMATES
                    } else {
                        &ag_params::CXDD_INDEX_CROP_ESTIMATES
                    };
                    let weights = ag_params::get_national_crop_weights(country);
                    let convert = |values: &ag_params::NationalCropWeights| CropValues {
                        maize: f64::from(values.maize),
                        rice: f64::from(values.rice),
                        wheat: f64::from(values.wheat),
                    };
                    Ok(BuiltinImpact::Crop(CropImpact {
                        weights: convert(weights),
                        country: convert(source),
                        yearly: convert(yearly),
                        index: convert(index),
                        historic_mean: baseline,
                    }))
                } else if factor == "cooling_degree_days" {
                    let energy_consumption =
                        f64::from(*energy_params::get_building_type_energy_consumption(
                            building_type,
                            country,
                        ));
                    let energy_price = f64::from(*energy_params::get_building_type_energy_price(
                        building_type,
                        country,
                    ));
                    let sector_energy = f64::from(*energy_params::get_building_type_energy(
                        building_type,
                        country,
                    ));
                    if sector_energy <= 0.0 || baseline <= 0.0 {
                        return Err(CrcError::MissingParameter(format!(
                            "cooling energy parameters are unavailable for {building_type} in {country}"
                        )));
                    }
                    let national_cost = energy_consumption * 1.0e9 * energy_price;
                    let asset_consumption = f64::from(
                        spei_params::get_asset_type_energy_consumption(building_type),
                    );
                    Ok(BuiltinImpact::Cooling(CoolingImpact {
                        cooling_cost: number(
                            "cooling_cost",
                            asset_consumption * national_cost / (sector_energy * 1.0e9),
                        ),
                        historic_mean: baseline,
                    }))
                } else {
                    Err(CrcError::Unsupported(
                        "maximum_consecutive_dry_days impact is available only for Agricultural buildings"
                            .into(),
                    ))
                }
            }
            _ => Err(CrcError::Unsupported(format!(
                "no built-in impact transform for factor {factor}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_override_changes_damage_cap() {
        let context = ImpactContext {
            factor: "fwi".into(),
            ..Default::default()
        };
        let transform =
            ImpactRegistry::resolve(&context, &HashMap::from([("maximum_damage".into(), 0.4)]))
                .unwrap();
        assert!(transform.transform_value(100.0) <= 0.4);
    }
}

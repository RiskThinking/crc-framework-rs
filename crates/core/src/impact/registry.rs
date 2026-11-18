use std::collections::BTreeMap;
use std::sync::OnceLock;

use h3o::{CellIndex, Resolution};
use serde_json::{Map, Value, json};

use super::ImpactDefinition;
use super::definition::override_rule;
use super::error::{ContractResult, fail};
use super::json::{Json, canonical_json, content_hash};
use super::reference::{CompiledReference, compile_reference};
use crate::reference::climate::{
    ag_params, energy_params, flood_params, precipitation_thresholds, spei_params, wind_params,
};

const REGISTRY_ID: &str = "crc-climate-impacts";
const REGISTRY_REVISION: &str = "2026.10.1";
const RESOLVER_VERSION: &str = "crc-impact-registry/1";
const FWI_MIDPOINT: f64 = 35.65;
const FWI_STEEPNESS: f64 = 0.2051873853077659;
const HOT_DAYS_SLOPE: f64 = 0.0813 / 365.0;
const SPEI_WATER_UNIT_COST: f64 = 2.9809;
const AGRICULTURAL: &str = "Agricultural buildings";

struct FactorMeta {
    factor: &'static str,
    context: &'static [&'static str],
    calibration: [&'static str; 4],
    branches: &'static [&'static str],
}

const FACTORS: &[FactorMeta] = &[
    FactorMeta {
        factor: "cflood",
        context: &["continent", "building_type"],
        calibration: ["flood_depth", "m", "damage_ratio", "1"],
        branches: &[],
    },
    FactorMeta {
        factor: "cooling_degree_days",
        context: &["country", "building_type", "historic_mean"],
        calibration: [
            "cooling_degree_days",
            "K.d",
            "normalized_cooling_cost",
            "crc-legacy-normalized-cost",
        ],
        branches: &["agricultural", "non_agricultural"],
    },
    FactorMeta {
        factor: "cyclone",
        context: &["country"],
        calibration: ["wind_speed", "m/s", "damage_ratio", "1"],
        branches: &[],
    },
    FactorMeta {
        factor: "daily_freezethaw_cycles",
        context: &["cell"],
        calibration: ["daily_freezethaw_cycles", "d", "damage_ratio", "1"],
        branches: &[],
    },
    FactorMeta {
        factor: "frost_days",
        context: &["cell"],
        calibration: ["frost_days", "d", "damage_ratio", "1"],
        branches: &[],
    },
    FactorMeta {
        factor: "fwi",
        context: &[],
        calibration: ["fwi", "1", "damage_ratio", "1"],
        branches: &[],
    },
    FactorMeta {
        factor: "hot_days",
        context: &[],
        calibration: ["hot_days", "d", "damage_ratio", "1"],
        branches: &[],
    },
    FactorMeta {
        factor: "inundation",
        context: &["continent", "building_type"],
        calibration: ["flood_depth", "m", "damage_ratio", "1"],
        branches: &[],
    },
    FactorMeta {
        factor: "maximum_consecutive_dry_days",
        context: &["country", "building_type", "historic_mean"],
        calibration: [
            "maximum_consecutive_dry_days",
            "d",
            "crop_yield_change",
            "1",
        ],
        branches: &["agricultural"],
    },
    FactorMeta {
        factor: "rflood",
        context: &["continent", "building_type"],
        calibration: ["flood_depth", "m", "damage_ratio", "1"],
        branches: &[],
    },
    FactorMeta {
        factor: "rx1day",
        context: &["cell"],
        calibration: ["rx1day", "mm", "damage_ratio", "1"],
        branches: &[],
    },
    FactorMeta {
        factor: "spei",
        context: &["country", "building_type"],
        calibration: [
            "spei",
            "1",
            "normalized_water_cost",
            "crc-legacy-normalized-cost",
        ],
        branches: &[],
    },
    FactorMeta {
        factor: "wind_max_daily_max",
        context: &["country"],
        calibration: ["wind_speed", "m/s", "damage_ratio", "1"],
        branches: &[],
    },
];

struct Installed {
    manifest_sha256: String,
    manifest_json: String,
    tables: BTreeMap<String, String>,
}

struct Trace {
    table_id: &'static str,
    key: Value,
}

/// Explicit caller context for resolving a pinned table reference.
#[derive(Clone, Debug, Default)]
pub struct ResolutionContext {
    pub cell: Option<String>,
    pub country: Option<String>,
    pub continent: Option<String>,
    pub building_type: Option<String>,
    pub historic_mean: Option<f64>,
}

/// A validated registry reference that resolves to a frozen impact definition.
#[derive(Clone, Debug)]
pub struct RegistryImpactReference {
    canonical: String,
    hash: String,
    compiled: CompiledReference,
}

impl RegistryImpactReference {
    pub fn from_json(text: &str) -> Result<Self, super::ImpactContractError> {
        let compiled = compile_reference(text)?;
        Ok(Self {
            canonical: compiled.canonical.clone(),
            hash: compiled.hash.clone(),
            compiled,
        })
    }

    pub fn canonical_json(&self) -> &str {
        &self.canonical
    }

    pub fn reference_hash(&self) -> &str {
        &self.hash
    }

    pub fn resolve(
        &self,
        context: &ResolutionContext,
    ) -> Result<ImpactDefinition, super::ImpactContractError> {
        resolve(&self.compiled, context)
    }
}

/// Hex digest of the installed registry manifest, without the `sha256:` prefix.
pub fn manifest_sha256() -> &'static str {
    installed().manifest_sha256.as_str()
}

/// Canonical JSON of the installed registry manifest.
pub fn manifest_json() -> &'static str {
    installed().manifest_json.as_str()
}

/// Installed registry id, revision, resolver semantics, and manifest digest.
pub fn registry_identity() -> (&'static str, &'static str, &'static str, &'static str) {
    (
        REGISTRY_ID,
        REGISTRY_REVISION,
        "crc-impact-registry/1",
        manifest_sha256(),
    )
}

/// Exact country keys for a built-in calibration and its selected building type.
#[doc(hidden)]
pub fn registry_country_choices(factor: &str, building: Option<&str>) -> Vec<&'static str> {
    let mut countries: Vec<_> = match (factor, building) {
        ("cyclone" | "wind_max_daily_max", None) => wind_table(factor).1.keys().copied().collect(),
        ("spei", None) => spei_params::NATIONAL_WATER_CONSUMPTION
            .keys()
            .copied()
            .filter(|country| spei_ratio(country).is_ok())
            .collect(),
        ("maximum_consecutive_dry_days" | "cooling_degree_days", Some(AGRICULTURAL)) => {
            let cooling = factor == "cooling_degree_days";
            ag_params::NATIONAL_CROP_WEIGHTS
                .keys()
                .copied()
                .filter(|country| crop_country_parameters(country, cooling).is_some())
                .collect()
        }
        ("cooling_degree_days", Some(building)) => {
            let Ok(tables) = energy_tables(building) else {
                return Vec::new();
            };
            energy_map(tables.0)
                .keys()
                .copied()
                .filter(|country| cooling_cost(country, building, tables).is_ok_and(f64::is_finite))
                .collect()
        }
        _ => Vec::new(),
    };
    countries.sort_unstable();
    countries
}

fn resolve(
    reference: &CompiledReference,
    context: &ResolutionContext,
) -> ContractResult<ImpactDefinition> {
    if reference.registry_id != REGISTRY_ID {
        return Err(fail(
            "registry_unavailable",
            "/registry_id",
            "registry is not installed",
        ));
    }
    if reference.revision != REGISTRY_REVISION {
        return Err(fail(
            "registry_unavailable",
            "/registry_revision",
            "registry revision is not installed",
        ));
    }
    if reference.manifest_sha256 != manifest_sha256() {
        return Err(fail(
            "registry_manifest_mismatch",
            "/registry_manifest_sha256",
            "installed manifest digest differs from the pinned digest",
        ));
    }
    validate_context(context)?;
    let meta = factor_meta(&reference.factor);
    require_context(meta, context)?;
    let agricultural = context.building_type.as_deref() == Some(AGRICULTURAL);
    if agricultural
        && reference
            .overrides
            .iter()
            .any(|(key, _)| key == "cooling_cost")
    {
        return Err(fail(
            "inapplicable_override",
            "/overrides/cooling_cost",
            "cooling_cost is not accepted for the agricultural branch",
        ));
    }
    if reference.factor == "maximum_consecutive_dry_days" && !agricultural {
        return Err(fail(
            "unsupported_value",
            "/context/building_type",
            "maximum_consecutive_dry_days is available only for agricultural buildings",
        ));
    }
    let (model, clamp, lookups) = match reference.factor.as_str() {
        "daily_freezethaw_cycles" | "frost_days" | "rx1day" => precipitation(reference, context)?,
        "cyclone" | "wind_max_daily_max" => wind(reference, context)?,
        "fwi" => (fwi_model(reference)?, None, Vec::new()),
        "hot_days" => hot_days(reference)?,
        "cflood" | "rflood" | "inundation" => flood(reference, context)?,
        "spei" => spei(reference, context)?,
        "cooling_degree_days" => cooling(reference, context, agricultural)?,
        "maximum_consecutive_dry_days" => crop(context, false)?,
        _ => {
            return Err(fail(
                "unsupported_value",
                "/factor",
                "unknown registry factor",
            ));
        }
    };
    freeze(
        reference,
        &used_context(meta, context),
        meta,
        model,
        clamp,
        lookups,
    )
}

fn used_context(meta: &FactorMeta, context: &ResolutionContext) -> ResolutionContext {
    let uses = |field: &str| meta.context.contains(&field);
    ResolutionContext {
        cell: uses("cell").then(|| context.cell.clone()).flatten(),
        country: uses("country").then(|| context.country.clone()).flatten(),
        continent: uses("continent")
            .then(|| context.continent.clone())
            .flatten(),
        building_type: uses("building_type")
            .then(|| context.building_type.clone())
            .flatten(),
        historic_mean: uses("historic_mean")
            .then_some(context.historic_mean)
            .flatten(),
    }
}

fn precipitation(
    reference: &CompiledReference,
    context: &ResolutionContext,
) -> ContractResult<(Value, Option<Value>, Vec<Trace>)> {
    let cell = context.cell.as_deref().expect("cell required");
    let index = cell.parse::<CellIndex>().map_err(|_| {
        fail(
            "invalid_value",
            "/context/cell",
            "must be a canonical H3 cell",
        )
    })?;
    if index.resolution() < Resolution::Four {
        return Err(fail(
            "invalid_value",
            "/context/cell",
            "cell is coarser than H3 resolution 4",
        ));
    }
    let parent = index.parent(Resolution::Four).ok_or_else(|| {
        fail(
            "invalid_value",
            "/context/cell",
            "cell cannot be normalized to H3 resolution 4",
        )
    })?;
    let parent_id = u64::from(parent);
    let table_id = match reference.factor.as_str() {
        "daily_freezethaw_cycles" => "precipitation.daily_freezethaw_cycles",
        "frost_days" => "precipitation.frost_days",
        _ => "precipitation.rx1day",
    };
    let lower = match reference.factor.as_str() {
        "daily_freezethaw_cycles" => precipitation_thresholds::dlyfrzthw_threshold(parent_id),
        "frost_days" => precipitation_thresholds::frost_days_threshold(parent_id),
        _ => precipitation_thresholds::rx1day_threshold(parent_id),
    };
    let Some(lower) = lower else {
        return Err(fail(
            "missing_table_key",
            "/context/cell",
            "no geographic damage threshold exists for this cell",
        ));
    };
    let parent_hex = cell_hex(parent);
    let (midpoint, steepness, maximum) = precipitation_coefficients(&reference.factor, lower);
    if !midpoint.is_finite() || !steepness.is_finite() {
        return Err(fail(
            "invalid_value",
            "/context/cell",
            "precipitation coefficients are not finite",
        ));
    }
    let model = sigmoid_model(reference, midpoint, steepness, 0.0, maximum, None)?;
    let key = json!({"cell": parent_hex, "h3_resolution": "4"});
    let lookup = Trace { table_id, key };
    Ok((model, None, vec![lookup]))
}

fn precipitation_coefficients(factor: &str, lower: f32) -> (f64, f64, f64) {
    let upper: f32 = if factor == "rx1day" { 1000.0 } else { 365.0 };
    let lower_damage = if factor == "rx1day" { 0.5 } else { 5.0 };
    let maximum = if factor == "rx1day" { 0.3737 } else { 1.0 };
    let midpoint = (f64::from(upper) + f64::from(lower)) / 2.0;
    let steepness = (1.0_f64 / (lower_damage / 100.0) - 1.0).ln() / (midpoint - f64::from(lower));
    (midpoint, steepness, maximum)
}

fn wind(
    reference: &CompiledReference,
    context: &ResolutionContext,
) -> ContractResult<(Value, Option<Value>, Vec<Trace>)> {
    let country = context.country.as_deref().expect("country required");
    let (table_id, params) = wind_table(&reference.factor);
    let params = params.get(country).ok_or_else(|| {
        fail(
            "missing_table_key",
            "/context/country",
            "no wind parameters exist for this country",
        )
    })?;
    let model = sigmoid_model(
        reference,
        f64::from(params.x0),
        f64::from(params.k),
        0.0,
        1.0,
        Some(f64::from(params.t1)),
    )?;
    Ok((model, None, vec![trace(table_id, json!(country))]))
}

fn wind_table(
    factor: &str,
) -> (
    &'static str,
    &'static phf::Map<&'static str, wind_params::WindParams>,
) {
    if factor == "cyclone" {
        ("wind.cyclone", &wind_params::CYCLONE_PARAMS)
    } else {
        (
            "wind.wind_max_daily_max",
            &wind_params::WIND_MAX_DAILY_MAX_PARAMS,
        )
    }
}

fn fwi_model(reference: &CompiledReference) -> ContractResult<Value> {
    sigmoid_model(reference, FWI_MIDPOINT, FWI_STEEPNESS, 0.0, 1.0, None)
}

fn hot_days(reference: &CompiledReference) -> ContractResult<(Value, Option<Value>, Vec<Trace>)> {
    let slope = number_override(reference, "slope", HOT_DAYS_SLOPE);
    let intercept = number_override(reference, "intercept", 0.0);
    let minimum = optional_override(reference, "minimum_damage", Some(0.0));
    let maximum = optional_override(reference, "maximum_damage", None);
    if minimum.is_some_and(|minimum| maximum.is_some_and(|maximum| minimum > maximum)) {
        return Err(fail(
            "invalid_value",
            "/overrides/maximum_damage",
            "minimum damage exceeds maximum damage",
        ));
    }
    let model = json!({"kind": "linear", "slope": slope, "intercept": intercept});
    let clamp = json!({"minimum": minimum, "maximum": maximum});
    Ok((model, Some(clamp), Vec::new()))
}

fn flood(
    reference: &CompiledReference,
    context: &ResolutionContext,
) -> ContractResult<(Value, Option<Value>, Vec<Trace>)> {
    let continent = context.continent.as_deref().expect("continent required");
    let building = context
        .building_type
        .as_deref()
        .expect("building type required");
    let params = flood_params::FLOOD_PARAMS.get(continent).ok_or_else(|| {
        fail(
            "missing_table_key",
            "/context/continent",
            "no flood damage curve exists for this region",
        )
    })?;
    let curve = flood_params::building_type_to_damage(building, params).ok_or_else(|| {
        fail(
            "missing_table_key",
            "/context/building_type",
            "no flood damage curve exists for this building type",
        )
    })?;
    let scale = number_override(reference, "maximum_damage", 1.0);
    let mut points = Vec::new();
    for (threshold, damage) in flood_params::FLOOD_DAMAGE_THRESHOLDS.into_iter().zip(curve) {
        let response = finite(
            f64::from(damage) * scale,
            "/overrides/maximum_damage",
            "scaled flood damage is not finite",
        )?;
        points.push(json!({
            "x": f64::from(threshold),
            "y": response,
        }));
    }
    Ok((
        json!({
            "kind": "tabulated",
            "interpolation": "linear",
            "outside_domain": "clamp",
            "points": points,
        }),
        None,
        vec![trace(
            "flood.damage",
            json!({"building_type": building, "continent": continent}),
        )],
    ))
}

fn spei(
    reference: &CompiledReference,
    context: &ResolutionContext,
) -> ContractResult<(Value, Option<Value>, Vec<Trace>)> {
    let country = context.country.as_deref().expect("country required");
    let building = context
        .building_type
        .as_deref()
        .expect("building type required");
    let ratio = spei_ratio(country)?;
    let asset = asset_energy(building)?;
    let consumption = finite(
        asset / ratio,
        "/context/country",
        "water consumption is not finite",
    )?;
    let cost = number_override(reference, "water_unit_cost", SPEI_WATER_UNIT_COST);
    let lookups = vec![
        trace("spei.national_energy", json!(country)),
        trace("spei.national_water", json!(country)),
        trace("spei.asset_energy", json!(building)),
    ];
    Ok((
        json!({
            "kind": "spei",
            "water_consumption_asset": consumption,
            "original_water_unit_cost": cost,
        }),
        None,
        lookups,
    ))
}

fn spei_ratio(country: &str) -> ContractResult<f64> {
    let energy = parameter(
        &spei_params::NATIONAL_ENERGY_CONSUMPTION,
        country,
        "/context/country",
        "no national energy consumption exists for this country",
    )?;
    let water = parameter(
        &spei_params::NATIONAL_WATER_CONSUMPTION,
        country,
        "/context/country",
        "no national water consumption exists for this country",
    )?;
    let ratio = energy / water;
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(fail(
            "invalid_value",
            "/context/country",
            "energy/water ratio is unavailable for this country",
        ));
    }
    Ok(ratio)
}

fn cooling(
    reference: &CompiledReference,
    context: &ResolutionContext,
    agricultural: bool,
) -> ContractResult<(Value, Option<Value>, Vec<Trace>)> {
    if agricultural {
        let (model, lookups) = crop_model(context, true)?;
        return Ok((model, None, lookups));
    }
    let country = context.country.as_deref().expect("country required");
    let building = context
        .building_type
        .as_deref()
        .expect("building type required");
    let historic = context.historic_mean.expect("historic mean required");
    if historic <= 0.0 {
        return Err(fail(
            "invalid_value",
            "/context/historic_mean",
            "historic mean must be positive",
        ));
    }
    let (consumption_id, price_id, sector_id) = energy_tables(building)?;
    let computed = cooling_cost(country, building, (consumption_id, price_id, sector_id))?;
    let cooling_cost = finite(
        number_override(reference, "cooling_cost", computed),
        "/overrides/cooling_cost",
        "cooling cost is not finite",
    )?;
    let lookups = vec![
        trace(consumption_id, json!(country)),
        trace(price_id, json!(country)),
        trace(sector_id, json!(country)),
        trace("spei.asset_energy", json!(building)),
    ];
    Ok((
        json!({"kind": "cooling", "cooling_cost": cooling_cost, "historic_mean": historic}),
        None,
        lookups,
    ))
}

fn cooling_cost(
    country: &str,
    building: &str,
    (consumption_id, price_id, sector_id): (&str, &str, &str),
) -> ContractResult<f64> {
    let consumption = energy_lookup(consumption_id, country)?;
    let price = energy_lookup(price_id, country)?;
    let sector = energy_lookup(sector_id, country)?;
    let asset = asset_energy(building)?;
    if sector <= 0.0 {
        return Err(fail(
            "invalid_value",
            "/context/country",
            "cooling energy parameters are unavailable",
        ));
    }
    let national_cost = consumption * 1.0e9 * price;
    Ok(asset * national_cost / (sector * 1.0e9))
}

fn crop(
    context: &ResolutionContext,
    cooling_branch: bool,
) -> ContractResult<(Value, Option<Value>, Vec<Trace>)> {
    let (model, lookups) = crop_model(context, cooling_branch)?;
    Ok((model, None, lookups))
}

fn crop_model(
    context: &ResolutionContext,
    cooling_branch: bool,
) -> ContractResult<(Value, Vec<Trace>)> {
    let country = context.country.as_deref().expect("country required");
    let historic = context.historic_mean.expect("historic mean required");
    let values_id = if cooling_branch {
        "crop.cdd"
    } else {
        "crop.cxdd"
    };
    let yearly_id = if cooling_branch {
        "crop.cdd_yearly"
    } else {
        "crop.cxdd_yearly"
    };
    let index_id = if cooling_branch {
        "crop.cdd_index"
    } else {
        "crop.cxdd_index"
    };
    let (values, weights) = crop_country_parameters(country, cooling_branch).ok_or_else(|| {
        fail(
            "missing_table_key",
            "/context/country",
            "no crop parameters exist for this country",
        )
    })?;
    let yearly = if cooling_branch {
        &ag_params::CDD_YEARLY_CROP_ESTIMATES
    } else {
        &ag_params::CXDD_YEARLY_CROP_ESTIMATES
    };
    let index = if cooling_branch {
        &ag_params::CDD_INDEX_CROP_ESTIMATES
    } else {
        &ag_params::CXDD_INDEX_CROP_ESTIMATES
    };
    let lookups = vec![
        trace(values_id, json!(country)),
        trace("crop.weights", json!(country)),
        trace(yearly_id, json!("global")),
        trace(index_id, json!("global")),
    ];
    Ok((
        json!({
            "kind": "crop",
            "weights": crop_value(weights),
            "country": crop_value(values),
            "yearly": crop_value(yearly),
            "index": crop_value(index),
            "historic_mean": historic,
        }),
        lookups,
    ))
}

fn crop_country_parameters(
    country: &str,
    cooling_branch: bool,
) -> Option<(
    &'static ag_params::NationalCropWeights,
    &'static ag_params::NationalCropWeights,
)> {
    let parameters = if cooling_branch {
        &ag_params::CDD_CROP_VALUES
    } else {
        &ag_params::CXDD_CROP_VALUES
    };
    Some((
        parameters.get(country)?,
        ag_params::NATIONAL_CROP_WEIGHTS.get(country)?,
    ))
}

fn energy_tables(building: &str) -> ContractResult<(&'static str, &'static str, &'static str)> {
    Ok(match building {
        "Residential buildings" => (
            "energy.residential_cooling",
            "energy.residential_price",
            "energy.residential_commercial_sector",
        ),
        "Commercial buildings" => (
            "energy.commercial_cooling",
            "energy.residential_price",
            "energy.residential_commercial_sector",
        ),
        "Industrial buildings" => (
            "energy.industrial_cooling",
            "energy.industrial_price",
            "energy.industrial_sector",
        ),
        _ => {
            return Err(fail(
                "missing_table_key",
                "/context/building_type",
                "no cooling energy parameters exist for this building type",
            ));
        }
    })
}

fn parameter(
    table: &phf::Map<&'static str, f32>,
    key: &str,
    path: &str,
    message: &str,
) -> ContractResult<f64> {
    table
        .get(key)
        .copied()
        .map(f64::from)
        .ok_or_else(|| fail("missing_table_key", path, message))
}

fn asset_energy(building: &str) -> ContractResult<f64> {
    parameter(
        &spei_params::ASSET_ENERGY_CONSUMPTION,
        building,
        "/context/building_type",
        "no asset energy consumption exists for this building type",
    )
}

fn energy_lookup(table_id: &str, country: &str) -> ContractResult<f64> {
    parameter(
        energy_map(table_id),
        country,
        "/context/country",
        "no cooling energy parameter exists for this country",
    )
}

fn trace(table_id: &'static str, key: Value) -> Trace {
    Trace { table_id, key }
}

fn sigmoid_model(
    reference: &CompiledReference,
    midpoint: f64,
    steepness: f64,
    minimum: f64,
    maximum: f64,
    zero_below: Option<f64>,
) -> ContractResult<Value> {
    let midpoint = finite(
        number_override(reference, "midpoint", midpoint),
        "/overrides/midpoint",
        "midpoint is not finite",
    )?;
    let steepness = finite(
        number_override(reference, "steepness", steepness),
        "/overrides/steepness",
        "steepness is not finite",
    )?;
    let minimum = finite(
        number_override(reference, "minimum_damage", minimum),
        "/overrides/minimum_damage",
        "minimum damage is not finite",
    )?;
    let maximum = finite(
        number_override(reference, "maximum_damage", maximum),
        "/overrides/maximum_damage",
        "maximum damage is not finite",
    )?;
    if maximum < minimum {
        return Err(fail(
            "invalid_value",
            "/overrides/maximum_damage",
            "maximum damage is below minimum damage",
        ));
    }
    let amplitude = finite(
        maximum - minimum,
        "/overrides/maximum_damage",
        "damage levels overflow",
    )?;
    let zero_below = match override_of(reference, "zero_below") {
        Override::Absent => zero_below,
        Override::Null => None,
        Override::Value(value) => Some(value),
    };
    Ok(json!({
        "kind": "sigmoid",
        "midpoint": midpoint,
        "steepness": steepness,
        "offset": minimum,
        "amplitude": amplitude,
        "origin_adjustment": "subtract_at_zero",
        "zero_below": zero_below,
    }))
}

enum Override {
    Absent,
    Null,
    Value(f64),
}

fn override_of(reference: &CompiledReference, key: &str) -> Override {
    match reference
        .overrides
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| *value)
    {
        None => Override::Absent,
        Some(None) => Override::Null,
        Some(Some(value)) => Override::Value(value),
    }
}

fn finite(value: f64, path: &str, message: &str) -> ContractResult<f64> {
    if value.is_finite() {
        Ok(if value == 0.0 { 0.0 } else { value })
    } else {
        Err(fail("invalid_value", path, message))
    }
}

fn number_override(reference: &CompiledReference, key: &str, default: f64) -> f64 {
    match override_of(reference, key) {
        Override::Value(value) => value,
        _ => default,
    }
}

fn optional_override(
    reference: &CompiledReference,
    key: &str,
    default_when_absent: Option<f64>,
) -> Option<f64> {
    match override_of(reference, key) {
        Override::Absent => default_when_absent,
        Override::Null => None,
        Override::Value(value) => Some(value),
    }
}

fn freeze(
    reference: &CompiledReference,
    context: &ResolutionContext,
    meta: &FactorMeta,
    model: Value,
    clamp: Option<Value>,
    lookups: Vec<Trace>,
) -> ContractResult<ImpactDefinition> {
    let mut calibration = meta.calibration;
    if reference.factor == "cooling_degree_days"
        && context.building_type.as_deref() == Some(AGRICULTURAL)
    {
        calibration[2] = "crop_yield_change";
        calibration[3] = "1";
    }
    let lookup_json = lookups
        .into_iter()
        .map(|lookup| {
            json!({
                "table_id": lookup.table_id,
                "table_sha256": installed().tables[lookup.table_id],
                "key": lookup.key,
            })
        })
        .collect::<Vec<_>>();
    let mut document = Map::new();
    document.insert("schema_version".to_string(), json!("1.0"));
    document.insert("semantics_version".to_string(), json!("crc-impact/1"));
    document.insert(
        "calibration".to_string(),
        json!({
            "exposure_variable": calibration[0],
            "exposure_unit": calibration[1],
            "impact_metric": calibration[2],
            "impact_unit": calibration[3],
        }),
    );
    document.insert("model".to_string(), model);
    if let Some(clamp) = clamp {
        document.insert("clamp".to_string(), clamp);
    }
    document.insert(
        "provenance".to_string(),
        json!({
            "origin": {
                "kind": "registry",
                "registry_id": REGISTRY_ID,
                "registry_revision": REGISTRY_REVISION,
                "registry_manifest_sha256": manifest_sha256(),
                "resolver_semantics_version": RESOLVER_VERSION,
                "factor": reference.factor,
                "context": {
                    "cell": context.cell,
                    "country": context.country,
                    "continent": context.continent,
                    "building_type": context.building_type,
                    "historic_mean": context.historic_mean,
                },
                "overrides": override_document(&reference.overrides),
                "lookups": lookup_json,
            },
            "sources": [],
            "derived_from": [],
        }),
    );
    ImpactDefinition::from_json(&serde_json::to_string(&Value::Object(document)).expect("json"))
}

fn override_document(overrides: &[(String, Option<f64>)]) -> Value {
    let mut object = Map::new();
    for (key, value) in overrides {
        object.insert(
            key.clone(),
            match value {
                Some(value) => json!(value),
                None => Value::Null,
            },
        );
    }
    Value::Object(object)
}

fn crop_value(values: &ag_params::NationalCropWeights) -> Value {
    json!({
        "maize": f64::from(values.maize),
        "rice": f64::from(values.rice),
        "wheat": f64::from(values.wheat),
    })
}

fn validate_context(context: &ResolutionContext) -> ContractResult<()> {
    if let Some(cell) = &context.cell {
        if !is_h3_cell(cell) {
            return Err(fail(
                "invalid_value",
                "/context/cell",
                "must be 15 lowercase hex characters",
            ));
        }
    }
    for (field, value) in [
        ("country", context.country.as_deref()),
        ("continent", context.continent.as_deref()),
        ("building_type", context.building_type.as_deref()),
    ] {
        if let Some(text) = value {
            if text.is_empty() || text.trim() != text {
                return Err(fail(
                    "invalid_value",
                    format!("/context/{field}"),
                    "must be non-empty without surrounding whitespace",
                ));
            }
        }
    }
    if context
        .historic_mean
        .is_some_and(|value| !value.is_finite())
    {
        return Err(fail(
            "non_finite_number",
            "/context/historic_mean",
            "numbers must be finite binary64 values",
        ));
    }
    Ok(())
}

fn require_context(meta: &FactorMeta, context: &ResolutionContext) -> ContractResult<()> {
    for field in meta.context {
        let present = match *field {
            "cell" => context.cell.is_some(),
            "country" => context.country.is_some(),
            "continent" => context.continent.is_some(),
            "building_type" => context.building_type.is_some(),
            "historic_mean" => context.historic_mean.is_some(),
            _ => false,
        };
        if !present {
            return Err(fail(
                "missing_context",
                format!("/context/{field}"),
                format!("{field} is required"),
            ));
        }
    }
    Ok(())
}

fn factor_meta(factor: &str) -> &'static FactorMeta {
    FACTORS
        .iter()
        .find(|meta| meta.factor == factor)
        .expect("factor catalog")
}

fn is_h3_cell(text: &str) -> bool {
    text.len() == 15
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn cell_hex(cell: CellIndex) -> String {
    format!("{:015x}", u64::from(cell))
}

fn installed() -> &'static Installed {
    static INSTALLED: OnceLock<Installed> = OnceLock::new();
    INSTALLED.get_or_init(build_installed)
}

fn build_installed() -> Installed {
    let mut tables = BTreeMap::new();
    tables.insert(
        "precipitation.daily_freezethaw_cycles".to_string(),
        hash_precip(precipitation_thresholds::DLYFRZTHW_THRESHOLDS),
    );
    tables.insert(
        "precipitation.frost_days".to_string(),
        hash_precip(precipitation_thresholds::FROST_DAYS_THRESHOLDS),
    );
    tables.insert(
        "precipitation.rx1day".to_string(),
        hash_precip(precipitation_thresholds::RX1DAY_THRESHOLDS),
    );
    tables.insert(
        "wind.cyclone".to_string(),
        hash_wind(&wind_params::CYCLONE_PARAMS),
    );
    tables.insert(
        "wind.wind_max_daily_max".to_string(),
        hash_wind(&wind_params::WIND_MAX_DAILY_MAX_PARAMS),
    );
    tables.insert("flood.damage".to_string(), hash_flood());
    tables.insert(
        "spei.national_energy".to_string(),
        hash_f32_map(&spei_params::NATIONAL_ENERGY_CONSUMPTION),
    );
    tables.insert(
        "spei.national_water".to_string(),
        hash_f32_map(&spei_params::NATIONAL_WATER_CONSUMPTION),
    );
    tables.insert(
        "spei.asset_energy".to_string(),
        hash_f32_map(&spei_params::ASSET_ENERGY_CONSUMPTION),
    );
    for (id, map) in energy_maps() {
        tables.insert(id.to_string(), hash_f32_map(map));
    }
    tables.insert(
        "crop.cdd".to_string(),
        hash_crop_map(&ag_params::CDD_CROP_VALUES),
    );
    tables.insert(
        "crop.cxdd".to_string(),
        hash_crop_map(&ag_params::CXDD_CROP_VALUES),
    );
    tables.insert(
        "crop.weights".to_string(),
        hash_crop_map(&ag_params::NATIONAL_CROP_WEIGHTS),
    );
    tables.insert(
        "crop.cdd_yearly".to_string(),
        hash_rows(vec![(
            "global".to_string(),
            crop_json(&ag_params::CDD_YEARLY_CROP_ESTIMATES),
        )]),
    );
    tables.insert(
        "crop.cxdd_yearly".to_string(),
        hash_rows(vec![(
            "global".to_string(),
            crop_json(&ag_params::CXDD_YEARLY_CROP_ESTIMATES),
        )]),
    );
    tables.insert(
        "crop.cdd_index".to_string(),
        hash_rows(vec![(
            "global".to_string(),
            crop_json(&ag_params::CDD_INDEX_CROP_ESTIMATES),
        )]),
    );
    tables.insert(
        "crop.cxdd_index".to_string(),
        hash_rows(vec![(
            "global".to_string(),
            crop_json(&ag_params::CXDD_INDEX_CROP_ESTIMATES),
        )]),
    );
    let manifest = manifest_document(&tables);
    let manifest_json = canonical_json(&manifest);
    let manifest_sha256 = content_hash(&manifest)["sha256:".len()..].to_string();
    Installed {
        manifest_sha256,
        manifest_json,
        tables,
    }
}

fn manifest_document(tables: &BTreeMap<String, String>) -> Json {
    let factors = FACTORS
        .iter()
        .map(|meta| {
            let mut entries = vec![
                ("factor".to_string(), Json::String(meta.factor.to_string())),
                (
                    "required_context".to_string(),
                    Json::Array(
                        meta.context
                            .iter()
                            .map(|field| Json::String((*field).to_string()))
                            .collect(),
                    ),
                ),
                (
                    "branches".to_string(),
                    Json::Array(
                        meta.branches
                            .iter()
                            .map(|branch| Json::String((*branch).to_string()))
                            .collect(),
                    ),
                ),
                ("overrides".to_string(), override_catalog(meta.factor)),
                (
                    "calibration".to_string(),
                    calibration_json(meta.calibration),
                ),
            ];
            if let Some(branches) = branch_calibrations(meta.factor) {
                entries.push(("branch_calibrations".to_string(), branches));
            }
            if let Some(coefficients) = coefficients(meta.factor) {
                entries.push(("coefficients".to_string(), coefficients));
            }
            Json::Object(entries)
        })
        .collect();
    let table_json = tables
        .iter()
        .map(|(id, digest)| table_entry(id, digest))
        .collect();
    Json::object([
        ("registry_id", Json::String(REGISTRY_ID.to_string())),
        (
            "registry_revision",
            Json::String(REGISTRY_REVISION.to_string()),
        ),
        (
            "resolver_semantics_version",
            Json::String(RESOLVER_VERSION.to_string()),
        ),
        ("factors", Json::Array(factors)),
        ("tables", Json::Array(table_json)),
    ])
}

fn calibration_json(values: [&str; 4]) -> Json {
    Json::object([
        ("exposure_variable", Json::String(values[0].to_string())),
        ("exposure_unit", Json::String(values[1].to_string())),
        ("impact_metric", Json::String(values[2].to_string())),
        ("impact_unit", Json::String(values[3].to_string())),
    ])
}

fn branch_calibrations(factor: &str) -> Option<Json> {
    match factor {
        "cooling_degree_days" => Some(Json::object([
            (
                "agricultural",
                calibration_json(["cooling_degree_days", "K.d", "crop_yield_change", "1"]),
            ),
            (
                "non_agricultural",
                calibration_json([
                    "cooling_degree_days",
                    "K.d",
                    "normalized_cooling_cost",
                    "crc-legacy-normalized-cost",
                ]),
            ),
        ])),
        _ => None,
    }
}

fn table_entry(id: &str, digest: &str) -> Json {
    let mut entries = vec![
        ("id".to_string(), Json::String(id.to_string())),
        ("sha256".to_string(), Json::String(digest.to_string())),
    ];
    if id.starts_with("precipitation.") {
        entries.push((
            "key_normalization".to_string(),
            Json::String("h3_parent_resolution_4".to_string()),
        ));
    }
    Json::Object(entries)
}

fn override_catalog(factor: &str) -> Json {
    let mut keys = [
        "cooling_cost",
        "intercept",
        "maximum_damage",
        "midpoint",
        "minimum_damage",
        "slope",
        "steepness",
        "water_unit_cost",
        "zero_below",
    ];
    keys.sort_unstable();
    Json::Array(
        keys.into_iter()
            .filter_map(|key| {
                override_rule(factor, key).map(|nullable| {
                    let mut entries = vec![
                        ("key".to_string(), Json::String(key.to_string())),
                        ("nullable".to_string(), Json::Bool(nullable)),
                    ];
                    if factor == "cooling_degree_days" && key == "cooling_cost" {
                        entries.push((
                            "branches".to_string(),
                            Json::Array(vec![Json::String("non_agricultural".to_string())]),
                        ));
                    }
                    Json::Object(entries)
                })
            })
            .collect(),
    )
}

fn coefficients(factor: &str) -> Option<Json> {
    match factor {
        "fwi" => Some(Json::object([
            ("maximum_damage", Json::number(1.0)),
            ("midpoint", Json::number(FWI_MIDPOINT)),
            ("minimum_damage", Json::number(0.0)),
            ("steepness", Json::number(FWI_STEEPNESS)),
        ])),
        "hot_days" => Some(Json::object([
            ("intercept", Json::number(0.0)),
            ("minimum_damage", Json::number(0.0)),
            ("slope", Json::number(HOT_DAYS_SLOPE)),
        ])),
        "rx1day" => Some(Json::object([
            ("lower_damage", Json::number(0.5)),
            ("maximum_damage", Json::number(0.3737)),
            ("upper", Json::number(1000.0)),
        ])),
        "daily_freezethaw_cycles" | "frost_days" => Some(Json::object([
            ("lower_damage", Json::number(5.0)),
            ("maximum_damage", Json::number(1.0)),
            ("upper", Json::number(365.0)),
        ])),
        "spei" => Some(Json::object([(
            "water_unit_cost",
            Json::number(SPEI_WATER_UNIT_COST),
        )])),
        _ => None,
    }
}

fn energy_maps() -> [(&'static str, &'static phf::Map<&'static str, f32>); 7] {
    [
        (
            "energy.residential_cooling",
            &energy_params::RESIDENTIAL_COOLING_ENERGY,
        ),
        (
            "energy.commercial_cooling",
            &energy_params::COMMERCIAL_COOLING_ENERGY,
        ),
        (
            "energy.industrial_cooling",
            &energy_params::INDUSTRIAL_COOLING_ENERGY,
        ),
        (
            "energy.residential_price",
            &energy_params::RESIDENTIAL_ENERGY_PRICE,
        ),
        (
            "energy.industrial_price",
            &energy_params::INDUSTRIAL_ENERGY_PRICE,
        ),
        (
            "energy.industrial_sector",
            &energy_params::INDUSTRIAL_ENERGY,
        ),
        (
            "energy.residential_commercial_sector",
            &energy_params::RESIDENTIAL_COMMERCIAL_ENERGY,
        ),
    ]
}

fn energy_map(table_id: &str) -> &'static phf::Map<&'static str, f32> {
    energy_maps()
        .into_iter()
        .find(|(id, _)| *id == table_id)
        .map(|(_, map)| map)
        .expect("energy table")
}

fn hash_f32_map(map: &phf::Map<&'static str, f32>) -> String {
    hash_rows(
        map.entries()
            .map(|(key, value)| ((*key).to_string(), Json::number(f64::from(*value))))
            .collect(),
    )
}

fn hash_wind(map: &phf::Map<&'static str, wind_params::WindParams>) -> String {
    hash_rows(
        map.entries()
            .map(|(key, params)| {
                (
                    (*key).to_string(),
                    Json::object([
                        ("k", Json::number(f64::from(params.k))),
                        ("t1", Json::number(f64::from(params.t1))),
                        ("t2", Json::number(f64::from(params.t2))),
                        ("x0", Json::number(f64::from(params.x0))),
                    ]),
                )
            })
            .collect(),
    )
}

fn hash_flood() -> String {
    hash_rows(
        flood_params::FLOOD_PARAMS
            .entries()
            .map(|(continent, params)| {
                (
                    (*continent).to_string(),
                    Json::object([
                        ("agriculture", f32_array(&params.agriculture_damage)),
                        ("commercial", f32_array(&params.commercial_damage)),
                        ("industrial", f32_array(&params.industrial_damage)),
                        ("infrastructure", f32_array(&params.infrastructure_damage)),
                        ("residential", f32_array(&params.residential_damage)),
                        ("transport", f32_array(&params.transport_damage)),
                    ]),
                )
            })
            .collect(),
    )
}

fn f32_array(values: &[f32]) -> Json {
    Json::Array(
        values
            .iter()
            .copied()
            .map(|value| Json::number(f64::from(value)))
            .collect(),
    )
}

fn hash_crop_map(map: &phf::Map<&'static str, ag_params::NationalCropWeights>) -> String {
    hash_rows(
        map.entries()
            .map(|(key, value)| ((*key).to_string(), crop_json(value)))
            .collect(),
    )
}

fn crop_json(values: &ag_params::NationalCropWeights) -> Json {
    Json::object([
        ("maize", Json::number(f64::from(values.maize))),
        ("rice", Json::number(f64::from(values.rice))),
        ("wheat", Json::number(f64::from(values.wheat))),
    ])
}

fn hash_precip(rows: &[(u64, u16)]) -> String {
    hash_rows(
        rows.iter()
            .map(|(cell, threshold)| {
                let index = CellIndex::try_from(*cell).expect("precipitation H3 cell");
                (
                    cell_hex(index),
                    Json::number(f64::from(*threshold as f32 * 0.1_f32)),
                )
            })
            .collect(),
    )
}

fn hash_rows(mut rows: Vec<(String, Json)>) -> String {
    rows.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
    let value = Json::Array(
        rows.into_iter()
            .map(|(key, item)| {
                Json::Object(vec![
                    ("key".to_string(), Json::String(key)),
                    ("value".to_string(), item),
                ])
            })
            .collect(),
    );
    content_hash(&value)["sha256:".len()..].to_string()
}

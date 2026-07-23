use crate::error::{CrcError, Result};

pub const HORIZONS: [i32; 17] = [
    2010, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070, 2075, 2080, 2085, 2090, 2095,
    2100,
];

pub const PATHWAYS: [(i32, &str); 16] = [
    (1, "SV"),
    (2, "Hot House"),
    (3, "Paris"),
    (4, "NDC"),
    (5, "ssp126"),
    (6, "ssp245"),
    (7, "ssp370"),
    (8, "ssp585"),
    (9, "<2 degrees"),
    (10, "2-3 degrees"),
    (11, "3-4 degrees"),
    (12, ">4 degrees"),
    (13, "historic"),
    (14, "ssp434"),
    (15, "ssp119"),
    (16, "ssp460"),
];

pub const RISK_FACTORS: [(i32, &str); 64] = [
    (1, "daily_freezethaw_cycles"),
    (2, "frost_days"),
    (3, "cyclone"),
    (4, "fwi"),
    (5, "rx1day"),
    (6, "wind_max_daily_max"),
    (7, "carbon_price"),
    (8, "hot_days"),
    (9, "cflood"),
    (10, "rflood"),
    (11, "inundation"),
    (12, "spei"),
    (13, "dc"),
    (14, "dmc"),
    (15, "ffmc"),
    (16, "isi"),
    (17, "bui"),
    (18, "rx5day"),
    (19, "wind_max_daily_mean"),
    (20, "cooling_degree_days"),
    (21, "tg_max"),
    (22, "tg_mean"),
    (23, "tg_min"),
    (24, "tx_max"),
    (25, "tx_mean"),
    (26, "tx_min"),
    (27, "tn_max"),
    (28, "tn_mean"),
    (29, "tn_min"),
    (30, "sdii"),
    (31, "liquidprcptot"),
    (32, "solidprcptot"),
    (33, "prcptot"),
    (34, "pet"),
    (35, "water_budget"),
    (37, "dtrmax"),
    (38, "dtrvar"),
    (39, "etr"),
    (40, "calm_days"),
    (41, "corn_heat_units"),
    (42, "wbgt"),
    (43, "windchill"),
    (44, "heat_index"),
    (45, "heat_wave_frequency"),
    (46, "heat_wave_total_length"),
    (47, "heat_wave_max_length"),
    (48, "heat_wave_index"),
    (49, "hot_spell_frequency"),
    (50, "hot_spell_max_length"),
    (51, "maximum_consecutive_frost_days"),
    (52, "cold_spell_days"),
    (53, "cold_spell_frequency"),
    (54, "maximum_consecutive_wet_days"),
    (55, "maximum_consecutive_dry_days"),
    (56, "at"),
    (57, "summer_days"),
    (58, "tropical_nights"),
    (60, "humidex"),
    (61, "heating_degree_days"),
    (62, "growing_degree_days"),
    (63, "ice_days"),
    (64, "dry_days"),
    (65, "wet_days"),
    (66, "dtr"),
];

pub fn pathway_name(id: i32) -> Result<&'static str> {
    PATHWAYS
        .iter()
        .find_map(|(candidate, name)| (*candidate == id).then_some(*name))
        .ok_or_else(|| CrcError::InvalidInput(format!("unknown pathway id {id}")))
}

pub fn pathway_id(name: &str) -> Result<i32> {
    PATHWAYS
        .iter()
        .find_map(|(id, candidate)| (*candidate == name).then_some(*id))
        .ok_or_else(|| CrcError::InvalidInput(format!("unknown pathway name {name}")))
}

pub fn risk_factor_name(id: i32) -> Result<&'static str> {
    RISK_FACTORS
        .iter()
        .find_map(|(candidate, name)| (*candidate == id).then_some(*name))
        .ok_or_else(|| CrcError::InvalidInput(format!("unknown risk factor id {id}")))
}

pub fn risk_factor_id(name: &str) -> Result<i32> {
    let canonical = match name {
        "dlyfrzthw" => "daily_freezethaw_cycles",
        "cflood_rp50" => "cflood",
        "rflood_rp50" => "rflood",
        "slr" => "inundation",
        other => other,
    };
    RISK_FACTORS
        .iter()
        .find_map(|(id, candidate)| (*candidate == canonical).then_some(*id))
        .ok_or_else(|| CrcError::InvalidInput(format!("unknown risk factor name {name}")))
}

pub fn validate_horizon(horizon: i32) -> Result<i32> {
    HORIZONS
        .contains(&horizon)
        .then_some(horizon)
        .ok_or_else(|| CrcError::InvalidInput(format!("unsupported horizon {horizon}")))
}

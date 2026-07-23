use h3o::{CellIndex, Resolution};

use crate::{
    error::{CrcError, Result},
    reference::spatial::{
        continent_r4 as continent_r4_data, geography_r5 as geography_r5_data, ipcc as ipcc_data,
    },
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Geography {
    pub continent: String,
    pub countries: Vec<String>,
}

pub fn parse_cell(value: &str) -> Result<CellIndex> {
    let raw = u64::from_str_radix(value.trim_start_matches("0x"), 16)
        .map_err(|_| CrcError::InvalidInput(format!("invalid H3 cell {value}")))?;
    CellIndex::try_from(raw).map_err(|_| CrcError::InvalidInput(format!("invalid H3 cell {value}")))
}

pub fn cell_from_u64(value: u64) -> Result<CellIndex> {
    CellIndex::try_from(value)
        .map_err(|_| CrcError::InvalidInput(format!("invalid H3 cell {value:x}")))
}

pub fn lookup_ipcc_region(cell: CellIndex) -> Result<Option<&'static str>> {
    let parent = parent_at(cell, Resolution::Four)?;
    let value = u64::from(parent) as i64;
    Ok(ipcc_data::CLIMATE_REGION_MAPPING
        .binary_search_by(|(start, end, _)| {
            if value < *start {
                std::cmp::Ordering::Greater
            } else if value > *end {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .ok()
        .map(|index| ipcc_data::CLIMATE_REGION_MAPPING[index].2))
}

pub fn lookup_continent(cell: CellIndex) -> Result<Option<&'static str>> {
    let parent = parent_at(cell, Resolution::Four)?;
    let value = u64::from(parent);
    Ok(continent_r4_data::HEX_TO_CONTINENT_MAPPING
        .binary_search_by(|(start, end, _)| {
            if value < *start {
                std::cmp::Ordering::Greater
            } else if value > *end {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .ok()
        .map(|index| continent_r4_data::HEX_TO_CONTINENT_MAPPING[index].2))
}

pub fn lookup_geography(cell: CellIndex) -> Result<Option<Geography>> {
    let parent = parent_at(cell, Resolution::Five)?;
    let value = u64::from(parent);
    let index = match geography_r5_data::CELL_RANGES.binary_search_by(|(start, end, _)| {
        if value < *start {
            std::cmp::Ordering::Greater
        } else if value > *end {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(index) => index,
        Err(_) => return Ok(None),
    };
    let payload = geography_r5_data::CELL_RANGES[index].2;
    let (continent, start, end) = geography_r5_data::PAYLOADS[payload];
    Ok(Some(Geography {
        continent: continent.to_owned(),
        countries: geography_r5_data::COUNTRIES[start..end]
            .iter()
            .map(|country| (*country).to_owned())
            .collect(),
    }))
}

fn parent_at(cell: CellIndex, resolution: Resolution) -> Result<CellIndex> {
    if cell.resolution() < resolution {
        return Err(CrcError::InvalidInput(format!(
            "H3 cell resolution {} is below required resolution {}",
            u8::from(cell.resolution()),
            u8::from(resolution)
        )));
    }
    cell.parent(resolution).ok_or_else(|| {
        CrcError::InvalidInput(format!(
            "could not normalize H3 cell to resolution {}",
            u8::from(resolution)
        ))
    })
}

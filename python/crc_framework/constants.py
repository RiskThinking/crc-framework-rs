from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union

from crc_framework import _core


class RiskFactor(str, Enum):
    DAILY_FREEZETHAW_CYCLES = "daily_freezethaw_cycles"
    FROST_DAYS = "frost_days"
    CYCLONE = "cyclone"
    FWI = "fwi"
    RX1DAY = "rx1day"
    WIND_MAX_DAILY_MAX = "wind_max_daily_max"
    CARBON_PRICE = "carbon_price"
    HOT_DAYS = "hot_days"
    CFLOOD = "cflood"
    RFLOOD = "rflood"
    INUNDATION = "inundation"
    SPEI = "spei"
    DC = "dc"
    DMC = "dmc"
    FFMC = "ffmc"
    ISI = "isi"
    BUI = "bui"
    RX5DAY = "rx5day"
    WIND_MAX_DAILY_MEAN = "wind_max_daily_mean"
    COOLING_DEGREE_DAYS = "cooling_degree_days"
    TG_MAX = "tg_max"
    TG_MEAN = "tg_mean"
    TG_MIN = "tg_min"
    TX_MAX = "tx_max"
    TX_MEAN = "tx_mean"
    TX_MIN = "tx_min"
    TN_MAX = "tn_max"
    TN_MEAN = "tn_mean"
    TN_MIN = "tn_min"
    SDII = "sdii"
    LIQUIDPRCPTOT = "liquidprcptot"
    SOLIDPRCPTOT = "solidprcptot"
    PRCPTOT = "prcptot"
    PET = "pet"
    WATER_BUDGET = "water_budget"
    DTRMAX = "dtrmax"
    DTRVAR = "dtrvar"
    ETR = "etr"
    CALM_DAYS = "calm_days"
    CORN_HEAT_UNITS = "corn_heat_units"
    WBGT = "wbgt"
    WINDCHILL = "windchill"
    HEAT_INDEX = "heat_index"
    HEAT_WAVE_FREQUENCY = "heat_wave_frequency"
    HEAT_WAVE_TOTAL_LENGTH = "heat_wave_total_length"
    HEAT_WAVE_MAX_LENGTH = "heat_wave_max_length"
    HEAT_WAVE_INDEX = "heat_wave_index"
    HOT_SPELL_FREQUENCY = "hot_spell_frequency"
    HOT_SPELL_MAX_LENGTH = "hot_spell_max_length"
    MAXIMUM_CONSECUTIVE_FROST_DAYS = "maximum_consecutive_frost_days"
    COLD_SPELL_DAYS = "cold_spell_days"
    COLD_SPELL_FREQUENCY = "cold_spell_frequency"
    MAXIMUM_CONSECUTIVE_WET_DAYS = "maximum_consecutive_wet_days"
    MAXIMUM_CONSECUTIVE_DRY_DAYS = "maximum_consecutive_dry_days"
    AT = "at"
    SUMMER_DAYS = "summer_days"
    TROPICAL_NIGHTS = "tropical_nights"
    HUMIDEX = "humidex"
    HEATING_DEGREE_DAYS = "heating_degree_days"
    GROWING_DEGREE_DAYS = "growing_degree_days"
    ICE_DAYS = "ice_days"
    DRY_DAYS = "dry_days"
    WET_DAYS = "wet_days"
    DTR = "dtr"


class Pathway(str, Enum):
    SV = "SV"
    HOT_HOUSE = "Hot House"
    PARIS = "Paris"
    NDC = "NDC"
    SSP126 = "ssp126"
    SSP245 = "ssp245"
    SSP370 = "ssp370"
    SSP585 = "ssp585"
    BELOW_2_DEGREES = "<2 degrees"
    BETWEEN_2_AND_3_DEGREES = "2-3 degrees"
    BETWEEN_3_AND_4_DEGREES = "3-4 degrees"
    ABOVE_4_DEGREES = ">4 degrees"
    HISTORIC = "historic"
    SSP434 = "ssp434"
    SSP119 = "ssp119"
    SSP460 = "ssp460"


@dataclass(frozen=True)
class Dimension:
    id: int
    name: str


PATHWAYS = tuple(Dimension(id_, name) for id_, name in _core.pathways())
RISK_FACTORS = tuple(Dimension(id_, name) for id_, name in _core.risk_factors())
HORIZONS = tuple(_core.horizons())


def pathway_from_id(id_: int) -> Pathway:
    try:
        return Pathway(next(item.name for item in PATHWAYS if item.id == id_))
    except StopIteration as error:
        raise ValueError(f"unknown pathway id {id_}") from error


def pathway_id(pathway: Union[Pathway, str]) -> int:
    value = str(pathway.value if isinstance(pathway, Pathway) else pathway)
    try:
        return next(item.id for item in PATHWAYS if item.name == value)
    except StopIteration as error:
        raise ValueError(f"unknown pathway {value}") from error


def risk_factor_from_id(id_: int) -> RiskFactor:
    try:
        return RiskFactor(next(item.name for item in RISK_FACTORS if item.id == id_))
    except StopIteration as error:
        raise ValueError(f"unknown risk factor id {id_}") from error


def risk_factor_id(factor: Union[RiskFactor, str]) -> int:
    value = str(factor.value if isinstance(factor, RiskFactor) else factor)
    try:
        return next(item.id for item in RISK_FACTORS if item.name == value)
    except StopIteration as error:
        raise ValueError(f"unknown risk factor {value}") from error

# Measurements and model requirements

[Documentation index](README.md) · [Python guide](python.md)

Measurements describe science independently of variable names. Use a typed record when its meaning applies, or `Measurement(quantity=..., unit=...)` for user-owned meanings. Unknown facts remain absent. Custom quantities and units require no taxonomy registration or extension schema.

```python
from crc_framework import (
    AssetType, BuiltinImpact, DepthAboveGround, FloodRegion, Impact,
    InundationContext, LengthUnit, Measurement, WaterSurfaceElevation,
)

custom = Measurement(quantity="my:quantity", unit="my:unit")
elevation = WaterSurfaceElevation(unit=LengthUnit.METRE, datum="my:vertical-datum")
depth = DepthAboveGround(unit=LengthUnit.METRE)
model = Impact.builtin(BuiltinImpact.INUNDATION)
model.requirements.bind(depth)
response = model.resolve(InundationContext(
    region=FloodRegion.EUROPE, asset_type=AssetType.RESIDENTIAL,
))
damage = response.evaluate(1.0)
```

`my:quantity`, `my:unit`, and `my:vertical-datum` are user data. Depth above ground and absolute water elevation are different quantities. Relabelling an elevation or a unit cannot perform a datum transformation or numerical unit conversion.

## Typed records

The following fields and units are generated from the same definitions used by JSON validation and model requirements. Required fields appear first. Built-in records reject unknown fields; custom `Measurement.fields` remains open and may optionally be validated by a local JSON Schema draft2020-12 extension.

| Record | Required fields | Optional fields | Units |
| --- | --- | --- | --- |
| `Measurement` |  | `fields`, `quantity`, `schema`, `unit` |  |
| `DepthAboveGround` | `unit` |  | `m`, `cm`, `mm`, `ft` |
| `WaterSurfaceElevation` | `unit`, `datum` |  | `m`, `cm`, `mm`, `ft` |
| `Ratio` | `unit` |  | `1`, `percent` |
| `TimeDuration` | `unit` |  | `s`, `h`, `d`, `month`, `yr` |
| `Temperature` | `unit` |  | `degC`, `K`, `degF` |
| `WindSpeed` | `unit` | `aggregation`, `averaging_duration` | `m/s`, `km/h` |
| `PrecipitationRate` | `unit`, `accumulation_duration`, `aggregation` |  | `mm/day`, `mm/h`, `kg m-2 s-1` |
| `PrecipitationAmount` | `unit`, `accumulation_duration` |  | `m`, `cm`, `mm`, `ft` |
| `EventCount` | `event`, `aggregation` | `duration`, `threshold` |  |
| `ClimateIndex` | `index` | `accumulation_duration`, `reference_period` |  |
| `DegreeDays` | `unit`, `threshold`, `duration` |  | `K.d`, `degC.d` |

`Duration(value, unit=...)` describes a scientific interval; `Threshold(value, unit=...)` describes an explicit comparison threshold. `ReferencePeriod(start, end)` uses inclusive years. These qualifiers are serialized inside the measurement once, not as a second corrective definition map.

## Published models

A reference exposes its concrete `measurement_type`, complete `measurement_schema`, expected `input`, accepted units, actual typed `context_type`, and allowed context choices. Inspect `model.requirements` or `help(model.requirements.context_type)` to discover the constructor. No anonymous context factory or lowercase registry singleton is required.

For country-based models, the country field in `model.requirements.context_fields` lists accepted table codes in `choices`. Cooling costs depend on the asset category. Index that field's `choices_by_asset_type` by an `AssetType` member to get the codes for a particular category.

Use `BuiltinImpact.COOLING_DEGREE_DAYS_CROP_YIELD` with `CropContext` for agricultural buildings. `BuiltinImpact.COOLING_DEGREE_DAYS` uses `CoolingCostContext` for residential, commercial, and industrial cooling costs.

A model binds data only when the data declares exactly the model's input, including its event threshold, counting or accumulation duration and wind averaging. The input facts below are the definitions the RiskThinking calibration assumed. Events are counted per calendar year: a frost day has a daily minimum temperature below the threshold; a freeze-thaw cycle has a daily minimum at or below the threshold and a daily maximum above it; a dry day has precipitation below the threshold.

The per-cell thresholds of frost days, freeze-thaw cycles and maximum daily precipitation are return levels of the 1985–2024 historic baseline: the 50-year level for frost days and freeze-thaw cycles, the 100-year level for precipitation. Cyclone winds are 10-minute sustained winds from STORM synthetic tracks, used without conversion; cyclone hazard does not depend on period or pathway. Daily maximum wind is used as delivered, with no averaging. SPEI is monthly and standardized over the same series, so it declares no reference period. The cooling degree-day cost compares against the historic mean the caller supplies in `CoolingCostContext`.

Where a source calibration does not establish a fact, `unestablished_facts` lists it. Scientific data binding rejects those models instead of guessing; supplying a plausible value on the data does not establish the published model calibration. Pure numerical `resolve` remains available for explicitly numerical work.

`BuiltinImpact.HOT_DAYS` is retired from the RiskThinking methodology. `Impact.builtin(BuiltinImpact.HOT_DAYS)` warns with `DeprecationWarning`, and binding it to data fails with `model_retired`. Saved documents that name it still read. crc-framework 2.0 removes it.

Maximum daily precipitation uses ETCCDI Rx1day: the maximum daily precipitation amount in millimetres accumulated over one day ([Climdex definition](https://www.climdex.org/learn/indices/)). Its numerical response consumes that amount directly; it does not integrate a precipitation rate. The reduction is recorded in data provenance.

| Model | Measurement | Input facts | Context | Unestablished calibration facts |
| --- | --- | --- | --- | --- |
| `crc:freeze_thaw_cycles` | `EventCount` | event `freeze_thaw_cycle`, aggregation `count`, threshold 0 degC, duration 1 yr | `CellContext` |  |
| `crc:frost_days` | `EventCount` | event `frost_day`, aggregation `count`, threshold 0 degC, duration 1 yr | `CellContext` |  |
| `crc:maximum_daily_precipitation` | `PrecipitationAmount` | unit `mm`, accumulation duration 1 d | `CellContext` |  |
| `crc:cyclone` | `WindSpeed` | unit `m/s`, aggregation `maximum`, averaging duration 600 s | `CountryContext` |  |
| `crc:maximum_daily_wind_speed` | `WindSpeed` | unit `m/s`, aggregation `maximum` | `CountryContext` |  |
| `crc:fire_weather_index` | `ClimateIndex` | index `fire_weather_index` | `EmptyContext` |  |
| `crc:hot_days` (retired) | `EventCount` | event `hot_day`, aggregation `count` | `EmptyContext` | event threshold, counting duration, calibration reference period |
| `crc:coastal_flood` | `DepthAboveGround` | unit `m` | `InundationContext` |  |
| `crc:river_flood` | `DepthAboveGround` | unit `m` | `InundationContext` |  |
| `crc:inundation` | `DepthAboveGround` | unit `m` | `InundationContext` |  |
| `crc:standardized_precipitation_evapotranspiration_index` | `ClimateIndex` | index `standardized_precipitation_evapotranspiration_index`, accumulation duration 1 month | `WaterCostContext` |  |
| `crc:cooling_degree_days` | `DegreeDays` | unit `K.d`, threshold 18 degC, duration 1 yr | `CoolingCostContext` |  |
| `crc:cooling_degree_days_crop_yield` | `DegreeDays` | unit `K.d` | `CropContext` | temperature threshold, accumulation duration, calibration reference period |
| `crc:consecutive_dry_days` | `EventCount` | event `dry_day`, aggregation `maximum` | `CropContext` | dry-day precipitation threshold, consecutive-event reduction, calibration reference period |

Flood regions and asset categories are closed typed choices. `FloodRegion.GLOBAL` selects the published global calibration explicitly. Agricultural land and agricultural buildings are distinct asset meanings. `CropContext` omits an asset-type choice because that model already fixes agricultural-building applicability.

Asset-column references round-trip as typed context bindings. Numerical resolution rejects unresolved bindings; data workflows resolve them against their actual asset table. H3 context cells are integers consistently across Python and persisted records.

Scientific identity includes the complete measurement, qualifiers and custom scientific schema. Display labels and provenance are outside that identity. Public hashes use `sha256:` followed by 64 lowercase hexadecimal digits.

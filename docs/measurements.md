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
| `PrecipitationRate` | `unit`, `accumulation_duration`, `aggregation` |  | `mm/day`, `mm/h` |
| `EventCount` | `event`, `aggregation` | `duration`, `threshold` |  |
| `ClimateIndex` | `index` | `accumulation_duration`, `reference_period` |  |
| `DegreeDays` | `unit`, `threshold`, `duration` |  | `K.d`, `degC.d` |

`Duration(value, unit=...)` describes a scientific interval; `Threshold(value, unit=...)` describes an explicit comparison threshold. `ReferencePeriod(start, end)` uses inclusive years. These qualifiers are serialized inside the measurement once, not as a second corrective definition map.

## Published models

A reference exposes its concrete `measurement_type`, complete `measurement_schema`, expected `input`, accepted units, actual typed `context_type`, and allowed context choices. Inspect `model.requirements` or `help(model.requirements.context_type)` to discover the constructor. No anonymous context factory or lowercase registry singleton is required.

Some source calibrations do not establish a threshold, averaging interval, accumulation timescale or reference period. `unestablished_facts` exposes these gaps. Scientific data binding rejects them instead of guessing; supplying a plausible value on the data does not establish the published model calibration. Pure numerical `resolve` remains available for explicitly numerical work.

| Model | Measurement | Context | Unestablished calibration facts |
| --- | --- | --- | --- |
| `crc:freeze_thaw_cycles` | `EventCount` | `CellContext` | event threshold, counting duration, calibration reference period |
| `crc:frost_days` | `EventCount` | `CellContext` | event threshold, counting duration, calibration reference period |
| `crc:maximum_daily_precipitation` | `PrecipitationRate` | `CellContext` | calibration reference period |
| `crc:cyclone` | `WindSpeed` | `CountryContext` | wind averaging duration, calibration reference period |
| `crc:maximum_daily_wind_speed` | `WindSpeed` | `CountryContext` | wind averaging duration, calibration reference period |
| `crc:fire_weather_index` | `ClimateIndex` | `EmptyContext` |  |
| `crc:hot_days` | `EventCount` | `EmptyContext` | event threshold, counting duration, calibration reference period |
| `crc:coastal_flood` | `DepthAboveGround` | `InundationContext` |  |
| `crc:river_flood` | `DepthAboveGround` | `InundationContext` |  |
| `crc:inundation` | `DepthAboveGround` | `InundationContext` |  |
| `crc:standardized_precipitation_evapotranspiration_index` | `ClimateIndex` | `WaterCostContext` | index accumulation duration, index reference period |
| `crc:cooling_degree_days` | `DegreeDays` | `CoolingCostContext` | temperature threshold, accumulation duration, calibration reference period |
| `crc:consecutive_dry_days` | `EventCount` | `CropContext` | dry-day precipitation threshold, consecutive-event reduction, calibration reference period |

Flood regions and asset categories are closed typed choices. `FloodRegion.GLOBAL` selects the published global calibration explicitly. Agricultural land and agricultural buildings are distinct asset meanings. `CropContext` omits an asset-type choice because that model already fixes agricultural-building applicability.

Asset-column references round-trip as typed context bindings. Numerical resolution rejects unresolved bindings; data workflows resolve them against their actual asset table. H3 context cells are integers consistently across Python and persisted records.

Scientific identity includes the complete measurement, qualifiers and custom scientific schema. Display labels and provenance are outside that identity. Public hashes use `sha256:` followed by 64 lowercase hexadecimal digits.

"""Generate current scientific schemas and cross-language conformance records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crc_framework import _measurement as science
from crc_framework import _impacts as impacts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "python/crc_framework/contracts/scientific"
PREFIX = "https://climateriskcommons.org/schemas/"


def _clean(value):
    if isinstance(value, dict):
        return {
            k: _clean(v)
            for k, v in value.items()
            if not k.startswith("x-crc-") and k != "default"
        }
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _impact_schema():
    private = json.loads(
        (
            ROOT
            / "python/crc_framework/contracts/impact/schemas/impact-definition-1.0.schema.json"
        ).read_text()
    )
    defs = _clean(private["$defs"])
    wanted = set()

    def visit(value):
        if isinstance(value, dict):
            ref = value.get("$ref", "")
            if ref.startswith("#/$defs/"):
                name = ref.removeprefix("#/$defs/")
                if name not in wanted:
                    wanted.add(name)
                    visit(defs[name])
            for v in value.values():
                visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)

    model = _clean(private["properties"]["model"])
    visit(model)
    clamp = science._object(
        {"minimum": {"type": "number"}, "maximum": {"type": "number"}}, ()
    )
    clamp["minProperties"] = 1
    reference = {"$ref": PREFIX + "impact-reference-2.0.schema.json"}
    provenance = science._object(
        {
            "model": reference,
            "context": {"$ref": PREFIX + "impact-context-1.0.schema.json"},
        },
        ("model", "context"),
    )
    root = science._object(
        {
            "schema_version": {"const": "2.0"},
            "input": {"$ref": PREFIX + "measurement-1.0.schema.json"},
            "output": {"$ref": PREFIX + "measurement-1.0.schema.json"},
            "model": model,
            "clamp": clamp,
            "provenance": provenance,
        },
        ("schema_version", "input", "output", "model"),
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": PREFIX + "impact-2.0.schema.json",
        "title": "Impact",
        "$defs": {k: defs[k] for k in sorted(wanted)},
        **root,
    }


def build():
    measurement = science._measurement_union_schema()
    context = science._context_union_schema()
    reference = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": PREFIX + "impact-reference-2.0.schema.json",
        "title": "ImpactReference",
        **science._object(
            {
                "schema_version": {"const": "2.0"},
                "id": {"enum": [v.value for v in impacts.BuiltinImpact]},
                "version": science._IDENTIFIER_SCHEMA,
                "manifest_sha256": {
                    "type": "string",
                    "pattern": "^sha256:[0-9a-f]{64}$",
                },
            },
            ("schema_version", "id", "version", "manifest_sha256"),
        ),
    }
    from crc_framework import _core

    _, version, _, digest = _core.impact_registry_identity()
    models = []
    for model in impacts.BuiltinImpact:
        ref = impacts.Impact.builtin(model)
        r = ref.requirements
        models.append(
            {
                "id": ref.id,
                "version": version,
                "manifest_sha256": "sha256:" + digest,
                "input": json.loads(r.input.to_json()),
                "measurement_type": r.measurement_type.__name__,
                "measurement_schema": json.loads(
                    json.dumps(science._plain(r.measurement_schema))
                ),
                "accepted_units": [
                    u.value if hasattr(u, "value") else u for u in r.accepted_units
                ],
                "context_type": r.context_type.__name__,
                "context_schema": science._context_schema(r.context_type._kind),
                "context_fields": [
                    {
                        "name": f.name,
                        "annotation": f.annotation,
                        "required": f.required,
                        "choices": list(f.choices),
                    }
                    for f in r.context_fields
                ],
                "unestablished_facts": list(r.unestablished_facts),
                "output": json.loads(r.output.to_json()),
            }
        )
    samples = [
        science.Measurement(),
        science.Measurement(
            quantity="user:quantity",
            unit="user:unit",
            fields={"definition": {"custom": True}},
        ),
        science.DepthAboveGround(unit=science.LengthUnit.METRE),
        science.WaterSurfaceElevation(
            unit=science.LengthUnit.FOOT, datum="user:vertical-datum"
        ),
        science.Ratio(),
        science.TimeDuration(unit=science.TimeUnit.DAY),
        science.Temperature(unit=science.TemperatureUnit.CELSIUS),
        science.WindSpeed(
            unit=science.SpeedUnit.METRES_PER_SECOND,
            averaging_duration=science.Duration(3, unit=science.TimeUnit.SECOND),
            aggregation=science.Aggregation.MAXIMUM,
        ),
        science.PrecipitationRate(
            unit=science.PrecipitationRateUnit.MILLIMETRES_PER_DAY,
            accumulation_duration=science.Duration(1, unit=science.TimeUnit.DAY),
            aggregation=science.Aggregation.MAXIMUM,
        ),
        science.PrecipitationAmount(
            unit=science.LengthUnit.MILLIMETRE,
            accumulation_duration=science.Duration(1, unit=science.TimeUnit.DAY),
        ),
        science.EventCount(
            event=science.Event.FROST_DAY,
            aggregation=science.Aggregation.COUNT,
            threshold=science.Threshold(0, unit=science.TemperatureUnit.CELSIUS),
            duration=science.Duration(1, unit=science.TimeUnit.YEAR),
        ),
        science.ClimateIndex(
            index=science.ClimateIndexKind.STANDARDIZED_PRECIPITATION_EVAPOTRANSPIRATION_INDEX,
            accumulation_duration=science.Duration(3, unit=science.TimeUnit.MONTH),
            reference_period=science.ReferencePeriod(1981, 2010),
        ),
        science.DegreeDays(
            unit=science.DegreeDayUnit.KELVIN_DAYS,
            threshold=science.Threshold(18, unit=science.TemperatureUnit.CELSIUS),
            duration=science.Duration(1, unit=science.TimeUnit.YEAR),
        ),
    ]
    fixtures = []
    for sample in samples:
        fixtures.append(
            {
                "name": type(sample).__name__,
                "schema": measurement["$id"],
                "valid": True,
                "document": json.loads(sample.to_json()),
            }
        )
    context_samples = [
        science.EmptyContext(),
        science.CellContext(cell=599686042433355775),
        science.CountryContext(country="CAN"),
        science.InundationContext(
            region=science.FloodRegion.EUROPE, asset_type=science.AssetType.INDUSTRIAL
        ),
        science.WaterCostContext(
            country="CAN", asset_type=science.AssetType.COMMERCIAL
        ),
        science.CoolingCostContext(
            country="CAN", asset_type=science.AssetType.RESIDENTIAL, historic_mean=20
        ),
        science.CropContext(country="USA", historic_mean=20),
    ]
    for sample in context_samples:
        fixtures.append(
            {
                "name": type(sample).__name__,
                "schema": context["$id"],
                "valid": True,
                "document": json.loads(sample.to_json()),
            }
        )
    bad = [
        {"kind": "depth_above_ground", "unit": "metres"},
        {"kind": "depth_above_ground", "unit": "m", "datum": "sea_level"},
        {"kind": "water_surface_elevation", "unit": "m"},
        {"kind": "time_duration", "unit": "days"},
        {
            "kind": "wind_speed",
            "unit": "m/s",
            "averaging_duration": {"value": 0, "unit": "s"},
        },
        {
            "kind": "event_count",
            "event": "frost_day",
            "aggregation": "count",
            "threshold": {"value": 0, "unit": "degC", "operator": "<"},
        },
        {"kind": "measurement", "quantity": " spaced "},
        {"kind": "measurement", "fields": [], "schema": {}},
        {"kind": "measurement", "quantity": None},
    ]
    for n, document in enumerate(bad):
        fixtures.append(
            {
                "name": f"invalid-measurement-{n}",
                "schema": measurement["$id"],
                "valid": False,
                "document": document,
            }
        )
    for n, document in enumerate(
        [
            {"kind": "cell_context", "cell": "852b83b3fffffff"},
            {
                "kind": "inundation_context",
                "continent": "Europe",
                "building_type": "residential",
            },
            {
                "kind": "inundation_context",
                "region": "EUROPE",
                "asset_type": "residential",
            },
            {
                "kind": "crop_context",
                "country": "USA",
                "historic_mean": 20,
                "asset_type": "agricultural_building",
            },
        ]
    ):
        fixtures.append(
            {
                "name": f"invalid-context-{n}",
                "schema": context["$id"],
                "valid": False,
                "document": document,
            }
        )
    semantic_cases = [
        (
            "custom-fields-violate-schema",
            measurement["$id"],
            {
                "kind": "measurement",
                "fields": {"duration": 0},
                "schema": {
                    "type": "object",
                    "properties": {"duration": {"type": "integer", "minimum": 1}},
                },
            },
        ),
        (
            "custom-schema-invalid-dialect",
            measurement["$id"],
            {
                "kind": "measurement",
                "schema": {"$schema": "http://json-schema.org/draft-07/schema#"},
            },
        ),
        (
            "custom-schema-external-ref",
            measurement["$id"],
            {
                "kind": "measurement",
                "schema": {"$ref": "https://example.invalid/remote"},
            },
        ),
        (
            "unordered-reference-period",
            measurement["$id"],
            {
                "kind": "climate_index",
                "index": "standardized_precipitation_evapotranspiration_index",
                "reference_period": {"start": 2010, "end": 1981},
            },
        ),
        ("invalid-h3-context", context["$id"], {"kind": "cell_context", "cell": 1}),
    ]
    for name, schema, document in semantic_cases:
        fixtures.append(
            {
                "name": name,
                "schema": schema,
                "schema_valid": True,
                "valid": False,
                "document": document,
            }
        )
    # JSON Schema integer values include integral JSON-number spellings. The
    # decoder hydrates typed integer fields without truncating fractional data.
    fixtures.append(
        {
            "name": "integral-reference-years",
            "schema": measurement["$id"],
            "valid": True,
            "document": {
                "kind": "climate_index",
                "index": "standardized_precipitation_evapotranspiration_index",
                "reference_period": {"start": 1981.0, "end": 2010.0},
            },
        }
    )

    for selected in impacts.BuiltinImpact:
        ref = impacts.Impact.builtin(selected)
        fixtures.append(
            {
                "name": "reference-" + selected.name,
                "schema": reference["$id"],
                "valid": True,
                "document": json.loads(ref.to_json()),
            }
        )
    functions = [
        impacts.Impact.linear(slope=2, input=science.Ratio(), output=science.Ratio()),
        impacts.Impact.sigmoid(
            midpoint=0, steepness=1, input=science.Ratio(), output=science.Ratio()
        ),
        impacts.Impact.tabulated(
            [(0, 0), (1, 1)], input=science.Ratio(), output=science.Ratio()
        ),
        impacts.Impact.binned(
            [0, 1], [0.5], input=science.Ratio(), output=science.Ratio()
        ),
        impacts.Impact.builtin(impacts.BuiltinImpact.INUNDATION).resolve(
            science.InundationContext(
                region=science.FloodRegion.EUROPE,
                asset_type=science.AssetType.RESIDENTIAL,
            )
        ),
    ]
    functions.insert(4, impacts.Impact.builtin(impacts.BuiltinImpact.MAXIMUM_DAILY_PRECIPITATION).resolve(science.CellContext(cell=594475159402840063)))
    impact_uri = PREFIX + "impact-2.0.schema.json"
    for n, function in enumerate(functions):
        fixtures.append(
            {
                "name": "impact-" + str(n),
                "schema": impact_uri,
                "valid": True,
                "document": json.loads(function.to_json()),
            }
        )
    raw_reference = json.loads(
        impacts.Impact.builtin(impacts.BuiltinImpact.INUNDATION).to_json()
    )
    raw_reference["manifest_sha256"] = raw_reference["manifest_sha256"].removeprefix(
        "sha256:"
    )
    fixtures.append(
        {
            "name": "reference-raw-digest",
            "schema": reference["$id"],
            "valid": False,
            "document": raw_reference,
        }
    )
    incorrect = json.loads(functions[0].to_json())
    incorrect["model"]["unrecognized"] = True
    fixtures.append(
        {
            "name": "impact-unknown-model-field",
            "schema": impact_uri,
            "valid": False,
            "document": incorrect,
        }
    )
    incorrect = json.loads(functions[-1].to_json())
    incorrect["model"]["points"][1]["y"] += 0.01
    fixtures.append(
        {
            "name": "impact-provenance-mismatch",
            "schema": impact_uri,
            "schema_valid": True,
            "valid": False,
            "document": incorrect,
        }
    )
    fixtures.append(
        {
            "name": "bound-context-columns",
            "schema": context["$id"],
            "valid": True,
            "document": {
                "kind": "inundation_context",
                "region": {"column": "region"},
                "asset_type": {"column": "use"},
            },
        }
    )
    opaque = {
        "kind": "measurement",
        "schema": {"$schema": "user-data"},
        "$ref": "literal-user-value",
    }
    opaque_schema = {
        "type": "object",
        "properties": {"payload": {"const": opaque}},
        "required": ["payload"],
        "additionalProperties": False,
    }
    fixtures.append(
        {
            "name": "custom-schema-literal-record-keys",
            "schema": measurement["$id"],
            "valid": True,
            "document": {
                "kind": "measurement",
                "fields": {"payload": opaque},
                "schema": opaque_schema,
            },
        }
    )
    files = {
        "measurement-1.0.schema.json": measurement,
        "impact-context-1.0.schema.json": context,
        "impact-reference-2.0.schema.json": reference,
        "impact-2.0.schema.json": _impact_schema(),
        "models-1.0.json": {"schema_version": "1.0", "models": models},
        "kernel-bindings-1.0.json": {
            "models": {
                model.value: factor for model, factor in impacts._FACTORS.items()
            },
            "asset_types": {
                asset.value: value for asset, value in impacts._ASSET_NAMES.items()
            },
            "regions": {
                region.value: value for region, value in impacts._REGION_NAMES.items()
            },
            "fixed_context": {
                impacts.BuiltinImpact.CONSECUTIVE_DRY_DAYS.value: {
                    "asset_type": science.AssetType.AGRICULTURAL_BUILDING.value
                }
            },
        },
        "fixtures-1.0.json": {"schema_version": "1.0", "cases": fixtures},
    }
    return {
        path: (
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode()
        for path, value in files.items()
    }


def reference():
    resources = build()
    measurements = json.loads(resources["measurement-1.0.schema.json"])["oneOf"]
    models = json.loads(resources["models-1.0.json"])["models"]
    lines = [
        "# Measurements and model requirements",
        "",
        "[Documentation index](README.md) · [Python guide](python.md)",
        "",
        "Measurements describe science independently of variable names. Use a typed record when its meaning applies, or `Measurement(quantity=..., unit=...)` for user-owned meanings. Unknown facts remain absent. Custom quantities and units require no taxonomy registration or extension schema.",
        "",
        "```python",
        "from crc_framework import (",
        "    AssetType, BuiltinImpact, DepthAboveGround, FloodRegion, Impact,",
        "    InundationContext, LengthUnit, Measurement, WaterSurfaceElevation,",
        ")",
        "",
        'custom = Measurement(quantity="my:quantity", unit="my:unit")',
        'elevation = WaterSurfaceElevation(unit=LengthUnit.METRE, datum="my:vertical-datum")',
        "depth = DepthAboveGround(unit=LengthUnit.METRE)",
        "model = Impact.builtin(BuiltinImpact.INUNDATION)",
        "model.requirements.bind(depth)",
        "response = model.resolve(InundationContext(",
        "    region=FloodRegion.EUROPE, asset_type=AssetType.RESIDENTIAL,",
        "))",
        "damage = response.evaluate(1.0)",
        "```",
        "",
        "`my:quantity`, `my:unit`, and `my:vertical-datum` are user data. Depth above ground and absolute water elevation are different quantities. Relabelling an elevation or a unit cannot perform a datum transformation or numerical unit conversion.",
        "",
        "## Typed records",
        "",
        "The following fields and units are generated from the same definitions used by JSON validation and model requirements. Required fields appear first. Built-in records reject unknown fields; custom `Measurement.fields` remains open and may optionally be validated by a local JSON Schema draft2020-12 extension.",
        "",
        "| Record | Required fields | Optional fields | Units |",
        "| --- | --- | --- | --- |",
    ]
    for schema in measurements:
        required = [v for v in schema["required"] if v != "kind"]
        optional = [
            v for v in schema["properties"] if v not in required and v != "kind"
        ]
        units = schema["properties"].get("unit", {}).get("enum", [])
        lines.append(
            "| `"
            + schema["title"]
            + "` | "
            + ", ".join("`" + n + "`" for n in required)
            + " | "
            + ", ".join("`" + n + "`" for n in optional)
            + " | "
            + ", ".join("`" + v + "`" for v in units)
            + " |"
        )
    lines += [
        "",
        "`Duration(value, unit=...)` describes a scientific interval; `Threshold(value, unit=...)` describes an explicit comparison threshold. `ReferencePeriod(start, end)` uses inclusive years. These qualifiers are serialized inside the measurement once, not as a second corrective definition map.",
        "",
        "## Published models",
        "",
        "A reference exposes its concrete `measurement_type`, complete `measurement_schema`, expected `input`, accepted units, actual typed `context_type`, and allowed context choices. Inspect `model.requirements` or `help(model.requirements.context_type)` to discover the constructor. No anonymous context factory or lowercase registry singleton is required.",
        "",
        "Some source calibrations do not establish a threshold, averaging interval, accumulation timescale or reference period. `unestablished_facts` exposes these gaps. Scientific data binding rejects them instead of guessing; supplying a plausible value on the data does not establish the published model calibration. Pure numerical `resolve` remains available for explicitly numerical work.",
        "",
        "Maximum daily precipitation uses ETCCDI Rx1day: the maximum daily precipitation amount in millimetres accumulated over one day ([Climdex definition](https://www.climdex.org/learn/indices/)). Its numerical response consumes that amount directly; it does not integrate a precipitation rate. The reduction is recorded in data provenance.",
        "",
        "| Model | Measurement | Context | Unestablished calibration facts |",
        "| --- | --- | --- | --- |",
    ]
    for model in models:
        lines.append(
            "| `"
            + model["id"]
            + "` | `"
            + model["measurement_type"]
            + "` | `"
            + model["context_type"]
            + "` | "
            + ", ".join(model["unestablished_facts"])
            + " |"
        )
    lines += [
        "",
        "Flood regions and asset categories are closed typed choices. `FloodRegion.GLOBAL` selects the published global calibration explicitly. Agricultural land and agricultural buildings are distinct asset meanings. `CropContext` omits an asset-type choice because that model already fixes agricultural-building applicability.",
        "",
        "Asset-column references round-trip as typed context bindings. Numerical resolution rejects unresolved bindings; data workflows resolve them against their actual asset table. H3 context cells are integers consistently across Python and persisted records.",
        "",
        "Scientific identity includes the complete measurement, qualifiers and custom scientific schema. Display labels and provenance are outside that identity. Public hashes use `sha256:` followed by64 lowercase hexadecimal digits.",
        "",
    ]
    return "\n".join(lines).replace("by64", "by 64").encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for path, content in build().items():
        target = OUT / path
        if args.check:
            if not target.exists() or target.read_bytes() != content:
                stale.append(path)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    document = ROOT / "docs/measurements.md"
    content = reference()
    if args.check:
        if not document.exists() or document.read_bytes() != content:
            stale.append("docs/measurements.md")
    else:
        document.write_bytes(content)
    if stale:
        raise SystemExit("stale scientific resources: " + ", ".join(stale))


if __name__ == "__main__":
    main()

"""Scientific records, model calibration, and shared schema conformance."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from crc_framework import (
    Aggregation,
    AssetType,
    BuiltinImpact,
    CellContext,
    ClimateIndex,
    ClimateIndexKind,
    CoolingCostContext,
    CropContext,
    DegreeDays,
    DegreeDayUnit,
    DepthAboveGround,
    Duration,
    Event,
    EventCount,
    FloodRegion,
    Impact,
    ImpactContext,
    ImpactError,
    ImpactReference,
    InundationContext,
    LengthUnit,
    Measurement,
    PrecipitationRate,
    PrecipitationRateUnit,
    ReferencePeriod,
    SpeedUnit,
    Temperature,
    TemperatureUnit,
    Threshold,
    TimeUnit,
    WaterSurfaceElevation,
    WindSpeed,
)
from crc_framework._measurement import _scientific_validator
from jsonschema import Draft202012Validator

from tools.scientific_contracts.generate import build

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "python/crc_framework/contracts/scientific"
CASES = json.loads((RESOURCES / "fixtures-1.0.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_scientific_fixtures_agree_with_schema_and_python(case) -> None:
    name = case["schema"].rsplit("/", 1)[-1]
    validator = _scientific_validator(name)
    assert validator.is_valid(case["document"]) is case.get(
        "schema_valid", case["valid"]
    )
    decoder = {
        "measurement-1.0.schema.json": Measurement.from_json,
        "impact-context-1.0.schema.json": ImpactContext.from_json,
        "impact-reference-2.0.schema.json": ImpactReference.from_json,
        "impact-2.0.schema.json": Impact.from_json,
    }[name]
    if case["valid"]:
        record = decoder(json.dumps(case["document"]))
        assert validator.is_valid(json.loads(record.to_json()))
        assert decoder(record.to_json()).to_json() == record.to_json()
    else:
        with pytest.raises((TypeError, ValueError, ImpactError)):
            decoder(json.dumps(case["document"]))


def test_scientific_resources_are_generated_from_the_active_types() -> None:
    for name, content in build().items():
        assert (RESOURCES / name).read_bytes() == content
    for filename in RESOURCES.glob("*.schema.json"):
        Draft202012Validator.check_schema(json.loads(filename.read_text()))


def test_custom_science_needs_no_registration_or_schema() -> None:
    record = Measurement(
        quantity="user 🌀 variable",
        unit="research-pressure",
        fields={"duration": {"value": 2, "unit": "seasons"}},
    )
    assert Measurement.from_json(record.to_json()) == record
    assert Measurement().to_json() == '{"kind":"measurement"}'
    assert "schema" not in json.loads(record.to_json())
    assert record.schema is None
    with pytest.raises(TypeError):
        record.fields["new"] = "value"


def test_custom_extension_schemas_validate_their_scientific_fields() -> None:
    schema = {
        "type": "object",
        "properties": {"duration": {"type": "integer", "minimum": 1}},
        "required": ["duration"],
        "additionalProperties": False,
    }
    valid = Measurement(quantity="my-quantity", fields={"duration": 3}, schema=schema)
    assert Measurement.from_json(valid.to_json()) == valid
    for fields in (
        {},
        {"duration": 0},
        {"duration": "3"},
        {"duration": 3, "surprise": True},
    ):
        with pytest.raises(ValueError, match="schema"):
            Measurement(fields=fields, schema=schema)
    with pytest.raises(ValueError, match="schema"):
        Measurement(schema={"type": "made-up"})
    with pytest.raises(ValueError, match="local references"):
        Measurement(schema={"$ref": "https://example.invalid/remote"})
    # JSON Schema keys include regexes and literal property names, not library
    # identifiers. An empty regex is valid and must not be trimmed or rejected.
    patterns = {"type": "object", "patternProperties": {"": {"type": "number"}}}
    custom = Measurement(fields={" literal field ": 2}, schema=patterns)
    assert Measurement.from_json(custom.to_json()) == custom


def test_elevation_does_not_become_ground_relative_depth() -> None:
    water = WaterSurfaceElevation(unit=LengthUnit.METRE, datum="my:vertical-datum")
    assert water.datum == "my:vertical-datum"
    with pytest.raises(ImpactError, match="DepthAboveGround"):
        Impact.builtin(BuiltinImpact.INUNDATION).requirements.bind(water)
    with pytest.raises(TypeError):
        WaterSurfaceElevation(unit=LengthUnit.METRE)


def test_units_cannot_be_relabelled_to_repair_measurement_facts() -> None:
    required = Impact.builtin(BuiltinImpact.INUNDATION).requirements
    required.bind(DepthAboveGround(unit=LengthUnit.METRE))
    with pytest.raises(ImpactError, match="convert_units"):
        required.bind(DepthAboveGround(unit=LengthUnit.CENTIMETRE))
    with pytest.raises(ImpactError, match="DepthAboveGround"):
        required.bind(Measurement(quantity="depth_above_ground", unit=LengthUnit.METRE))


def test_named_qualifiers_are_validated_and_serialized_once() -> None:
    measurements = [
        Temperature(unit=TemperatureUnit.CELSIUS),
        WindSpeed(
            unit=SpeedUnit.METRES_PER_SECOND,
            averaging_duration=Duration(3, unit=TimeUnit.SECOND),
            aggregation=Aggregation.MAXIMUM,
        ),
        PrecipitationRate(
            unit=PrecipitationRateUnit.MILLIMETRES_PER_DAY,
            accumulation_duration=Duration(1, unit=TimeUnit.DAY),
            aggregation=Aggregation.MAXIMUM,
        ),
        EventCount(
            event=Event.FROST_DAY,
            aggregation=Aggregation.COUNT,
            threshold=Threshold(0, unit=TemperatureUnit.CELSIUS),
        ),
        ClimateIndex(
            index=ClimateIndexKind.STANDARDIZED_PRECIPITATION_EVAPOTRANSPIRATION_INDEX,
            accumulation_duration=Duration(3, unit=TimeUnit.DAY),
            reference_period=ReferencePeriod(1981, 2010),
        ),
        DegreeDays(
            unit=DegreeDayUnit.KELVIN_DAYS,
            threshold=Threshold(18, unit=TemperatureUnit.CELSIUS),
            duration=Duration(1, unit=TimeUnit.YEAR),
        ),
    ]
    for measurement in measurements:
        document = json.loads(measurement.to_json())
        assert (
            "quantity" not in document
            and "fields" not in document
            and "schema" not in document
        )
        assert Measurement.from_json(measurement.to_json()) == measurement
        assert measurement.schema["additionalProperties"] is False
    with pytest.raises(TypeError, match="SpeedUnit"):
        WindSpeed(unit="m/s")
    with pytest.raises(ValueError, match="positive"):
        Duration(0, unit=TimeUnit.DAY)
    with pytest.raises(ValueError, match="ordered"):
        ReferencePeriod(2010, 1981)


def test_context_factories_have_real_canonical_types_and_exact_asset_choices() -> None:
    required = Impact.builtin(BuiltinImpact.INUNDATION).requirements
    assert required.context_type is InundationContext
    assert InundationContext.__module__ == "crc_framework"
    signature = inspect.signature(required.context_type)
    assert list(signature.parameters) == ["region", "asset_type"]
    assert "building_type" not in signature.parameters
    choices = next(
        field.choices for field in required.context_fields if field.name == "asset_type"
    )
    enum_schema = required.context_type(
        region=FloodRegion.GLOBAL, asset_type=AssetType.ROADS
    ).schema["properties"]["asset_type"]["oneOf"][0]["enum"]
    assert tuple(choices) == tuple(enum_schema)
    with pytest.raises(ValueError, match="agricultural_building"):
        InundationContext(
            region=FloodRegion.EUROPE, asset_type=AssetType.AGRICULTURAL_BUILDING
        )
    with pytest.raises(TypeError, match="FloodRegion"):
        InundationContext(region="Europe", asset_type=AssetType.RESIDENTIAL)
    with pytest.raises(TypeError, match="H3 integer"):
        CellContext(cell="852b83b3fffffff")


@pytest.mark.parametrize(
    ("model", "asset_type", "context_values"),
    [
        (BuiltinImpact.CYCLONE, None, {}),
        (BuiltinImpact.MAXIMUM_DAILY_WIND_SPEED, None, {}),
        (
            BuiltinImpact.STANDARDIZED_PRECIPITATION_EVAPOTRANSPIRATION_INDEX,
            None,
            {"asset_type": AssetType.COMMERCIAL},
        ),
        (BuiltinImpact.CONSECUTIVE_DRY_DAYS, None, {"historic_mean": 25}),
        (BuiltinImpact.COOLING_DEGREE_DAYS_CROP_YIELD, None, {"historic_mean": 25}),
        *[
            (
                BuiltinImpact.COOLING_DEGREE_DAYS,
                asset_type,
                {"asset_type": asset_type, "historic_mean": 25},
            )
            for asset_type in (
                AssetType.RESIDENTIAL,
                AssetType.COMMERCIAL,
                AssetType.INDUSTRIAL,
            )
        ],
    ],
)
def test_country_requirements_agree_with_table_resolution_and_model_schema(
    model, asset_type, context_values
) -> None:
    reference = Impact.builtin(model)
    requirements = reference.requirements
    country = next(f for f in requirements.context_fields if f.name == "country")
    choices = (
        country.choices_by_asset_type[asset_type] if asset_type else country.choices
    )
    assert choices == tuple(sorted(set(choices)))
    assert "USA" in choices
    models = json.loads((RESOURCES / "models-1.0.json").read_text())["models"]
    registered = next(v for v in models if v["id"] == reference.id)
    validator = Draft202012Validator(registered["context_schema"])
    for code in choices:
        context = requirements.context_type(country=code, **context_values)
        assert validator.is_valid(json.loads(context.to_json()))
        assert isinstance(reference.resolve(context), Impact)

    if asset_type:
        different_branch = sorted(set(country.choices) - set(choices))
    else:
        different_branch = []
    invalid_code = different_branch[0] if different_branch else "ZZZ"
    context = requirements.context_type(country=invalid_code, **context_values)
    assert not validator.is_valid(json.loads(context.to_json()))
    with pytest.raises(
        ImpactError, match="model.requirements.context_fields"
    ) as caught:
        reference.resolve(context)
    assert caught.value.code == "missing_table_key"
    assert caught.value.path == "/context/country"


def test_cooling_degree_days_models_discover_their_own_context_and_output() -> None:
    crop = Impact.builtin(BuiltinImpact.COOLING_DEGREE_DAYS_CROP_YIELD)
    cooling = Impact.builtin(BuiltinImpact.COOLING_DEGREE_DAYS)

    assert crop.requirements.context_type is CropContext
    assert crop.requirements.output.quantity == "crop_yield_change"
    result = crop.resolve(CropContext(country="USA", historic_mean=25))
    assert Impact.from_json(result.to_json()).output == crop.requirements.output

    assert cooling.requirements.context_type is CoolingCostContext
    assert cooling.requirements.output.quantity == "normalized_cooling_cost"
    with pytest.raises(ValueError, match="agricultural_building"):
        CoolingCostContext(
            country="USA", asset_type=AssetType.AGRICULTURAL_BUILDING, historic_mean=25
        )


def test_column_bindings_round_trip_but_are_not_resolved_by_numerical_models() -> None:
    @dataclass(frozen=True)
    class Column:
        name: str

    context = InundationContext(region=Column("region"), asset_type=Column("use"))
    restored = ImpactContext.from_json(context.to_json())
    assert restored.to_json() == context.to_json()
    with pytest.raises(ImpactError, match="unresolved_context"):
        Impact.builtin(BuiltinImpact.INUNDATION).resolve(restored)


def test_unknown_calibration_facts_are_not_invented_from_data() -> None:
    requirement = Impact.builtin(BuiltinImpact.HOT_DAYS).requirements
    assert requirement.unestablished_facts
    supplied = EventCount(
        event=Event.HOT_DAY,
        aggregation=Aggregation.COUNT,
        threshold=Threshold(35, unit=TemperatureUnit.CELSIUS),
        duration=Duration(1, unit=TimeUnit.YEAR),
    )
    with pytest.raises(ImpactError, match="calibration_unestablished"):
        requirement.bind(supplied)
    # Numerical resolution is distinct from asserting a calibrated data match.
    assert isinstance(Impact.builtin(BuiltinImpact.HOT_DAYS).resolve(), Impact)


def test_pinned_provenance_rejects_a_different_numerical_definition() -> None:
    resolved = Impact.builtin(BuiltinImpact.INUNDATION).resolve(
        InundationContext(region=FloodRegion.EUROPE, asset_type=AssetType.RESIDENTIAL)
    )
    document = json.loads(resolved.to_json())
    assert set(document["provenance"]) == {"model", "context"}
    assert document["provenance"]["model"]["manifest_sha256"].startswith("sha256:")
    document["model"]["points"][1]["y"] += 0.01
    with pytest.raises(ImpactError, match="provenance_mismatch"):
        Impact.from_json(json.dumps(document))
    reference = json.loads(Impact.builtin(BuiltinImpact.INUNDATION).to_json())
    reference["manifest_sha256"] = reference["manifest_sha256"].removeprefix("sha256:")
    with pytest.raises(ImpactError):
        ImpactReference.from_json(json.dumps(reference))


def test_scientific_numeric_spelling_is_canonical_and_bool_remains_distinct() -> None:
    assert Measurement(fields={"x": 3}) == Measurement(fields={"x": 3.0})
    assert Measurement(fields={"x": False}) != Measurement(fields={"x": 0})
    assert (
        Measurement(fields={"x": -0.0}).to_json()
        == Measurement(fields={"x": 0}).to_json()
    )
    response = Impact.builtin(BuiltinImpact.INUNDATION).resolve(
        InundationContext(region=FloodRegion.EUROPE, asset_type=AssetType.RESIDENTIAL)
    )
    document = json.loads(response.to_json())
    # The JSON schema and numerical model both treat integral number spellings
    # identically; reconstruction returns the trusted canonical definition.
    document["model"]["points"][0]["x"] = 0.0
    reconstructed = Impact.from_json(json.dumps(document))
    assert reconstructed.to_json() == response.to_json()
    assert reconstructed.sha256 == response.sha256


def test_unknown_nested_fields_and_superseded_records_are_rejected() -> None:
    for payload in [
        {
            "kind": "water_surface_elevation",
            "unit": "m",
            "datum": "x",
            "definition": {},
        },
        {
            "kind": "wind_speed",
            "unit": "m/s",
            "averaging_duration": {"value": 3, "unit": "s", "calendar": "gregorian"},
        },
        {
            "kind": "measurement",
            "quantity": "x",
            "unit": "u",
            "fields": {},
            "schema": None,
        },
        {
            "kind": "inundation_context",
            "continent": "Europe",
            "building_type": "residential",
        },
    ]:
        decoder = (
            ImpactContext.from_json
            if payload["kind"].endswith("context")
            else Measurement.from_json
        )
        with pytest.raises((ValueError, TypeError)):
            decoder(json.dumps(payload))

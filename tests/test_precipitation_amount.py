"""Precipitation accumulation and empty-evidence fitting are explicit science."""

import json

import pytest
from crc_framework import (
    Aggregation,
    BuiltinImpact,
    CellContext,
    DepthAboveGround,
    DistributionFamily,
    Duration,
    FitError,
    FitFailureDiagnostic,
    FitRecord,
    Impact,
    ImpactError,
    LengthUnit,
    Measurement,
    PrecipitationAmount,
    PrecipitationRate,
    PrecipitationRateUnit,
    SampleFit,
    SpeedUnit,
    TimeUnit,
    fit_samples,
)


def test_precipitation_amount_is_neither_depth_above_ground_nor_a_rate() -> None:
    amount = PrecipitationAmount(
        unit=LengthUnit.MILLIMETRE, accumulation_duration=Duration(1, unit=TimeUnit.DAY)
    )
    assert amount.quantity == "precipitation_amount"
    assert amount != DepthAboveGround(unit=LengthUnit.MILLIMETRE)
    assert Measurement.from_json(amount.to_json()) == amount
    assert json.loads(amount.to_json()) == {
        "kind": "precipitation_amount",
        "unit": "mm",
        "accumulation_duration": {"value": 1, "unit": "d"},
    }


def test_amount_requires_real_length_units_and_an_explicit_interval() -> None:
    with pytest.raises(TypeError):
        PrecipitationAmount(
            unit=SpeedUnit.METRES_PER_SECOND,  # type: ignore[arg-type]
            accumulation_duration=Duration(1, unit=TimeUnit.HOUR),
        )
    with pytest.raises(ValueError):
        Measurement.from_json('{"kind":"precipitation_amount","unit":"mm"}')


def test_no_usable_samples_produces_an_authoritative_failed_fit_record() -> None:
    spec = SampleFit(family=DistributionFamily.GUMBEL_RIGHT)
    with pytest.raises(FitError) as raised:
        fit_samples([], spec)
    record = raised.value.record
    assert not record.converged and record.value_interval is None
    assert record.evidence_count == record.used_count == record.excluded_count == 0
    assert record.attempts[0].specification == spec
    diagnostic = record.attempts[0].diagnostic
    assert isinstance(diagnostic, FitFailureDiagnostic)
    assert "requires at least" in diagnostic.message
    assert FitRecord.from_json(record.to_json()) == record


def test_maximum_daily_precipitation_model_requires_a_one_day_amount() -> None:
    reference = Impact.builtin(BuiltinImpact.MAXIMUM_DAILY_PRECIPITATION)
    amount = PrecipitationAmount(
        unit=LengthUnit.MILLIMETRE, accumulation_duration=Duration(1, unit=TimeUnit.DAY)
    )
    requirements = reference.requirements
    assert requirements.measurement_type is PrecipitationAmount
    assert requirements.input == amount
    assert requirements.accepted_units == (LengthUnit.MILLIMETRE,)
    with pytest.raises(ImpactError) as wrong_type:
        requirements.bind(
            PrecipitationRate(
                unit=PrecipitationRateUnit.MILLIMETRES_PER_DAY,
                accumulation_duration=Duration(1, unit=TimeUnit.DAY),
                aggregation=Aggregation.MAXIMUM,
            )
        )
    assert wrong_type.value.code == "measurement_type_mismatch"
    # Source tables do not establish their calibration period. Correct units
    # and quantity do not manufacture that missing model-owned fact.
    with pytest.raises(ImpactError) as missing_calibration:
        requirements.bind(amount)
    assert missing_calibration.value.code == "calibration_unestablished"
    resolved = reference.resolve(CellContext(cell=594475159402840063))
    assert resolved.input == amount
    assert 0 <= resolved.evaluate(100.0) <= 1

"""Persisted fitting specifications and records remain truthful and strict."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from crc_framework import (
    CensoredQuantileFit,
    Distribution,
    DistributionFamily,
    FitConstraints,
    FitError,
    FitRecord,
    FitSpec,
    HurdleQuantileFit,
    ModelSelection,
    QuantileCurve,
    QuantileFit,
    SampleEstimator,
    SampleFit,
    SelectionCriterion,
    _core,
    fit_quantiles,
    fit_samples,
)
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(
    (
        ROOT / "python/crc_framework/contracts/numerical/fitting-1.0.schema.json"
    ).read_text()
)
REGISTRY = Registry().with_resource(SCHEMA["$id"], Resource.from_contents(SCHEMA))
RECORD_VALIDATOR = Draft202012Validator(SCHEMA, registry=REGISTRY)
SPEC_VALIDATOR = Draft202012Validator(
    {"$ref": SCHEMA["$id"] + "#/$defs/specification"}, registry=REGISTRY
)


def specifications():
    family = DistributionFamily.GUMBEL_RIGHT
    return [
        SampleFit(
            family=family,
            estimator=SampleEstimator.L_MOMENTS,
            fallbacks=(SampleFit(family=family),),
        ),
        QuantileFit(
            family=family,
            weights=(1, 2, 3, 4),
            constraints=FitConstraints(probability=0.99, maximum_value=100),
        ),
        CensoredQuantileFit(
            family=DistributionFamily.GENERALIZED_EXTREME_VALUE, censor_location=0
        ),
        HurdleQuantileFit(family=family, atom_probability=0.25),
        ModelSelection(
            candidates=(
                SampleFit(family=family),
                SampleFit(family=DistributionFamily.GUMBEL_LEFT),
            )
        ),
        ModelSelection(
            candidates=(
                QuantileFit(family=family),
                QuantileFit(family=DistributionFamily.GENERALIZED_EXTREME_VALUE),
            ),
            criterion=SelectionCriterion.NORMALIZED_ROOT_MEAN_SQUARE_ERROR,
        ),
    ]


@pytest.mark.parametrize("specification", specifications(), ids=lambda s: s.kind.value)
def test_specs_round_trip_through_schema_and_generic_decoder(specification):
    document = json.loads(specification.to_json())
    assert SPEC_VALIDATOR.is_valid(document)
    restored = FitSpec.from_json(specification.to_json())
    assert type(restored) is type(specification)
    assert restored.to_json() == specification.to_json()
    concrete = type(specification).from_json(specification.to_json())
    assert type(concrete) is type(specification)


@pytest.mark.parametrize(
    "factory",
    [SampleFit, QuantileFit, CensoredQuantileFit, HurdleQuantileFit, ModelSelection],
)
def test_concrete_decoder_rejects_another_discriminator(factory):
    other = next(s for s in specifications() if type(s) is not factory)
    with pytest.raises(ValueError):
        factory.from_json(other.to_json())


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("constraints",),
        ("fallbacks", 0),
        ("fallbacks", 0, "constraints"),
    ],
)
def test_specs_reject_unknown_nested_fields(path):
    sample = SampleFit(
        family=DistributionFamily.GUMBEL_RIGHT,
        constraints=FitConstraints(probability=0.99),
        fallbacks=(
            SampleFit(
                family=DistributionFamily.GUMBEL_RIGHT,
                constraints=FitConstraints(probability=0.9),
            ),
        ),
    )
    document = json.loads(sample.to_json())
    node = document
    for part in path:
        node = node[part]
    node["obsolete"] = True
    assert not SPEC_VALIDATOR.is_valid(document)
    with pytest.raises(ValueError):
        FitSpec.from_json(json.dumps(document))


def sample_record():
    return fit_samples(
        list(range(1, 9)),
        SampleFit(
            family=DistributionFamily.GUMBEL_RIGHT, estimator=SampleEstimator.L_MOMENTS
        ),
    ).record


def quantile_record():
    law = Distribution.parametric(DistributionFamily.GUMBEL_RIGHT, location=0, scale=1)
    q = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    curve = QuantileCurve(q, np.asarray(law.quantile(q)))
    return fit_quantiles(
        curve,
        QuantileFit(family=DistributionFamily.GUMBEL_RIGHT, weights=(1, 0, 2, 3, 4, 5)),
    ).record


@pytest.mark.parametrize("make_record", [sample_record, quantile_record])
def test_successful_records_round_trip_with_exact_attempt_and_weight_identity(
    make_record,
):
    record = make_record()
    assert RECORD_VALIDATOR.is_valid(json.loads(record.to_json()))
    restored = FitRecord.from_json(record.to_json())
    assert restored.to_json() == record.to_json()
    assert restored.selected_attempt == record.selected_attempt
    assert (
        restored.attempts[restored.selected_attempt].specification.to_json()
        == record.attempts[record.selected_attempt].specification.to_json()
    )
    assert restored.used_count + restored.excluded_count == restored.evidence_count


def test_failed_record_round_trips_without_success_or_actual_estimator():
    requested = SampleFit(family=DistributionFamily.GUMBEL_RIGHT)
    with pytest.raises(FitError) as caught:
        fit_samples([2] * 8, requested)
    record = caught.value.record
    restored = FitRecord.from_json(record.to_json())
    assert not restored.converged
    assert restored.selected_attempt is None and restored.estimator is None
    assert restored.to_json() == record.to_json()


def test_fallback_record_identifies_the_explicitly_executed_alternative():
    fallback = SampleFit(
        family=DistributionFamily.GUMBEL_RIGHT, estimator=SampleEstimator.L_MOMENTS
    )
    requested = SampleFit(
        family=DistributionFamily.GUMBEL_RIGHT,
        minimum_samples=20,
        fallbacks=(fallback,),
    )
    record = fit_samples(list(range(1, 9)), requested).record
    restored = FitRecord.from_json(record.to_json())
    assert restored.selected_attempt == 1
    assert restored.attempts[1].specification.to_json() == fallback.to_json()
    assert restored.estimator is SampleEstimator.L_MOMENTS


def test_same_family_candidates_remain_disambiguated_by_selected_attempt():
    family = DistributionFamily.GUMBEL_RIGHT
    candidates = (
        SampleFit(family=family, estimator=SampleEstimator.L_MOMENTS),
        SampleFit(
            family=family,
            estimator=SampleEstimator.L_MOMENTS,
            constraints=FitConstraints(probability=0.99, maximum_value=100),
        ),
    )
    requested = ModelSelection(candidates=candidates)
    record = fit_samples(list(range(1, 9)), requested).record
    restored = FitRecord.from_json(record.to_json())
    assert restored.specification.to_json() == requested.to_json()
    assert len(restored.attempts) == 2
    assert (
        restored.attempts[0].specification.to_json()
        != restored.attempts[1].specification.to_json()
    )
    assert (
        restored.attempts[restored.selected_attempt].specification.to_json()
        == record.attempts[record.selected_attempt].specification.to_json()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("selected_attempt", 99),
        ("selected_attempt", True),
        ("selected_attempt", None),
        ("converged", False),
        ("converged", 1),
        ("estimator", "mle"),
        ("evidence_count", 9),
        ("used_count", -1),
        ("excluded_count", 3),
        ("weights", [1]),
        ("weights", [0.2] * 8),
        ("weights", [None] * 8),
        ("evidence_support", [2, 1]),
        ("evidence_support", [1]),
        ("attempts", []),
    ],
)
def test_record_rejects_contradictory_counts_selection_and_estimator(field, value):
    document = json.loads(sample_record().to_json())
    document[field] = value
    with pytest.raises((TypeError, ValueError)):
        FitRecord.from_json(json.dumps(document))


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("attempts", 0),
        ("attempts", 0, "diagnostic"),
        ("attempts", 0, "specification"),
        ("specification",),
    ],
)
def test_record_rejects_unknown_fields_at_every_declared_boundary(path):
    document = json.loads(sample_record().to_json())
    node = document
    for part in path:
        node = node[part]
    node["obsolete"] = True
    assert not RECORD_VALIDATOR.is_valid(document)
    with pytest.raises(ValueError):
        FitRecord.from_json(json.dumps(document))


def test_attempts_must_come_from_the_requested_candidate_or_explicit_fallback_graph():
    document = json.loads(sample_record().to_json())
    document["attempts"][0]["specification"]["family"] = "gumbel_l"
    assert RECORD_VALIDATOR.is_valid(document)
    with pytest.raises(ValueError):
        FitRecord.from_json(json.dumps(document))


def test_successful_attempt_cannot_carry_a_failure_reason():
    document = json.loads(sample_record().to_json())
    document["attempts"][0]["reason"] = "optimizer failed"
    with pytest.raises(ValueError):
        FitRecord.from_json(json.dumps(document))


def test_failed_record_cannot_hide_a_converged_attempt_without_selection():
    document = json.loads(sample_record().to_json())
    document.update(selected_attempt=None, converged=False, estimator=None)
    with pytest.raises(ValueError):
        FitRecord.from_json(json.dumps(document))


def test_quantile_record_requires_probability_support_and_effective_weights():
    document = json.loads(quantile_record().to_json())
    for field, value in [
        ("evidence_support", [-1, 2]),
        ("evidence_support", [0.5, 0.5]),
        ("weights", None),
    ]:
        malformed = deepcopy(document)
        malformed[field] = value
        with pytest.raises(ValueError):
            FitRecord.from_json(json.dumps(malformed))


def test_effective_weights_match_the_selected_quantile_specification():
    document = json.loads(quantile_record().to_json())
    _core.validate_document(SCHEMA["$id"], json.dumps(document))
    document["weights"][0], document["weights"][-1] = (
        document["weights"][-1],
        document["weights"][0],
    )
    assert RECORD_VALIDATOR.is_valid(document)
    with pytest.raises(ValueError, match="effective weights contradict"):
        FitRecord.from_json(json.dumps(document))
    with pytest.raises(ValueError, match="effective weights contradict"):
        _core.validate_document(SCHEMA["$id"], json.dumps(document))


def test_failed_quantile_record_still_validates_executed_weights():
    with pytest.raises(FitError) as caught:
        fit_quantiles(
            QuantileCurve([0.1, 0.3, 0.5, 0.7, 0.9], [1, 2, 3, 4, 5]),
            QuantileFit(
                family=DistributionFamily.GUMBEL_RIGHT,
                weights=(1, 2, 3, 4, 5),
                constraints=FitConstraints(minimum_value=1, maximum_value=2),
            ),
        )
    record = caught.value.record
    assert FitRecord.from_json(record.to_json()) == record
    document = json.loads(record.to_json())
    document["weights"].reverse()
    with pytest.raises(ValueError, match="effective weights contradict"):
        FitRecord.from_json(json.dumps(document))


@pytest.mark.parametrize(
    "diagnostic",
    [None, {}, {"message": "fit failed"}, {"ks_statistic": 0.25}],
)
def test_attempts_require_their_outcome_diagnostic(diagnostic):
    document = json.loads(sample_record().to_json())
    document["attempts"][0]["diagnostic"] = diagnostic
    assert not RECORD_VALIDATOR.is_valid(document)
    with pytest.raises(ValueError):
        FitRecord.from_json(json.dumps(document))
    with pytest.raises(ValueError):
        _core.validate_document(SCHEMA["$id"], json.dumps(document))


def test_model_selection_record_must_name_the_first_best_converged_attempt():
    candidates = tuple(
        SampleFit(family=family, estimator=SampleEstimator.L_MOMENTS)
        for family in (DistributionFamily.GUMBEL_RIGHT, DistributionFamily.GUMBEL_LEFT)
    )
    samples = [1, 1.1, 1.2, 1.3, 2, 3, 10]
    record = fit_samples(samples, ModelSelection(candidates=candidates)).record
    assert record.selected_attempt == 0
    document = json.loads(record.to_json())
    document["selected_attempt"] = 1
    assert RECORD_VALIDATOR.is_valid(document)
    with pytest.raises(ValueError, match="selection criterion"):
        FitRecord.from_json(json.dumps(document))
    with pytest.raises(ValueError, match="selection criterion"):
        _core.validate_document(SCHEMA["$id"], json.dumps(document))


@pytest.mark.parametrize(
    "make_record,field",
    [(sample_record, "minimum_samples"), (quantile_record, "minimum_knots")],
)
def test_record_cannot_claim_convergence_without_the_required_evidence(
    make_record, field
):
    document = json.loads(make_record().to_json())
    document["specification"][field] = document["attempts"][0]["specification"][
        field
    ] = 100
    assert RECORD_VALIDATOR.is_valid(document)
    with pytest.raises(ValueError, match="less than its required evidence"):
        FitRecord.from_json(json.dumps(document))
    with pytest.raises(ValueError, match="less than its required evidence"):
        _core.validate_document(SCHEMA["$id"], json.dumps(document))


def test_sample_record_uses_available_observations_after_exclusion():
    document = json.loads(sample_record().to_json())
    document.update(evidence_count=9, excluded_count=1)
    assert FitRecord.from_json(json.dumps(document)).used_count == 8

    document.update(used_count=2, excluded_count=7)
    with pytest.raises(ValueError, match="less than its required evidence"):
        FitRecord.from_json(json.dumps(document))


def test_sample_record_cannot_claim_quantile_weights():
    document = json.loads(sample_record().to_json())
    document["weights"] = [1 / 8] * 8
    with pytest.raises(ValueError):
        FitRecord.from_json(json.dumps(document))


def test_integral_json_number_counts_are_hydrated_without_truncation():
    document = json.loads(sample_record().to_json())
    for field in ("selected_attempt", "evidence_count", "used_count", "excluded_count"):
        document[field] = float(document[field])
    document["specification"]["minimum_samples"] = 3.0
    document["attempts"][0]["specification"]["minimum_samples"] = 3.0
    document["attempts"][0]["diagnostic"]["iterations"] = 0.0
    assert RECORD_VALIDATOR.is_valid(document)
    restored = FitRecord.from_json(json.dumps(document))
    assert type(restored.evidence_count) is int
    assert type(restored.selected_attempt) is int
    assert restored.specification.minimum_samples == 3
    assert type(restored.attempts[0].diagnostic.iterations) is int
    document["used_count"] = 7.5
    with pytest.raises(ValueError):
        FitRecord.from_json(json.dumps(document))


def test_fallback_cannot_be_recorded_after_its_parent_already_converged():
    fallback = SampleFit(
        family=DistributionFamily.GUMBEL_LEFT, estimator=SampleEstimator.L_MOMENTS
    )
    requested = SampleFit(
        family=DistributionFamily.GUMBEL_RIGHT,
        estimator=SampleEstimator.L_MOMENTS,
        fallbacks=(fallback,),
    )
    document = json.loads(fit_samples(list(range(1, 9)), requested).record.to_json())
    extra = deepcopy(document["attempts"][0])
    extra["specification"] = json.loads(fallback.to_json())
    document["attempts"].append(extra)
    with pytest.raises(ValueError):
        FitRecord.from_json(json.dumps(document))


def test_model_selection_cannot_claim_candidates_that_were_never_attempted():
    candidates = (
        SampleFit(
            family=DistributionFamily.GUMBEL_RIGHT, estimator=SampleEstimator.L_MOMENTS
        ),
        SampleFit(
            family=DistributionFamily.GUMBEL_LEFT, estimator=SampleEstimator.L_MOMENTS
        ),
    )
    document = json.loads(sample_record().to_json())
    document["specification"] = json.loads(
        ModelSelection(candidates=candidates).to_json()
    )
    with pytest.raises(ValueError):
        FitRecord.from_json(json.dumps(document))

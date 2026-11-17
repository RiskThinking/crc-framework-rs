"""Scientific estimator behavior and auditable execution outcomes."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from crc_framework import (
    CensoredQuantileFit,
    Distribution,
    DistributionFamily,
    FitConstraints,
    FitError,
    FitStatus,
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
from crc_framework._diagnostics import ScientificValue


def _number(value: ScientificValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


@pytest.mark.parametrize(
    ("family", "shape"),
    [
        (DistributionFamily.GENERALIZED_EXTREME_VALUE, 0.1),
        (DistributionFamily.WEIBULL_MINIMUM, 2.2),
        (DistributionFamily.WEIBULL_MAXIMUM, 1.7),
        (DistributionFamily.SKEW_NORMAL, 1.5),
        (DistributionFamily.GUMBEL_RIGHT, None),
        (DistributionFamily.GUMBEL_LEFT, None),
        (DistributionFamily.GENERALIZED_PARETO, 0.15),
    ],
)
def test_sample_mle_recovers_requested_family_and_reports_actual_optimizer(
    family: DistributionFamily, shape: float | None
) -> None:
    law = Distribution.parametric(family, location=2.0, scale=3.0, shape=shape)
    observations = np.asarray(
        law.quantile((np.arange(400) + 0.5) / 400), dtype=np.float64
    )
    specification = SampleFit(family=family)
    result = fit_samples(observations, specification)
    np.testing.assert_allclose(
        result.distribution.quantile([0.1, 0.5, 0.9]),
        law.quantile([0.1, 0.5, 0.9]),
        atol=0.04,
    )
    assert result.record.specification is specification
    assert result.record.estimator is SampleEstimator.MAXIMUM_LIKELIHOOD
    assert result.record.converged
    assert result.record.evidence_count == 400
    assert result.record.used_count == 400
    assert result.record.excluded_count == 0
    assert result.record.weights is None
    assert _number(result.diagnostics[0].details["iterations"]) > 0
    assert _number(result.diagnostics[0].details["evaluations"]) > 0
    assert result.record.attempts[0].status is FitStatus.CONVERGED


def test_fixed_family_constraints_are_fitted_inside_requested_family() -> None:
    samples = [-0.7, 0.0, 0.2, 0.9, 1.4, 2.0, 2.7, 3.0]
    specification = SampleFit(
        family=DistributionFamily.GUMBEL_RIGHT,
        constraints=FitConstraints(probability=0.99, maximum_value=4.0),
    )
    result = fit_samples(samples, specification)
    assert result.distribution.quantile(0.99) <= 4.0 + 1e-10
    assert len(result.record.attempts) == 1
    assert result.record.attempts[0].specification is specification
    assert (
        json.loads(result.distribution.to_json())["family"]
        == DistributionFamily.GUMBEL_RIGHT.value
    )


def test_failed_mle_does_not_masquerade_as_moments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise ValueError("likelihood optimization did not converge")

    monkeypatch.setattr(_core, "fit_samples", fail)
    specification = SampleFit(family=DistributionFamily.GUMBEL_RIGHT)
    with pytest.raises(FitError) as caught:
        fit_samples([0.1, 0.2, 0.5, 1.2, 2.1], specification)
    error = caught.value
    assert error.code == "fit.no_converged_candidate"
    assert error.record.specification is specification
    assert error.record.estimator is None
    assert error.record.converged is False
    assert error.record.selected_attempt is None
    assert len(error.record.attempts) == 1
    assert error.record.attempts[0].status is FitStatus.FAILED
    assert "did not converge" in error.diagnostics[0].message


def test_only_explicit_alternative_estimator_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = _core.fit_samples
    calls: list[str] = []

    def likelihood_fails(
        samples: list[float], family: str, estimator: str, **options: object
    ) -> _core.NativeFitResult:
        calls.append(estimator)
        if estimator == "mle":
            raise ValueError("deliberate nonconvergence")
        return actual(samples, family, estimator)

    monkeypatch.setattr(_core, "fit_samples", likelihood_fails)
    alternative = SampleFit(
        family=DistributionFamily.GUMBEL_RIGHT, estimator=SampleEstimator.L_MOMENTS
    )
    requested = SampleFit(
        family=DistributionFamily.GUMBEL_RIGHT, fallbacks=(alternative,)
    )
    result = fit_samples([0.1, 0.2, 0.5, 1.2, 2.1], requested)
    assert calls == ["mle", "lmoments"]
    assert result.record.specification is requested
    assert result.record.estimator is SampleEstimator.L_MOMENTS
    assert [attempt.status for attempt in result.record.attempts] == [
        FitStatus.FAILED,
        FitStatus.CONVERGED,
    ]
    assert result.record.attempts[1].specification is alternative


def test_sample_selection_has_only_the_requested_candidates() -> None:
    specifications = (
        SampleFit(
            family=DistributionFamily.GUMBEL_RIGHT, estimator=SampleEstimator.L_MOMENTS
        ),
        SampleFit(
            family=DistributionFamily.GUMBEL_LEFT, estimator=SampleEstimator.L_MOMENTS
        ),
    )
    result = fit_samples(
        [0.1, 0.2, 0.5, 1.2, 2.1, 5.0], ModelSelection(candidates=specifications)
    )
    assert (
        tuple(attempt.specification for attempt in result.record.attempts)
        == specifications
    )
    assert result.record.converged


def test_quantile_fit_recovers_values_and_reports_effective_weights() -> None:
    law = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=2.0, scale=3.0
    )
    probabilities = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    curve = QuantileCurve(
        probabilities, np.asarray(law.quantile(probabilities), dtype=np.float64)
    )
    specification = QuantileFit(
        family=DistributionFamily.GUMBEL_RIGHT, weights=(1.0, 0.0, 2.0, 3.0, 4.0, 5.0)
    )
    result = fit_quantiles(curve, specification)
    np.testing.assert_allclose(
        result.distribution.quantile(probabilities), curve.values, atol=1e-6
    )
    assert result.record.evidence_count == 6
    assert result.record.used_count == 5
    assert result.record.excluded_count == 1
    assert result.record.weights == pytest.approx(
        (1 / 15, 0, 2 / 15, 3 / 15, 4 / 15, 5 / 15)
    )
    assert result.record.evidence_support == (0.1, 0.99)
    assert result.record.estimator is None
    assert "ks_pvalue" not in result.diagnostics[0].details


def test_quantile_constraint_refines_only_one_requested_law() -> None:
    probabilities = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    law = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=0.0, scale=1.0
    )
    curve = QuantileCurve(
        probabilities, np.asarray(law.quantile(probabilities), dtype=np.float64)
    )
    result = fit_quantiles(
        curve,
        QuantileFit(
            family=DistributionFamily.GUMBEL_RIGHT,
            constraints=FitConstraints(probability=0.99, maximum_value=4.5),
        ),
    )
    assert result.distribution.quantile(0.99) <= 4.5 + 1e-10
    assert len(result.record.attempts) == 1


def test_censored_quantile_fit_has_operation_specific_diagnostics() -> None:
    curve = QuantileCurve(
        [0.5, 0.8, 0.9, 0.96, 0.98, 0.99, 0.996, 0.998, 0.999],
        [0, 0, 0, 0, 0, 0, 0.03, 0.06, 0.10],
    )
    specification = CensoredQuantileFit(
        family=DistributionFamily.GENERALIZED_EXTREME_VALUE, censor_location=0.0
    )
    result = fit_quantiles(curve, specification)
    details = result.diagnostics[0].details
    assert details["censored_count"] == 6
    assert details["uncensored_count"] == 3
    assert _number(details["residual_scale"]) >= _number(
        details["residual_scale_floor"]
    )
    assert "negative_log_likelihood" in details
    assert "atom_probability" not in details


def test_hurdle_quantile_fit_reports_atom_and_true_tail() -> None:
    source = Distribution.hurdle(
        Distribution.parametric(
            DistributionFamily.GUMBEL_RIGHT, location=0.0, scale=1.0
        ),
        atom_probability=0.6,
    )
    probabilities = np.array([0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    curve = QuantileCurve(
        probabilities, np.asarray(source.quantile(probabilities), dtype=np.float64)
    )
    result = fit_quantiles(
        curve,
        HurdleQuantileFit(family=DistributionFamily.GUMBEL_RIGHT, atom_probability=0.6),
    )
    np.testing.assert_allclose(
        result.distribution.quantile(probabilities), curve.values, atol=1e-6
    )
    assert result.distribution.mass(0.0) == pytest.approx(0.6)
    assert result.diagnostics[0].details["atom_count"] == 4
    assert result.diagnostics[0].details["tail_count"] == 5


def test_scientific_inputs_and_choices_are_explicit() -> None:
    with pytest.raises(TypeError):
        SampleFit(family="gumbel_r")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SampleFit(family=DistributionFamily.GUMBEL_RIGHT, estimator="mle")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ModelSelection(candidates=())
    with pytest.raises(ValueError):
        FitConstraints(probability=1.1)
    with pytest.raises(TypeError):
        fit_samples(
            QuantileCurve([0.1, 0.9], [1.0, 2.0]),  # type: ignore[arg-type]
            SampleFit(family=DistributionFamily.GUMBEL_RIGHT),
        )
    with pytest.raises(TypeError):
        fit_quantiles(
            QuantileCurve([0.1, 0.3, 0.6, 0.9], [1.0, 2.0, 3.0, 4.0]),
            ModelSelection(
                candidates=(SampleFit(family=DistributionFamily.GUMBEL_RIGHT),)
            ),
        )
    specification = QuantileFit(family=DistributionFamily.GUMBEL_RIGHT)
    with pytest.raises(FrozenInstanceError):
        specification.family = DistributionFamily.GUMBEL_LEFT  # type: ignore[misc]


def test_endpoint_evidence_is_never_silently_discarded() -> None:
    curve = QuantileCurve(
        [0.0, 0.1, 0.3, 0.5, 0.9, 1.0], [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    )
    with pytest.raises(FitError) as caught:
        fit_quantiles(curve, QuantileFit(family=DistributionFamily.GUMBEL_RIGHT))
    assert caught.value.record.evidence_count == 6
    assert (
        "endpoint evidence cannot be silently excluded"
        in caught.value.diagnostics[0].message
    )


def test_specification_json_has_one_discriminator_and_canonical_enum_values() -> None:
    specification = ModelSelection(
        candidates=(
            SampleFit(
                family=DistributionFamily.GENERALIZED_EXTREME_VALUE,
                estimator=SampleEstimator.L_MOMENTS,
            ),
            SampleFit(family=DistributionFamily.GUMBEL_RIGHT),
        )
    )
    payload = json.loads(specification.to_json())
    assert payload["kind"] == "model_selection"
    assert payload["criterion"] == "ks"
    assert payload["candidates"][0]["kind"] == "samples"
    assert payload["candidates"][0]["family"] == "genextreme"
    assert payload["candidates"][0]["estimator"] == "lmoments"


def test_quantile_model_selection_uses_its_explicit_evidence_criterion() -> None:
    probabilities = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    source = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=2.0, scale=3.0
    )
    curve = QuantileCurve(
        probabilities, np.asarray(source.quantile(probabilities), dtype=np.float64)
    )
    candidates = (
        QuantileFit(family=DistributionFamily.GUMBEL_LEFT),
        QuantileFit(family=DistributionFamily.GUMBEL_RIGHT),
    )
    with pytest.raises(ValueError, match="NORMALIZED_ROOT_MEAN_SQUARE_ERROR"):
        ModelSelection(candidates=candidates)
    specification = ModelSelection(
        candidates=candidates,
        criterion=SelectionCriterion.NORMALIZED_ROOT_MEAN_SQUARE_ERROR,
    )
    result = fit_quantiles(curve, specification)
    np.testing.assert_allclose(
        result.distribution.quantile(probabilities), curve.values, atol=1e-6
    )
    assert (
        tuple(attempt.specification for attempt in result.record.attempts) == candidates
    )
    assert _number(result.diagnostics[0].details["normalized_rmse"]) < 1e-6
    assert "ks_statistic" not in result.diagnostics[0].details
    with pytest.raises(ValueError, match="KOLMOGOROV_SMIRNOV"):
        ModelSelection(
            candidates=(SampleFit(family=DistributionFamily.GUMBEL_RIGHT),),
            criterion=SelectionCriterion.NORMALIZED_ROOT_MEAN_SQUARE_ERROR,
        )


def test_zero_weight_endpoints_are_explicit_exclusions() -> None:
    source = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=2.0, scale=3.0
    )
    interior = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    interior_values = np.asarray(source.quantile(interior), dtype=np.float64)
    curve = QuantileCurve(
        np.concatenate(([0.0], interior, [1.0])),
        np.concatenate(([-100.0], interior_values, [100.0])),
    )
    specification = QuantileFit(
        family=DistributionFamily.GUMBEL_RIGHT,
        weights=(0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
    )
    result = fit_quantiles(curve, specification)
    np.testing.assert_allclose(
        result.distribution.quantile(interior), interior_values, atol=1e-6
    )
    assert result.record.evidence_count == 8
    assert result.record.used_count == 6
    assert result.record.excluded_count == 2
    assert result.record.weights == pytest.approx(
        (0.0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 0.0)
    )
    assert result.record.evidence_support == (0.0, 1.0)
    assert result.record.selected_attempt == 0


def test_same_family_selection_identifies_the_selected_attempt() -> None:
    source = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=2.0, scale=3.0
    )
    probabilities = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    curve = QuantileCurve(
        probabilities, np.asarray(source.quantile(probabilities), dtype=np.float64)
    )
    constrained = QuantileFit(
        family=DistributionFamily.GUMBEL_RIGHT,
        constraints=FitConstraints(probability=0.99, maximum_value=10.0),
    )
    unconstrained = QuantileFit(family=DistributionFamily.GUMBEL_RIGHT)
    selection = ModelSelection(
        candidates=(constrained, unconstrained),
        criterion=SelectionCriterion.NORMALIZED_ROOT_MEAN_SQUARE_ERROR,
    )
    result = fit_quantiles(curve, selection)
    assert all(
        attempt.status is FitStatus.CONVERGED for attempt in result.record.attempts
    )
    assert result.record.selected_attempt == 1
    assert (
        result.record.attempts[result.record.selected_attempt].specification
        is unconstrained
    )
    np.testing.assert_allclose(
        result.distribution.quantile(probabilities), curve.values, atol=1e-6
    )


def test_effective_weights_are_invariant_to_a_large_common_scale() -> None:
    source = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=2.0, scale=3.0
    )
    probabilities = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    curve = QuantileCurve(
        probabilities, np.asarray(source.quantile(probabilities), dtype=np.float64)
    )
    result = fit_quantiles(
        curve, QuantileFit(family=DistributionFamily.GUMBEL_RIGHT, weights=(1e308,) * 6)
    )
    assert result.record.weights == pytest.approx((1 / 6,) * 6)
    np.testing.assert_allclose(
        result.distribution.quantile(probabilities), curve.values, atol=1e-6
    )


def test_rejected_lmoment_constraints_and_evidence_counts_are_truthful() -> None:
    specification = SampleFit(
        family=DistributionFamily.GUMBEL_RIGHT,
        estimator=SampleEstimator.L_MOMENTS,
        constraints=FitConstraints(probability=0.5, maximum_value=-100.0),
    )
    with pytest.raises(FitError) as caught:
        fit_samples([0.1, 0.2, 0.5, 1.2, 2.1], specification)
    assert caught.value.record.attempts[0].status is FitStatus.REJECTED
    assert caught.value.record.used_count == 5
    assert caught.value.record.selected_attempt is None
    with pytest.raises(FitError) as insufficient:
        fit_samples(
            [0.1, 0.2, 0.5],
            SampleFit(family=DistributionFamily.GUMBEL_RIGHT, minimum_samples=4),
        )
    assert insufficient.value.record.evidence_count == 3
    assert insufficient.value.record.used_count == 0

"""Public insurance terms and empirical annual-loss workflows."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import permutations
from math import inf, nextafter

import numpy as np
import pytest
from crc_framework import (
    Distribution,
    PolicyTerms,
    Statistic,
    Tail,
    summarize_annual_losses,
)


def test_policy_terms_apply_to_scalar_or_array_damage_ratios() -> None:
    terms = PolicyTerms(deductible=5_000, limit=15_000, share=0.8)
    assert terms.insured_loss(100_000, 0.26) == 12_000
    ratios = np.array([[0.0, 0.26], [0.05, 0.15]])[:, ::-1]
    losses = terms.insured_loss(100_000, ratios)
    assert isinstance(losses, np.ndarray)
    np.testing.assert_array_equal(losses, [[12_000, 0], [8_000, 0]])
    assert not losses.flags.writeable
    assert PolicyTerms().insured_loss(100_000, 0.26) == 26_000
    assert PolicyTerms(limit=0).insured_loss(100_000, 1) == 0
    assert PolicyTerms(share=0).insured_loss(100_000, 1) == 0
    with pytest.raises(ValueError, match="damage ratio"):
        terms.insured_loss(100_000, [0.1, 1.2])
    with pytest.raises(ValueError, match="deductible"):
        PolicyTerms(deductible=-1)


def test_annual_loss_summary_uses_the_empirical_law_and_exact_tail_mass() -> None:
    first = [0, 0, 0, 10, 0, 20, 0, 20]
    second = [0, 0, 0, 0, 10, 10, 30, 30]
    totals = [a + b for a, b in zip(first, second, strict=True)]
    summary = summarize_annual_losses(
        totals,
        total_insured_value=500,
        levels=[0.8125, 0.99],
        return_periods=[100],
        components={"a": first, "b": second},
    )
    law = Distribution.empirical(totals)
    stats = law.statistics(
        metrics=[Statistic.MEAN, Statistic.STANDARD_DEVIATION, Statistic.CVAR],
        level=0.8125,
        tail=Tail.UPPER,
    ).values
    assert summary.trial_count == 8
    assert summary.average_annual_loss == stats[Statistic.MEAN].value == 16.25
    assert summary.standard_deviation == stats[Statistic.STANDARD_DEVIATION].value
    assert summary.coefficient_of_variation == pytest.approx(
        summary.standard_deviation / 16.25
    )
    assert summary.loss_cost == pytest.approx(16.25 / 500)
    assert summary.diagnostics == ()
    level = summary.at(0.8125)
    assert level.value_at_risk == law.quantile(0.8125) == 30
    assert level.conditional_value_at_risk == pytest.approx(130 / 3)
    assert level.conditional_value_at_risk == stats[Statistic.CVAR].value
    assert level.nominal_tail_years == 1.5
    assert level.boundary_years == 2
    assert level.boundary_weight == pytest.approx(0.25)
    assert level.contributions == pytest.approx({"a": 50 / 3, "b": 80 / 3})
    assert level.contributions is not None
    with pytest.raises(TypeError):
        level.contributions["a"] = 0.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        level.value_at_risk = 0.0  # type: ignore[misc]
    point = summary.aep[0]
    assert point.return_period == 100
    assert point.loss == summary.at(0.99).value_at_risk == law.quantile(0.99)
    assert point.nominal_exceedance_probability == 0.01
    assert point.exceedance_probability == law.exceedance(point.loss) == 0
    assert point.nominal_tail_years == pytest.approx(0.08)
    with pytest.raises(KeyError):
        summary.at(0.5)

    # The tied 30-loss years differ in composition. Reordering them preserves
    # each component's share of the portfolio tail.
    for fifth, sixth in permutations((5, 6)):
        order = [0, 1, 2, 3, 4, fifth, sixth, 7]
        reordered = summarize_annual_losses(
            [totals[i] for i in order],
            total_insured_value=500,
            levels=[0.8125],
            return_periods=[],
            components={
                "a": [first[i] for i in order],
                "b": [second[i] for i in order],
            },
        )
        assert reordered.at(0.8125).contributions == level.contributions


def test_signed_diversification_and_zero_loss_cost_denominator() -> None:
    first = [10] * 4 + [0] * 96
    second = [0] * 4 + [10] * 4 + [0] * 92
    summary = summarize_annual_losses(
        [a + b for a, b in zip(first, second, strict=True)],
        total_insured_value=20,
        components={"a": first, "b": second},
    )
    assert summary.at(0.95).diversification_benefit == -10
    zero = summarize_annual_losses([0, 0], total_insured_value=0)
    assert zero.average_annual_loss == zero.standard_deviation == 0
    assert zero.coefficient_of_variation == 0
    assert zero.loss_cost is None
    assert [item.code for item in zero.diagnostics] == ["zero_total_insured_value"]
    assert zero.at(0.95).contributions is None
    assert zero.at(0.95).diversification_benefit is None


def test_annual_loss_threshold_at_an_exact_empirical_rank() -> None:
    level = 5 / 6
    below = nextafter(level, 0)
    above = nextafter(level, inf)
    result = summarize_annual_losses(
        [0, 0, 2, 4, 4, 7],
        total_insured_value=100,
        levels=[below, level, above],
        return_periods=[6],
        components={"a": [0, 0, 2, 4, 0, 5], "b": [0, 0, 0, 0, 4, 2]},
    )
    assert [item.value_at_risk for item in result.levels] == [4, 4, 7]
    cutoff = result.at(level)
    assert cutoff.conditional_value_at_risk == 7
    assert cutoff.diversification_benefit == 2
    assert cutoff.boundary_years == 2
    assert cutoff.boundary_weight == 0
    assert cutoff.nominal_tail_years == 1
    assert cutoff.contributions == {"a": 5, "b": 2}
    assert result.at(below).boundary_weight > 0
    (point,) = result.aep
    assert point.loss == 4
    assert point.exceedance_probability == point.nominal_exceedance_probability == 1 / 6
    assert point.nominal_tail_years == 1

    thirds = summarize_annual_losses(
        [0, 1, 2], total_insured_value=10, levels=[2 / 3], return_periods=[3]
    )
    assert thirds.aep[0].loss == thirds.at(2 / 3).value_at_risk == 1
    assert thirds.aep[0].exceedance_probability == 1 / 3


def test_components_must_reconcile_for_each_year() -> None:
    with pytest.raises(ValueError, match="reconcile"):
        summarize_annual_losses(
            [0, 3],
            total_insured_value=10,
            components={"interest": [3, 0]},
        )

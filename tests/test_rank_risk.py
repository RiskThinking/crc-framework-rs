"""Rank labels, independent risk, and reference geography through the public API."""

import math
from dataclasses import FrozenInstanceError

import pytest
from crc_framework import (
    BinaryOutcome,
    Distribution,
    compute_risk,
    lookup_continent,
    lookup_geography,
    lookup_ipcc_region,
    rank_scenarios,
)


def outcome(id: str, probability: float, impact: float) -> BinaryOutcome:
    return BinaryOutcome(
        id,
        lower_rank_mass=1 - probability,
        lower_impact=0,
        upper_rank_mass=probability,
        upper_impact=impact,
    )


def test_atom_thresholds_are_distinct_from_rank_labels() -> None:
    law = Distribution.empirical([0, 0, 0, 1])
    scenarios = rank_scenarios(law, probabilities=[0.1, 0.5, 0.9])
    assert scenarios[1].value == 0
    assert scenarios[1].lower_rank_mass == 0.5
    assert scenarios[1].upper_rank_mass == 0.5
    assert law.cdf(0) == 0.75
    assert law.cdf(0, inclusive=False) == 0
    assert law.mass(0) == 0.75
    assert not hasattr(scenarios[0], "probability_below")
    with pytest.raises(FrozenInstanceError):
        scenarios[0].value = 1  # type: ignore[misc]


def test_endpoint_scenarios_keep_infinite_values() -> None:
    from crc_framework import DistributionFamily

    law = Distribution.parametric(DistributionFamily.GUMBEL_RIGHT, location=0, scale=1)
    scenarios = rank_scenarios(law, probabilities=[0, 1])
    assert scenarios[0].value == -math.inf
    assert scenarios[1].value == math.inf


@pytest.mark.parametrize(
    "probabilities", [[], [0.2, 0.2], [0.5, 0.1], [-0.1], [math.nan], [True]]
)
def test_invalid_rank_grid_is_rejected(probabilities) -> None:
    with pytest.raises(ValueError):
        rank_scenarios(Distribution.point_mass(1), probabilities=probabilities)


def test_risk_retains_rare_mass() -> None:
    result = compute_risk(
        [outcome("custom:catastrophe", 1e-6, 1e6)], levels=[0.95, 0.9999995]
    )
    assert result.at(0.95).value_at_risk == 0
    assert result.at(0.95).conditional_value_at_risk == pytest.approx(20)
    assert result.at(0.9999995).value_at_risk == 1e6
    assert result.branch_count == 2
    with pytest.raises(KeyError):
        result.at(0.951)


def test_contributions_sum_without_implicit_averaging() -> None:
    result = compute_risk(
        [outcome("a", 0.1, 1), outcome("b", 0.2, 2)], levels=[0.5, 0.95]
    )
    level = result.at(0.95)
    assert level.value_at_risk == 2
    assert level.conditional_value_at_risk == pytest.approx(2.4)
    assert sum(item.value_at_risk for item in level.attribution) == level.value_at_risk
    assert sum(
        item.conditional_value_at_risk for item in level.attribution
    ) == pytest.approx(level.conditional_value_at_risk)


def test_tied_attribution_is_independent_of_input_order() -> None:
    a, b = outcome("a", 0.5, 1), outcome("b", 0.5, 1)
    for values in ([a, b], [b, a]):
        level = compute_risk(values, levels=[0.5]).at(0.5)
        assert {part.id: part.value_at_risk for part in level.attribution} == {
            "a": 0.5,
            "b": 0.5,
        }


def test_invalid_ids_and_risk_inputs() -> None:
    for id in ("", " a", "a "):
        with pytest.raises(ValueError):
            outcome(id, 0.1, 1)
    with pytest.raises(ValueError, match="unique"):
        compute_risk([outcome("a", 0.1, 1)] * 2, levels=[0.95])
    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            compute_risk([outcome("a", 0.1, 1)], levels=[0.95], max_branches=limit)
    for levels in ([], [0], [1], [math.nan], [True]):
        with pytest.raises(ValueError):
            compute_risk([outcome("a", 0.1, 1)], levels=levels)


@pytest.mark.parametrize("limit", [None, 2**40, 2**100])
def test_huge_branch_requests_raise_without_allocating(limit) -> None:
    with pytest.raises(ValueError, match="spanning branches"):
        compute_risk(
            [outcome(str(i), 0.1, 1) for i in range(70)],
            levels=[0.95],
            max_branches=limit,
        )


def test_reference_geography_uses_integer_h3_and_no_implicit_defaults() -> None:
    assert lookup_continent(594475228122316799) == "Europe"
    assert lookup_ipcc_region(594479566039285759) == "Greenland/Iceland"
    geography = lookup_geography(600550049193132031)
    assert geography is not None
    assert geography.continent == "Africa"
    assert geography.countries == ("NER",)
    with pytest.raises(TypeError):
        lookup_continent("8400001ffffffff")  # type: ignore[arg-type]

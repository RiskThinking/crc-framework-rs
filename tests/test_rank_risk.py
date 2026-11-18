"""Rank labels, independent risk, and reference geography through the public API."""

import math
from dataclasses import FrozenInstanceError

import numpy as np
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
    "probabilities", [[], [0.2, 0.2], [0.5, 0.1], [-0.1], [math.nan]]
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


def test_var_keeps_the_lower_cutoff_when_prefix_mass_equals_level() -> None:
    a = BinaryOutcome(
        "a", lower_rank_mass=0.5, lower_impact=-1, upper_rank_mass=0.5, upper_impact=1
    )
    above = math.nextafter(0.5, 1)
    for b, c in (("b", "c"), ("c", "b")):
        # Write the reflected mass explicitly so the two joint tails agree.
        down = BinaryOutcome(
            b,
            lower_rank_mass=0.007,
            lower_impact=-2,
            upper_rank_mass=0.993,
            upper_impact=0,
        )
        up = BinaryOutcome(
            c,
            lower_rank_mass=0.993,
            lower_impact=0,
            upper_rank_mass=0.007,
            upper_impact=2,
        )
        result = compute_risk([a, down, up], levels=[0.5, above])
        assert result.at(0.5).value_at_risk == -1
        assert result.at(above).value_at_risk == 1


def test_var_compares_exact_joint_mass_at_rounding_boundary() -> None:
    slow = outcome("large", 0.75, 10)
    small = BinaryOutcome(
        "small",
        lower_rank_mass=0.2,
        lower_impact=0,
        upper_rank_mass=0.8,
        upper_impact=1,
    )
    above = math.nextafter(0.25, 1.0)
    result = compute_risk([slow, small], levels=[0.25, above])
    assert result.at(0.25).value_at_risk == 1
    assert result.at(above).value_at_risk == 10


@pytest.mark.parametrize("q", [1e-20, math.ulp(0.0)])
def test_cvar_retains_small_excluded_prefix_at_large_impact_scale(q: float) -> None:
    symmetric = BinaryOutcome(
        "symmetric",
        lower_rank_mass=0.5,
        lower_impact=-1e308,
        upper_rank_mass=0.5,
        upper_impact=1e308,
    )
    level = compute_risk([symmetric], levels=[q]).at(q)
    expected = 1e308 * q
    assert level.value_at_risk == -1e308
    assert level.conditional_value_at_risk == pytest.approx(expected, rel=1e-15, abs=0)
    assert (
        level.attribution[0].conditional_value_at_risk
        == level.conditional_value_at_risk
    )


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
        assert level.conditional_value_at_risk == 1.5
        assert {
            part.id: part.conditional_value_at_risk for part in level.attribution
        } == {"a": 0.75, "b": 0.75}


def test_impossible_branches_do_not_overflow() -> None:
    result = compute_risk(
        [outcome("a", 0, 1e308), outcome("b", 0, 1e308)], levels=[0.5]
    )
    assert result.branch_count == 4
    assert result.at(0.5).value_at_risk == 0
    assert result.at(0.5).conditional_value_at_risk == 0


@pytest.mark.parametrize(
    "impacts, total",
    [
        ([1e16, 1, -1e16], 1),
        ([1e308, 1e308, -1e308], 1e308),
        ([1e308, 1e308, -1e308, -1e308, math.ulp(0.0)], math.ulp(0.0)),
    ],
)
def test_realizable_branches_sum_stably(impacts: list[float], total: float) -> None:
    values = [outcome(str(i), 1, impact) for i, impact in enumerate(impacts)]
    for ordered in (values, values[::-1]):
        level = compute_risk(ordered, levels=[0.5]).at(0.5)
        assert level.value_at_risk == total
        assert level.conditional_value_at_risk == total


def test_invalid_ids_and_risk_inputs() -> None:
    for id in ("", " a", "a "):
        with pytest.raises(ValueError):
            outcome(id, 0.1, 1)
    with pytest.raises(ValueError, match="unique"):
        compute_risk([outcome("a", 0.1, 1)] * 2, levels=[0.95])
    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            compute_risk([outcome("a", 0.1, 1)], levels=[0.95], max_branches=limit)
    for levels in ([], [0], [1], [math.nan]):
        with pytest.raises(ValueError):
            compute_risk([outcome("a", 0.1, 1)], levels=levels)


@pytest.mark.parametrize("value", [True, np.bool_(True), np.array(0.5), "0.5"])
def test_risk_levels_require_real_scalar_values(value: float) -> None:
    with pytest.raises(TypeError, match="risk level must be a real number"):
        compute_risk([outcome("a", 0.1, 1)], levels=[value])
    with pytest.raises(TypeError, match="probability must be a real number"):
        rank_scenarios(Distribution.point_mass(1), probabilities=[value])


@pytest.mark.parametrize("value", [True, np.bool_(True), "1"])
def test_binary_outcome_rejects_non_numeric_masses_and_impacts(value: object) -> None:
    for field in ("lower_rank_mass", "upper_rank_mass", "lower_impact", "upper_impact"):
        numbers: dict[str, object] = {
            "lower_rank_mass": 0.0,
            "lower_impact": 0.0,
            "upper_rank_mass": 1.0,
            "upper_impact": 1.0,
        }
        numbers[field] = value
        with pytest.raises(TypeError, match=f"{field} must be a real number"):
            BinaryOutcome("custom:outcome", **numbers)  # type: ignore[arg-type]


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

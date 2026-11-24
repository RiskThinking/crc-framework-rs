"""Compare annual loss metrics with pinned Ecofusion v4 spanning-set exports."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import product
from math import fsum, lcm
from pathlib import Path
from typing import Any

import pytest
from crc_framework import (
    BinaryOutcome,
    Distribution,
    Statistic,
    Tail,
    compute_risk,
    summarize_annual_losses,
)

_EXPORT = Path(__file__).with_name("fixtures") / "ecofusion-v4-spanning-set.json"
_CASES = {
    case["name"]: case
    for case in json.loads(_EXPORT.read_text(encoding="utf-8"))["cases"]
}
# Ecofusion emits binary32 probabilities, impacts, and scores.
_EXPORT_REL_TOL = 2e-7


@pytest.mark.parametrize(
    ("name", "tail_contributions", "historical_cvar_delta"),
    [
        (
            "upstream-flattening",
            ((0.02, 0.02), (0.1, 0.02), (0.1, 0.05)),
            (0.02, 0.01, 0.0),
        ),
        (
            "equal-branch-losses",
            ((0.875, 1.625), (2.0, 2.0), (2.0, 2.0)),
            (0.1, 0.0, 0.0),
        ),
    ],
)
def test_annual_losses_match_independent_ecofusion_branches(
    name: str,
    tail_contributions: tuple[tuple[float, ...], ...],
    historical_cvar_delta: tuple[float, ...],
) -> None:
    case = _CASES[name]
    factors = [BinaryOutcome(**factor) for factor in case["factors"]]
    branch_states: list[tuple[bool, ...]] = []
    branch_masses: list[Fraction] = []
    branch_losses: list[tuple[float, ...]] = []
    branches: list[dict[str, Any]] = case["branches"]
    for branch in branches:
        indices = {entry["index"]: entry for entry in branch["indices"]}
        assert indices.keys() == {factor.id for factor in factors}
        states = tuple(indices[factor.id]["is_downside"] for factor in factors)
        mass = Fraction(1)
        losses: list[float] = []
        for factor, downside in zip(factors, states, strict=True):
            probability, impact = (
                (factor.upper_rank_mass, factor.upper_impact)
                if downside
                else (factor.lower_rank_mass, factor.lower_impact)
            )
            assert indices[factor.id]["probability"] == pytest.approx(
                probability, rel=_EXPORT_REL_TOL
            )
            assert indices[factor.id]["impact"] == pytest.approx(
                impact, rel=_EXPORT_REL_TOL
            )
            mass *= Fraction(str(probability))
            losses.append(impact)
        assert branch["joint_probability"] == pytest.approx(
            float(mass), rel=_EXPORT_REL_TOL
        )
        assert branch["total_impact"] == pytest.approx(
            fsum(losses), rel=_EXPORT_REL_TOL
        )
        branch_states.append(states)
        branch_masses.append(mass)
        branch_losses.append(tuple(losses))
    assert sorted(branch_states) == list(product((False, True), repeat=len(factors)))
    assert sorted(row["branch_index"] for row in branches) == list(range(len(branches)))
    assert sum(branch_masses) == 1

    # The nominal input probabilities are exact finite fractions. Repeating a
    # branch by its count preserves its mass and alignment in the annual sample.
    years = lcm(*(mass.denominator for mass in branch_masses))
    annual_rows = [
        losses
        for mass, losses in zip(branch_masses, branch_losses, strict=True)
        for _ in range(int(mass * years))
    ]
    components = {
        factor.id: [row[position] for row in annual_rows]
        for position, factor in enumerate(factors)
    }
    levels = [report["level"] for report in case["historical_levels"]]
    annual = summarize_annual_losses(
        [fsum(row) for row in annual_rows],
        total_insured_value=10,
        levels=levels,
        return_periods=[2, 20, 100],
        components=components,
    )
    rank = compute_risk(factors, levels=levels)
    assert rank.branch_count == len(branches)
    assert annual.trial_count == years
    assert annual.average_annual_loss == pytest.approx(
        case["expected_impact"], rel=_EXPORT_REL_TOL
    )

    exported_masses = [row["joint_probability"] for row in branches]
    exported_total = fsum(exported_masses)
    weighted = Distribution.discrete(
        [row["total_impact"] for row in branches],
        [mass / exported_total for mass in exported_masses],
    )
    for position, legacy in enumerate(case["historical_levels"]):
        level = legacy["level"]
        report = annual.at(level)
        independent = rank.at(level)
        weighted_cvar = (
            weighted.statistics(metrics=[Statistic.CVAR], level=level, tail=Tail.UPPER)
            .values[Statistic.CVAR]
            .value
        )
        assert annual.aep[position].loss == report.value_at_risk
        assert report.value_at_risk == pytest.approx(independent.value_at_risk)
        assert report.value_at_risk == pytest.approx(
            weighted.quantile(level), rel=_EXPORT_REL_TOL
        )
        assert report.value_at_risk == pytest.approx(legacy["var"], rel=_EXPORT_REL_TOL)
        assert report.conditional_value_at_risk == pytest.approx(
            weighted_cvar, rel=_EXPORT_REL_TOL
        )

        expected_parts = dict(
            zip(components, tail_contributions[position], strict=True)
        )
        assert report.contributions == pytest.approx(expected_parts)
        assert {
            part.id: part.conditional_value_at_risk for part in independent.attribution
        } == pytest.approx(expected_parts)
        assert report.conditional_value_at_risk == pytest.approx(
            fsum(expected_parts.values())
        )
        assert independent.conditional_value_at_risk == pytest.approx(
            report.conditional_value_at_risk
        )

        # V4 includes the full cutoff branch in its tail, which can change CVaR.
        assert legacy["cvar"] == pytest.approx(
            report.conditional_value_at_risk - historical_cvar_delta[position],
            rel=_EXPORT_REL_TOL,
        )
        assert fsum(legacy["tail_contributions"].values()) == pytest.approx(
            legacy["cvar"], rel=_EXPORT_REL_TOL
        )

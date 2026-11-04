"""Regression tests for transform, risk-metric and vectorised-evaluation fixes."""

import math
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pytest

from crc_framework import (
    BinaryOutcome,
    EmpiricalDistribution,
    FittedDistribution,
    HurdleDistribution,
    PointMassDistribution,
    TabulatedDistribution,
    compute_risk,
    compute_spanning_set,
    generate_microscores,
)
from crc_framework.metrics import conditional_value_at_risk
from crc_framework.transforms import LinearImpact, SigmoidImpact


def gumbel() -> FittedDistribution:
    return FittedDistribution("gumbel_r", location=0.0, scale=1.0)


# C1: decreasing transforms on asymmetric probability grids.


def test_decreasing_linear_impact_uses_reflected_quantiles() -> None:
    exposure = gumbel()
    grid = [0.5, 0.9, 0.99]
    impact = LinearImpact(slope=-1.0, intercept=0.0, minimum=None)(exposure, probabilities=grid)
    np.testing.assert_allclose(impact.probabilities, grid)
    expected = [-float(exposure.ppf(1.0 - p)) for p in grid]
    np.testing.assert_allclose(impact.values, expected, rtol=1e-12)
    # q99 of -Gumbel(0, 1) is -Q(0.01) = ln(-ln 0.01).
    assert float(impact.ppf(0.99)) == pytest.approx(math.log(-math.log(0.01)), rel=1e-12)


def test_decreasing_sigmoid_impact_uses_reflected_quantiles() -> None:
    exposure = gumbel()
    grid = np.array([0.5, 0.9, 0.99])
    transform = SigmoidImpact(midpoint=1.0, steepness=-2.0)
    impact = transform(exposure, probabilities=grid)
    expected = transform.evaluate(exposure.quantiles(1.0 - grid))
    np.testing.assert_allclose(impact.values, expected, rtol=1e-12)
    assert np.all(np.diff(impact.values) >= 0)


def test_symmetric_grid_is_unchanged() -> None:
    exposure = TabulatedDistribution([0.1, 0.5, 0.9], [1.0, 2.0, 3.0])
    impact = LinearImpact(slope=-1.0, intercept=10.0, minimum=None)(exposure)
    np.testing.assert_allclose(impact.probabilities, [0.1, 0.5, 0.9])
    np.testing.assert_allclose(impact.values, [7.0, 8.0, 9.0])


def test_decreasing_impact_on_bounded_return_period_table_reflects_knots() -> None:
    exposure = TabulatedDistribution.from_return_periods(
        [10, 50, 100, 500], [1.0, 2.0, 3.0, 4.0], tail="upper"
    )
    impact = LinearImpact(slope=-1.0, intercept=10.0, minimum=None)(exposure)
    np.testing.assert_allclose(impact.probabilities, [0.002, 0.01, 0.02, 0.1])
    np.testing.assert_allclose(impact.values, [6.0, 7.0, 8.0, 9.0])


# C2: CVaR quadrature must not evaluate the quantile function at one.


def test_cvar_of_heavy_tail_is_finite_and_accurate() -> None:
    # GEV with scipy shape c=-0.3 (xi=0.3): CVaR_0.95 = 8.3276 by adaptive quadrature (SciPy).
    heavy = FittedDistribution("genextreme", location=0.0, scale=1.0, shape=-0.3)
    assert conditional_value_at_risk(heavy, 0.95) == pytest.approx(8.3276, rel=0.01)
    # Gumbel: 3.9831 by the same SciPy quadrature.
    assert conditional_value_at_risk(gumbel(), 0.95) == pytest.approx(3.9831, rel=0.002)


# C3: huge branch limits raise instead of aborting the process.


@pytest.mark.parametrize("max_branches", [None, 2**40, 2**63])
def test_huge_branch_limits_raise(max_branches: Optional[int]) -> None:
    outcomes = [BinaryOutcome(f"factor_{index}", 0.1, 1.0) for index in range(40)]
    with pytest.raises(ValueError, match="spanning branches"):
        compute_risk(outcomes, levels=(0.95,), max_branches=max_branches)
    with pytest.raises(ValueError, match="spanning branches"):
        compute_spanning_set(outcomes, max_branches=max_branches)


def test_risk_outputs_are_stable() -> None:
    outcomes = [BinaryOutcome("a", 0.1, 1.0), BinaryOutcome("b", 0.2, 2.0)]
    result = compute_risk(outcomes, levels=(0.5, 0.95))
    assert result.branch_count == 4
    assert [level.var for level in result.levels] == [0.0, 1.0]
    branches = compute_spanning_set(outcomes)
    assert [branch.impact for branch in branches] == [0.0, 0.5, 1.0, 1.5]
    assert branches[1].choices == (True, False)


# C10: a pure-Python distribution passed as an impact is not called as a transform.


class _PythonDistribution:
    def __init__(self, inner: FittedDistribution) -> None:
        self.inner = inner

    def pdf(self, x: object) -> Union[float, npt.NDArray[np.float64]]:
        return self.inner.pdf(x)  # type: ignore[arg-type]

    def cdf(self, x: object) -> Union[float, npt.NDArray[np.float64]]:
        return self.inner.cdf(x)  # type: ignore[arg-type]

    def ppf(self, probability: object) -> Union[float, npt.NDArray[np.float64]]:
        return self.inner.ppf(probability)  # type: ignore[arg-type]

    def quantiles(self, probabilities: object) -> npt.NDArray[np.float64]:
        return self.inner.quantiles(probabilities)  # type: ignore[arg-type]

    def sample(self, size: int, **_: object) -> npt.NDArray[np.float64]:
        return self.inner.sample(size)


def test_python_distribution_impact_is_used_as_distribution() -> None:
    impact = _PythonDistribution(FittedDistribution("gumbel_r", location=5.0, scale=2.0))
    suite = generate_microscores(gumbel(), impact=impact, probabilities=[0.5, 0.9])
    expected = impact.quantiles([0.5, 0.9])
    np.testing.assert_allclose([score.impact for score in suite.scores], expected)


# CP3 / C8: vectorised evaluation matches scalar evaluation, NaN propagates.


def _distributions() -> list[object]:
    return [
        gumbel(),
        FittedDistribution("skewnorm", location=0.0, scale=1.0, shape=3.0),
        EmpiricalDistribution([1.0, 2.0, 2.0, 3.0, 5.0]),
        TabulatedDistribution([0.1, 0.5, 0.9], [1.0, 2.0, 3.0]),
        PointMassDistribution(2.0),
        HurdleDistribution(
            FittedDistribution("weibull_min", location=0.0, scale=1.0, shape=1.5),
            atom_probability=0.3,
        ),
    ]


@pytest.mark.parametrize("distribution", _distributions())
def test_array_evaluation_matches_scalar(distribution: object) -> None:
    x = np.linspace(-1.0, 6.0, 12).reshape(3, 4)
    for method in ("pdf", "cdf"):
        function = getattr(distribution, method)
        result = function(x)
        assert isinstance(result, np.ndarray) and result.shape == (3, 4)
        assert result.dtype == np.float64
        expected = [function(float(value)) for value in x.flat]
        np.testing.assert_array_equal(result.reshape(-1), expected)
        assert function(x.tolist()).shape == (3, 4)
        assert function(np.array([], dtype=np.float64)).shape == (0,)
        assert isinstance(function(1.5), float)
        # Non-contiguous and integer inputs are converted.
        np.testing.assert_array_equal(function(x[:, ::2]), result[:, ::2])
        np.testing.assert_array_equal(function(np.array([1, 2])), function([1.0, 2.0]))
    p = np.array([[0.2, 0.5], [0.6, 0.8]])
    np.testing.assert_array_equal(
        distribution.ppf(p).reshape(-1),  # type: ignore[attr-defined]
        [distribution.ppf(float(value)) for value in p.flat],  # type: ignore[attr-defined]
    )


@pytest.mark.parametrize("distribution", _distributions())
def test_cdf_of_nan_is_nan(distribution: object) -> None:
    assert math.isnan(distribution.cdf(float("nan")))  # type: ignore[attr-defined]
    result = distribution.cdf(np.array([np.nan, 2.0]))  # type: ignore[attr-defined]
    assert math.isnan(result[0]) and not math.isnan(result[1])


def test_array_ppf_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError):
        gumbel().ppf(np.array([0.5, 1.5]))


def test_weibull_rejects_non_positive_shape() -> None:
    for family in ("weibull_min", "weibull_max"):
        with pytest.raises(ValueError):
            FittedDistribution(family, location=0.0, scale=1.0, shape=0.0)
        with pytest.raises(ValueError):
            FittedDistribution(family, location=0.0, scale=1.0, shape=-1.0)

"""Independent numerical references for the standalone native estimator."""

import math

import numpy as np
import pytest
from crc_framework import EmpiricalDistribution, FitConstraints, fit_distribution


# Unbiased PWMs evaluated separately in NumPy, with the exact GEV L-skewness
# equation solved by SciPy brentq and parameters evaluated with scipy.special.gamma.
# No SciPy runtime dependency is needed to check these fixed reference cases.
@pytest.mark.parametrize(
    "samples, expected",
    [
        (
            [1, 2, 3, 4, 5, 6, 7, 8],
            (0.2837755261699676, 3.5673697128875066, 2.6485046284617866),
        ),
        (
            [0, 0, 1, 3, 8, 15, 30, 100],
            (-0.6906224839693698, 2.5435518253418365, 6.22052453291623),
        ),
        (
            [1, 2, 2, 3, 5, 8, 13, 21, 34, 55],
            (-0.46999345573819784, 4.659674734457967, 6.785717967583603),
        ),
        (
            [-100, -30, -15, -8, -3, -1, 0, 0],
            (2.343489515427684, -6.78103358608292, 16.64903073472396),
        ),
    ],
)
def test_gev_reference_parameters(samples, expected):
    fit = fit_distribution(samples, "genextreme", method="lmoments")
    d = fit.distribution
    np.testing.assert_allclose([d.shape, d.location, d.scale], expected, rtol=2e-12)
    assert all(math.isfinite(x) for x in vars(fit.diagnostics).values())
    # Sorting is internal and the estimator is deterministic.
    other = fit_distribution(
        samples[::-1], "genextreme", method="lmoments"
    ).distribution
    assert (d.shape, d.location, d.scale) == (other.shape, other.location, other.scale)


def test_gumbel_analytic_and_reflected_fit():
    # For [1,...,8], L1 = 4.5 and unbiased L2 = 1.5.
    d = fit_distribution(list(range(1, 9)), "gumbel_r", method="lmoments").distribution
    scale = 1.5 / math.log(2)
    assert d.scale == pytest.approx(scale, rel=1e-14)
    assert d.location == pytest.approx(4.5 - 0.5772156649015329 * scale, rel=1e-14)
    reflected = fit_distribution(
        list(range(-8, 0)), "gumbel_l", method="lmoments"
    ).distribution
    assert reflected.location == pytest.approx(-d.location)
    assert reflected.scale == pytest.approx(d.scale)


@pytest.mark.parametrize("shape", [-0.4, -0.2, 0.0, 0.2, 1.2])
def test_recovers_population_quantiles(shape):
    from crc_framework import FittedDistribution

    original = FittedDistribution("genextreme", shape=shape, location=10, scale=3)
    data = original.quantiles((np.arange(20000) + 0.5) / 20000)
    fitted = fit_distribution(data, "genextreme", method="lmoments").distribution
    # Heavy tails converge slowly; compare return levels within the sample record.
    np.testing.assert_allclose(
        fitted.quantiles([0.5, 0.9, 0.99]),
        original.quantiles([0.5, 0.9, 0.99]),
        rtol=0.035,
    )


@pytest.mark.parametrize("family", ["genextreme", "gumbel_r", "gumbel_l"])
@pytest.mark.parametrize("scale, offset", [(1e-20, 0), (1e20, 1e22), (0.1, 1e5)])
def test_affine_equivariance(family, scale, offset):
    values = np.array([0, 0, 1, 3, 8, 15, 30, 100], dtype=float)
    base = fit_distribution(values, family, method="lmoments").distribution
    actual = fit_distribution(
        values * scale + offset, family, method="lmoments"
    ).distribution
    assert actual.shape == pytest.approx(base.shape, abs=1e-10)
    assert actual.location == pytest.approx(base.location * scale + offset, rel=1e-10)
    assert actual.scale == pytest.approx(base.scale * scale, rel=1e-10)


@pytest.mark.parametrize(
    "samples", [[], [1, 2, 3], [2] * 8, [0, 1, 2, np.nan], [0, 1, 2, np.inf]]
)
def test_invalid_records(samples):
    with pytest.raises(ValueError):
        fit_distribution(samples, "genextreme", method="lmoments")


def test_explicit_policy_and_constraints():
    values = EmpiricalDistribution([1, 2, 3, 4, 5, 6, 7, 8])
    with pytest.raises(ValueError, match="explicit family"):
        fit_distribution(values, method="lmoments")
    with pytest.raises(ValueError, match="no candidates"):
        fit_distribution(values, "gumbel_r", candidates=["gumbel_r"], method="lmoments")
    with pytest.raises(ValueError, match="supports"):
        fit_distribution(values, "genpareto", method="lmoments")
    with pytest.raises(ValueError, match="unknown fitting method"):
        fit_distribution(values, "gumbel_r", method="not-an-estimator")
    fit_distribution(
        values,
        "gumbel_r",
        method="lmoments",
        constraints=FitConstraints(minimum_value=0),
    )
    with pytest.raises(ValueError, match="constraints"):
        fit_distribution(
            values,
            "gumbel_r",
            method="lmoments",
            constraints=FitConstraints(maximum_value=0),
        )


def test_heavy_tail_estimate_converges_with_record_length():
    from crc_framework import FittedDistribution

    original = FittedDistribution("genextreme", shape=-0.8, location=10, scale=3)
    target = original.ppf(0.99)
    errors = []
    for n in (1000, 20000):
        values = original.quantiles((np.arange(n) + 0.5) / n)
        estimated = fit_distribution(
            values, "genextreme", method="lmoments"
        ).distribution
        errors.append(abs(estimated.ppf(0.99) - target))
    assert errors[1] < errors[0]


@pytest.mark.parametrize("values", [[0, 0, 0, 1], [0, 1, 1, 1]])
def test_degenerate_skewness_is_rejected(values):
    with pytest.raises(ValueError, match="L-skewness"):
        fit_distribution(values, "genextreme", method="lmoments")

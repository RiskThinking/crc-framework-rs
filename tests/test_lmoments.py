"""Independent numerical references for the explicit L-moment estimator."""

import math

import numpy as np
import pytest
from crc_framework import (
    Distribution,
    DistributionFamily,
    FitConstraints,
    SampleEstimator,
    SampleFit,
    fit_samples,
)


def specification(
    family=DistributionFamily.GENERALIZED_EXTREME_VALUE, *, constraints=None
):
    return SampleFit(
        family=family, estimator=SampleEstimator.L_MOMENTS, constraints=constraints
    )


# Unbiased PWMs evaluated independently in NumPy. GEV L-skewness roots and
# gamma factors were evaluated with SciPy; SciPy is not a runtime dependency.
@pytest.mark.parametrize(
    "samples,parameters",
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
def test_gev_reference_quantiles(samples, parameters):
    shape, location, scale = parameters
    reference = Distribution.parametric(
        DistributionFamily.GENERALIZED_EXTREME_VALUE,
        shape=shape,
        location=location,
        scale=scale,
    )
    fit = fit_samples(samples, specification())
    qs = [0.01, 0.1, 0.5, 0.9, 0.99]
    np.testing.assert_allclose(
        fit.distribution.quantile(qs), reference.quantile(qs), rtol=2e-12, atol=2e-12
    )
    other = fit_samples(samples[::-1], specification())
    np.testing.assert_array_equal(
        other.distribution.quantile(qs), fit.distribution.quantile(qs)
    )
    assert fit.record.estimator is SampleEstimator.L_MOMENTS
    assert fit.record.converged


def test_gumbel_analytic_and_reflected_fit():
    scale = 1.5 / math.log(2)
    reference = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT,
        scale=scale,
        location=4.5 - 0.5772156649015329 * scale,
    )
    fit = fit_samples(list(range(1, 9)), specification(DistributionFamily.GUMBEL_RIGHT))
    qs = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(
        fit.distribution.quantile(qs), reference.quantile(qs), rtol=1e-14
    )
    reflected = fit_samples(
        list(range(-8, 0)), specification(DistributionFamily.GUMBEL_LEFT)
    )
    np.testing.assert_allclose(
        reflected.distribution.quantile(qs),
        -fit.distribution.quantile(1 - qs),
        rtol=1e-14,
    )


@pytest.mark.parametrize("shape", [-0.4, -0.2, 0.0, 0.2, 1.2])
def test_recovers_population_quantiles(shape):
    original = Distribution.parametric(
        DistributionFamily.GENERALIZED_EXTREME_VALUE, shape=shape, location=10, scale=3
    )
    samples = original.quantile((np.arange(20000) + 0.5) / 20000)
    fitted = fit_samples(samples, specification()).distribution
    np.testing.assert_allclose(
        fitted.quantile([0.5, 0.9, 0.99]),
        original.quantile([0.5, 0.9, 0.99]),
        rtol=0.035,
    )


@pytest.mark.parametrize(
    "family",
    [
        DistributionFamily.GENERALIZED_EXTREME_VALUE,
        DistributionFamily.GUMBEL_RIGHT,
        DistributionFamily.GUMBEL_LEFT,
    ],
)
@pytest.mark.parametrize("scale,offset", [(1e-20, 0), (1e20, 1e22), (0.1, 1e5)])
def test_affine_equivariance(family, scale, offset):
    values = np.array([0, 0, 1, 3, 8, 15, 30, 100], dtype=float)
    qs = [0.1, 0.5, 0.9]
    base = fit_samples(values, specification(family)).distribution
    actual = fit_samples(values * scale + offset, specification(family)).distribution
    np.testing.assert_allclose(
        actual.quantile(qs), base.quantile(qs) * scale + offset, rtol=1e-10
    )


@pytest.mark.parametrize(
    "samples", [[], [1, 2, 3], [2] * 8, [0, 1, 2, np.nan], [0, 1, 2, np.inf]]
)
def test_invalid_evidence_fails_explicitly(samples):
    with pytest.raises(ValueError):
        fit_samples(samples, specification())


def test_quantile_constraint_can_reject_l_moment_estimate():
    with pytest.raises(ValueError, match="constraint"):
        fit_samples(
            list(range(1, 9)),
            specification(
                DistributionFamily.GUMBEL_RIGHT,
                constraints=FitConstraints(maximum_value=0, probability=0.99),
            ),
        )


def test_heavy_tail_estimate_converges_with_record_length():
    original = Distribution.parametric(
        DistributionFamily.GENERALIZED_EXTREME_VALUE, shape=-0.8, location=10, scale=3
    )
    target = original.quantile(0.99)
    errors = []
    for n in (1000, 20000):
        values = original.quantile((np.arange(n) + 0.5) / n)
        estimate = fit_samples(values, specification()).distribution
        errors.append(abs(estimate.quantile(0.99) - target))
    assert errors[1] < errors[0]


@pytest.mark.parametrize("values", [[0, 0, 0, 1], [0, 1, 1, 1]])
def test_degenerate_skewness_is_rejected(values):
    with pytest.raises(ValueError, match="L-skewness"):
        fit_samples(values, specification())

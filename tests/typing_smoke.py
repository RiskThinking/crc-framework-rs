from typing import reveal_type

from crc_framework import (
    Distribution,
    DistributionFamily,
    Impact,
    QuantileCurve,
    QuantileFit,
    Ratio,
    SampleEstimator,
    SampleFit,
    Statistic,
    TailCompletion,
    fit_quantiles,
    fit_samples,
    rank_scenarios,
)

law = Distribution.empirical([1.0, 2.0, 2.0, 3.0])
quantile = law.quantile(0.5)
reveal_type(quantile)
statistics = law.statistics(metrics=[Statistic.MEAN])
reveal_type(statistics.values[Statistic.MEAN].value)
fit = fit_samples(
    [1.0, 2.0, 2.0, 3.0],
    SampleFit(
        family=DistributionFamily.GUMBEL_RIGHT, estimator=SampleEstimator.L_MOMENTS
    ),
)
reveal_type(fit.record.converged)
curve = QuantileCurve([0.1, 0.3, 0.6, 0.9, 0.99], [0.0, 0.2, 0.8, 1.6, 2.5])
completed = curve.complete(TailCompletion(lower=-0.5, upper=4.0))
quantile_fit = fit_quantiles(curve, QuantileFit(family=DistributionFamily.GUMBEL_RIGHT))
reveal_type(quantile_fit.distribution)
impact = Impact.linear(slope=0.5, input=Ratio(), output=Ratio())
reveal_type(impact.evaluate([0.5, 1.0]))
reveal_type(impact.pushforward(completed))
reveal_type(rank_scenarios(completed, probabilities=[0.5, 0.95])[0].upper_rank_mass)

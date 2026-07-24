from crc_framework import (
    EmpiricalDistribution,
    TabulatedDistribution,
    RiskFactor,
    ScenarioMetadata,
    TransformContext,
    fit_distribution,
    fit_hurdle_quantiles,
    fit_quantiles,
    generate_microscores,
    impacts,
)

exposure = EmpiricalDistribution([1.0, 2.0, 3.0, 4.0])
fitted = fit_distribution(exposure).distribution
impact = impacts.for_factor(
    RiskFactor.FWI,
    context=TransformContext(),
)(fitted)
suite = generate_microscores(
    fitted,
    impact=impact,
    probabilities=[0.95],
    metadata=ScenarioMetadata(factor=RiskFactor.FWI.value),
)
outcome = suite.at(0.95)
reveal_type(outcome.downside_impact)

knots = TabulatedDistribution(
    [0.1, 0.3, 0.6, 0.9, 0.99],
    [0.0, 0.2, 0.8, 1.6, 2.5],
)
quantile_fit = fit_quantiles(knots, family="gumbel_r")
reveal_type(quantile_fit.diagnostics.rmse)
hurdle_fit = fit_hurdle_quantiles(
    [0.2, 0.5, 0.7, 0.8, 0.9, 0.97],
    [0.0, 0.0, 0.2, 0.5, 1.0, 2.0],
    "gumbel_r",
    atom_probability=0.5,
)
reveal_type(hurdle_fit.distribution.atom_probability)

from crc_framework import (
    EmpiricalDistribution,
    RiskFactor,
    ScenarioMetadata,
    TransformContext,
    fit_distribution,
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

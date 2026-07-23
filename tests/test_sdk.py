import unittest

import numpy as np

from crc_framework import (
    BinaryOutcome,
    EmpiricalDistribution,
    FitConstraints,
    RiskFactor,
    ScenarioMetadata,
    TabulatedDistribution,
    TransformContext,
    compute_risk,
    compute_spanning_set,
    fit_distribution,
    generate_microscores,
    impacts,
    lookup_continent,
    lookup_geography,
    lookup_ipcc_region,
    quality_metrics,
)


class DistributionTests(unittest.TestCase):
    def test_return_period_input_uses_canonical_probability(self):
        distribution = TabulatedDistribution.from_return_periods(
            [10, 100, 1000],
            [0.5, 2.0, 5.0],
            tail="upper",
        )
        self.assertAlmostEqual(distribution.ppf(0.99), 2.0)
        with self.assertRaises(ValueError):
            distribution.ppf(0.5)

    def test_auto_fit_exposes_diagnostics_and_vector_sampling(self):
        source = EmpiricalDistribution(np.linspace(1.0, 20.0, 100))
        result = fit_distribution(source)
        self.assertGreaterEqual(result.diagnostics.ks_pvalue, 0.0)
        self.assertEqual(result.distribution.quantiles([0.1, 0.5, 0.9]).shape, (3,))
        self.assertEqual(result.distribution.sample(4, seed=42).shape, (4,))
        diagnostics = quality_metrics(source, result.distribution)
        self.assertAlmostEqual(diagnostics.ks_statistic, result.diagnostics.ks_statistic)
        constrained = fit_distribution(
            source,
            constraints=FitConstraints(probability=0.99, maximum_value=100.0),
        )
        self.assertLessEqual(constrained.distribution.ppf(0.99), 100.0)


class PipelineTests(unittest.TestCase):
    def test_flood_bypasses_fitting_and_generates_one_probability_surface(self):
        exposure = TabulatedDistribution.from_return_periods(
            [10, 20, 50, 100, 200, 500],
            [0.1, 0.2, 0.5, 0.9, 1.4, 2.0],
            tail="upper",
        )
        impact = impacts.for_factor(
            RiskFactor.CFLOOD.value,
            context=TransformContext(
                continent="Europe", building_type="Commercial buildings"
            ),
            overrides={"maximum_damage": 0.8},
        )(exposure)
        suite = generate_microscores(
            exposure,
            impact=impact,
            probabilities=[0.95, 0.99],
            metadata=ScenarioMetadata(factor=RiskFactor.CFLOOD.value),
        )
        outcome = suite.at(0.95)
        self.assertAlmostEqual(outcome.downside_probability, 0.05)
        self.assertGreater(outcome.downside_impact, 0.0)
        self.assertAlmostEqual(impact.ppf(0.98), 0.12, places=6)

    def test_spanning_set_and_dynamic_risk_levels(self):
        outcomes = [
            BinaryOutcome("flood", 0.05, 0.4),
            BinaryOutcome("fire", 0.1, 0.2),
        ]
        self.assertEqual(len(compute_spanning_set(outcomes)), 4)
        result = compute_risk(outcomes, levels=[0.8, 0.95])
        self.assertEqual(result.branch_count, 4)
        self.assertGreaterEqual(result.at(0.95).cvar, result.at(0.95).var)


class SpatialAndConstantTests(unittest.TestCase):
    def test_reference_spatial_tables_are_exposed_without_silent_defaults(self):
        self.assertEqual(lookup_continent(594475228122316799), "Europe")
        self.assertEqual(
            lookup_ipcc_region(594479566039285759), "Greenland/Iceland"
        )
        geography = lookup_geography(600550049193132031)
        self.assertIsNotNone(geography)
        self.assertEqual(geography.continent, "Africa")
        self.assertEqual(geography.countries, ("NER",))


if __name__ == "__main__":
    unittest.main()

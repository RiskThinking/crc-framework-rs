import unittest

import numpy as np
from crc_framework import (
    BinaryOutcome,
    EmpiricalDistribution,
    FitConstraints,
    FittedDistribution,
    HurdleDistribution,
    PointMassDistribution,
    RiskFactor,
    ScenarioMetadata,
    TabulatedDistribution,
    TransformContext,
    compute_risk,
    compute_spanning_set,
    fit_distribution,
    fit_hurdle_quantiles,
    fit_quantiles,
    generate_microscores,
    impacts,
    lookup_continent,
    lookup_geography,
    lookup_ipcc_region,
    quality_metrics,
)


class DistributionTests(unittest.TestCase):
    def test_point_mass_is_exact_at_every_probability(self):
        distribution = PointMassDistribution(0.0)

        np.testing.assert_array_equal(
            distribution.quantiles([0.0, 0.5, 0.999, 1.0]),
            np.zeros(4),
        )
        self.assertEqual(distribution.cdf(-0.1), 0.0)
        self.assertEqual(distribution.cdf(0.0), 1.0)
        self.assertEqual(distribution.point_mass(0.0), 1.0)
        np.testing.assert_array_equal(distribution.sample(3), np.zeros(3))

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
        self.assertAlmostEqual(
            diagnostics.ks_statistic, result.diagnostics.ks_statistic
        )
        constrained = fit_distribution(
            source,
            constraints=FitConstraints(probability=0.99, maximum_value=100.0),
        )
        self.assertLessEqual(constrained.distribution.ppf(0.99), 100.0)

    def test_quantile_fit_accepts_arrays_and_tabulated_distribution(self):
        source = FittedDistribution.from_parameters(
            "gumbel_r", location=1.25, scale=0.7
        )
        probabilities = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        values = source.quantiles(probabilities)

        arrays = fit_quantiles(probabilities, values.tolist(), "gumbel_r")
        tabulated = fit_quantiles(
            TabulatedDistribution(probabilities, values), family="gumbel_r"
        )

        self.assertLess(arrays.diagnostics.normalized_rmse, 1.0e-6)
        self.assertAlmostEqual(
            arrays.diagnostics.rmse, tabulated.diagnostics.rmse, places=10
        )
        np.testing.assert_allclose(
            arrays.distribution.quantiles(probabilities), values, atol=1.0e-5
        )

    def test_sample_fit_rejects_probability_knots(self):
        tabulated = TabulatedDistribution([0.1, 0.3, 0.6, 0.9], [0.0, 1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "fit_quantiles"):
            fit_distribution(tabulated)  # type: ignore[arg-type]

    def test_hurdle_constructor_preserves_atom_and_truncated_tail(self):
        base = FittedDistribution.from_parameters("gumbel_r", location=0.5, scale=1.2)
        hurdle = HurdleDistribution(base, atom_probability=0.4)
        self.assertEqual(hurdle.ppf(0.4), 0.0)
        self.assertEqual(hurdle.point_mass(0.0), 0.4)
        self.assertEqual(hurdle.pdf(0.0), 0.0)
        for probability in [0.5, 0.75, 0.95]:
            self.assertAlmostEqual(
                hurdle.cdf(hurdle.ppf(probability)), probability, places=9
            )

    def test_wri_hurdle_fit_uses_original_probabilities(self):
        periods = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
        levels = [
            0.0,
            0.1595597267150879,
            0.5769345760345459,
            1.0778119564056396,
            1.4315228462219238,
            1.7589995861053467,
            2.1495182514190674,
            2.4226200580596924,
            2.6714894771575928,
        ]
        source = TabulatedDistribution.from_return_periods(
            periods, levels, tail="upper"
        )
        result = fit_hurdle_quantiles(
            source,
            family="gumbel_r",
            atom_probability=0.5,
        )
        fitted = result.distribution.quantiles(source.probabilities)
        self.assertEqual(fitted[0], 0.0)
        self.assertLess(result.diagnostics.tail.rmse, 0.11)
        self.assertLess(np.max(np.abs(fitted[1:] - source.values[1:])), 0.25)
        self.assertEqual(result.diagnostics.atom_probability_lower_bound, 0.5)


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
        self.assertEqual(lookup_ipcc_region(594479566039285759), "Greenland/Iceland")
        geography = lookup_geography(600550049193132031)
        self.assertIsNotNone(geography)
        self.assertEqual(geography.continent, "Africa")
        self.assertEqual(geography.countries, ("NER",))


if __name__ == "__main__":
    unittest.main()

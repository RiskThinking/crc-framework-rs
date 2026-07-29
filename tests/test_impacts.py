import unittest

import numpy as np

from crc_framework import (
    CallableImpact,
    CallableTransform,
    ImpactFunction,
    LinearImpact,
    PiecewiseLinearImpact,
    SigmoidImpact,
    TabulatedDistribution,
    TransformContext,
    impacts,
)


class PointImpactTests(unittest.TestCase):
    def test_linear_evaluate_preserves_scalar_and_array_shapes(self):
        impact = LinearImpact(slope=0.5, intercept=0.1, maximum=1.0)

        self.assertAlmostEqual(impact.evaluate(0.4), 0.3)
        values = np.array([[0.0, 0.5], [1.0, 3.0]])
        evaluated = impact.evaluate(values)

        self.assertIsInstance(evaluated, np.ndarray)
        np.testing.assert_allclose(evaluated, [[0.1, 0.35], [0.6, 1.0]])
        self.assertEqual(evaluated.shape, values.shape)

    def test_all_builtin_impacts_evaluate_points(self):
        sigmoid = SigmoidImpact(midpoint=1.0, steepness=2.0)
        piecewise = PiecewiseLinearImpact(
            exposure=[0.0, 1.0, 2.0],
            impact=[0.0, 0.25, 1.0],
        )

        sigmoid_values = sigmoid.evaluate(np.array([0.0, 1.0, 2.0]))
        piecewise_values = piecewise.evaluate(np.array([-1.0, 0.5, 3.0]))

        self.assertIsInstance(sigmoid_values, np.ndarray)
        self.assertLess(sigmoid_values[0], sigmoid_values[1])
        self.assertLess(sigmoid_values[1], sigmoid_values[2])
        np.testing.assert_allclose(piecewise_values, [0.0, 0.125, 1.0])

    def test_decreasing_event_values_are_not_distribution_quantiles(self):
        impact = LinearImpact(
            slope=-1.0,
            intercept=10.0,
            minimum=None,
            maximum=None,
        )
        exposure = TabulatedDistribution(
            probabilities=[0.1, 0.5, 0.9],
            values=[1.0, 2.0, 3.0],
        )

        np.testing.assert_allclose(impact.evaluate([1.0, 2.0, 3.0]), [9.0, 8.0, 7.0])
        np.testing.assert_allclose(
            impact(exposure).quantiles([0.1, 0.5, 0.9]), [7.0, 8.0, 9.0]
        )

    def test_callable_impact_validates_shape_and_finiteness(self):
        impact = CallableImpact(lambda exposure: np.clip(exposure / 2.0, 0.0, 1.0))

        self.assertIsInstance(impact, ImpactFunction)
        np.testing.assert_allclose(impact.evaluate([0.0, 1.0, 4.0]), [0.0, 0.5, 1.0])
        with self.assertRaisesRegex(ValueError, "must be finite"):
            impact.evaluate([np.nan])
        with self.assertRaisesRegex(ValueError, "preserve"):
            CallableImpact(lambda exposure: np.array([1.0])).evaluate([1.0, 2.0])

    def test_callable_transform_remains_distribution_compatible(self):
        exposure = TabulatedDistribution([0.1, 0.5, 0.9], [0.0, 1.0, 2.0])
        impact = CallableTransform(lambda values: values / 2.0)

        np.testing.assert_allclose(impact.evaluate([0.0, 1.0]), [0.0, 0.5])
        np.testing.assert_allclose(
            impact(exposure).quantiles([0.1, 0.5, 0.9]), [0.0, 0.5, 1.0]
        )

    def test_registry_evaluate_uses_native_vector_path_and_call_context(self):
        impact = impacts.for_factor("inundation")
        context = TransformContext(
            continent="Europe",
            building_type="Commercial buildings",
        )
        exposure = np.array([[0.0, 0.5], [1.0, 2.0]])

        evaluated = impact.evaluate(exposure, context=context)

        self.assertIsInstance(evaluated, np.ndarray)
        self.assertEqual(evaluated.shape, exposure.shape)
        self.assertEqual(evaluated[0, 0], 0.0)
        self.assertGreater(evaluated[-1, -1], evaluated[0, 1])
        self.assertIsInstance(impact.evaluate(1.0, context=context), float)


if __name__ == "__main__":
    unittest.main()

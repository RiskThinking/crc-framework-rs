# Python guide

[Documentation index](README.md) · [Project README](../README.md)

## Probability laws and quantile evidence

`Distribution` represents a complete probability law. `QuantileCurve` represents
probability-labelled evidence over its actual interval. Choose according to what
you know about the data:

| Available data | Constructor |
| --- | --- |
| Observed samples, including ties | `Distribution.empirical(samples)` |
| Values with atom probabilities | `Distribution.discrete(values, probabilities)` |
| Uniform density within bins | `Distribution.histogram(edges, probabilities)` |
| A constant | `Distribution.point_mass(value)` |
| Known family and parameters | `Distribution.parametric(family, location=..., scale=..., shape=...)` |
| An atom followed by a conditional tail | `Distribution.hurdle(base, atom_probability=..., atom_location=...)` |
| Probability-labelled values | `QuantileCurve(probabilities, values)` |
| Explicit interpolated sample quantiles | `QuantileCurve.from_samples(samples)` |

All complete laws share `quantile`, `cdf`, `exceedance`, `mass`, `sample`, and
`statistics`. Scalar evaluations return floats; array evaluations preserve shape,
including empty and noncontiguous NumPy arrays. Probabilities are non-exceedance
ranks in `[0, 1]`.

```python
from crc_framework import Distribution

law = Distribution.empirical([0, 0, 0, 4])
assert law.quantile(0.5) == 0
assert law.cdf(0) == 0.75                  # P(X <= 0)
assert law.cdf(0, inclusive=False) == 0   # P(X < 0)
assert law.exceedance(0) == 0.25          # P(X > 0)
assert law.exceedance(0, inclusive=True) == 1
assert law.mass(0) == 0.75
assert law.sample((2, 3), seed=42).shape == (2, 3)
```

The empirical constructor always uses the ECDF: ties are atoms and the mean is
the observed sample mean. Histogram probabilities are masses of bins, rather
than heights. A hurdle places its declared atom at the threshold and conditions
the base strictly above that threshold before allocating the remaining mass.

## Use supplied quantiles directly

A quantile curve interpolates linearly in probability between supplied knots.
For upper-tail return period `T`, explicitly calculate `q = 1 - 1/T`:

```python
import numpy as np
from crc_framework import QuantileCurve

periods = np.array([10, 20, 50, 100, 200, 1000], dtype=float)
curve = QuantileCurve(1 - 1 / periods, [0.5, 0.8, 1.4, 2.0, 3.0, 5.0])
assert curve.quantile(0.9) == 0.5
assert curve.quantile(0.99) == 2.0
```

A query outside `curve.probability_interval` raises `ValueError`. Curves have no
whole-law sampling, CDF, or statistics. They do not implicitly fit, extrapolate,
or clamp. Make completion or conditioning an explicit scientific choice:

```python
from crc_framework import TailCompletion

bounded = curve.complete(TailCompletion(lower=0, upper=8))
conditional = curve.condition(q_min=0.9, q_max=0.999)
```

Finite-endpoint completion adds linear tails to probabilities zero and one.
Conditioning rescales only the chosen probability interval to a complete law.
Both operations retain the original evidence and the modelling choice in their
serialized numerical representation.

## Statistics retain probability mass

Statistics integrate the law, including rare atoms, rather than sampling a
midpoint grid. Each result has an explicit `StatisticStatus`: finite, positive
infinity, negative infinity, or undefined. Undefined values carry diagnostics;
they are not replaced with ordinary zeros.
Standard deviation can remain finite when variance exceeds the floating-point
range.

```python
from crc_framework import Distribution, Statistic, StatisticStatus, Tail

catastrophe = Distribution.discrete([0, 1_000_000], [0.999999, 0.000001])
summary = catastrophe.statistics(
    metrics=[Statistic.MEAN, Statistic.MAXIMUM, Statistic.CVAR],
    level=0.95,
    tail=Tail.UPPER,
)
assert summary.values[Statistic.MEAN].status is StatisticStatus.FINITE
assert abs(summary.values[Statistic.MEAN].value - 1) < 1e-8
assert summary.values[Statistic.MAXIMUM].value == 1_000_000
assert abs(summary.values[Statistic.CVAR].value - 20) < 1e-8
```

For upper-tail CVaR, `level=0.95` averages the upper 5% of rank mass; lower-tail
CVaR averages the lower 5%. A partially included atom contributes exactly its
included mass. Infinite mathematical endpoints remain infinite.

## Fit the evidence you actually have

Fitting requires a family or explicit candidates. Framework fitting
specifications are also the SDK's scientific specifications.

```python
from crc_framework import (
    DistributionFamily, SampleEstimator, SampleFit, fit_samples,
)

fit = fit_samples(
    [0.1, 0.2, 0.4, 0.7, 1.1, 1.8],
    SampleFit(
        family=DistributionFamily.GENERALIZED_EXTREME_VALUE,
        estimator=SampleEstimator.L_MOMENTS,
    ),
)
median = fit.distribution.quantile(0.5)
assert fit.record.converged
assert fit.record.estimator is SampleEstimator.L_MOMENTS
ks = fit.diagnostic.ks_statistic
```

Maximum likelihood requires successful likelihood optimization. Moment
initialization is not reported as successful maximum likelihood. Estimator
alternatives execute only when supplied through `SampleFit.fallbacks`, and all
attempts remain in the fit record. `FitError.record` contains those attempts
and their failure messages.

For probability-labelled evidence, use `fit_quantiles` and the appropriate
specification. Supplied quantile knots are never treated as observations:

```python
from crc_framework import DistributionFamily, QuantileFit, fit_quantiles

fit = fit_quantiles(
    curve,
    QuantileFit(family=DistributionFamily.GUMBEL_RIGHT),
)
```

`CensoredQuantileFit` models knots recorded at or below a declared censoring
location. `HurdleQuantileFit` uses a declared atom probability and a fitted tail;
it does not infer an atom from repeated sparse knots. `ModelSelection` executes
only its explicit candidate specifications. Sample selection uses
`SelectionCriterion.KOLMOGOROV_SMIRNOV`; quantile selection explicitly uses
`SelectionCriterion.NORMALIZED_ROOT_MEAN_SQUARE_ERROR`. Quantile evidence is
not treated as an independent sample for a KS test. `FitConstraints` apply to the
requested fit; they do not authorize fitting other families.

A `FitResult` contains the law, the selected attempt's typed `diagnostic`, and
a `FitRecord` with the requested
specification, actual estimator, convergence, attempts, the selected attempt
index, effective weights, evidence counts, and engine version. The
`value_interval` records the observed value range. For a quantile fit it
reflects the selected attempt's positive-weight knots and any declared
censoring. Failed quantile fits record the last attempt that used knots. The
`probability_interval` describes the original supplied knots and applies only
to quantile evidence. Reopening a fitted curve checks that its support covers
the recorded values when `require_sample_support=True`.
Sample attempts report their KS statistic and p-value. Quantile attempts report
the residual measures for the requested operation and compare predicted
quantiles at every supplied interior probability. This comparison weights the
knots equally and divides RMSE by their observed value span, so candidates are
scored on the same evidence even when their estimation weights differ.
For censored selection, every candidate must use the same `censor_location`.
Scores compare observations and predicted quantiles at the bound wherever they
are left-censored. Values supplied below the bound are placeholders for the
same observation.
Model selection picks the first converged candidate with the lowest recorded
score. Distribution parameters are available on the fitted law.

The sample fit record counts observations supplied to `fit_samples`. An attempt
that meets its minimum uses the full sample. If every attempted specification
requires more observations, the record keeps the supplied sample range and
reports zero used observations.

## Evaluate impacts explicitly

A resolved `Impact` is a context-free mathematical function. Its input and
output measurements declare the science independently of variable names:

```python
from crc_framework import Distribution, Impact, Ratio

impact = Impact.linear(slope=0.5, input=Ratio(), output=Ratio())
assert impact.evaluate(0.8) == 0.4
law = Distribution.discrete([0, 1], [0.75, 0.25])
assert impact.expectation(law) == 0.125
response = impact.pushforward(law)
assert response.mass(0.5) == 0.25
```

`Impact.sigmoid`, `Impact.tabulated`, and `Impact.binned` share these same
operations. Pushforwards preserve atoms, including decreasing functions at atom
boundaries. An unsupported exact operation raises a capability error; it does
not silently tabulate an approximation.

Built-in references are selected with `Impact.builtin(BuiltinImpact.INUNDATION)`.
Inspect `reference.requirements` for the input measurement, accepted units,
output, and the actual typed `context_type` constructor. Resolving a reference
validates the required context and produces the same `Impact` type. A variable
name alone never chooses a model.

## Rank scenarios and independent risk

```python
from crc_framework import Distribution, rank_scenarios

scenarios = rank_scenarios(
    Distribution.empirical([0, 0, 0, 4]), probabilities=[0.5, 0.95],
)
assert scenarios[0].lower_rank_mass == 0.5
```

Rank splits label ranks, not threshold probabilities. In this example the mass
at or below the median value is 0.75, despite the lower rank mass being 0.5.
Use `cdf`, `exceedance`, and `mass` for actual threshold events. Request
statistics separately through `Distribution.statistics`.

`BinaryOutcome` declares two states with explicit rank masses and impacts.
`compute_risk(outcomes, levels=[...])` sums independent contributions in their
supplied units and returns exact VaR, upper-tail CVaR, and attribution. VaR
attribution averages equally valued states by their probability masses. When
the CVaR cutoff includes only part of a tied value's mass, each tied state
contributes the same fraction of its mass. Branch limits are enforced before
allocation. Outcome identities must be unique.

## Serialization and reference geography

`Distribution`, `QuantileCurve`, fitting specifications, and impacts have typed
`to_json` representations. Numerical objects use format `1.0`; resolved impacts
use format `2.0`. Load them with the corresponding type's `from_json`; historical
curve dictionaries and impact envelopes are rejected.

`lookup_continent`, `lookup_ipcc_region`, and `lookup_geography` accept integer H3
cells and return reference values or `None`. They do not invent geographic
context when an entry is absent.

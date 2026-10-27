# Python guide

[Documentation index](README.md) · [Project README](../README.md)

## Distributions

All calculations are expressed through a distribution interface with `pdf`,
`cdf`, `ppf`, `quantiles`, and `sample` operations. Scalar and NumPy array
inputs are supported for the evaluation methods.

Choose a distribution according to the form of the available data:

- `EmpiricalDistribution` for raw observations.
- `TabulatedDistribution` for explicit probability/value pairs.
- `FittedDistribution` for a known parametric family and its parameters.
- `HurdleDistribution` for a point mass followed by a truncated parametric
  tail.
- `PointMassDistribution` for a value that is constant at every probability.

`TabulatedDistribution` does not extrapolate by default. Querying outside its
declared probability range raises an error unless `extrapolate=True` is set.

## Probability convention

All metric APIs use non-exceedance probability `q` in `[0, 1]`. Return periods
are accepted only when constructing a tabulated distribution. For upper-tail
hazards, a return period `T` is converted to `q = 1 - 1/T`; therefore, the
upper-tail 100-year return level is queried at `q=0.99`.

## Fitting

Fitting is always explicit. Metrics sample the distribution supplied by the
caller and never silently substitute a fitted curve. This makes the modelling
choice auditable.

`fit_distribution` accepts raw observations or an `EmpiricalDistribution` and
selects among the supported families using the
Kolmogorov–Smirnov p-value. The supported families are `genextreme`,
`weibull_min`, `weibull_max`, `skewnorm`, `gumbel_r`, `gumbel_l`, and
`genpareto`. The returned `FitResult` includes KS, RMSE, and R-squared
diagnostics. Passing a tabulated or fitted distribution is an error: those
values are not independent observations.

### Fit observations

```python
import numpy as np

from crc_framework import EmpiricalDistribution, FitConstraints, fit_distribution

observations = np.array([0.1, 0.2, 0.4, 0.7, 1.1, 1.8])
empirical = EmpiricalDistribution(observations)
fit = fit_distribution(
    empirical,
    family="auto",
    constraints=FitConstraints(probability=0.99, maximum_value=3.0),
)
median = fit.distribution.ppf(0.5)
print(fit.distribution.family, fit.diagnostics.ks_pvalue, median)
```

Pass `family="auto"` (the default) to select a family, a family name to fit a
specific family, or use `fit_all` to inspect every candidate. Use
`quality_metrics` to calculate diagnostics for a selected fitted distribution.

### L-moment estimation

For a standalone L-moment estimate of GEV or Gumbel observations, select the
estimator explicitly:

```python
from crc_framework import fit_distribution

annual_extremes = [0.1, 0.2, 0.4, 0.7, 1.1, 1.8]
result = fit_distribution(annual_extremes, family="genextreme", method="lmoments")
```

This uses unbiased sample L-moments (Hosking, 1990), with an exact numerical
solution of the GEV L-skewness equation. It supports `genextreme`, `gumbel_r`
and `gumbel_l`, requires at least four finite, nonconstant observations, and
performs no likelihood optimization or automatic family selection. GEV shape
uses the SciPy convention (`c = -xi`); the finite-mean domain is `c > -1`.
Degenerate L-skewness is rejected rather than clipped. Fit constraints can
accept or reject the estimated distribution without refitting it.

The native implementation sorts once, uses compensated sums in scaled and
centered coordinates, and reuses that sort for diagnostics. Work is O(n log n)
with O(n) storage. The Python binding releases the GIL during native fitting.
The returned KS p-value is a descriptive diagnostic, not a calibrated hypothesis
test after parameter estimation. See [NIST's L-moments reference](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/lmoment.htm)
for background and the Hosking references.

### Fit quantile curves

Use `fit_quantiles` for probability/value knots. It minimizes weighted
value-space differences between the supplied values and the selected family's
PPF, requires an explicit family, and reports residual diagnostics rather than
KS statistics:

```python
from crc_framework import TabulatedDistribution, fit_quantiles

curve = TabulatedDistribution.from_return_periods(
    [5, 10, 25, 50, 100],
    [0.2, 0.6, 1.1, 1.5, 1.9],
    tail="upper",
)
fit = fit_quantiles(curve, family="gumbel_r")
```

### Fit a known point mass and tail

For a known point mass, use `fit_hurdle_quantiles` and supply its probability;
the framework does not infer an exact mass from sparse zero-valued knots:

```python
from crc_framework import fit_hurdle_quantiles

fit = fit_hurdle_quantiles(
    curve,
    family="gumbel_r",
    atom_probability=0.5,
    atom_location=0.0,
)
```

### Fit censored quantiles

For sparse return-period curves whose lower values mean "at or below the
physical detection bound", use `fit_censored_quantiles`. It fits every knot
with a Gaussian left-censored likelihood, rather than treating repeated zeros
as an exact atom or discarding them from the fit:

```python
from crc_framework import fit_censored_quantiles

fit = fit_censored_quantiles(
    [0.5, 0.8, 0.9, 0.96, 0.98, 0.99, 0.996, 0.998, 0.999],
    [0, 0, 0, 0, 0, 0, 0.03, 0.06, 0.10],
    family="genextreme",
    censor_location=0.0,
)
```

The residual scale has a reproducible positive floor (by default `1e-3` times
the robust value scale) because sparse tails may otherwise make the likelihood
unbounded as residual variance approaches zero. The fitted object is the latent
parametric distribution; producers should clamp it at the censor location when
storing a non-negative observed curve.

`TabulatedDistribution` remains the lossless interpolation of the supplied
knots. Parametric and hurdle fits are lossy, so systems that require source
reconstruction must persist the original probability/value pairs separately.

## Create microscores from flood exposure

```python
from crc_framework import (
    RiskFactor,
    ScenarioMetadata,
    TabulatedDistribution,
    TransformContext,
    generate_microscores,
    impacts,
)

exposure = TabulatedDistribution.from_return_periods(
    periods=[10, 20, 50, 100, 200, 500],
    values=[0.1, 0.2, 0.5, 0.9, 1.4, 2.0],
    tail="upper",
)

impact = impacts.for_factor(
    RiskFactor.CFLOOD,
    context=TransformContext(
        continent="Europe",
        building_type="Commercial buildings",
    ),
)(exposure)

suite = generate_microscores(
    exposure,
    impact=impact,
    probabilities=[0.95, 0.99],
    metadata=ScenarioMetadata(factor=RiskFactor.CFLOOD.value),
)
```

`generate_microscores` returns a `MicroscoreSuite`. Use `suite.scores` to
inspect sampled exposure and impact values, `suite.at(q)` to obtain the binary
outcome at a requested probability, and `exposure_statistics` and
`impact_statistics` for summary diagnostics.

## Evaluate impacts

Impact functions expose two deliberately distinct operations:

- `impact.evaluate(values, context=...)` evaluates event-aligned exposure
  values. Input shape and order are preserved, including for decreasing or
  non-monotonic callables. This is the appropriate operation when return
  periods continue to identify the source hazard events.
- `impact(distribution, probabilities=..., context=...)` transforms an
  exposure distribution into an impact distribution. Decreasing built-in
  transforms reorder the resulting quantiles so the output remains a valid
  distribution for risk metrics.

The built-in `LinearImpact`, `SigmoidImpact`, and `PiecewiseLinearImpact`
support both operations. `CallableImpact` adapts a vectorized NumPy callable
for point evaluation, while the backward-compatible `CallableTransform` also
supports distribution transformation.

```python
import numpy as np

from crc_framework import (
    CallableImpact, LinearImpact, TabulatedDistribution, generate_microscores,
)

exposure = TabulatedDistribution([0.5, 0.9, 0.99], [0.5, 1.0, 2.0])
impact_function = LinearImpact(slope=0.25, maximum=1.0)
event_impacts = impact_function.evaluate(np.array([0.5, 2.0]))
impact = impact_function(exposure)
suite = generate_microscores(
    exposure,
    impact=impact,
    probabilities=[0.95, 0.99],
)

custom = CallableImpact(lambda values: np.clip(values / 2.0, 0.0, 1.0))
custom_event_impacts = custom.evaluate(np.array([0.5, 2.0]))
```

`impacts.for_factor(...)` selects a registry-backed climate transform. Supply a
`TransformContext` with the relevant geography, building type, and historical
values; pass `overrides` when an application needs to replace transform
parameters. Registry-backed impacts support the same point and distribution
interfaces, with point evaluation delegated directly to the native registry.

## Impact contracts and provenance

See the [impact contracts guide](impact-contracts.md) for schemas, identities,
provenance, conformance fixtures, and native capability coverage.

## Aggregate factor outcomes

Each microscore can be converted to a `BinaryOutcome`. Combine outcomes into
the full set of independent downside/upside branches, then calculate VaR and
CVaR at one or more confidence levels:

```python
from crc_framework import BinaryOutcome, compute_risk, compute_spanning_set

outcomes = [
    BinaryOutcome("flood", downside_probability=0.05, downside_impact=0.4),
    BinaryOutcome("fire", downside_probability=0.10, downside_impact=0.2),
]

branches = compute_spanning_set(outcomes)
risk = compute_risk(outcomes, levels=[0.80, 0.95])
level_95 = risk.at(0.95)

print(len(branches))  # 4: two possible outcomes per factor
print(level_95.var, level_95.cvar)
```

Branch generation grows as `2**n` for `n` factors. Set `max_branches` on
`compute_spanning_set` or `compute_risk` to enforce an application-specific
limit. `RiskLevel.attribution` reports each factor's contribution to VaR and
CVaR.

## Spatial helpers

Spatial helpers accept an H3 cell as an integer or string, normalize it to H3
resolution 4, and return `None` when no reference mapping exists.

```python
from crc_framework import lookup_geography, lookup_ipcc_region

cell = 600550049193132031
geography = lookup_geography(cell)
region = lookup_ipcc_region(cell)

if geography is not None:
    print(geography.continent, geography.countries)
print(region)
```

`lookup_continent` returns a continent name, `lookup_ipcc_region` returns the
IPCC region, and `lookup_geography` returns a `Geography` object with the
continent and intersecting ISO3 country codes.

## Public API

The package exports the following primary interfaces:

- Distributions: `EmpiricalDistribution`, `TabulatedDistribution`,
  `FittedDistribution`, `HurdleDistribution`, `PointMassDistribution`,
  `fit_distribution`, `fit_quantiles`, `fit_censored_quantiles`,
  `fit_hurdle_quantiles`, `fit_all`, and `quality_metrics`.
- Transforms: `impacts`, `ImpactRegistry`, `ImpactFunction`, `ClimateImpact`,
  `LinearImpact`, `SigmoidImpact`, `PiecewiseLinearImpact`, `CallableImpact`,
  and `CallableTransform`.
- Metrics: `generate_microscores`, `compute_spanning_set`, and `compute_risk`.
- Models and constants: `ScenarioMetadata`, `TransformContext`, `RiskFactor`,
  `Pathway`, `RISK_FACTORS`, `PATHWAYS`, and `HORIZONS`.
- Spatial lookup: `lookup_continent`, `lookup_ipcc_region`, and
  `lookup_geography`.

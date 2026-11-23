# Migrating to crc-framework 1.0

[Documentation index](README.md) · [Python guide](python.md) · [JavaScript guide](javascript.md)

Version 1.0 replaces the 0.x Python and JavaScript interfaces. Most changes are
renames into a smaller set of types, but a few change numbers silently. Read
[Changes that alter results](#changes-that-alter-results) first, then work
through the sections that match the code you use.

Examples marked **Before** use crc-framework 0.6. Examples marked **After** run
against 1.0 and are checked by `tools/check_examples.py`.

## Upgrade checklist

1. Pin `crc-framework>=1,<2` (Python 3.11+) or
   `@climateriskcommons/crc-framework@^1` (npm). If you also use crc-sdk,
   upgrade both packages together; crc-sdk 1.0 requires crc-framework 1.x.
   Until 1.0.0 is final, these ranges do not match the prerelease: use
   `crc-framework>=1.0.0a1,<2` and `@climateriskcommons/crc-framework@alpha`.
   Building from source now needs Rust 1.87 or later (0.6 needed 1.85).
2. Import everything from `crc_framework`. The submodules
   `crc_framework.distributions`, `.transforms`, `.metrics`, `.impacts`,
   `.models`, `.constants`, `.spatial`, and `.statistics` no longer exist.
3. Replace the five distribution classes with `Distribution` or `QuantileCurve`.
4. Replace `ppf`/`quantiles` with `quantile`.
5. Pass a fitting specification (`SampleFit`, `QuantileFit`, …) to
   `fit_samples` or `fit_quantiles`.
6. Give every impact an input and output measurement, and resolve built-in
   models with a typed context.
7. Rename risk fields (`var` → `value_at_risk`, and so on) and check the
   [aggregation change](#risk-aggregation-sums-instead-of-averaging).
8. Regenerate stored curves and impact definitions. Historical curve
   dictionaries (`curve_kind`, `curve_type`, …) and impact schema 1.0 documents
   are rejected.
9. In JavaScript, pass numerical JSON documents instead of flat `curve_*`
   properties, and handle out-of-support queries as errors instead of `null`.

## Changes that alter results

These changes do not raise an error. Compare a few outputs before and after
upgrading.

| Area | 0.x behaviour | 1.0 behaviour | To keep the old result |
| --- | --- | --- | --- |
| Return-period curves | `TabulatedDistribution.from_return_periods` interpolated in log return period | `QuantileCurve` interpolates linearly in probability | Supply denser knots; values at the knots themselves are unchanged |
| `LinearImpact` | Clamped below at `minimum=0.0` by default | `Impact.linear` has no lower bound unless you pass `minimum=` | Pass `minimum=0.0` |
| `compute_risk` | Weighted average of factor impacts: `Σ wᵢ·impactᵢ / Σ wᵢ` | Plain sum of impacts in their supplied units; no `weight` | Pre-scale each `upper_impact` by `wᵢ / Σ wᵢ` |
| `compute_risk` levels | Defaulted to `(0.5, 0.95, 0.99)` | `levels=` is required | Pass the levels explicitly |
| Tabulated queries | `extrapolate=True` extended the curve | No implicit extrapolation; queries outside the knots raise `ValueError` | Use `curve.complete(TailCompletion(...))` |

For example, a curve with 10-year and 100-year depths of 0.5 m and 2.0 m gave a
20-year depth of about 0.95 m in 0.6 and gives 1.33 m in 1.0.

## Distributions

One `Distribution` type represents every complete probability law. A
`QuantileCurve` represents probability-labelled evidence over the interval you
supplied; it has no CDF, sampling, or statistics until you complete it.

| 0.x | 1.0 |
| --- | --- |
| `EmpiricalDistribution(samples)` | `Distribution.empirical(samples)` |
| `TabulatedDistribution(probabilities, values)` | `QuantileCurve(probabilities, values)` |
| `TabulatedDistribution.from_return_periods(periods, values, tail="upper")` | `QuantileCurve(1 - 1 / periods, values)` |
| `FittedDistribution("gumbel_r", location=..., scale=...)` | `Distribution.parametric(DistributionFamily.GUMBEL_RIGHT, location=..., scale=...)` |
| `HurdleDistribution(base, atom_probability=..., atom_location=...)` | `Distribution.hurdle(base, atom_probability=..., atom_location=...)` |
| `PointMassDistribution(value)` | `Distribution.point_mass(value)` |
| `DiscreteDistribution`, `HistogramExposure` | `Distribution.discrete(values, probabilities)`, `Distribution.histogram(edges, probabilities)` |
| `dist.ppf(q)`, `dist.quantiles(qs)` | `dist.quantile(q)` (scalar or array) |
| `dist.pdf(x)` | `dist.mass(x)` for atoms, `dist.cdf(b) - dist.cdf(a)` for intervals; no density (see below) |
| `distribution_statistics(...)`, `interval_statistics(...)` | `dist.statistics(metrics=[...], level=..., tail=...)` |
| `as_ecdf()`, `as_interpolated()` | `Distribution.empirical(...)`, `QuantileCurve.from_samples(...)` |
| Family strings such as `"genextreme"` | `DistributionFamily.GENERALIZED_EXTREME_VALUE` |

**Before**

```py
from crc_framework import FittedDistribution, TabulatedDistribution

curve = TabulatedDistribution.from_return_periods(
    [10, 100, 1000], [0.5, 2.0, 5.0], tail="upper"
)
print(curve.ppf(0.99))  # 2.0

law = FittedDistribution("gumbel_r", location=1.0, scale=2.0)
print(law.quantiles([0.5, 0.99]))
```

**After**

```python
import numpy as np
from crc_framework import Distribution, DistributionFamily, QuantileCurve

periods = np.array([10, 100, 1000], dtype=float)
curve = QuantileCurve(1 - 1 / periods, [0.5, 2.0, 5.0])
assert curve.quantile(0.99) == 2.0

law = Distribution.parametric(
    DistributionFamily.GUMBEL_RIGHT, location=1.0, scale=2.0
)
print(law.quantile([0.5, 0.99]))
```

For a lower-tail hazard such as drought, use `q = 1 / T` instead of `1 - 1 / T`.

`Distribution` answers probability questions only, so there is no `pdf`. Use
`mass(x)` for the probability of exactly `x`, which is nonzero only at atoms of
discrete, empirical, point-mass and hurdle laws, and CDF differences for the
probability of an interval. `mass` is not a density: it is `0.0` everywhere on a
continuous law.

```python
from crc_framework import Distribution, DistributionFamily

gumbel = Distribution.parametric(
    DistributionFamily.GUMBEL_RIGHT, location=1.0, scale=2.0
)
assert gumbel.mass(1.0) == 0.0
interval = gumbel.cdf(3.0) - gumbel.cdf(1.0)  # P(1 < X <= 3)
assert 0.0 < interval < 1.0

hurdle = Distribution.hurdle(gumbel, atom_probability=0.3, atom_location=0.0)
assert hurdle.mass(0.0) == 0.3
assert list(Distribution.discrete([0, 2], [0.9, 0.1]).mass([0, 1, 2])) == [0.9, 0.0, 0.1]
```

If you need a density, for example to plot a fitted law, evaluate it outside
the framework. Parametric family names and parameters follow SciPy, including
the GEV shape sign (`c = -xi`), so the serialized law maps directly onto
`scipy.stats`:

```py
import json
from scipy import stats

law = json.loads(gumbel.to_json())
family = getattr(stats, law["family"])
shape = (law["shape"],) if "shape" in law else ()
density = family.pdf(x, *shape, loc=law["location"], scale=law["scale"])
```

Empirical laws have no density; a smoothed estimate such as a kernel density is
a modelling choice you make outside the framework.

### Statistics, tails, and new queries

Statistics now integrate the whole law and report a status, so an infinite or
undefined mean is visible instead of being returned as a number. Partial curves
need an explicit tail choice before you can compute statistics or sample them.

```python
from crc_framework import Statistic, StatisticStatus, TailCompletion

complete = curve.complete(TailCompletion(lower=0.0, upper=8.0))
summary = complete.statistics(metrics=[Statistic.MEAN, Statistic.CVAR], level=0.95)
mean = summary.values[Statistic.MEAN]
assert mean.status is StatisticStatus.FINITE
print(mean.value, summary.values[Statistic.CVAR].value)

assert complete.exceedance(2.0) == 1 - complete.cdf(2.0)  # P(X > 2)
print(complete.mass(0.0))                                  # P(X == 0)
```

`curve.condition(q_min=..., q_max=...)` is the other explicit option. It
rescales the chosen probability interval to a complete law, so its quantiles
differ from the original knots; use `complete` when the knot values must stay
where they are.

## Fitting

Fitting functions take the evidence plus one specification object. There is no
automatic family selection: list the candidates you want to compare.

| 0.x | 1.0 |
| --- | --- |
| `fit_distribution(samples, family="gumbel_r")` | `fit_samples(samples, SampleFit(family=DistributionFamily.GUMBEL_RIGHT))` |
| `fit_distribution(samples, family="genextreme", method="lmoments")` | `SampleFit(family=..., estimator=SampleEstimator.L_MOMENTS)` |
| `fit_distribution(samples, family="auto")`, `fit_all(samples)` | `fit_samples(samples, ModelSelection(candidates=(SampleFit(...), ...)))` |
| `fit_quantiles(curve, family="gumbel_r")` | `fit_quantiles(curve, QuantileFit(family=...))` |
| `fit_censored_quantiles(p, v, family=..., censor_location=0.0)` | `fit_quantiles(QuantileCurve(p, v), CensoredQuantileFit(family=..., censor_location=0.0))` |
| `fit_hurdle_quantiles(curve, family=..., atom_probability=...)` | `fit_quantiles(curve, HurdleQuantileFit(family=..., atom_probability=...))` |
| `fit.diagnostics` | `fit.diagnostic` (typed per fit kind) and `fit.record` (all attempts) |
| `quality_metrics(...)` | Read `fit.diagnostic` |
| Fit failure exception | `FitError`, with `.record` listing every attempt |

Maximum likelihood (`SampleEstimator.MAXIMUM_LIKELIHOOD`) is the default sample
estimator. If you relied on an estimator falling back silently, declare the
fallback with `SampleFit(..., fallbacks=(SampleFit(...),))`.

**Before**

```py
from crc_framework import fit_censored_quantiles, fit_distribution

annual_maxima = [0.1, 0.2, 0.4, 0.7, 1.1, 1.8]
best = fit_distribution(annual_maxima, family="auto")
print(best.distribution.family, best.diagnostics.ks_pvalue)

censored = fit_censored_quantiles(
    [0.5, 0.8, 0.9, 0.96, 0.98, 0.99, 0.996, 0.998, 0.999],
    [0, 0, 0, 0, 0, 0, 0.03, 0.06, 0.10],
    family="genextreme",
    censor_location=0.0,
)
```

**After**

```python
from crc_framework import (
    CensoredQuantileFit, DistributionFamily, ModelSelection, QuantileCurve,
    SampleFit, fit_quantiles, fit_samples,
)

annual_maxima = [0.1, 0.2, 0.4, 0.7, 1.1, 1.8]
best = fit_samples(
    annual_maxima,
    ModelSelection(candidates=(
        SampleFit(family=DistributionFamily.GENERALIZED_EXTREME_VALUE),
        SampleFit(family=DistributionFamily.GUMBEL_RIGHT),
    )),
)
chosen = best.record.attempts[best.record.selected_attempt]
print(chosen.specification.family, best.diagnostic.ks_pvalue)
print(best.distribution.to_json())  # family and fitted parameters

censored = fit_quantiles(
    QuantileCurve(
        [0.5, 0.8, 0.9, 0.96, 0.98, 0.99, 0.996, 0.998, 0.999],
        [0, 0, 0, 0, 0, 0, 0.03, 0.06, 0.10],
    ),
    CensoredQuantileFit(
        family=DistributionFamily.GENERALIZED_EXTREME_VALUE, censor_location=0.0
    ),
)
assert censored.record.converged
```

Specifications serialize with `to_json()`, so you can store the exact fitting
choice next to its result.

## Impacts

Impacts are one `Impact` type with explicit input and output measurements. The
three operations that used to share `__call__` now have separate names.

| 0.x | 1.0 |
| --- | --- |
| `LinearImpact(slope=..., maximum=...)` | `Impact.linear(slope=..., maximum=..., input=..., output=...)` |
| `SigmoidImpact(midpoint, steepness, minimum=a, maximum=b)` | `Impact.sigmoid(midpoint=..., steepness=..., offset=a, amplitude=b - a, input=..., output=...)` |
| `PiecewiseLinearImpact(exposure=xs, impact=ys)` | `Impact.tabulated(list(zip(xs, ys)), input=..., output=...)` |
| `CallableImpact`, `CallableTransform` | Removed; use `Impact.tabulated` or `Impact.binned` |
| `impact.evaluate(values)` | `impact.evaluate(values)` (unchanged) |
| `impact(distribution)`, `transform_distribution` | `impact.pushforward(distribution)` |
| `expected_value(...)` | `impact.expectation(distribution)` |
| `impacts.for_factor(RiskFactor.X, context=TransformContext(...))` | `Impact.builtin(BuiltinImpact.X).resolve(<typed context>)` |
| `ImpactExpression`, `RegistryImpactReference` | `Impact` / `ImpactReference` with `to_json()` / `from_json()` |
| `overrides=` on registry impacts | Removed; build your own `Impact` when you need different parameters |

`SigmoidImpact(zero_below=...)` has no direct equivalent; use
`Impact.tabulated` if you need a hard cut-off.

Like 0.x `expected_value`, `expectation` uses an exact calculation when one is
available and adaptive quadrature otherwise, for example for a sigmoid over a
parametric law. Pass `method=ExpectationMethod.EXACT` to require an exact
result; it raises `ImpactError` with code `exact_method_unavailable` when there
is none. The expectation equals the mean of the pushforward, up to
integration accuracy:

```python
import math

from crc_framework import (
    Distribution, DistributionFamily, ExpectationMethod, Impact, ImpactError,
    Ratio, Statistic,
)

hazard = Distribution.parametric(
    DistributionFamily.GUMBEL_RIGHT, location=1.0, scale=0.5
)
sigmoid = Impact.sigmoid(midpoint=1.0, steepness=2.0, input=Ratio(), output=Ratio())
expected = sigmoid.expectation(hazard)  # quadrature
response = sigmoid.pushforward(hazard)
mean = response.statistics(metrics=[Statistic.MEAN]).values[Statistic.MEAN].value
assert math.isclose(expected, mean, rel_tol=1e-6)

try:
    sigmoid.expectation(hazard, method=ExpectationMethod.EXACT)
except ImpactError as error:
    assert error.code == "exact_method_unavailable"
```

**Before**

```py
from crc_framework import (
    LinearImpact, RiskFactor, TabulatedDistribution, TransformContext, impacts,
)

exposure = TabulatedDistribution([0.5, 0.9, 0.99], [0.5, 1.0, 2.0])
linear = LinearImpact(slope=0.25, maximum=1.0)
print(linear.evaluate([0.5, 2.0]))

flood = impacts.for_factor(
    RiskFactor.INUNDATION,
    context=TransformContext(continent="Europe", building_type="Commercial buildings"),
)
print(flood.evaluate([1.0]))
```

**After**

```python
from crc_framework import (
    AssetType, BuiltinImpact, DepthAboveGround, FloodRegion, Impact,
    InundationContext, LengthUnit, Ratio,
)

depth = DepthAboveGround(unit=LengthUnit.METRE)
linear = Impact.linear(
    slope=0.25, minimum=0.0, maximum=1.0, input=depth, output=Ratio()
)
print(linear.evaluate([0.5, 2.0]))

model = Impact.builtin(BuiltinImpact.INUNDATION)
print(model.requirements.context_type)  # InundationContext
flood = model.resolve(
    InundationContext(region=FloodRegion.EUROPE, asset_type=AssetType.COMMERCIAL)
)
print(flood.evaluate([1.0]))
```

### Built-in model names

`RiskFactor`, `Pathway`, `RISK_FACTORS`, `PATHWAYS`, and `HORIZONS` are removed.
Built-in impacts are listed in `BuiltinImpact`; scenarios and years are data
declarations in crc-sdk.

| 0.x `RiskFactor` | 1.0 `BuiltinImpact` | Context |
| --- | --- | --- |
| `CFLOOD` | `COASTAL_FLOOD` | `InundationContext(region, asset_type)` |
| `RFLOOD` | `RIVER_FLOOD` | `InundationContext(region, asset_type)` |
| `INUNDATION` | `INUNDATION` | `InundationContext(region, asset_type)` |
| `RX1DAY` | `MAXIMUM_DAILY_PRECIPITATION` | `CellContext(cell)` |
| `WIND_MAX_DAILY_MAX` | `MAXIMUM_DAILY_WIND_SPEED` | `CountryContext(country)` |
| `CYCLONE` | `CYCLONE` | `CountryContext(country)` |
| `FWI` | `FIRE_WEATHER_INDEX` | `EmptyContext()` |
| `HOT_DAYS` | `HOT_DAYS` | `EmptyContext()` |
| `FROST_DAYS` | `FROST_DAYS` | `CellContext(cell)` |
| `DAILY_FREEZETHAW_CYCLES` | `FREEZE_THAW_CYCLES` | `CellContext(cell)` |
| `SPEI` | `STANDARDIZED_PRECIPITATION_EVAPOTRANSPIRATION_INDEX` | `WaterCostContext(country, asset_type)` |
| `COOLING_DEGREE_DAYS` (non-agricultural) | `COOLING_DEGREE_DAYS` | `CoolingCostContext(country, asset_type, historic_mean)` |
| `COOLING_DEGREE_DAYS` (agricultural) | `COOLING_DEGREE_DAYS_CROP_YIELD` | `CropContext(country, historic_mean)` |
| `MAXIMUM_CONSECUTIVE_DRY_DAYS` | `CONSECUTIVE_DRY_DAYS` | `CropContext(country, historic_mean)` |

The old free-text `continent` and `building_type` strings are now
`FloodRegion` and `AssetType` enums. H3 cells in contexts are integers.

Ten of these models list `requirements.unestablished_facts`, such as the
calibration reference period. `resolve()` still gives you a numerical `Impact`
for direct calculations, but crc-sdk will not bind those models to datasets.
Only the three flood models and `FIRE_WEATHER_INDEX` bind to data in 1.0. See
the [measurements guide](measurements.md#published-models).

## Microscores and risk

`generate_microscores`, `MicroscoreSuite`, `ScenarioMetadata`, and
`compute_spanning_set` are removed. Use `rank_scenarios` for the exposure at
chosen ranks, `Impact.evaluate` for the impact at that exposure, and
`Distribution.statistics` for summaries.

### Risk aggregation sums instead of averaging

`BinaryOutcome` now names both states explicitly. The old "downside" is the
upper state; the old "upside" is the lower state.

| 0.x | 1.0 |
| --- | --- |
| `BinaryOutcome(factor, downside_probability=p, downside_impact=x)` | `BinaryOutcome(id, lower_rank_mass=1 - p, lower_impact=0.0, upper_rank_mass=p, upper_impact=x)` |
| `upside_probability`, `upside_impact` | `lower_rank_mass`, `lower_impact` |
| `weight` | Removed; scale impacts yourself |
| `level.var`, `level.cvar` | `level.value_at_risk`, `level.conditional_value_at_risk` |
| `attribution.factor`, `.var_impact`, `.cvar_impact` | `attribution.id`, `.value_at_risk`, `.conditional_value_at_risk` |

**Before**

```py
from crc_framework import (
    BinaryOutcome, LinearImpact, TabulatedDistribution, compute_risk,
    generate_microscores,
)

exposure = TabulatedDistribution([0.5, 0.9, 0.99], [0.5, 1.0, 2.0])
impact = LinearImpact(slope=0.25, maximum=1.0)
suite = generate_microscores(exposure, impact=impact(exposure), probabilities=[0.95])
flood = suite.at(0.95)  # BinaryOutcome with downside impact ~0.389

outcomes = [flood, BinaryOutcome("fire", downside_probability=0.10, downside_impact=0.2)]
level = compute_risk(outcomes).at(0.95)
print(level.var, level.cvar)
```

**After**

```python
from crc_framework import (
    BinaryOutcome, Impact, QuantileCurve, Ratio, TailCompletion, compute_risk,
    rank_scenarios,
)

exposure = QuantileCurve([0.5, 0.9, 0.99], [0.5, 1.0, 2.0]).complete(
    TailCompletion(lower=0.0, upper=3.0)
)
impact = Impact.linear(
    slope=0.25, minimum=0.0, maximum=1.0, input=Ratio(), output=Ratio()
)
(scenario,) = rank_scenarios(exposure, probabilities=[0.95])
flood = BinaryOutcome(
    "flood",
    lower_rank_mass=scenario.lower_rank_mass,
    lower_impact=0.0,
    upper_rank_mass=scenario.upper_rank_mass,
    upper_impact=float(impact.evaluate(scenario.value)),
)
fire = BinaryOutcome(
    "fire", lower_rank_mass=0.90, lower_impact=0.0, upper_rank_mass=0.10, upper_impact=0.2
)
level = compute_risk([flood, fire], levels=[0.95]).at(0.95)
print(level.value_at_risk, level.conditional_value_at_risk)
```

The 1.0 numbers are twice the 0.6 numbers here, because two factors are now
summed rather than averaged. To reproduce the 0.x weighted average, multiply
each `upper_impact` by `weight / total_weight` before calling `compute_risk`.

## Geography

`lookup_continent`, `lookup_ipcc_region`, and `lookup_geography` accept only
integer H3 cells. Convert hexadecimal strings with `int(cell, 16)`.

```python
from crc_framework import lookup_geography

geography = lookup_geography(int("85283473fffffff", 16))
if geography is not None:
    print(geography.continent, geography.countries)
```

## Stored data and JSON

All 1.0 objects round-trip through `to_json()` and the matching `from_json()`.
Numerical documents use format `1.0` and resolved impacts use format `2.0`.

```python
from crc_framework import Distribution

restored = Distribution.from_json(law.to_json())
assert restored.quantile(0.5) == law.quantile(0.5)
```

`from_json()` checks the whole document before loading it. If you assemble
documents by hand, nest only the bare law: a hurdle `base` written as a full
`to_json()` document, with its own `object` and `schema_version` fields, is
rejected.

The 0.x canonical columns (`curve_kind`, `curve_type`, `curve_location`,
`curve_scale`, `curve_shape`, `curve_atom_*`, `curve_probabilities`,
`curve_values`) and impact schema 1.0 documents cannot be loaded. Regenerate
stored datasets with crc-sdk 1.0, or rebuild each curve from its old columns
with the constructors above.

## JavaScript and WebAssembly

The JavaScript package reads the same numerical JSON documents that Python
writes. Flat `curve_*` objects, including PMTiles/MVT feature properties in the
0.x layout, are no longer accepted.

| 0.x | 1.0 |
| --- | --- |
| `evaluateCurve(properties, q)` | `quantile(numericalJSONText, q)` |
| `evaluateCurves(rows, q)` | `prepareNumericals(arrayJSONText).quantile(q)` |
| `prepareCurve(properties).evaluate(q)` | `prepareNumerical(text).quantile(q)` |
| `prepareCurves(rows).evaluate(q)` | `prepareNumericals(arrayJSONText).quantile(q)` |
| Returned `null` outside support or for no-data | Throws; missing data is a dataset failure, not a curve |
| `initSync({ module })` | Removed; use `await init(bytesOrURL)` |
| `init({ module_or_path: url })` | `init(url)` |

**Before**

```js
import init, { prepareCurves } from "@climateriskcommons/crc-framework";

await init();
const curves = prepareCurves(tileFeatures.map((f) => f.properties));
const values = curves.evaluate(0.99); // Array<number | null>
curves.free();
```

**After**

```js
import init, { prepareNumericals } from "@climateriskcommons/crc-framework";

await init();
// Each item is a numerical document, e.g. the output of Python's to_json().
const curves = prepareNumericals(JSON.stringify(documents));
try {
  const values = curves.quantile(0.99); // Float64Array
} finally {
  curves.free();
}
```

`validateDocument(DocumentSchema.DATASET, text)` validates dataset, numerical,
and impact documents with the same rules as Python.

## New in 1.0

- `QuantileCurve` keeps supplied evidence separate from complete laws, with
  explicit `complete` and `condition` operations.
- `exceedance`, `mass`, and `cdf(inclusive=...)` answer threshold questions
  directly, including at atoms.
- `statistics` integrates rare masses exactly and reports finite, infinite, or
  undefined results through `StatisticStatus`.
- Typed fitting specifications, `ModelSelection`, declared fallbacks, and a
  `FitRecord` that lists every attempt.
- Exact `pushforward` for impacts, including decreasing functions and atoms;
  `expectation` is exact where possible, and you can require it with
  `ExpectationMethod.EXACT`; `Impact.binned` for step functions.
- Typed measurements and units (`DepthAboveGround`, `Temperature`,
  `PrecipitationAmount`, …) and model requirements you can inspect before
  binding data.
- JavaScript `validateDocument` and packaged JSON Schemas shared with Python
  and crc-sdk.

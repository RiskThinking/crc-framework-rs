# Climate Risk Commons Framework

The Climate Risk Commons Framework provides a reusable Rust computation core
and typed Python bindings for climate-risk calculations. It turns
climate-hazard distributions into comparable impact microscores, scenario
branches, and portfolio-level VaR/CVaR risk metrics.

This repository owns the native computation layer. The Python distribution is
named `crc-framework` and imported as `crc_framework`. It packages the PyO3
extension together with a typed binding layer. Higher-level, application-
specific Python functionality is intended to live in a separate package that
depends on and may re-export this API.

## Features

- Empirical, tabulated, and parametric probability distributions
- Explicit parametric fitting with diagnostics and acceptance constraints
- Built-in and composable exposure-to-impact transforms
- Climate impact registry keyed by risk factor and scenario context
- Microscores that turn distribution quantiles into binary risk outcomes
- Spanning sets, VaR, CVaR, and factor-level attribution
- H3-based IPCC region, continent, and country lookups

## Installation

Install the published Python wheel with:

```shell
python -m pip install crc-framework
```

The package requires Python 3.9 or later. Installing a compatible prebuilt
wheel does not require a Rust toolchain. Building from source requires Rust
1.85 or later and a C compiler suitable for building Python extensions.

Verify the installation:

```shell
python -c "import crc_framework; print(crc_framework.__doc__)"
```

For local development, create an environment and install the project in
editable mode:

```shell
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip maturin
.venv/bin/python -m pip install -e ".[test]"
```

After changing Rust code, rebuild the extension:

```shell
.venv/bin/maturin develop
```

## Core concepts

### Distributions

All calculations are expressed through a distribution interface with `pdf`,
`cdf`, `ppf`, `quantiles`, and `sample` operations. Scalar and NumPy array
inputs are supported for the evaluation methods.

Choose a distribution according to the form of the available data:

- `EmpiricalDistribution` for raw observations.
- `TabulatedDistribution` for explicit probability/value pairs.
- `FittedDistribution` for a known parametric family and its parameters.
- `HurdleDistribution` for a point mass followed by a truncated parametric
  tail.

`TabulatedDistribution` does not extrapolate by default. Querying outside its
declared probability range raises an error unless `extrapolate=True` is set.

### Probability convention

All metric APIs use non-exceedance probability `q` in `[0, 1]`. Return periods
are accepted only when constructing a tabulated distribution. For upper-tail
hazards, a return period `T` is converted to `q = 1 - 1/T`; therefore, the
upper-tail 100-year return level is queried at `q=0.99`.

### Fitting

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

## Fit observations explicitly

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

from crc_framework import CallableImpact, LinearImpact, generate_microscores

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
  `FittedDistribution`, `HurdleDistribution`, `fit_distribution`,
  `fit_quantiles`, `fit_hurdle_quantiles`, `fit_all`, and `quality_metrics`.
- Transforms: `impacts`, `ImpactRegistry`, `ImpactFunction`, `ClimateImpact`,
  `LinearImpact`, `SigmoidImpact`, `PiecewiseLinearImpact`, `CallableImpact`,
  and `CallableTransform`.
- Metrics: `generate_microscores`, `compute_spanning_set`, and `compute_risk`.
- Models and constants: `ScenarioMetadata`, `TransformContext`, `RiskFactor`,
  `Pathway`, `RISK_FACTORS`, `PATHWAYS`, and `HORIZONS`.
- Spatial lookup: `lookup_continent`, `lookup_ipcc_region`, and
  `lookup_geography`.

## Package structure

- `crates/core` contains the reusable Rust algorithms and reference data.
- `src/lib.rs` exposes the Rust core through PyO3.
- `python/crc_framework` contains the typed Python binding layer and type
  information shipped in the wheel.

Consumers should import the stable `crc_framework` API rather than importing
`crc_framework._core` directly. The `_core` module is an implementation detail
and may change between releases.

## License

CRC Framework is licensed under the GNU Affero General Public License, version
3 or any later version (`AGPL-3.0-or-later`). See [LICENSE](LICENSE).

Parts of the numerical distribution implementation are derived from or
informed by SciPy and Cephes. Their applicable copyright notices and license
terms are preserved in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Development

```shell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
.venv/bin/python -m mypy
.venv/bin/python -m pytest
.venv/bin/maturin build --release
```

Run `maturin develop` again after changing the Rust extension. Python-only
changes are available directly from the editable installation.

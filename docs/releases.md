# Versioning and releases

[Documentation index](README.md) · [Project README](../README.md)

## 1.2.0a4

Fourth prerelease of 1.2.0, with no interface or format changes. Python
bookkeeping around native quantile fits is lighter: weights are normalized
with NumPy, fit records reuse the documents of their candidate
specifications, and lists of floats are type-checked in C. A twelve-candidate
selection on real 201-knot curves runs 12% faster, with identical results.

## 1.2.0a3

Third prerelease of 1.2.0, with no interface or format changes. Gumbel and
generalized Pareto hurdle tails each search a single parameter, and that
search now runs to full precision. On exact tails they again fit at least as
closely as in 1.2.0a1.

## 1.2.0a2

Second prerelease of 1.2.0. Install it the same way as 1.2.0a1. Interfaces and
formats are unchanged from 1.2.0a1.

- Hurdle tails are fitted by variable projection. For each trial shape and
  truncation point the scale and location are solved exactly, so a 999-knot
  GEV hurdle fit takes 5.7 ms instead of 197 ms. On real quantile tables no
  fit is worse than before, and many fit more closely.
- Fits under `FitBounds(maximum_value=...)` keep the support at or below the
  maximum exactly, in the same profile, instead of falling back to a general
  search.

## 1.2.0a1

Prerelease of 1.2.0 for fitting large quantile tables. Install it with
`python -m pip install --upgrade --pre crc-framework` or
`npm install @climateriskcommons/crc-framework@alpha`. Full notes come with
1.2.0.

- Quantile fits without a hurdle find the least-squares optimum directly: for
  each trial shape the location and scale are solved exactly, so only the shape
  is searched. On 999-knot curves, GEV fits take 1.1 ms instead of 21 ms and
  Gumbel fits 0.12 ms instead of 1.4 ms, at the same optimum. Skew-normal fits,
  hurdle tails and fits whose optimum violates their constraints search as
  before.
- `HurdleQuantileFit(atom_probability=None)` reads the point mass from the
  evidence: the largest probability among the knots at `atom_location`, which
  stays an explicit value with default `0`. It needs at least two knots there,
  so one specification can fit curves with and without a run of zeros inside a
  `ModelSelection`.

The fitting format moves to `1.1`, which allows a null `atom_probability`.
Every 1.0 fitting document is a valid 1.1 document, and documents that do not
use the inferred atom still read with 1.0 readers. Dataset, taxonomy and other
formats keep their versions.

## 1.1.0

Built-in impact models now carry the calibration facts the RiskThinking
science team confirmed, so more of them bind to data. Fire-weather damage
applies only in the wildland-urban interface.

- Seven models establish their input facts and bind to data that declares
  them exactly: frost days and freeze-thaw cycles (0 °C, per year), maximum
  daily precipitation (1-day accumulation), cyclone wind (10-minute sustained,
  annual maximum), maximum daily wind (no averaging), SPEI (monthly) and
  cooling degree days (18 °C base, per year). Their per-cell thresholds are
  return levels of the 1985–2024 historic baseline. The
  [measurements guide](measurements.md#published-models) lists each model's
  input facts.
- `BuiltinImpact.FIRE_WEATHER_INDEX` requires
  `WildfireContext(wildland_urban_interface=...)` and refuses to resolve
  without it. Outside the interface the response is zero damage. Bind the
  field to an asset column to decide it per asset. A saved fire-weather
  impact resolved without this context no longer re-resolves.
- `BuiltinImpact.HOT_DAYS` is retired. `Impact.builtin` warns with
  `DeprecationWarning`, binding it to data fails with `model_retired`, and
  saved documents that name it still read. crc-framework 2.0 removes it.
- The crop-yield models (`COOLING_DEGREE_DAYS_CROP_YIELD`,
  `CONSECUTIVE_DRY_DAYS`) keep their unestablished facts until the fitting
  definitions are published.

The impact context format moves to `1.1`, which adds `wildfire_context`. Every
1.0 context document is a valid 1.1 document. Other formats are unchanged.

## 1.0.1

A performance release with no interface or format changes. Distributions are
built from typed values instead of a JSON round trip, and a hurdle reuses its
base law. `Distribution.from_json` validates its text once. Scalar queries
call native code directly, and array queries pass contiguous bytes instead of
Python lists. Long arrays are evaluated across CPU threads without the GIL,
with results identical to single-threaded evaluation. Transformed laws release
the GIL during quantile and inverse-survival inversion.

## 1.0.0

This major release replaces the public Python and JavaScript interfaces. Python
exposes complete probability laws as `Distribution` and finite probability
evidence as `QuantileCurve`. Fitting uses typed specifications. Impacts use
typed input and output measurements and resolve built-in reference tables
through explicit model context. JavaScript loads the same numerical documents
in Node and browsers, and validates the shared scientific and data contracts.

The framework supports numerical and fitting format `1.0`, impact format
`2.0`, and typed measurement and context format `1.0`. Dataset schema `2.0`
and taxonomy schema `1.0` are shared with CRC-SDK. See the [Python guide](python.md)
and the [JavaScript guide](javascript.md) for runnable workflows. To upgrade existing code, follow the
[1.0 migration guide](migration-v1.md).

## Release contract

The package and artifact versions are declared in
[`release.json`](../python/crc_framework/contracts/api/release.json). The
framework Python package, Rust workspace, WASM package, and generated JavaScript
package use `1.2.0-alpha.4` (Python `1.2.0a4`), a prerelease of 1.2.0. Each
artifact format has its own version in that ledger.
Changing a package version does not automatically increment every format.

Unshipped definitions are replaced in place. An earlier branch assigning a
number does not justify another major version or a transitional parser. The
public API has one declared surface and current artifact formats; old exports,
aliases, and wire layouts impose no preservation requirement.

The package version source is `[workspace.package].version` in root
[`Cargo.toml`](../Cargo.toml), which must match the ledger. All three Rust crates
inherit it; maturin derives the Python version, and the JS build derives the npm
version. Source `js/package.json` stays private and versionless; the build writes
the public manifest to ignored `js/dist/`.

Before a release, verify declarations and the installed package's actual public
surface:

```sh
.venv/bin/python tools/api_contracts/check.py
/path/to/wheel-venv/bin/python -I /path/to/repository/tools/api_contracts/check.py --runtime
```

The runtime check validates exports, signatures, enums, and Python/native
versions against the checkout. It also checks that the wheel contains the
current Python modules, stubs, and contract resources. Run the examples with
the same wheel:

```sh
/path/to/wheel-venv/bin/python -I /path/to/repository/tools/check_examples.py
```

The Python release workflow tests each wheel and also builds and tests a wheel
from the source distribution. The JavaScript workflow tests the packed tarball
with Node, TypeScript, Vite server-side rendering, and Chromium.
Both publishers require the shared `conformance.yaml` checks to pass on their
release tag. These checks exercise the installed Python package, native Rust
contracts, and WebAssembly package.

## Publishing

1. Set the one release number in the ledger and `Cargo.toml`. Refresh
   `Cargo.lock` with Cargo and review generated package versions. For a
   prerelease, the install commands in `README.md`, `js/README.md`, and
   `docs/javascript.md` use `--pre` and the npm prerelease channel; a final
   release uses the plain commands.
2. Build and verify the Python wheel and JavaScript package for that exact
   version, including their public runtime contracts.
3. Create the GitHub release from the tested commit using the exact Cargo tag,
   such as `v1.0.0`. Mark alpha, beta, or release-candidate packages as
   prereleases.
4. Check both **Package release** and **JavaScript WASM** workflow results.
   Publishing is independent, so one registry can succeed while another fails.

| Destination | Version | Release behavior |
| --- | --- | --- |
| Rust workspace | `1.2.0-alpha.4` | Versioned source; no crates.io publishing workflow is configured |
| PyPI `crc-framework` | `1.2.0a4` | `release.yaml` builds wheels and an sdist, attaches assets, and publishes to PyPI |
| npm `@climateriskcommons/crc-framework` | `1.2.0-alpha.4` (`alpha` dist-tag) | `npm.yaml` publishes the tested generated package |

Use `v` plus the exact Cargo version for the tag. Cargo `1.0.0-alpha.1` maps to
Python `1.0.0a1`; the Python spelling is an output conversion, not another version
source. Stable packages use npm's `latest` channel; prereleases use their
corresponding channel. No hand-written npm version or CI-generated version commit
is required.

npm publishing uses GitHub OIDC with the `production` environment. The trusted
publisher authorizes organization `RiskThinking`, repository `crc-framework-rs`,
workflow `npm.yaml`, environment `production`, and direct `npm publish`. Python
publishing uses its separate workflow and the same environment.

Released tags remain fixed. An implementation fix after publication requires a
new release; rewriting an unshipped definition does not.

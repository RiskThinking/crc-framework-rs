# Development and contribution

[Documentation index](README.md) · [Project README](../README.md)

## Repository layout

| Path | Responsibility |
| --- | --- |
| `crates/core` | Shared Rust algorithms and reference data |
| `src/lib.rs` | PyO3 extension |
| `python/crc_framework` | Typed Python API shipped with the extension |
| `python/crc_framework/contracts/impact` | Generated impact schemas, catalogs, conformance fixtures, and digest manifest |
| `tools/impact_contracts` | Contract generator, schema sources, fixture cases, and independent Python reference implementation |
| `crates/wasm` | WASM exports backed by the Rust core |
| `js` | JS adapter, TypeScript declarations, build tooling, and consumer tests |
| `tests` | Python tests and shared parity fixtures |

Python consumers should import `crc_framework`; `crc_framework._core` is an
implementation detail that may change between releases.

## Python setup and checks

Run these commands from the repository root:

```sh
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip maturin
.venv/bin/python -m pip install -e ".[test]"

cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
.venv/bin/python -m mypy
.venv/bin/python -m pytest
.venv/bin/maturin build --release --locked
```

After changing Rust code, run `.venv/bin/maturin develop` to rebuild the Python
extension. Python-only edits are available directly from the editable install.

## Impact contract generation and regression checks

Edit the sources in [`tools/impact_contracts`](../tools/impact_contracts), then
regenerate the artifact from the repository root:

```sh
.venv/bin/python tools/impact_contracts/generate.py
```

Commit the generator sources and generated artifact together. The generator and
reference implementation are development tooling; Python consumers load the
generated resources through `crc_framework.contracts`. No separate tools package
is needed. When the manifest changes, update the SDK's pinned manifest digest
together with any required consumer changes.

After the Python setup above, run the focused checks:

```sh
.venv/bin/python tools/impact_contracts/generate.py --check
.venv/bin/python -m pytest tests/test_impact_contracts.py -q
cargo test -p crc-framework-core --test impact_contract_canonical --locked
cargo test -p crc-framework-core --no-default-features --test impact_contract_canonical --locked
```

The Python suite checks generation freshness, schema agreement, canonical
identities, validation errors, numerical reference cases, and provenance rules.
Numerical cases specify tolerances; canonical bytes and hashes require exact
agreement. Monotonicity cases allow documented weaker classifications, including
both monotone directions for constant functions. Rust tests consume the same
artifact to check canonical serialization, native integer limits, and source
distribution reconstruction with and without default features.

## JavaScript setup and checks

Use the Node version in [`js/.nvmrc`](../js/.nvmrc), install the
`wasm32-unknown-unknown` Rust target and `wasm-pack 0.15.0`, then:

```sh
cd js
npm ci
npm run build
npm test
npx playwright install chromium
npm run test:browser
npm run test:consumer
```

The consumer test packs the generated distribution and checks it with Vite,
TypeScript, SSR, and Chromium. See the [JavaScript build and test guide](development.md#javascript-packaging)
for packaging and validation commands.

Native and WASM transcendental math can differ slightly. Numerical regression
and parity tests use explicit tolerances; structural validation stays exact.
The [quantile fitting golden test](../crates/core/tests/quantile_fit_golden.rs)
documents the optional exact-comparison mode for refactors in a fixed environment.


## JavaScript packaging


Use the Node version in `.nvmrc`, install Rust's `wasm32-unknown-unknown` target
and `wasm-pack 0.15.0`, then run from this directory:

```sh
npm ci
npm run build
npm test
npx playwright install chromium
npm run test:browser
npm run test:consumer # packed Vite + TypeScript + SSR/browser checks
npm run benchmark
npm run pack -- --dry-run
npm run pack
```

The tracked `package.json` is a private, versionless development manifest;
`package-lock.json` pins its build/test dependencies. Keep both in Git. The build
reads Cargo metadata and writes the public package to ignored `dist/`, including
the derived version, JS adapter, declarations, WASM glue/asset, and licenses.
Development scripts, dev dependencies, and the private flag are omitted.
Source manifests and lockfiles are not rewritten by the build.

`npm run pack` (or `npm pack ./dist`) requires a successful build for the current
workspace version and creates the tarball under `dist/`. The consumer test
installs that artifact. The private development package itself cannot be published.


## Contributor credits

The roster in `.all-contributorsrc` generates the contributor tables in both
`README.md` (also the Python package README) and `js/README.md` (the npm README).
Run from the repository root:

```sh
npx --yes all-contributors-cli@6.26.1 add HANDLE code,review
npx --yes all-contributors-cli@6.26.1 add HANDLE research
npx --yes all-contributors-cli@6.26.1 generate
```

Use `code,review` for code contributors and specify other types explicitly.
Edit the roster for corrections, then regenerate both tables; do not edit the
generated tables directly.

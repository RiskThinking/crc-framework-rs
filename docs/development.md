# Development and contribution

[Documentation index](README.md) · [Project README](../README.md)

## Repository layout

| Path | Responsibility |
| --- | --- |
| `crates/core` | Shared Rust algorithms and reference data |
| `src/lib.rs` | PyO3 extension |
| `python/crc_framework` | Typed Python API shipped with the extension |
| `python/crc_framework/contracts/numerical` | Current numerical/fitting schemas and accepted/rejected query fixtures |
| `tools/numerical_contracts` | Numerical/fitting schema and fixture generators |
| `python/crc_framework/contracts/scientific` | Current measurements, contexts, model requirements, impact schemas, and shared fixtures |
| `tools/scientific_contracts` | Scientific resource and measurement-reference generator |
| `python/crc_framework/contracts/dataset`, `python/crc_framework/contracts/taxonomy` | Byte-identical copies of CRC-SDK's canonical data contracts |
| `python/crc_framework/contracts/impact` | Private numerical-kernel/table conformance resources and digest manifest |
| `tools/impact_contracts` | Private kernel schema, table/fixture generator and independent numerical reference |
| `crates/wasm` | WASM exports backed by the Rust core |
| `js` | JS adapter, TypeScript declarations, build tooling, and consumer tests |
| `tests` | Python tests and shared parity fixtures |

Python consumers should import `crc_framework`; `crc_framework._core` is an
implementation detail that may change between releases.

## Python interface contract

The authoritative public interface is
[`public-api.json`](../python/crc_framework/contracts/api/public-api.json).
It declares constructors, methods, properties, annotations, defaults, enum
members, semantic requirements, and removed import paths. Annotation aliases
inside the document describe types; they are not additional root exports.

[`release.json`](../python/crc_framework/contracts/api/release.json) records the
single package and artifact version ledger. All changes to unshipped definitions
replace those definitions in place rather than introducing transitional formats
or another major version.

Validate declarations and source version ownership without importing numerical
implementations:

```sh
.venv/bin/python tools/api_contracts/check.py
.venv/bin/python -m pytest tests/test_api_contracts.py -q
```

Enforce the full interface on an installed package before release:

```sh
/path/to/wheel-venv/bin/python -I /path/to/repository/tools/api_contracts/check.py --runtime
```

Runtime enforcement loads the contract and version ledger from the imported
package itself. It checks exact exports, removed attributes and modules, method
bindings and signatures, properties, enums, and Python/native version agreement.
`--package-root /path/to/site-packages` optionally selects and asserts the
installation directory; in declaration mode it reads an unpacked wheel directory
containing `crc_framework`. Without that explicit option it uses the interpreter's
installed package and never inserts a checkout into Python's import path. The
installed Python modules, stubs, API declarations, and schema resources must
match the release checkout.

Do not regenerate declarations from the runtime to conceal a mismatch. Numerical
regressions and cross-language fixtures test scientific behavior; an old symbol
remaining importable is not an acceptance requirement.

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

## Current interchange contracts

The current packaged contracts define numerical/fitting format `1.0`, scientific
declarations `1.0`, Impact `2.0`, dataset `2.0`, and taxonomy `1.0`. Their schemas and fixtures are packaged with
the wheel and embedded in Rust/WASM; consumers apply scientific invariants in
addition to JSON Schema structural validation. A tagged numerical object carries
only its active representation. Typed measurement/context fields are generated
from the same definitions used by runtime validation and model inspection.

Regenerate numerical and scientific resources after changing their definitions:

```sh
.venv/bin/python tools/numerical_contracts/generate.py
.venv/bin/python tools/numerical_contracts/fixtures.py
.venv/bin/python tools/scientific_contracts/generate.py
```

The scientific generator also writes [measurements and model requirements](measurements.md).
Commit generated resources with their sources. Verify freshness and consumer
agreement with:

```sh
.venv/bin/python tools/numerical_contracts/generate.py --check
.venv/bin/python tools/numerical_contracts/fixtures.py --check
.venv/bin/python tools/scientific_contracts/generate.py --check
.venv/bin/python -m pytest tests/test_measurements.py tests/test_fitting_serialization.py tests/test_numerical_impacts.py -q
cargo test -p crc-framework-core --test canonical_consumer --locked
cargo test -p crc-framework-core --test dataset_contract --locked
```

CRC-SDK owns dataset declarations, Arrow storage and taxonomy. The schema sources
are `crc_sdk/_declarations.py`, `_catalog.py` and `_schema/contract.py`; the dataset
generator is `tools/generate_dataset_contracts.py`. Its schema and fixture files,
and the taxonomy resources checked by `tests/test_catalog.py`, are vendored
unchanged into this repository's `python/crc_framework/contracts/dataset` and
`python/crc_framework/contracts/taxonomy` directories. Synchronization requires an explicit matching
SDK checkout path and copies the exact schema and fixture bytes:

```sh
.venv/bin/python tools/sync_data_contracts.py --sdk ../crc-sdk
.venv/bin/python tools/sync_data_contracts.py --sdk ../crc-sdk --check
```

Use the matching SDK environment to check its dataset generator and consumers:

```sh
../crc-sdk/.venv/bin/python ../crc-sdk/tools/generate_dataset_contracts.py --check
../crc-sdk/.venv/bin/python -m pytest ../crc-sdk/tests/test_dataset_schema.py ../crc-sdk/tests/test_declarations.py ../crc-sdk/tests/test_catalog.py -q
```

## Private table and kernel conformance

Reference-table and private kernel fixtures are maintained in
[`tools/impact_contracts`](../tools/impact_contracts). Regenerate those internal
assets from the repository root:

```sh
.venv/bin/python tools/impact_contracts/generate.py
```

Commit the generator sources and generated artifact together. The generator and
reference implementation are development tooling. Generated native-kernel
resources are loaded internally; the public Python numerical interface lives at
the `crc_framework` root. When reference tables change, review their pinned
identities and numerical conformance fixtures together.

After the Python setup above, run the focused checks:

```sh
.venv/bin/python tools/impact_contracts/generate.py --check
.venv/bin/python -m pytest tests/test_kernel_artifacts.py tests/test_numerical_impacts.py -q
cargo test -p crc-framework-core --test impact_contract_canonical --locked
cargo test -p crc-framework-core --no-default-features --test impact_contract_canonical --locked
```

The Python suites check generation freshness, calibrated impact identities,
validation errors, point values, exact expectations, pushforward atoms, and
representation-independent distribution capabilities.
Numerical cases specify tolerances; canonical bytes and hashes require exact
agreement. Monotonicity cases allow documented weaker classifications, including
both monotone directions for constant functions. Rust tests consume the same
artifact to check canonical serialization, native integer limits, and source
distribution reconstruction with and without default features.

## JavaScript setup and checks

Use the Node version in [`js/.nvmrc`](../js/.nvmrc), install Rust's
`wasm32-unknown-unknown` target and `wasm-pack 0.15.0`, then:

```sh
cd js
npm ci
npm run build
npm test
npx playwright install chromium
npm run test:browser
npm run test:consumer   # packed Vite + TypeScript + SSR/browser checks
npm run benchmark
npm run pack -- --dry-run
```

The tracked `package.json` is a private, versionless development manifest and
`package-lock.json` pins build/test dependencies; keep both in Git. The build
reads Cargo metadata and writes the public package (derived version, JS
adapter, declarations, WASM glue and asset, licences) to ignored `dist/`,
without development scripts or dependencies. `npm run pack` requires a build
for the current workspace version and creates the tarball that the consumer
test installs.

Native and WASM transcendental math can differ slightly. Numerical regression
and parity tests use explicit tolerances; structural validation stays exact.
The [quantile fitting golden test](../crates/core/tests/quantile_fit_golden.rs)
documents the optional exact-comparison mode for refactors in a fixed environment.

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

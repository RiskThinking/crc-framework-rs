# Agent guide

crc-framework is the numerical core of Climate Risk Commons: a Rust crate with
Python (PyO3) and JavaScript/WASM bindings. Data access and workflows live in
[crc-sdk](https://github.com/RiskThinking/crc-sdk).

## Read first

- [docs/README.md](docs/README.md): reading order and source-of-truth map.
- [docs/python.md](docs/python.md): Python API, probability conventions and the
  public export list.
- [docs/javascript.md](docs/javascript.md): current numerical inputs and errors.
- [docs/measurements.md](docs/measurements.md): typed scientific records and model-owned requirements.
- [python/crc_framework/contracts/api/public-api.json](python/crc_framework/contracts/api/public-api.json): authoritative Python exports, call signatures, enum members, and removals.
- [python/crc_framework/contracts/api/release.json](python/crc_framework/contracts/api/release.json): package and artifact version ledger.

## Layout

| Path | Contents |
| --- | --- |
| `crates/core` | All algorithms and reference data (edit here first) |
| `src/` | PyO3 bindings (`crc_framework._core`) |
| `python/crc_framework` | Typed Python API; `_core.pyi` must match `src/` |
| `crates/wasm`, `js/` | WASM exports and the JS adapter |
| `python/crc_framework/contracts/numerical` | Numerical/fitting schemas and shared query fixtures; edit `tools/numerical_contracts/` |
| `python/crc_framework/contracts/scientific` | Measurement/context/impact schemas, model science, private table bindings, and shared fixtures; edit `tools/scientific_contracts/` and typed Python records |
| `python/crc_framework/contracts/dataset`, `python/crc_framework/contracts/taxonomy` | Dataset/taxonomy resources shared with CRC-SDK; keep vendored bytes identical |
| `python/crc_framework/contracts/impact` | Private numerical-kernel and reference-table conformance assets; edit `tools/impact_contracts/` |
| `python/crc_framework/contracts/api` | Python interface declarations and the release version ledger |
| `tools/api_contracts/check.py` | Declaration validation and installed-package release enforcement |
| `tools/sync_data_contracts.py` | Sync/check dataset and taxonomy resources from an explicit matching SDK checkout |

## Checks

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
.venv/bin/maturin develop --release && .venv/bin/python -m pytest -q
.venv/bin/python -m mypy
.venv/bin/python tools/api_contracts/check.py
.venv/bin/python tools/numerical_contracts/generate.py --check
.venv/bin/python tools/numerical_contracts/fixtures.py --check
.venv/bin/python tools/scientific_contracts/generate.py --check
```

## Rules

- Probabilities are non-exceedance `q` in `[0, 1]`; upper-tail return period
  `T` is `q = 1 - 1/T`.
- Keep Python, JSON Schema, Arrow, Rust and WASM admission consistent. Run the
  shared accepted/rejected fixtures through their structural and scientific
  validators; numeric queries use explicit tolerances.
- Regenerate current contract resources from their owners, including the
  generated measurement reference. Dataset and taxonomy definitions belong to
  CRC-SDK; use `tools/sync_data_contracts.py --sdk PATH --check` to verify the
  framework copies are byte-identical consumers. Private table assets
  are implementation inputs and never a second public interchange contract.
- Long native calls release the GIL (`py.detach`).
- Implement the declared Python and numerical contracts directly. Remove
  superseded exports, aliases, schemas, defaults, and compatibility adapters;
  old export lists and the old JS `CurveProperties` shape impose no preservation
  requirement.
- Use one public name per concept and the canonical root Python import. Closed
  library choices use enums; user identities and custom scientific values stay
  open. Do not hide public input or result types behind `Any`.
- Keep scientific choices separate from resource handling. Fit specifications
  belong to the framework and are consumed unchanged by the SDK and catalogs.
- Follow the single release ledger. Unshipped contract revisions replace their
  definitions in place; they do not cause another major version increment.
- Installed-package releases must pass
  `tools/api_contracts/check.py --runtime`; declaration checks alone do not prove
  that an implementation satisfies the public contract.
- Docs, code and commit messages are in English.

# Agent guide

crc-framework is the numerical core of Climate Risk Commons: a Rust crate with
Python (PyO3) and JavaScript/WASM bindings. Data access and workflows live in
[crc-sdk](https://github.com/RiskThinking/crc-sdk).

## Read first

- [docs/README.md](docs/README.md): reading order and source-of-truth map.
- [docs/python.md](docs/python.md): Python API, probability conventions and the
  public export list.
- [docs/javascript.md](docs/javascript.md): canonical curve inputs and errors.

## Layout

| Path | Contents |
| --- | --- |
| `crates/core` | All algorithms and reference data (edit here first) |
| `src/` | PyO3 bindings (`crc_framework._core`) |
| `python/crc_framework` | Typed Python API; `_core.pyi` must match `src/` |
| `crates/wasm`, `js/` | WASM exports and the JS adapter |
| `python/crc_framework/contracts/impact` | Generated; edit `tools/impact_contracts/` and regenerate |

## Checks

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
.venv/bin/maturin develop --release && .venv/bin/python -m pytest -q
.venv/bin/python -m mypy
```

## Rules

- Probabilities are non-exceedance `q` in `[0, 1]`; upper-tail return period
  `T` is `q = 1 - 1/T`.
- Keep Python, Rust and WASM numerically consistent; parity tests use explicit
  tolerances.
- Long native calls release the GIL (`py.detach`).
- Do not break exported Python names or the JS `CurveProperties` contract.
- Docs, code and commit messages are in English.

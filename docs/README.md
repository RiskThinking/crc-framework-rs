# CRC Framework documentation

Start at the [project README](../README.md) for package choices, installation,
contributors, and runnable quick starts. Then follow the guide for your task:

1. **Use the framework:** [Python](python.md) or [JavaScript / WebAssembly](javascript.md).
2. **Declare science:** [measurements and model requirements](measurements.md).
3. **Understand shared contracts:** [impact mathematics and identities](impact-contracts.md).
4. **Change the framework:** [development and contribution](development.md).
5. **Ship a version:** [versioning and releases](releases.md).

## Ownership and source map

| Concern | Source of truth |
| --- | --- |
| Numerical algorithms, canonical validation, reference data | `crates/core/` |
| Typed Python interface and native binding | `python/crc_framework/`, `src/` |
| Public Python names, signatures, enum members, and removals | `python/crc_framework/contracts/api/public-api.json` |
| Package and artifact versions | `python/crc_framework/contracts/api/release.json` |
| WASM exports and JavaScript adapter | `crates/wasm/`, `js/src/` |
| Numerical/fitting schemas and query fixtures | `python/crc_framework/contracts/numerical/`; edit `tools/numerical_contracts/` |
| Measurement/context/impact schemas and model requirements | `python/crc_framework/contracts/scientific/`; edit typed Python records and `tools/scientific_contracts/` |
| Dataset/taxonomy schemas and fixtures | CRC-SDK owns the definitions; vendored copies in `python/crc_framework/contracts/dataset/` and `python/crc_framework/contracts/taxonomy/` feed native/WASM consumers |
| Private numerical-kernel/table conformance assets | `python/crc_framework/contracts/impact/`; edit `tools/impact_contracts/` |
| Package version declaration | Root `Cargo.toml`, `[workspace.package].version`; must match the release ledger |
| Contributor roster | `.all-contributorsrc`; generate both package READMEs |

Paths above are relative to the repository root. Humans and agents should use
these guides before following source details. Treat generated artifacts as outputs;
change their documented sources and regenerate them.

This directory covers the framework only. Dataset discovery, ingestion, and
Parquet workflows belong to CRC-SDK; tile loading, caches, and UI behavior belong
to consuming applications. These boundaries keep a future shared documentation
structure possible without duplicating other projects' guides here.

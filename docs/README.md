# CRC Framework documentation

Start at the [project README](../README.md) for package choices, installation,
contributors, and runnable quick starts. Then follow the guide for your task:

1. **Use the framework:** [Python](python.md) or [JavaScript / WebAssembly](javascript.md).
2. **Understand shared contracts:** [impact contracts and provenance](impact-contracts.md).
3. **Change the framework:** [development and contribution](development.md).
4. **Ship a version:** [versioning and releases](releases.md).

## Ownership and source map

| Concern | Source of truth |
| --- | --- |
| Numerical algorithms, canonical validation, reference data | `crates/core/` |
| Typed Python interface and native binding | `python/crc_framework/`, `src/` |
| Public Python names, signatures, enum members, and removals | `python/crc_framework/contracts/api/public-api.json` |
| Package and artifact versions | `python/crc_framework/contracts/api/release.json` |
| WASM exports and JavaScript adapter | `crates/wasm/`, `js/src/` |
| Impact schemas, catalogs, fixtures, manifest | Generated `python/crc_framework/contracts/impact/`; edit `tools/impact_contracts/` |
| Package version declaration | Root `Cargo.toml`, `[workspace.package].version`; must match the release ledger |
| Contributor roster | `.all-contributorsrc`; generate both package READMEs |

Paths above are relative to the repository root. Humans and agents should use
these guides before following source details. Treat generated artifacts as outputs;
change their documented sources and regenerate them.

This directory covers the framework only. Dataset discovery, ingestion, and
Parquet workflows belong to CRC-SDK; tile loading, caches, and UI behavior belong
to consuming applications. These boundaries keep a future shared documentation
structure possible without duplicating other projects' guides here.

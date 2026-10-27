# Impact contracts and provenance

[Documentation index](README.md) · [Project README](../README.md)

`crc_framework.contracts` publishes the `crc-impact-contracts` artifact, currently
version `1.0.0`, with JSON Schemas, accepted and rejected examples, canonical JSON
and SHA-256 identities, numerical reference cases, and catalogs of errors, units,
and versioned semantics.

The contracts separate the reusable function from caller data, execution
provenance, and application policy:

| Contract | Responsibility |
| --- | --- |
| `impact_definition` | Calibration quantity and units, model parameters, final clamp, and function provenance |
| `registry_impact_reference` | Pinned registry revision and manifest, factor, context defaults, overrides, and fallback policy |
| `impact_evaluation_record` | Function and source identities, operation, preprocessing, engine, diagnostics, and replay information |
| `spatial_combination_policy` | Application choice to combine aligned point exposures before evaluation or combine impacts afterward |
| `spatial_impact_evaluation_record` | Spatial policy identity, contributors, geometry coverage, weights, and child evaluation records |

A definition describes the impact function independently of a dataset or output
format. Evaluation records describe how it was applied. Spatial combination is a
separate application policy; combination of whole source distributions remains
deferred.

Artifact versions, wire `schema_version`, and numerical `semantics_version` have
separate roles. Registry references also pin resolver semantics and reference
data. Persist these identities with the evaluation provenance to make numerical
choices and data dependencies explicit.

Inspect the packaged artifact without importing development tooling:

```python
from crc_framework.contracts import (
    impact_contract_manifest_sha256,
    load_impact_fixtures,
    load_impact_schema,
    verify_impact_contracts,
)

verify_impact_contracts()
manifest_sha256 = impact_contract_manifest_sha256()
schema = load_impact_schema("impact_definition")
examples = load_impact_fixtures("impact_definition")
```

Consumers should compare the manifest SHA-256 with their pinned value and verify
the listed file digests before using the artifact. The SDK's internal contract
loader performs both checks. A verified manifest identifies the exact contract
files; replay also depends on the source data and provenance in each record.

The artifact supplies contracts and conformance expectations. Executable impacts
continue to use the classes described above. The
[native capability catalog](../python/crc_framework/contracts/impact/catalogs/native-capabilities.json)
tracks which fixture families the native implementation currently supports.

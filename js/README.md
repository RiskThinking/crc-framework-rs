# CRC Framework for JavaScript

Consume the same current numerical and scientific documents used by Python and
canonical datasets. The package includes WebAssembly, TypeScript declarations,
and the current JSON Schemas. Dataset workflows and data acquisition belong to
CRC-SDK.

```sh
npm install @climateriskcommons/crc-framework
```

Create numerical evidence in Python without authoring a serialization envelope:

```python
from pathlib import Path
from crc_framework import QuantileCurve

curve = QuantileCurve([0.9, 0.99, 0.999], [0.5, 2, 5])
Path("depth.numerical.json").write_text(curve.to_json(), encoding="utf-8")
```

Read that document in Node:

```javascript
import { readFileSync } from "node:fs";
import init, { prepareNumerical } from "@climateriskcommons/crc-framework";

const wasm = readFileSync(new URL(import.meta.resolve(
  "@climateriskcommons/crc-framework/crc_framework_bg.wasm"
)));
await init({ module_or_path: wasm });
const text = readFileSync("depth.numerical.json", "utf8");
const curve = prepareNumerical(text);
try {
  console.log(curve.quantile(0.99)); // 2: the supplied 100-year upper-tail level
  console.log(curve.quantile([0.9, 0.99])); // Float64Array
} finally {
  curve.free();
}
```

Imports are safe during server rendering; operations require initialization.
`init()` is shared and idempotent. In browsers, its default loads the packaged
WASM asset; bundlers can also supply bytes through `module_or_path`.

`quantile(text, probabilities)` performs a one-shot query. `prepareNumericals`
accepts an array of numerical documents as JSON text and returns one result per
document for each scalar probability query. Prepared handles own their native
memory; `free()` is idempotent and further queries fail.

Partial evidence is queried only on its actual interval. Invalid coordinates,
malformed documents, historical records, and out-of-support queries fail.
Mathematical infinite endpoints remain JavaScript infinities. Missing data is a
retained dataset failure, never a fictional numerical distribution or a silent
null result.

Use shared structural and scientific validation without guessing schema URIs:

```javascript
import { DocumentSchema, validateDocument } from "@climateriskcommons/crc-framework";
const datasetJSONText = await (await fetch("/dataset.json")).text();
validateDocument(DocumentSchema.DATASET, datasetJSONText);
```

Supply original JSON text to preserve unsigned64 cell identifiers and arbitrary
scientific integers. JavaScript's ordinary JSON parser cannot represent every
integer exactly. Current schemas are also available under the package's
`contracts/` export. Model provenance, scientific references, fit execution
records, spatial support, and row-key integrity are validated consistently with
Python.

See the repository's [scientific measurement reference](../docs/measurements.md)
and [Python guide](../docs/python.md) for scientific declarations and fitting.

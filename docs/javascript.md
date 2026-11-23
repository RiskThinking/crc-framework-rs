# JavaScript and WebAssembly

Consume the same current numerical and scientific documents used by Python and
canonical datasets. The package includes WebAssembly, TypeScript declarations,
and the current JSON Schemas. Dataset workflows and data acquisition belong to
CRC-SDK.

```sh
npm install @climateriskcommons/crc-framework@alpha
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

await init();
const text = readFileSync("depth.numerical.json", "utf8");
const curve = prepareNumerical(text);
try {
  console.log(curve.quantile(0.99)); // 2: the supplied 100-year upper-tail level
  console.log(curve.quantile([0.9, 0.99])); // Float64Array
} finally {
  curve.free();
}
```

Imports are safe during server rendering. Call `await init()` before operations.
Initialization loads the packaged WASM asset in Node and in browsers. It is
shared and idempotent. To load a particular WASM file, pass its URL, bytes, or
compiled module to `init(wasmInput)`.

`quantile(text, probabilities)` performs a one-shot query. `prepareNumericals`
accepts an array of numerical documents as JSON text and returns one result per
document for each scalar probability query. Prepared handles own their native
memory; `free()` is idempotent and further queries fail.

Partial evidence is queried only on its actual interval. Invalid coordinates,
malformed documents, historical records, and out-of-support queries fail.
Mathematical infinite endpoints remain JavaScript infinities. Dataset failures
distinguish source unavailability (`kind: "source_unavailable"`) from failed
fits with an execution record.

Use shared structural and scientific validation without guessing schema URIs:

```javascript
import init, { DocumentSchema, validateDocument } from "@climateriskcommons/crc-framework";

await init();
const datasetJSONText = await (await fetch("/dataset.json")).text();
validateDocument(DocumentSchema.DATASET, datasetJSONText);
```

Supply original JSON text to preserve unsigned64 cell identifiers and arbitrary
scientific integers. JavaScript's ordinary JSON parser cannot represent every
integer exactly. Current schemas are also available under the package's
`contracts/` export. Model provenance, scientific references, fit execution
records, spatial support, and row-key integrity are validated consistently with
Python.

See the repository's [scientific measurement reference](measurements.md)
and [Python guide](python.md) for scientific declarations and fitting.

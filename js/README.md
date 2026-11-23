# CRC Framework for JavaScript

Consume the same current numerical and scientific documents used by Python and
canonical datasets. The package includes WebAssembly, TypeScript declarations,
and the current JSON Schemas. Dataset workflows and data acquisition belong to
CRC-SDK.

## Contributors

Thanks to the people who shape this project through code, reviews, research,
and internal discussions and ideas.

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ChrisHampu"><img src="https://avatars.githubusercontent.com/u/6600532?s=80" width="80" alt="@ChrisHampu" /><br /><sub><b>@ChrisHampu</b></sub></a><br /><a href="https://github.com/RiskThinking/crc-framework-rs/commits?author=ChrisHampu" title="Code">💻</a> <a href="https://github.com/RiskThinking/crc-framework-rs/pulls?q=is%3Apr+reviewed-by%3AChrisHampu" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/woozyking"><img src="https://avatars.githubusercontent.com/u/2837532?s=80" width="80" alt="@woozyking" /><br /><sub><b>@woozyking</b></sub></a><br /><a href="https://github.com/RiskThinking/crc-framework-rs/commits?author=woozyking" title="Code">💻</a> <a href="https://github.com/RiskThinking/crc-framework-rs/pulls?q=is%3Apr+reviewed-by%3Awoozyking" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/awiebe42"><img src="https://avatars.githubusercontent.com/u/65562157?s=80" width="80" alt="@awiebe42" /><br /><sub><b>@awiebe42</b></sub></a><br /><a href="https://github.com/RiskThinking/crc-framework-rs/commits?author=awiebe42" title="Code">💻</a> <a href="https://github.com/RiskThinking/crc-framework-rs/pulls?q=is%3Apr+reviewed-by%3Aawiebe42" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/awiebe42" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kylalangotsky"><img src="https://avatars.githubusercontent.com/u/91707255?s=80" width="80" alt="@kylalangotsky" /><br /><sub><b>@kylalangotsky</b></sub></a><br /><a href="https://github.com/kylalangotsky" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/reshmasoman"><img src="https://avatars.githubusercontent.com/u/102987481?s=80" width="80" alt="@reshmasoman" /><br /><sub><b>@reshmasoman</b></sub></a><br /><a href="https://github.com/reshmasoman" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rkodamana"><img src="https://avatars.githubusercontent.com/u/84455891?s=80" width="80" alt="@rkodamana" /><br /><sub><b>@rkodamana</b></sub></a><br /><a href="https://github.com/rkodamana" title="Research">🔬</a> <a href="https://github.com/RiskThinking/crc-framework-rs/pulls?q=is%3Apr+reviewed-by%3Arkodamana" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dilshank"><img src="https://avatars.githubusercontent.com/u/1534562?s=80" width="80" alt="@dilshank" /><br /><sub><b>@dilshank</b></sub></a><br /><a href="https://github.com/dilshank" title="Research">🔬</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

Contribution types: 💻 code · 👀 review · 🔬 research. We recognize contributions
both on and off GitHub using [All Contributors](https://allcontributors.org/).

## Installation and quick start

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
import { readFile } from "node:fs/promises";
import init, { prepareNumerical } from "@climateriskcommons/crc-framework";

await init();
const text = await readFile("depth.numerical.json", "utf8");
const curve = prepareNumerical(text);
try {
  console.log(curve.quantile(0.99)); // 2: the supplied 100-year upper-tail level
  console.log(curve.quantile([0.9, 0.99])); // Float64Array
} finally {
  curve.free();
}
```

Import the package during server rendering or in browser code. Call
`await init()` before running queries or validation. Node.js and browser
bundlers load the packaged WebAssembly asset. To use your own copy, pass a
WebAssembly module, bytes, or a URL to `init`.

`quantile(text, probabilities)` and `prepareNumerical(text)` are strict: an
invalid document or an out-of-support query throws. `prepareNumericals` accepts
a JSON array of numerical documents and answers every document on its own, for
map tiles and other batches where one curve must not fail the rest.
`quantile(probability)` returns `{ values, outcomes }`: a `Float64Array` and a
`Uint8Array` with one entry per document. `values[i]` is NaN unless
`outcomes[i]` is `QuantileOutcome.VALUE`; the other outcomes are
`BELOW_SUPPORT`, `ABOVE_SUPPORT`, `INVALID_DOCUMENT`, and `UNAVAILABLE`.
`invalidDocuments` lists each rejected document's index and validation message.
Only text that is not a JSON array, or an invalid probability, throws. Prepared
handles own their native memory; `free()` is idempotent and further queries fail.

Partial evidence is queried only on its actual interval. Invalid coordinates,
malformed documents, historical records, and out-of-support queries fail.
Mathematical infinite endpoints remain JavaScript infinities. Missing data is a
retained dataset failure, never a fictional numerical distribution or a silent
null result.

Use shared structural and scientific validation without guessing schema URIs:

```javascript
import init, { DocumentSchema, validateDocument } from "@climateriskcommons/crc-framework";

await init();
const response = await fetch("/dataset.json");
validateDocument(DocumentSchema.DATASET, await response.text());
```

Supply original JSON text to preserve unsigned 64-bit cell identifiers and arbitrary
scientific integers. JavaScript's ordinary JSON parser cannot represent every
integer exactly. Current schemas are also available under the package's
`contracts/` export. Model provenance, scientific references, fit execution
records, spatial support, and row-key integrity are validated consistently with
Python.

See the repository's [scientific measurement reference](../docs/measurements.md)
and [Python guide](../docs/python.md) for scientific declarations and fitting.

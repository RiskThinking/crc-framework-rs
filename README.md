# Climate Risk Commons Framework

CRC Framework provides a shared Rust computation core with official Python and
JavaScript/WebAssembly bindings for climate-risk calculations. It turns hazard
distributions into explicit impact laws, rank scenarios, and
portfolio-level VaR/CVaR risk metrics.

| Interface | Package | Scope |
| --- | --- | --- |
| Python | `crc-framework` on PyPI; import `crc_framework` | Distributions, quantile evidence, fitting, impacts, risk aggregation, and geography |
| JavaScript / WASM | `@climateriskcommons/crc-framework` on npm | Numerical document queries (quantiles, batches, prepared handles), shared schema validation, and TypeScript declarations |
| Rust | `crates/core` | Shared algorithms and reference data used by both bindings |

The JavaScript binding reads the numerical documents that Python writes and
validates the shared data contracts. It does not read the flat `curve_*`
properties of 0.x PMTiles/MVT features. Python also supports fitting, impacts,
and portfolio risk.
Application-specific workflows belong in packages built on this framework.

[Installation](#installation) · [Quick start](#quick-start) · [Guides](#guides-and-next-steps)

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

## Installation

### Python

Until 1.0.0 is final, a plain install resolves to the 0.x line, which does not
have the interfaces shown here. Install the 1.0 prerelease:

```sh
python -m pip install --upgrade --pre crc-framework
```

Python 3.11 or later is required. Compatible prebuilt wheels need no Rust
toolchain. Building from source requires Rust 1.87 or later and a C compiler
suitable for Python extensions.

### JavaScript / WebAssembly

```sh
npm install @climateriskcommons/crc-framework@alpha
```

Stable releases use npm's `latest` channel; the 1.0 prerelease is published
under `alpha`. The package is ESM, requires WebAssembly, and includes TypeScript
declarations. See the [JavaScript guide](docs/javascript.md) for browser, bundler, Node,
and SSR setup, canonical input rules, and prepared collection APIs.

## Quick start

### Python: evaluate a return-period curve

```python
from crc_framework import QuantileCurve

exposure = QuantileCurve(
    probabilities=[0.9, 0.95, 0.98, 0.99, 0.995, 0.998],
    values=[0.1, 0.2, 0.5, 0.9, 1.4, 2.0],
)
print(exposure.quantile(0.99))  # 100-year upper-tail return level: 0.9
```

### JavaScript: consume a numerical document

```javascript
import init, { prepareNumerical } from "@climateriskcommons/crc-framework";

await init();
const text = await (await fetch("/depth.numerical.json")).text();
const curve = prepareNumerical(text);
try {
  console.log(curve.quantile(0.99));
} finally {
  curve.free();
}
```

Create `depth.numerical.json` with Python's `QuantileCurve.to_json()` or use an
existing current numerical document. The [JavaScript guide](docs/javascript.md)
covers Node initialization, array queries, and shared schema validation.

## Guides and next steps

| Task | Read next |
| --- | --- |
| Use distributions, fitting, impacts, rank scenarios, or risk metrics | [Python guide](docs/python.md) |
| Initialize WASM, prepare collections, or understand canonical inputs and errors | [JavaScript guide](docs/javascript.md) |
| Pin schemas, validate identities, or record impact provenance | [Impact contracts](docs/impact-contracts.md) |
| Set up the repository, run checks, or update contributor credits | [Development](docs/development.md) |
| Check release contracts or publish a release | [Releases](docs/releases.md) |

Start with the [documentation index](docs/README.md) for the reading order and
repository boundaries.

## License

CRC Framework is licensed under the GNU Affero General Public License, version
3 or any later version (`AGPL-3.0-or-later`). See [LICENSE](LICENSE).

Parts of the numerical distribution implementation are derived from or
informed by SciPy and Cephes. Their applicable copyright notices and license
terms are preserved in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

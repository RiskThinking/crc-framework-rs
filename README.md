# Climate Risk Commons Framework

CRC Framework provides a shared Rust computation core with official Python and
JavaScript/WebAssembly bindings for climate-risk calculations. It turns hazard
distributions into comparable impact microscores, scenario branches, and
portfolio-level VaR/CVaR risk metrics.

| Interface | Package | Scope |
| --- | --- | --- |
| Python | `crc-framework` on PyPI; import `crc_framework` | Distributions, fitting, impacts, microscores, risk aggregation, and geography |
| JavaScript / WASM | `@climateriskcommons/crc-framework` on npm | Canonical CRC curve consumption: quantile evaluation, batches, prepared handles, and TypeScript declarations |
| Rust | `crates/core` | Shared algorithms and reference data used by both bindings |

The JavaScript binding consumes canonical curves, including PMTiles/MVT
properties. Python also supports fitting, impacts, and portfolio risk.
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

```sh
python -m pip install crc-framework
# Include prereleases during the 0.5.3 alpha cycle:
python -m pip install --upgrade --pre crc-framework
```

Python 3.11 or later is required. Compatible prebuilt wheels need no Rust
toolchain. Building from source requires Rust 1.85 or later and a C compiler
suitable for Python extensions.

### JavaScript / WebAssembly

```sh
npm install @climateriskcommons/crc-framework@alpha
```

Use `@alpha` for the current prerelease channel; stable releases use npm's
`latest` channel. The package is ESM, requires WebAssembly, and includes TypeScript
declarations. See the [JavaScript guide](docs/javascript.md) for browser, bundler, Node,
and SSR setup, canonical input rules, and prepared collection APIs.

## Quick start

### Python: evaluate a return-period curve

```python
from crc_framework import TabulatedDistribution

exposure = TabulatedDistribution.from_return_periods(
    periods=[10, 20, 50, 100, 200, 500],
    values=[0.1, 0.2, 0.5, 0.9, 1.4, 2.0],
    tail="upper",
)
print(exposure.ppf(0.99))  # 100-year return level: 0.9
```

### JavaScript: consume a canonical curve

```js
import init, { evaluateCurve } from '@climateriskcommons/crc-framework';

await init(); // initialize once at client startup
const properties = {
  curve_kind: 'fitted',
  curve_type: 'gumbel_r',
  curve_location: 1,
  curve_scale: 2,
};
const value = evaluateCurve(properties, 0.99);
```

Evaluation is synchronous after initialization. Imports are SSR-safe; initialize
in the client, or load WASM bytes explicitly for Node evaluation. For repeated
viewport or percentile queries, use `prepareCurves` to validate and transfer a
collection once, then call `evaluate(q)` and free the handle when finished.

## Guides and next steps

| Task | Read next |
| --- | --- |
| Use distributions, fitting, impacts, microscores, or risk metrics | [Python guide](docs/python.md) |
| Initialize WASM, prepare collections, or understand canonical inputs and errors | [JavaScript guide](docs/javascript.md) |
| Pin schemas, validate identities, or record impact provenance | [Impact contracts](docs/impact-contracts.md) |
| Set up the repository, run checks, or update contributor credits | [Development](docs/development.md) |
| Bump versions, migrate the npm scope, or publish a release | [Releases](docs/releases.md) |

Start with the [documentation index](docs/README.md) for the reading order and
repository boundaries. Historical implementation notes are linked there separately.

## License

CRC Framework is licensed under the GNU Affero General Public License, version
3 or any later version (`AGPL-3.0-or-later`). See [LICENSE](LICENSE).

Parts of the numerical distribution implementation are derived from or
informed by SciPy and Cephes. Their applicable copyright notices and license
terms are preserved in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

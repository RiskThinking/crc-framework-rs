# @climateriskcommons/crc-framework

Official JavaScript/WebAssembly bindings for CRC Framework: consume canonical
climate curves, evaluate quantiles, and reuse prepared collections. The ESM
package includes TypeScript declarations and has no runtime npm dependencies.

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
npm install @climateriskcommons/crc-framework
```

```js
import init, { evaluateCurve, evaluateCurves, prepareCurve, prepareCurves } from '@climateriskcommons/crc-framework';

await init(); // client startup, once; imports themselves are safe during SSR
const properties = {
  curve_kind: 'fitted', curve_type: 'gumbel_r',
  curve_location: 1, curve_scale: 2,
};
const value = evaluateCurve(properties, 0.99);
const values = evaluateCurves([properties, properties], 0.99);
const curve = prepareCurve(properties);
try {
  const quantiles = curve.quantiles([0.5, 0.9, 0.99]);
} finally {
  curve.free(); // prepared handles own WASM memory
}
```

Use `@alpha` for prereleases and omit it for stable releases. From
`0.5.3-alpha.1`, replace imports of `@riskthinking.ai/crc-framework` with
`@climateriskcommons/crc-framework`, including the WASM asset subpath.

## Guides

- [JavaScript guide](../docs/javascript.md): browser/bundler/Node initialization,
  prepared handles, canonical inputs, support ranges, and errors.
- [Project README](../README.md): Python quick start and framework scope.
- [Documentation index](../docs/README.md): reading order and responsibility boundaries.
- [Development](../docs/development.md): build, test, and packaging.
- [Releases](../docs/releases.md): versions and publishing.

## License

AGPL-3.0-or-later. See the packaged `LICENSE` and `THIRD_PARTY_NOTICES.md`.

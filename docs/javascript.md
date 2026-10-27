# JavaScript / WebAssembly guide

[Documentation index](README.md) · [Project README](../README.md)

For installation and a runnable first example, start with the
[JavaScript package README](../js/README.md). The examples below assume the
package exports have been imported as shown there.

## Initialization

`q` is a non-exceedance probability in `[0, 1]`. Evaluation is synchronous after
initialization. Concurrent calls to `init` share a promise; failed loads can be
retried. The package is ESM, requires WebAssembly, has no runtime npm dependencies,
and does not initialize on import. Vite handles the WASM asset through `new URL`.
Other bundlers must copy/serve that asset or pass an explicit URL:

```js
await init({ module_or_path: new URL('/assets/crc_framework_bg.wasm', location.href) });
```

Serve `.wasm` as `application/wasm` for streaming compilation. Follow the host
application's CSP policy for WebAssembly execution. Node and SSR code that
explicitly needs evaluation can load bytes instead of fetching a `file:` URL:

```js
import { readFile } from 'node:fs/promises';
const bytes = await readFile(new URL(import.meta.resolve('@climateriskcommons/crc-framework/crc_framework_bg.wasm')));
await init({ module_or_path: bytes });
// Or initSync({ module: bytes }) for synchronous startup.
```

## Input and errors

For repeated tile queries or percentile changes, prepare the collection once:

```js
const curves = prepareCurves(propertiesArray);
try {
  const median = curves.evaluate(0.5);
  const tail = curves.evaluate(0.99);
} finally {
  curves.free();
}
```

Preparation validates and copies a snapshot of the canonical inputs. Subsequent
`evaluate(q)` calls process the collection inside one WASM call, preserve input
order, and return `Array<number | null>`. Changes to the original objects do not
change the prepared snapshot; prepare a new collection when tile data changes.
Both prepared APIs require `free()` when finished; double-free is safe and use
after free throws `Error`. Ordinary `evaluateCurves` is a one-shot convenience API
that includes conversion/validation on every call.

The binding transfers contiguous float64 buffers internally. Optional-field
presence is explicit, so supplied NaN cannot masquerade as missing input. Rust
still owns all canonical validation, reconstruction and numerical algorithms.
Numeric codes/names are supplied by Rust during initialization. The numeric wire
format is private; consumers continue to pass canonical objects. MVT string-array
decoding has a bounded internal cache. Public missing values remain null.

The public TypeScript declarations describe `CurveProperties`. Unknown feature
properties are ignored; omitted optional fields equal null. Numeric fields must
be numbers, not numeric strings. Array fields accept arrays or MVT JSON strings.
All rules follow CRC-SDK's canonical representation:

- Fitted: one of `genextreme`, `genpareto`, `weibull_min`, `weibull_max`, `gumbel_r`,
  `gumbel_l`, `skewnorm`; finite location, positive scale, required shape except
  Gumbel. Gumbel shape is normalized away by Rust.
- Hurdle: fitted base plus finite atom location and atom probability strictly in
  `(0, 1)`; the base must have positive tail probability above the atom.
- Point mass: type `point_mass`, finite location, no shape, scale `0`, atom
  probability `1`, atom location equal to location. Include these canonical fields
  even though older frontend evaluators accepted abbreviated point masses.
- Tabulated: type `linear_probability`, at least two finite matching knots,
  strictly increasing probabilities in `[0,1]`, nondecreasing values; no scalars.
- No data: a declared canonical reason, no parameters.

Invalid probabilities throw `RangeError`. Malformed records and unsupported
kinds/families throw `TypeError`. A malformed record aborts the entire batch;
empty batches still validate `q`. Calls before initialization throw `Error`.

`null` means no data or a valid probability outside curve support: fitted excludes
both endpoints, hurdle excludes `1`, point mass includes both, and tabulated
supports only its stored knot interval (including endpoints). No clamping or
extrapolation occurs. Malformed records are validated even outside support.
Native and WASM transcendental math can differ slightly; parity tests use tolerances.

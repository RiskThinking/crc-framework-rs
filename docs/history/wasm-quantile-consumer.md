# Historical rollout notes

[Documentation index](../README.md)

This records the October 2026 initial rollout, its measurements, and decisions.
Package names and release setup below describe that rollout. For current usage,
read the [project README](../../README.md), [JavaScript guide](../javascript.md),
and [release guide](../releases.md).

# Official CRC Framework JavaScript / WebAssembly binding

Created: 2026-10-09
Updated: 2026-10-16
Release target: CRC Framework 0.4.0; initial package 0.4.0-alpha.1, optimized binding 0.4.0-alpha.2.

## Objective and scope

Extend the official high-level CRC Framework binding from Python to JavaScript /
WebAssembly, published as `@riskthinking.ai/crc-framework`. The initial 0.4.0
rollout consumes canonical CRC curves already present in PMTiles. Later releases
will add broader Python parity and canonicalization so a browser-installed CRC-SDK
can combine with tools such as DuckDB-WASM for client-side climate analytics.

The initial release does not require dataset discovery, ingestion, fitting,
Parquet access or Python SDK workflow extraction. Numerical algorithms already
live in `crc-framework-core`. The extraction needed now is the transport-neutral
canonical record validation/reconstruction contract from
`crc-sdk/crc_sdk/types/hazard.py::CurveParameters`.

## Implemented initial rollout

- `crates/core/src/canonical.rs`: public `CurveParameters::reconstruct()` and
  `CanonicalCurve::quantile(q) -> Result<Option<f64>>`. Rust owns canonical
  invariants and dispatch to existing fitted, hurdle, point-mass and tabulated
  distributions. Unknown feature properties are ignored; optional omissions equal
  null. Gumbel shape normalization is in Rust.
- `crates/wasm`: separate `wasm-bindgen` crate, depending on the core without its
  default `sampling` feature. Existing native/Python builds retain sampling;
  quantile consumption needs no entropy, random polyfill or `getrandom` override.
- `js`: ESM npm package with TypeScript declarations, browser-safe imports,
  explicit asynchronous/synchronous initialization, scalar and batch APIs, and
  explicitly freed prepared handles for repeated queries. Only MVT array-string
  decoding, numeric buffer packing and language error conversion live in the JS adapter.
- `visual-eyes`: replace the handwritten evaluator with the package. Initialize
  from client map startup (single and compare views), before evaluation handlers
  are enabled. Reuse up to 4,096 prepared curves with LRU eviction/freeing and
  cache the last probability/result per curve for repeated idle/hover scans.
  Paint, extrema, legend timeline and building joins retain their existing caller
  interfaces. Bulk cell collection now uses prepared collections with a separate
  bounded viewport cache (four collections / 32,768 records). Direct canonical
  field comparison permits reuse across new MapLibre property objects and
  percentile changes without formatting per-curve cache keys. Curve null results
  override stale hover properties/feature state;
  null must also remain null rather than becoming `Number(null) === 0`.
- `.github/workflows/npm.yaml`: native/Python parity, WASM/Node tests, real Chromium
  tests, isolated packed Vite production consumer and TS declarations check;
  publish the tested tarball on matching version tags using npm OIDC.

Public package API:

```ts
import init, {
  evaluateCurve, evaluateCurves, prepareCurve, prepareCurves,
} from '@riskthinking.ai/crc-framework';

await init();
evaluateCurve(properties, q);       // number | null
evaluateCurves(propertiesArray, q); // Array<number | null>
const curve = prepareCurve(properties);
try {
  curve.evaluate(q);
  curve.quantiles([0.5, 0.99]);
} finally {
  curve.free();
}
const collection = prepareCurves(propertiesArray);
try {
  collection.evaluate(0.5);
  collection.evaluate(0.99); // one WASM call for the whole prepared collection
} finally {
  collection.free();
}
```

Input fields are `curve_kind`, `curve_type`, `curve_shape`, `curve_location`,
`curve_scale`, `curve_atom_probability`, `curve_atom_location`,
`curve_probabilities`, `curve_values`. Numeric fields require JS numbers; arrays
can be native arrays or MVT JSON strings. Transport coercion of numeric strings
is deliberately not part of the canonical contract.

### Evaluation contract

`q` is a non-exceedance probability in `[0,1]`, not a percentile or return period.

| Kind | Quantile behavior | Supported q |
| --- | --- | --- |
| fitted | Existing Rust fitted-family PPF | `0 < q < 1` |
| hurdle | Atom, then base conditioned above the atom | `0 <= q < 1` |
| point_mass | Exact location, including signed zero | `0 <= q <= 1` |
| tabulated | Linear probability interpolation, including plateaus | Stored knot interval, endpoints included |
| no_data | null | All valid q |

Invalid q throws `RangeError`; malformed parameters, kinds or families throw
`TypeError`. Pre-initialization calls throw `Error`. A malformed row aborts a
batch, preserving the scalar contract; successful batches preserve input order.
Empty batches still validate q. Validation occurs before checking curve support.
Rust distinguishes malformed input (`Err`) from absent evaluation (`Ok(None)`).
Valid out-of-support queries return null, without clamping/extrapolation.

The canonical point-mass record includes zero scale, atom probability one and
atom location equal to location. An abbreviated frontend-only point mass is
rejected. No-data rows require a recognized reason and no parameters. Other
cross-field invariants match the SDK's current representation.

For a tabulated record with probabilities `[0.998,0.999]` and values
`[0.1597156259295522,0.17687592042871514]`, q=0.9985 yields approximately
`0.16829577317913367`; q=0.5, 0.99 and 0.9995 return null. No shared evaluator can
recover information not stored in a canonical curve.

### Verification and performance

Shared fixtures: `tests/fixtures/canonical-quantiles.json`, regenerated by
`tools/generate_consumer_fixtures.py` through native Python bindings. Native Rust,
Python and Node/Chromium WASM test the same references with numerical tolerances.
Cases cover every kind and family, negative/zero skew, deep-tail hurdles, shape
zero, atom boundaries, signed zero, tabulated plateaus/endpoints/interpolation,
no-data reasons, optional omissions, MVT strings, malformed records, scalar/batch
agreement and prepared handle lifetime. Packaged-consumer tests verify SSR-safe
imports, TypeScript exports, Vite production bundling and actual WASM loading.
The original plan's external VE fixture JSON/generator is absent in the current
checkout; this release establishes tracked framework fixtures instead.

Alpha.1 baseline measurements on 2026-10-16 (Apple Silicon, Node 24.15.0; synthetic 10,000-row
populations, median of 20 warm runs, excluding I/O):

| Measurement | Result |
| --- | --- |
| WASM asset | 89,281 bytes; 39,941 bytes gzip |
| Whole npm tarball | About 62 KB compressed |
| Node byte initialization / Chromium local HTTP initialization | About 14 / 13 ms |
| 10,000 ordinary fitted scalar queries | 12.15 ms |
| 10,000 ordinary fitted batch queries | 11.69 ms |
| 10,000 prepared ordinary fitted queries | 1.64 ms |
| 100 prepared skew-normal queries | 58.59 ms |

The synthetic benchmark is reproducible with `cd js && npm run benchmark`.
A real PMTiles check sampled 24 tiles around Toronto, Ottawa and southwest
Ontario at z3/z5/z6 from the local prototype archive
`prototypes/hazard-curve-aggregation/artifacts/ontario-hot-days-2050-ssp245/hot_days_2050_ssp245_ontario_z10.pmtiles`.
Its 5,367 records comprised 4,443 hurdles, 104 fitted curves and 820 tabulated
curves. At q=0.99, median times were 8.69 ms scalar, 8.69 ms batch, 0.66 ms prepared,
and 4.47 ms through the frontend's bounded cache. Scalar/batch missing-data
semantics agreed; the largest difference from the previous JS evaluator was
1.42e-14. The old JS evaluator took about 0.83 ms on this particular Gumbel-heavy
sample: WASM adoption establishes numerical ownership, not a universal speedup.
Reproduce the local-tile benchmark from `visual-eyes` using
`node --experimental-strip-types scripts/benchmark-crc-curves.mjs <archive-path>`.

Prepared handles are justified by the measured repeated-query cost. The frontend
cache uses typed, length-prefixed identities to avoid repeated MVT decoding and
Rust reconstruction while keeping signed zero and malformed values distinct.
Skew-normal still integrates its CDF numerically and remains costly on first
evaluation. Bisection now stops when its f64 bounds cannot refine further,
approximately halving this cost without changing reference results. Workers and
a faster shared skew-normal algorithm remain possible follow-ups if production
tile profiles require them. Do not change the math only in JS.

### Alpha.2 performance investigation

The alpha.1 batch API crossed into WASM once but still walked JS objects and
reconstructed every distribution through Serde on every call. On the real tile
sample, JS normalization/JSON decoding alone cost about 0.76 ms; Serde plus
reconstruction/evaluation without that normalization cost about 7.10 ms, versus
0.61 ms for already-prepared scalar Rust evaluation. VELO's string cache keys
added several milliseconds even on cache hits. This explains why a simple
Gumbel-heavy JS evaluator beat the first WASM interface.

Alpha.2 replaces that bridge with private contiguous f64 buffers and explicit
optional-field presence bits. Rust supplies the kind/type codes at startup and
retains all canonical invariants/math; JS does only language/transport conversion.
MVT array parsing has a bounded cache. Prepared collections retain Rust-owned
records and return one contiguous numeric result buffer per query; the public
adapter restores null for missing values. Serde is no longer a WASM dependency.
The scalar and one-shot APIs remain compatible; `prepareCurves(rows)` is the
recommended API for repeated queries. Prepared collections copy a snapshot,
require `free()`, preserve order/float64/signed zero and reject use after free.

Production Chromium / Vite benchmark on the same 5,367 real records, 50 warm
samples after 10 warmups, fresh property objects and alternating q=.5/.9/.99:

| Path | Median |
| --- | --- |
| Previous JS evaluator | 0.58 ms |
| New one-shot scalar WASM calls | 3.60 ms |
| New one-shot WASM batch | 1.46 ms |
| Prepared WASM collection | 0.155 ms |
| VELO prepared viewport including field comparisons | 0.255 ms |
| Previous complete H3 cell collection | 1.37 ms |
| New complete H3 cell collection | 1.10 ms |

The real frontend hot path exceeds parity: about 2.3x faster for evaluation and
20% faster for complete cell collection. First viewport preparation, including
initial JS JIT/cache warmup, took about 5.1 ms and is still slower than the old
JS path. One-shot APIs still pay conversion/validation/reconstruction on every
call. Do not claim all call patterns are faster; amortizing preparation is the
intended interactive contract. Rust numerical algorithms are unchanged in this
performance pass. The WASM asset falls to about 74 KB / 32 KB gzip, and the npm
tarball to about 53 KB.

Reproduce actual rows with VELO's tile benchmark, then run from `crc-framework-rs/js`:
`node test/performance.mjs <visual-eyes-repo> <exported-rows.json> <old-evaluator-git-ref>`.
It builds an isolated production Vite page, runs Chromium with high-resolution
timing, compares numerical/null semantics and full cell collections, and prevents
identity-only or same-probability caches from hiding evaluation work. Keep the
old evaluator only in a temporary benchmark checkout; no duplicate formulas are
restored in production. Timing thresholds are not enforced in CI. The benchmark's old-code reference was
VELO commit `f77b563d15d89f4447d30497779b4f4437bc718b`.

Verification includes all shared references, invalid/empty batches, mixed kinds,
strict numeric conversion, omitted vs non-finite fields, signed zero, snapshots,
array mutation/cache invalidation and handle lifetime. The browser and packed
Vite/TypeScript/SSR tests now cover prepared collections too. VELO's production
build and 41 tests pass. Its full typecheck baseline previously had 388 existing
diagnostics; the rollout introduces no new diagnostic messages.

### Release operation

Build with wasm-pack 0.15.0 and locked Cargo dependencies. `js/build.mjs` derives
npm's version from the root Cargo manifest and checks both Rust crate versions.
Rust and npm accept the same SemVer prerelease (`0.4.0-alpha.1`), so no custom
Python-style `alpha -> a` translation is needed. Publish prereleases under their
identifier (`alpha`, `beta`, `rc`) and stable versions under `latest`.

The first release is manually published from the tested tarball using the logged
in npm account under `@riskthinking.ai`. Subsequent matching `v<version>` tags
publish through `npm.yaml`, independently of the Python wheel workflow. Manual
workflow runs validate/build without publishing. Merge source and workflow before
subsequent tags; do not move existing release tags to trigger another ecosystem.

The initial manual publish completed on 2026-10-16 as `0.4.0-alpha.1` under tag `alpha`.
The optimized `0.4.0-alpha.2` publish completed the same day under `alpha`;
its public registry integrity matches the tested tarball, and VELO installs this
exact registry version. Final repeated browser measurements were 0.60 ms legacy
vs 0.265 ms new viewport evaluation, and 1.51 ms vs 1.18 ms complete cell collection
(first viewport preparation 5.61 ms).
Public registry metadata and the tarball are available; the tarball checksum
matches the tested local artifact.
The npm trusted publisher was created and verified through `npm trust`:
GitHub repository `RiskThinking/crc-framework-rs`, workflow `npm.yaml`, environment
`production`, with direct publishing allowed. The GitHub environment already
exists. No further npm token/GitHub secret setup is required. The source and
workflow still need to be committed/merged before a future release tag can run
this CI. See `../../js/README.md` for build/install/manual publish commands.

## Responsibility boundaries

| Responsibility | Owner |
| --- | --- |
| Numerical distributions, canonical reconstruction, quantiles | Framework Rust core |
| JS/Python conversion, initialization, language errors | Framework bindings |
| MVT array-string decoding | JS package adapter |
| Schema metadata, providers, dataset discovery, ingestion, Parquet workflows | CRC-SDK |
| Tile loading, frontend cache, feature state, building joins, percentile keys | Frontend |

The Python SDK currently retains its existing Pydantic validation/dispatch; its
numerics already use Rust. Initial WASM consumption does not need Python's
higher-level discovery/I/O code moved into the framework. Until the SDK delegates
its validation/dispatch to the new canonical core API, claim shared numerical
implementation and a Rust-owned JS reconstruction contract, not a fully unified
SDK/JS validation implementation.

## Later phase: comprehensive Python parity and writing

This phase is planning only and is not required for the 0.4.0 initial rollout.

1. Expose the canonical consumer through PyO3, then have SDK `CurveParameters`
   delegate validation and reconstruction while retaining its frozen public
   Pydantic model, aliases/defaults and error compatibility. Gate the change on
   SDK regression tests for accepted/rejected records, all no-data reasons,
   normalization and existing `to_distribution()` behavior. Coordinate the SDK's
   minimum framework version. Remove SDK-owned canonical dispatch rules only
   after compatibility is proven; this completes single ownership of validation.
2. Add CDF/PDF, survival functions, inverse survival, distribution statistics and
   vectorized multi-q interfaces to WASM from the same core. Decide memory
   ownership/typed-array APIs from measured browser workloads. Preserve supported
   domains, null semantics and language-specific error mapping.
3. Add canonicalization/writing: fitted, censored/hurdle fitting, tabulation,
   no-data classification, canonical parameter extraction and deterministic
   serialization. Extract only algorithmic/schema invariants from the Python SDK;
   leave providers and orchestration language-specific. Verify round trips across
   native/Python/WASM using shared valid/invalid fixtures and explicit tolerances.
4. Design a browser-installable CRC-SDK JS layer for dataset discovery, fetching,
   schema/version checks and canonical read/write workflows. Integrate Arrow or
   DuckDB-WASM through interoperable batches/typed arrays; avoid a second
   implementation of curve math or canonicalization. Prove an end-to-end browser
   example: load canonical rows, query/join in DuckDB, evaluate curves, export
   canonical results. Define download size, memory and offline/browser-worker
   behavior before selecting Parquet dependencies.
5. Add sampling (with an explicit browser entropy strategy), transformations,
   risk aggregation, microscores and spatial utilities by priority. Feature-gate
   heavier dependencies and split optional I/O integrations from core consumption.
6. Define dataset-aware return-period queries with explicit exceedance semantics,
   tail and conversion convention. Do not infer these from curve parameters.

Frontend follow-ups remain independent: arbitrary-percentile selector/URL UI,
actual mean evaluation (legacy mean aliases currently select median q=0.5),
production-tile performance profiling, worker execution if needed, and a faster
shared skew-normal inversion with separate numerical review.

## Tooling references

- [wasm-pack build targets and packaging](https://wasm-bindgen.github.io/wasm-pack/book/commands/build.html)
- [wasm-bindgen structured data with Serde](https://wasm-bindgen.github.io/wasm-bindgen/reference/arbitrary-data-with-serde.html)
- [npm trusted publishing](https://docs.npmjs.com/trusted-publishers/)
- [npm distribution tags](https://docs.npmjs.com/adding-dist-tags-to-packages/)

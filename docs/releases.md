# Versioning and releases

[Documentation index](README.md) · [Project README](../README.md)

## Release contract

The package and artifact versions are declared in
[`release.json`](../python/crc_framework/contracts/api/release.json). The
framework Python package, Rust workspace, WASM package, and generated JavaScript
package use `1.0.0`. Each artifact format has its own version in that ledger.
Changing a package version does not automatically increment every format.

Unshipped definitions are replaced in place. An earlier branch assigning a
number does not justify another major version or a transitional parser. The
public API has one declared surface and current artifact formats; old exports,
aliases, and wire layouts impose no preservation requirement.

The package version source is `[workspace.package].version` in root
[`Cargo.toml`](../Cargo.toml), which must match the ledger. All three Rust crates
inherit it; maturin derives the Python version, and the JS build derives the npm
version. Source `js/package.json` stays private and versionless; the build writes
the public manifest to ignored `js/dist/`.

Before a release, verify declarations and the installed package's actual public
surface:

```sh
.venv/bin/python tools/api_contracts/check.py
/path/to/wheel-venv/bin/python -I /path/to/repository/tools/api_contracts/check.py --runtime
```

The runtime check loads its declarations from the installed package and rejects
missing methods, incorrect signatures or defaults, extra enum aliases, removed
names that remain importable, and Python/native version mismatches. Passing a
static declaration check does not authorize publishing an incomplete interface.
Numerical, schema, native/WASM parity, and installed-consumer checks must also
pass.

## Publishing

1. Set the one release number in the ledger and `Cargo.toml`. Refresh
   `Cargo.lock` with Cargo and review generated package versions.
2. Build and verify the Python wheel and JavaScript package for that exact
   version, including their public runtime contracts.
3. Create the GitHub release from the tested commit using the exact Cargo tag,
   such as `v1.0.0`. Mark alpha, beta, or release-candidate packages as
   prereleases.
4. Check both **Package release** and **JavaScript WASM** workflow results.
   Publishing is independent, so one registry can succeed while another fails.

| Destination | Version | Release behavior |
| --- | --- | --- |
| Rust workspace | `1.0.0` | Versioned source; no crates.io publishing workflow is configured |
| PyPI `crc-framework` | `1.0.0` | `release.yaml` builds wheels and an sdist, attaches assets, and publishes to PyPI |
| npm `@climateriskcommons/crc-framework` | `1.0.0` | `npm.yaml` publishes the tested generated package |

Use `v` plus the exact Cargo version for the tag. Cargo `1.0.0-alpha.1` maps to
Python `1.0.0a1`; the Python spelling is an output conversion, not another version
source. Stable packages use npm's `latest` channel; prereleases use their
corresponding channel. No hand-written npm version or CI-generated version commit
is required.

npm publishing uses GitHub OIDC with the `production` environment. The trusted
publisher authorizes organization `RiskThinking`, repository `crc-framework-rs`,
workflow `npm.yaml`, environment `production`, and direct `npm publish`. Python
publishing uses its separate workflow and the same environment.

Released tags remain fixed. An implementation fix after publication requires a
new release; rewriting an unshipped definition does not.

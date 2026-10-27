# Versioning and releases

[Documentation index](README.md) · [Project README](../README.md)

The single version source is `[workspace.package].version` in the root
[`Cargo.toml`](../Cargo.toml). All three Rust crates inherit it; maturin derives the
Python version, and the JS build derives the npm version. Source
`js/package.json` stays private and versionless; the build generates the public
package manifest in ignored `js/dist/`.

To release, for example, `0.5.3`:

1. Edit only the workspace version in `Cargo.toml`, then run
   `cargo check -p crc-framework-core` to refresh `Cargo.lock`.
2. Review and commit `Cargo.toml` and `Cargo.lock` with the release changes.
   Push and run the JavaScript WASM workflow on the branch; wait for checks to pass.
3. In GitHub Releases, create **`v0.5.3`** targeting the tested branch
   or commit, marking alpha/beta/rc releases as **prereleases**. Releases can
   target the prepared release branch before it is merged into `main`.
4. Check both **Package release** and **JavaScript WASM** workflow results.
   Publishing is independent, so one registry can succeed while another fails.

| Destination | Version in this example | Release behavior |
| --- | --- | --- |
| Rust workspace | `0.5.3` | Versioned source; no crates.io publishing workflow is configured |
| PyPI: `crc-framework` | `0.5.3` | `release.yaml` builds wheels and an sdist, attaches assets to the GitHub release, and publishes to PyPI |
| npm: `@climateriskcommons/crc-framework` | `0.5.3` | `npm.yaml` builds/tests and publishes the tested tarball under the `latest` dist-tag |

Use **`v` + the exact Cargo version** for the Git tag. For prereleases,
`0.5.3-alpha.1` maps to Python's `0.5.3a1`; the Python spelling is an output
conversion, not the release-tag format. The npm publisher rejects
a tag that differs from its generated package version. Stable `0.5.3` uses tag
`v0.5.3` and npm's `latest` channel; `beta` and `rc` prereleases use their
respective channels. No npm version edits, extra versioning tools, or CI-generated
version commits are needed.

npm publishing uses GitHub OIDC with the `production` environment. The npm trusted
publisher must authorize organization `RiskThinking`, repository
`crc-framework-rs`, workflow `npm.yaml`, environment `production`, and direct
`npm publish`; no npm token secret is needed. Python publishing also uses the
`production` environment through its separate workflow.

Tags trigger publishing. Re-running an existing tag uses the code at that tag;
a new fix requires a new release version. Keep released tags fixed and increment
the version for subsequent releases.


### npm scope migration and initial setup

Starting with `0.5.3-alpha.1`, install `@climateriskcommons/crc-framework`.
Replace imports from `@riskthinking.ai/crc-framework`, including the WASM asset
subpath, and update the application lockfile. The old package is historical;
existing versions are not moved or republished.

Trusted publisher configuration is per package and does not carry over from the
old scope. For the first release, push the tag, download the `npm-package` artifact
from its successful build job, and publish that exact tarball as an npm maintainer
with access to `@climateriskcommons`:

```sh
npm login
gh run download RUN_ID --name npm-package --dir /tmp/crc-npm-release
npm publish /tmp/crc-npm-release/climateriskcommons-crc-framework-0.5.3-alpha.1.tgz --access public --tag alpha
npm trust github @climateriskcommons/crc-framework --repo RiskThinking/crc-framework-rs --file npm.yaml --env production --allow-publish
```

Complete any npm browser/2FA prompts. Verify the package and trusted publisher
then rerun the failed publish job (`gh run rerun RUN_ID --failed`). The workflow
recognizes an already published version
only when its registry integrity matches the exact tested tarball; a different
artifact fails the release. Subsequent versions publish directly through OIDC.
See [npm trusted publishing](https://docs.npmjs.com/trusted-publishers/).

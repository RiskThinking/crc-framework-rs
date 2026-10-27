import {
  readFileSync, writeFileSync, copyFileSync, cpSync, existsSync,
  mkdirSync, readdirSync, rmSync,
} from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const directory = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(directory);
const output = path.join(directory, 'dist');
// Let Cargo resolve workspace inheritance rather than interpreting TOML ourselves.
const metadata = JSON.parse(execFileSync('cargo', [
  'metadata', '--locked', '--no-deps', '--format-version', '1',
], { cwd: root, encoding: 'utf8' }));
const framework = metadata.packages.find(pkg => pkg.manifest_path === path.join(root, 'Cargo.toml'));
const version = framework.version;
for (const name of ['crc-framework-core', 'crc-framework-wasm']) {
  if (metadata.packages.find(pkg => pkg.name === name)?.version !== version) {
    throw new Error(`Version mismatch: ${name}`);
  }
}
const manifest = JSON.parse(readFileSync(path.join(directory, 'package.json'), 'utf8'));
if (manifest.private !== true || 'version' in manifest) {
  throw new Error('The development package must be private and versionless');
}
const checking = process.argv.includes('--check') || process.argv.includes('--pack');
if (checking) {
  const manifestPath = path.join(output, 'package.json');
  const built = existsSync(manifestPath) && JSON.parse(readFileSync(manifestPath, 'utf8'));
  if (!built || built.name !== manifest.name || built.version !== version || built.private ||
      !existsSync(path.join(output, 'wasm/crc_framework_bg.wasm'))) {
    throw new Error('Run npm run build before packing');
  }
} else {
  rmSync(output, { recursive: true, force: true });
  execFileSync('wasm-pack', ['build', 'crates/wasm', '--target', 'web', '--release',
    '--out-dir', '../../js/wasm', '--out-name', 'crc_framework', '--', '--locked'], { cwd: root, stdio: 'inherit' });
  const wasm = path.join(directory, 'wasm');
  const wasmManifest = JSON.parse(readFileSync(path.join(wasm, 'package.json'), 'utf8'));
  if (wasmManifest.version !== version) throw new Error('wasm-pack version differs from the workspace');
  mkdirSync(path.join(output, 'wasm'), { recursive: true });
  cpSync(path.join(directory, 'src'), path.join(output, 'src'), { recursive: true });
  for (const name of readdirSync(wasm)) {
    if (/\.(?:js|wasm|ts)$/.test(name)) copyFileSync(path.join(wasm, name), path.join(output, 'wasm', name));
  }
  for (const name of ['LICENSE', 'THIRD_PARTY_NOTICES.md']) copyFileSync(path.join(root, name), path.join(output, name));
  // Registry README links must work without a repository-relative base URL.
  const readme = readFileSync(path.join(directory, 'README.md'), 'utf8')
    .replace(/\]\(\.\.\/([^)]*)\)/g, `](https://github.com/RiskThinking/crc-framework-rs/blob/v${version}/$1)`);
  writeFileSync(path.join(output, 'README.md'), readme);
  // The checked-in manifest describes development. Only the generated artifact
  // is publishable, with no development scripts/dependencies or private flag.
  const { private: _private, scripts, devDependencies, ...published } = manifest;
  writeFileSync(path.join(output, 'package.json'), JSON.stringify({ ...published, version }, null, 2) + '\n');
}
const tag = version.includes('-') ? version.split('-')[1].split('.')[0] : 'latest';
console.log(`${manifest.name}@${version} (npm tag: ${tag})`);
if (process.argv.includes('--pack')) {
  const args = ['pack'];
  if (process.argv.includes('--dry-run')) args.push('--dry-run');
  execFileSync('npm', args, { cwd: output, stdio: 'inherit' });
}

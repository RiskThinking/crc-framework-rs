import { readFileSync, writeFileSync, copyFileSync, existsSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const directory = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(directory);
const read = name => readFileSync(path.join(root, name), 'utf8');
const versionOf = name => read(name).match(/^version = "([^"]+)"/m)?.[1];
const version = versionOf('Cargo.toml');
for (const crate of ['crates/core/Cargo.toml', 'crates/wasm/Cargo.toml']) {
  if (versionOf(crate) !== version) throw new Error(`Version mismatch: ${crate}`);
}
const manifestPath = path.join(directory, 'package.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
const checking = process.argv.includes('--check');
if (checking) {
  if (manifest.version !== version || !existsSync(path.join(directory, 'wasm/crc_framework_bg.wasm'))) {
    throw new Error('Run npm run build before packing');
  }
} else {
  manifest.version = version;
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
  const lockPath = path.join(directory, 'package-lock.json');
  if (existsSync(lockPath)) {
    const lock = JSON.parse(readFileSync(lockPath, 'utf8'));
    lock.version = version;
    lock.packages[''].version = version;
    writeFileSync(lockPath, JSON.stringify(lock, null, 2) + '\n');
  }
  execFileSync('wasm-pack', ['build', 'crates/wasm', '--target', 'web', '--release',
    '--out-dir', '../../js/wasm', '--out-name', 'crc_framework', '--', '--locked'], { cwd: root, stdio: 'inherit' });
  for (const name of ['package.json', '.gitignore', 'README.md']) {
    rmSync(path.join(directory, 'wasm', name), { force: true });
  }
  for (const name of ['LICENSE', 'THIRD_PARTY_NOTICES.md']) copyFileSync(path.join(root, name), path.join(directory, name));
}
const tag = version.includes('-') ? version.split('-')[1].split('.')[0] : 'latest';
console.log(`${manifest.name}@${version} (npm tag: ${tag})`);

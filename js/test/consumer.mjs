import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
import { build, preview } from 'vite';

const root = fileURLToPath(new URL('../', import.meta.url));
const manifest = JSON.parse(await readFile(path.join(root, 'dist/package.json'), 'utf8'));
const ledger = JSON.parse(await readFile(new URL(
  '../../python/crc_framework/contracts/api/release.json', import.meta.url,
), 'utf8'));
const version = ledger.packages.framework;
assert.equal(manifest.name, '@climateriskcommons/crc-framework');
assert.equal(manifest.version, version);
for (const field of ['private', 'devDependencies', 'dependencies', 'scripts']) {
  assert.equal(manifest[field], undefined);
}
const archive = path.join(root, 'dist', `climateriskcommons-crc-framework-${version}.tgz`);
const directory = await mkdtemp(path.join(tmpdir(), 'crc-wasm-consumer-'));
const packageDirectory = path.join(directory, 'node_modules', manifest.name);
const pointMass = '{"schema_version":"1.0","object":"distribution","kind":"point_mass","value":2}';
let server;
let browser;

try {
  await writeFile(path.join(directory, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
  execFileSync('npm', ['install', '--offline', '--ignore-scripts', '--no-audit', '--no-fund', archive], {
    cwd: directory,
    stdio: 'inherit',
  });
  assert.deepEqual(JSON.parse(await readFile(path.join(packageDirectory, 'package.json'), 'utf8')), manifest);

  await writeFile(path.join(directory, 'node.mjs'), `
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import init, * as crc from '${manifest.name}';

const text = '${pointMass}';
assert.deepEqual(Object.keys(crc).sort(), [
  'DocumentSchema', 'default', 'prepareNumerical', 'prepareNumericals', 'quantile', 'validateDocument',
]);
assert.throws(() => crc.quantile(text, .5), /await init/);
await assert.rejects(init(new Uint8Array(1)), WebAssembly.CompileError);
const pending = init();
assert.equal(init(), pending);
await pending;
assert.equal(crc.quantile(text, .5), 2);
crc.validateDocument(crc.DocumentSchema.NUMERICAL, text);

const asset = new URL(import.meta.resolve('${manifest.name}/crc_framework_bg.wasm'));
const { version } = await import(new URL('crc_framework.js', asset));
assert.equal(version(), '${version}');
const schemaPath = new URL(import.meta.resolve('${manifest.name}/contracts/numerical/numerical-1.0.schema.json'));
assert.equal(JSON.parse(await readFile(schemaPath, 'utf8')).$id, crc.DocumentSchema.NUMERICAL);
`);
  execFileSync(process.execPath, ['node.mjs'], { cwd: directory, stdio: 'inherit' });

  await writeFile(path.join(directory, 'smoke.ts'), `
import init, {quantile, prepareNumerical, prepareNumericals, DocumentSchema, validateDocument} from '${manifest.name}';
const text='${pointMass}';
await init();
await init(new URL('https://example.com/crc_framework_bg.wasm'));
await init(fetch('https://example.com/crc_framework_bg.wasm'));
await init(await WebAssembly.compile(new Uint8Array(0)));
const scalar: number = quantile(text, .5);
const values: Float64Array = quantile(text, new Float64Array([.1,.9]));
const prepared = prepareNumerical(text);
prepared.quantile([.1, .9]);
prepared.free();
const batch = prepareNumericals('[' + text + ']');
const batchValues: Float64Array = batch.quantile(.5);
batch.free();
validateDocument(DocumentSchema.NUMERICAL, text);
`);
  const tsc = [
    path.join(root, 'node_modules/typescript/bin/tsc'), '--noEmit', '--strict',
    '--module', 'nodenext', '--moduleResolution', 'nodenext', '--target', 'es2022',
  ];
  execFileSync(process.execPath, [...tsc, path.join(directory, 'smoke.ts')], { stdio: 'inherit' });
  await writeFile(path.join(directory, 'node.ts'), `
import init, { quantile } from '${manifest.name}';
await init(new Uint8Array(0));
const value: number = quantile('${pointMass}', .5);
`);
  execFileSync(process.execPath, [...tsc, '--lib', 'es2022', path.join(directory, 'node.ts')], { stdio: 'inherit' });

  await writeFile(path.join(directory, 'server.mjs'), `
import init, { quantile } from '${manifest.name}';
export async function render() {
  await init();
  return '<p>' + quantile('${pointMass}', .5) + '</p>';
}
`);
  const vite = { root: directory, configFile: false, logLevel: 'warn' };
  await build({
    ...vite,
    build: {
      ssr: 'server.mjs',
      outDir: 'ssr',
      rollupOptions: { output: { entryFileNames: 'server.mjs' } },
    },
  });
  const { render } = await import(pathToFileURL(path.join(directory, 'ssr/server.mjs')));
  assert.equal(await render(), '<p>2</p>');
  console.log('Packed Node, TypeScript, and Vite SSR consumers passed');

  const pageSource = await readFile(new URL('./browser.html', import.meta.url), 'utf8');
  await writeFile(path.join(directory, 'index.html'), pageSource
    .replace("'../src/index.js'", `'${manifest.name}'`)
    .replace("'../../python/crc_framework/contracts/numerical/fixtures-1.0.json'", "'/numerical-fixtures.json'"));
  const publicDirectory = path.join(directory, 'public');
  await mkdir(publicDirectory);
  await copyFile(
    path.join(packageDirectory, 'contracts/numerical/fixtures-1.0.json'),
    path.join(publicDirectory, 'numerical-fixtures.json'),
  );
  await build(vite);
  server = await preview({ ...vite, preview: { port: 0, host: '127.0.0.1' } });
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  await page.waitForFunction(() => window.testResult);
  const result = await page.evaluate(() => window.testResult);
  assert.equal(result.status, 'passed', JSON.stringify(result));
  assert.ok(result.assertions > 0);
  console.log('Packed Vite browser consumer:', result);
} finally {
  await browser?.close();
  if (server) await new Promise(resolve => server.httpServer.close(resolve));
  await rm(directory, { recursive: true, force: true });
}

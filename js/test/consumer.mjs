// Install the packed artifact into an isolated Vite consumer, then test production
// asset loading and SSR-safe imports. No Rust or Python exists in the consumer.
import {execFileSync} from 'node:child_process';
import {mkdtemp, mkdir, readFile, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {build, preview} from 'vite';
import {chromium} from 'playwright';
const root=fileURLToPath(new URL('../',import.meta.url));
const manifest=JSON.parse(await readFile(path.join(root,'package.json'),'utf8'));
const filename=`riskthinking.ai-crc-framework-${manifest.version}.tgz`;
const directory=await mkdtemp(path.join(tmpdir(),'crc-wasm-consumer-'));
let server;
let browser;
try {
  await writeFile(path.join(directory,'package.json'), JSON.stringify({private:true,type:'module'}));
  const packageDirectory=path.join(directory,'node_modules/@riskthinking.ai/crc-framework');
  await mkdir(packageDirectory,{recursive:true});
  execFileSync('tar',['-xzf',path.join(root,filename),'-C',packageDirectory,'--strip-components=1']);
  const imported=await import(pathToFileURL(path.join(packageDirectory,'src/index.js')).href);
  assert.throws(()=>imported.evaluateCurve({},.5),/not initialized/);
  let html=await readFile(new URL('./browser.html',import.meta.url),'utf8');
  html=html.replace("'../src/index.js'","'@riskthinking.ai/crc-framework'")
    .replace("'../../tests/fixtures/canonical-quantiles.json'","'./canonical-quantiles.json'");
  await writeFile(path.join(directory,'index.html'),html);
  await mkdir(path.join(directory,'public'));
  await writeFile(path.join(directory,'public/canonical-quantiles.json'),
    await readFile(new URL('../../tests/fixtures/canonical-quantiles.json',import.meta.url)));
  await writeFile(path.join(directory,'smoke.ts'), `
import init, {evaluateCurve, evaluateCurves, prepareCurve, prepareCurves, type CurveProperties} from '@riskthinking.ai/crc-framework';
const row: CurveProperties = {curve_kind:'fitted',curve_type:'gumbel_r',curve_location:0,curve_scale:1};
await init();
const value: number | null = evaluateCurve(row, .5);
const values: Array<number | null> = evaluateCurves([row], .5);
const handle = prepareCurve(row);
handle.quantiles(new Float64Array([.5]));
handle.free();
const collection = prepareCurves([row]);
const batch: Array<number | null> = collection.evaluate(.5);
collection.free();
`);
  execFileSync(process.execPath,[path.join(root,'node_modules/typescript/bin/tsc'),
    '--noEmit','--strict','--module','nodenext','--moduleResolution','nodenext',
    '--target','es2022',path.join(directory,'smoke.ts')],{stdio:'inherit'});
  await build({root:directory,configFile:false,logLevel:'warn'});
  server=await preview({root:directory,configFile:false,logLevel:'warn',preview:{port:0,host:'127.0.0.1'}});
  browser=await chromium.launch({headless:true});
  const page=await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  await page.waitForFunction(()=>window.testResult);
  const result=await page.evaluate(()=>window.testResult);
  assert.equal(result.status,'passed',JSON.stringify(result));
  console.log('Packed Vite production consumer:',result);
} finally {
  await browser?.close();
  await new Promise(resolve=>server?server.httpServer.close(resolve):resolve());
  await rm(directory,{recursive:true,force:true});
}

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
const manifest=JSON.parse(await readFile(path.join(root,'dist/package.json'),'utf8'));
const filename=`climateriskcommons-crc-framework-${manifest.version}.tgz`;
assert.equal(manifest.private, undefined);
assert.equal(manifest.devDependencies, undefined);
assert.equal(manifest.scripts, undefined);
assert.match(manifest.version, /^\d+\.\d+\.\d+/);
const directory=await mkdtemp(path.join(tmpdir(),'crc-wasm-consumer-'));
let server;
let browser;
try {
  await writeFile(path.join(directory,'package.json'), JSON.stringify({private:true,type:'module'}));
  const packageDirectory=path.join(directory,'node_modules/@climateriskcommons/crc-framework');
  await mkdir(packageDirectory,{recursive:true});
  execFileSync('tar',['-xzf',path.join(root,'dist',filename),'-C',packageDirectory,'--strip-components=1']);
  const imported=await import(pathToFileURL(path.join(packageDirectory,'src/index.js')).href);
  assert.throws(()=>imported.quantile("{}",.5),/await init/);
  let html=await readFile(new URL('./browser.html',import.meta.url),'utf8');
  html=html.replace("'../src/index.js'","'@climateriskcommons/crc-framework'")
    .replace("'../../python/crc_framework/contracts/numerical/fixtures-1.0.json'","'./numerical-fixtures.json'");
  await writeFile(path.join(directory,'index.html'),html);
  await mkdir(path.join(directory,'public'));
  await writeFile(path.join(directory,'public/numerical-fixtures.json'),
    await readFile(new URL('../../python/crc_framework/contracts/numerical/fixtures-1.0.json',import.meta.url)));
  await writeFile(path.join(directory,'smoke.ts'), `
import init, {quantile,prepareNumerical,prepareNumericals,DocumentSchema,validateDocument} from '@climateriskcommons/crc-framework';
const text='{"schema_version":"1.0","object":"distribution","kind":"point_mass","value":2}';
await init();
const scalar:number=quantile(text,.5);
const array:Float64Array=quantile(text,new Float64Array([.1,.9]));
const prepared=prepareNumerical(text);
prepared.quantile([.1,.9]);prepared.free();
const batch=prepareNumericals('['+text+']');
const values:Float64Array=batch.quantile(.5);batch.free();
validateDocument(DocumentSchema.NUMERICAL,text);
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

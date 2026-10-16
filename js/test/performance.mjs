// Reproducible production-Chromium comparison against a historical VELO
// evaluator. Supply actual canonical tile rows exported by VELO's tile benchmark.
// node test/performance.mjs /path/to/visual-eyes /path/to/rows.json [baseline-ref]
import { execFileSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import assert from "node:assert/strict";
import { build, preview } from "vite";
import { chromium } from "playwright";
const [consumer, rowFile, ref = "HEAD"] = process.argv.slice(2);
if (!consumer || !rowFile)
  throw new Error("Pass consumer repo and tile rows JSON paths");
const directory = await mkdtemp(path.join(tmpdir(), "crc-performance-"));
let browser;
let server;
try {
  const packagePath = path.join(
    consumer,
    "node_modules/@riskthinking.ai/crc-framework/src/index.js",
  );
  const frontendPath = path.join(
    consumer,
    "app/utils/pmtiles-map/curve-evaluator.ts",
  );
  const require = createRequire(path.join(consumer, "package.json"));
  const baseline = execFileSync(
    "git",
    ["show", `${ref}:app/utils/pmtiles-map/curve-evaluator.ts`],
    { cwd: consumer, encoding: "utf8" },
  );
  await writeFile(path.join(directory, "legacy.ts"), baseline);
  const collector = execFileSync(
    "git",
    ["show", `${ref}:app/utils/pmtiles-map/hazard-curve.ts`],
    { cwd: consumer, encoding: "utf8" },
  )
    .replace('"./curve-evaluator"', '"./legacy.ts"')
    .replace('"h3-js"', JSON.stringify(require.resolve("h3-js")));
  await writeFile(path.join(directory, "legacy-collector.ts"), collector);
  await writeFile(
    path.join(directory, "package.json"),
    JSON.stringify({ type: "module" }),
  );
  await mkdir(path.join(directory, "public"));
  await writeFile(
    path.join(directory, "public/rows.json"),
    await readFile(rowFile),
  );
  const source = `
import init, {evaluateCurve, evaluateCurves, prepareCurves} from ${JSON.stringify(packagePath)};
import {evaluateCurves as evaluateViewport} from ${JSON.stringify(frontendPath)};
import {evaluateCurve as legacy} from './legacy.ts';
import {collectDisplayCells as oldCollect} from './legacy-collector.ts';
import {collectDisplayCells as newCollect} from ${JSON.stringify(path.join(consumer, "app/utils/pmtiles-map/hazard-curve.ts"))};
await init();
const rows=await (await fetch('./rows.json')).json();
const prepared=prepareCurves(rows);
function previous(rows,q) { return rows.map(row=>{try{return legacy(row,q);}catch(error){if(error instanceof RangeError)return null;throw error;}}); }
const methods={
  'legacy JS':previous,
  'WASM scalar':(rows,q)=>rows.map(row=>evaluateCurve(row,q)),
  'WASM one-shot batch':evaluateCurves,
  'WASM prepared batch':(_rows,q)=>prepared.evaluate(q),
  'VELO prepared viewport':evaluateViewport,
  'legacy full cell collection':(rows,q)=>Array.from(oldCollect(rows.map(properties=>({properties})),q).values.values(),cell=>cell.value),
  'VELO full cell collection':(rows,q)=>Array.from(newCollect(rows.map(properties=>({properties})),q).values.values(),cell=>cell.value),
};
const times=Object.fromEntries(Object.keys(methods).map(name=>[name,[]]));
let checksum=0;
const coldStart=performance.now();
evaluateViewport(rows,.99);
const coldViewportMs=performance.now()-coldStart;
for(let i=0;i<60;i++) {
  const q=[.5,.9,.99][i%3];
  // Recreated objects defeat identity-only caches; changing q defeats value memoization.
  const fresh=rows.map(row=>({...row}));
  for(const [name,method] of Object.entries(methods)) {
    const start=performance.now();
    const result=method(fresh,q);
    const elapsed=performance.now()-start;
    checksum+=result.length+(result[0]??0);
    if(i>=10) times[name].push(elapsed);
  }
}
for(const q of [0,.1,.2,.5,.9,.99,.9985,1]) {
  const expected=rows.map(row=>evaluateCurve(row,q));
  const features=rows.map(properties=>({properties}));
  const oldCells=oldCollect(features,q);
  const newCells=newCollect(features,q);
  if(oldCells.resolution!==newCells.resolution || oldCells.values.size!==newCells.values.size) throw new Error('Cell collection structure mismatch');
  for(const [cell,record] of oldCells.values) {
    const actual=newCells.values.get(cell)?.value;
    const expected=record.value;
    if(expected===null ? actual!==null : typeof actual!=='number'||Math.abs(actual-expected)>2e-10*Math.max(1,Math.abs(expected))) throw new Error('Cell collection parity mismatch');
  }
  for(const [name,method] of Object.entries(methods)) {
    // Older fitted/hurdle endpoints throw; the legacy frontend soft-failed them.
    if(name.includes('full cell collection')) continue;
    const actual=method(rows,q);
    actual.forEach((value,index)=>{
      const target=expected[index];
      if(target===null ? value!==null : typeof value!=='number'||Math.abs(value-target)>2e-10*Math.max(1,Math.abs(target))) {
        throw new Error(name+' parity mismatch at q='+q+' row='+index);
      }
    });
  }
}
prepared.free();
const median=values=>values.sort((a,b)=>a-b)[Math.floor(values.length/2)];
window.benchmarkResult={records:rows.length,coldViewportMs,medianMs:Object.fromEntries(Object.entries(times).map(([name,values])=>[name,median(values)])),checksum};
`;
  await writeFile(
    path.join(directory, "index.html"),
    `<!doctype html><script type="module">${source}</script>`,
  );
  await build({ root: directory, configFile: false, logLevel: "warn" });
  server = await preview({
    root: directory,
    configFile: false,
    logLevel: "warn",
    preview: {
      port: 0,
      host: "127.0.0.1",
      headers: {
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
      },
    },
  });
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  await page
    .waitForFunction(() => window.benchmarkResult, undefined, {
      timeout: 60000,
    })
    .catch((error) => {
      throw new Error(errors.join("\n") || String(error));
    });
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify(await page.evaluate(() => window.benchmarkResult), null, 2),
  );
} finally {
  await browser?.close();
  await new Promise((resolve) =>
    server ? server.httpServer.close(resolve) : resolve(),
  );
  await rm(directory, { recursive: true, force: true });
}

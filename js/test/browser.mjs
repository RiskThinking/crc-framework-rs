import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
const root=fileURLToPath(new URL('../../',import.meta.url));
const server=createServer(async(req,res)=>{
  const filename=path.resolve(root, '.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));
  if(!filename.startsWith(root)) {res.writeHead(403).end();return;}
  try {
    const bytes=await readFile(filename);
    const type=filename.endsWith('.wasm')?'application/wasm':filename.endsWith('.js')?'text/javascript':filename.endsWith('.json')?'application/json':'text/html';
    res.writeHead(200,{'Content-Type':type}).end(bytes);
  } catch {res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
let browser;
try {
  browser=await chromium.launch({headless:true});
  const page=await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/js/test/browser.html`);
  await page.waitForFunction(()=>window.testResult);
  const result=await page.evaluate(()=>window.testResult);
  assert.equal(result.status,'passed',JSON.stringify(result));
  console.log(result);
} finally {
  await browser?.close();
  server.close();
}

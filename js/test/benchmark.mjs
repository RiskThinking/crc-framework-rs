import {readFileSync} from 'node:fs';
import {gzipSync} from 'node:zlib';
import init,{quantile,prepareNumerical,prepareNumericals} from '../src/node.js';
const bytes=readFileSync(new URL('../wasm/crc_framework_bg.wasm',import.meta.url));
const start=performance.now();await init();
console.log(`WASM ${bytes.length} bytes (${gzipSync(bytes).length} gzip); init ${performance.now()-start}ms`);
const cases=JSON.parse(readFileSync(new URL('../../python/crc_framework/contracts/numerical/fixtures-1.0.json',import.meta.url)));
const documents=cases.accepted.filter(c=>c.document.kind==='parametric').map(c=>c.document);
const rows=Array.from({length:1000},(_,i)=>documents[i%documents.length]);
function measure(name,fn) {const start=performance.now();for(let i=0;i<5;i++)fn();console.log(`${name}: ${(performance.now()-start)/5}ms`);}
measure('1000 one-shot numerical documents',()=>rows.map(r=>quantile(JSON.stringify(r),.99)));
const prepared=rows.map(r=>prepareNumerical(JSON.stringify(r)));
try {measure('1000 prepared numerical documents',()=>prepared.map(r=>r.quantile(.99)));} finally {prepared.forEach(r=>r.free());}
const batch=prepareNumericals(JSON.stringify(rows));
try {measure('1000 prepared documents in one WASM call',()=>batch.quantile(.99));} finally {batch.free();}

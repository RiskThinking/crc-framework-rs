import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import init, * as api from '../src/index.js';
const directory=new URL('../../python/crc_framework/contracts/',import.meta.url);
// Retain exact large integer tokens when passing source documents through JS.
function fixtures(path) {
 return JSON.parse(readFileSync(new URL(path,directory),'utf8'),(key,value,context)=>
   typeof value==='number' && Number.isInteger(value) && !Number.isSafeInteger(value)
     ? JSON.rawJSON(context.source) : value);
}
const numerical=fixtures('numerical/fixtures-1.0.json');
await init({module_or_path:readFileSync(new URL('../wasm/crc_framework_bg.wasm',import.meta.url))});
for (const fixture of numerical.accepted) test(`numerical ${fixture.name}`,()=>{
 const text=JSON.stringify(fixture.document);
 api.validateDocument(api.DocumentSchema.NUMERICAL,text);
 const prepared=api.prepareNumerical(text);
 try {
  for(const query of fixture.queries) {
   if(query.outcome==='out_of_support') {assert.throws(()=>prepared.quantile(query.probability),RangeError);continue;}
   const expected=query.outcome==='positive_infinity'?Infinity:query.outcome==='negative_infinity'?-Infinity:Number(query.value?.rawJSON??query.value);
   for(const actual of [prepared.quantile(query.probability),api.quantile(text,query.probability)]) {
    if(!Number.isFinite(expected)) assert.equal(actual,expected);
    else assert.ok(Math.abs(actual-expected)<=2e-10*Math.max(1,Math.abs(expected)),`${actual} != ${expected}`);
   }
  }
 } finally {prepared.free();prepared.free();}
 assert.throws(()=>prepared.quantile(.5),/freed/);
});
for(const fixture of numerical.rejected) test(`reject numerical ${fixture.name}`,()=>assert.throws(()=>api.prepareNumerical(JSON.stringify(fixture.document)))) ;
for(const fixture of numerical.rejected_text) test(`reject text ${fixture.name}`,()=>assert.throws(()=>api.prepareNumerical(fixture.text)));
for(const fixture of fixtures('scientific/fixtures-1.0.json').cases) test(`scientific ${fixture.name}`,()=>{
 const call=()=>api.validateDocument(fixture.schema,JSON.stringify(fixture.document));
 if(fixture.valid) assert.doesNotThrow(call);else assert.throws(call);
});
for(const [path,schema] of [['dataset/fixtures.json',api.DocumentSchema.DATASET],['taxonomy/fixtures.json',api.DocumentSchema.CATALOG]]) {
 const cases=fixtures(path);
 for(const fixture of cases.accepted) test(`accept ${path} ${fixture.name}`,()=>assert.doesNotThrow(()=>api.validateDocument(schema,JSON.stringify(fixture.document))));
 for(const fixture of cases.rejected) test(`reject ${path} ${fixture.name}`,()=>assert.throws(()=>api.validateDocument(schema,JSON.stringify(fixture.document))));
}
test('scalar and batch queries share one name and preserve infinities',()=>{
 const text=JSON.stringify(numerical.accepted.find(x=>x.name==='point').document);
 assert.deepEqual([...api.quantile(text,new Float64Array([0,.5,1]))],[0,0,0]);
 const prepared=api.prepareNumericals(`[${text},${text}]`);
 try {assert.equal(prepared.quantile(.5).length,2);} finally {prepared.free();}
 for(const name of ['evaluateCurve','evaluateCurves','prepareCurve','prepareCurves']) assert.equal(api[name],undefined);
});

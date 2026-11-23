import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import init, * as api from '../src/node.js';
const directory=new URL('../../python/crc_framework/contracts/',import.meta.url);
// Retain exact large integer tokens when passing source documents through JS.
function fixtures(path) {
 return JSON.parse(readFileSync(new URL(path,directory),'utf8'),(key,value,context)=>
   typeof value==='number' && Number.isInteger(value) && !Number.isSafeInteger(value)
     ? JSON.rawJSON(context.source) : value);
}
const numerical=fixtures('numerical/fixtures-1.0.json');
await init();
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
 try {
  assert.equal(prepared.length,2);
  const {values,outcomes}=prepared.quantile(.5);
  assert.deepEqual([...values],[0,0]);
  assert.deepEqual([...outcomes],[api.QuantileOutcome.VALUE,api.QuantileOutcome.VALUE]);
 } finally {prepared.free();}
 for(const name of ['evaluateCurve','evaluateCurves','prepareCurve','prepareCurves']) assert.equal(api[name],undefined);
});

test('WASM queries reject values that JavaScript could coerce to a probability', () => {
 const text = JSON.stringify(numerical.accepted.find(fixture => fixture.name === 'point').document);
 const single = api.prepareNumerical(text);
 const batch = api.prepareNumericals(`[${text}]`);
 try {
  for (const invalid of ['0.5', null, true, undefined]) {
   assert.throws(() => single.quantile(invalid), TypeError);
   assert.throws(() => single.quantile([invalid]), TypeError);
   assert.throws(() => batch.quantile(invalid), TypeError);
  }
  assert.throws(() => single.quantile(new Array(1)), TypeError);
  for (const invalid of [NaN, Infinity, -.1, 1.1]) {
   assert.throws(() => single.quantile(invalid), RangeError);
   assert.throws(() => single.quantile(new Float64Array([invalid])), RangeError);
   assert.throws(() => batch.quantile(invalid), RangeError);
  }
 } finally {
  single.free();
  batch.free();
 }
});

test('batched queries report a value or a reason for every document', () => {
 const accepted = numerical.accepted.map(fixture => JSON.stringify(fixture.document));
 const rejected = numerical.rejected.map(fixture => JSON.stringify(fixture.document));
 const documents = [...accepted, ...rejected];
 const batch = api.prepareNumericals(`[${documents.join(',')}]`);
 try {
  assert.equal(batch.length, documents.length);
  assert.deepEqual(batch.invalidDocuments.map(item => item.index),
   rejected.map((_, i) => accepted.length + i));
  for (const item of batch.invalidDocuments) assert.equal(typeof item.message, 'string');
  const probabilities = [...new Set(numerical.accepted.flatMap(f => f.queries.map(q => q.probability)))];
  for (const probability of probabilities) {
   const {values, outcomes} = batch.quantile(probability);
   assert.ok(values instanceof Float64Array && outcomes instanceof Uint8Array);
   accepted.forEach((text, i) => {
    let expected;
    try { expected = api.quantile(text, probability); } catch (error) {
     assert.ok(error instanceof RangeError, String(error));
     assert.ok(Number.isNaN(values[i]));
     assert.ok([api.QuantileOutcome.BELOW_SUPPORT, api.QuantileOutcome.ABOVE_SUPPORT].includes(outcomes[i]));
     return;
    }
    assert.equal(outcomes[i], api.QuantileOutcome.VALUE);
    assert.ok(Object.is(values[i], expected), `${i}: ${values[i]} != ${expected}`);
   });
   rejected.forEach((_, i) => {
    assert.equal(outcomes[accepted.length + i], api.QuantileOutcome.INVALID_DOCUMENT);
    assert.ok(Number.isNaN(values[accepted.length + i]));
   });
  }
 } finally { batch.free(); }
 assert.throws(() => batch.quantile(.5), /freed/);
});

test('batch outcomes tell below-support from above-support queries', () => {
 const partial = '{"object":"quantile_curve","probabilities":[0.1,0.9],"schema_version":"1.0","values":[1,3]}';
 const batch = api.prepareNumericals(`[${partial}]`);
 try {
  assert.equal(batch.quantile(.05).outcomes[0], api.QuantileOutcome.BELOW_SUPPORT);
  assert.equal(batch.quantile(.95).outcomes[0], api.QuantileOutcome.ABOVE_SUPPORT);
  assert.equal(batch.quantile(.5).values[0], 2);
  assert.throws(() => api.prepareNumericals(partial), TypeError);
  assert.throws(() => api.prepareNumericals('[1,'), TypeError);
 } finally { batch.free(); }
});

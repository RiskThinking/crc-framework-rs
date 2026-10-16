import { readFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import init, {
  evaluateCurve,
  evaluateCurves,
  prepareCurve,
  prepareCurves,
} from "../src/index.js";
const bytes = readFileSync(
  new URL("../wasm/crc_framework_bg.wasm", import.meta.url),
);
const start = performance.now();
await init({ module_or_path: bytes });
console.log(
  `WASM ${bytes.length} bytes (${gzipSync(bytes).length} gzip); init ${(performance.now() - start).toFixed(2)} ms`,
);
const cases = JSON.parse(
  readFileSync(
    new URL("../../tests/fixtures/canonical-quantiles.json", import.meta.url),
  ),
);
const families = cases.filter(
  (c) =>
    c.properties.curve_kind === "fitted" &&
    c.properties.curve_type !== "skewnorm",
);
const rows = Array.from(
  { length: 10000 },
  (_, i) => families[i % families.length].properties,
);
const prepared = rows.map(prepareCurve);
function measure(name, fn, iterations = 20) {
  for (let i = 0; i < 3; i++) fn();
  const times = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    fn();
    times.push(performance.now() - start);
  }
  times.sort((a, b) => a - b);
  console.log(
    `${name}: ${times[Math.floor(times.length / 2)].toFixed(2)} ms median`,
  );
}
measure("10k scalar records", () =>
  rows.map((row) => evaluateCurve(row, 0.99)),
);
measure("10k batch records", () => evaluateCurves(rows, 0.99));
measure("10k prepared queries", () =>
  prepared.map((curve) => curve.evaluate(0.99)),
);
prepared.forEach((curve) => curve.free());
const collection = prepareCurves(rows);
measure("10k prepared batch queries", () => collection.evaluate(0.99));
collection.free();
const skew = prepareCurve(
  cases.find((c) => c.properties.curve_type === "skewnorm").properties,
);
measure(
  "100 skew-normal prepared queries",
  () => Array.from({ length: 100 }, () => skew.evaluate(0.99)),
  5,
);
skew.free();

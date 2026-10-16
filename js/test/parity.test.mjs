import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import init, {
  evaluateCurve,
  evaluateCurves,
  prepareCurve,
  prepareCurves,
} from "../src/index.js";

assert.throws(() => evaluateCurve({}, 0.5), /not initialized/);
await Promise.all([
  init({
    module_or_path: readFileSync(
      new URL("../wasm/crc_framework_bg.wasm", import.meta.url),
    ),
  }),
  init(),
]);
export const cases = JSON.parse(
  readFileSync(
    new URL("../../tests/fixtures/canonical-quantiles.json", import.meta.url),
  ),
);

export function close(actual, expected) {
  if (expected === null) return assert.equal(actual, null);
  assert.equal(typeof actual, "number");
  assert.ok(
    Math.abs(actual - expected) <= 2e-10 * Math.max(1, Math.abs(expected)),
    `${actual} != ${expected}`,
  );
}

for (const fixture of cases) {
  test(`Python/native/WASM parity: ${fixture.name}`, () => {
    const curve = prepareCurve(fixture.properties);
    try {
      const quantiles = curve.quantiles(fixture.probabilities);
      fixture.probabilities.forEach((q, index) => {
        const expected = fixture.expected[index];
        close(evaluateCurve(fixture.properties, q), expected);
        close(evaluateCurves([fixture.properties], q)[0], expected);
        close(curve.evaluate(q), expected);
        close(quantiles[index], expected);
      });
    } finally {
      curve.free();
    }
    assert.throws(() => curve.evaluate(0.5), /freed/);
    curve.free();
  });
}

test("MVT string arrays, unrelated properties, omitted nulls and batch ordering", () => {
  const row = {
    curve_kind: "tabulated",
    curve_type: "linear_probability",
    curve_probabilities: "[0.2,0.8]",
    curve_values: "[1,4]",
    h3_cell: "arbitrary",
    building_count: 10,
  };
  evaluateCurves([row, cases.at(-1).properties, row], 0.5).forEach((value, i) =>
    close(value, [2.5, null, 2.5][i]),
  );
  assert.equal(evaluateCurve(row, 0.1), null);
  const gumbel = cases.find(
    (c) => c.properties.curve_type === "gumbel_r",
  ).properties;
  assert.equal(
    evaluateCurve({ ...gumbel, curve_shape: 999 }, 0.5),
    evaluateCurve(gumbel, 0.5),
  );
  const { curve_shape, ...omitted } = gumbel;
  assert.equal(evaluateCurve(omitted, 0.5), evaluateCurve(gumbel, 0.5));
});

test("signed zero is preserved through the WASM boundary", () => {
  const fixture = cases.find((c) => c.name === "point-mass-negative-zero");
  assert.ok(Object.is(evaluateCurve(fixture.properties, 0.5), -0));
});

test("malformed inputs throw TypeError; invalid q throws RangeError; batches abort", () => {
  const fitted = cases[0].properties;
  const tabulated = cases.find((c) => c.name === "tabulated-narrow").properties;
  for (const q of [NaN, Infinity, -0.1, 1.1, null, "0.5", undefined]) {
    assert.throws(() => evaluateCurve(fitted, q), RangeError);
    assert.throws(() => evaluateCurves([], q), RangeError);
  }
  for (const row of [
    null,
    [],
    {},
    { ...fitted, curve_scale: 0 },
    { ...fitted, curve_scale: Infinity },
    { ...fitted, curve_type: "invalid" },
    { ...fitted, curve_shape: null },
    { ...fitted, curve_location: "1" },
    { ...fitted, curve_atom_probability: 0.5 },
    { ...tabulated, curve_probabilities: "[bad" },
    { ...tabulated, curve_values: "{}" },
    { ...tabulated, curve_values: [4, 1] },
    { ...tabulated, curve_probabilities: [0.5, 0.5] },
    { ...tabulated, curve_values: [1, null] },
    { ...tabulated, curve_probabilities: [-0.1, 0.9] },
    { curve_kind: "no_data", curve_type: "invalid" },
    { ...cases.at(-1).properties, curve_location: 0 },
    { curve_kind: "point_mass", curve_type: "point_mass", curve_location: 0 },
  ]) {
    assert.throws(() => evaluateCurve(row, 0.5), TypeError);
    assert.throws(() => evaluateCurves([fitted, row, fitted], 0.5), TypeError);
  }
});

test("prepared collections match every family, preserve order/null/signed zero and own a snapshot", () => {
  const rows = cases.map((fixture) => structuredClone(fixture.properties));
  const collection = prepareCurves(rows);
  try {
    for (const q of [0, 0.1, 0.2, 0.5, 0.99, 0.9985, 1]) {
      const expected = rows.map((row) => evaluateCurve(row, q));
      const actual = collection.evaluate(q);
      assert.deepEqual(actual, expected);
    }
    const before = collection.evaluate(0.5);
    rows[0].curve_location = 999;
    assert.deepEqual(collection.evaluate(0.5), before);
    for (const q of [NaN, Infinity, -0.1, 1.1, "0.5", null]) {
      assert.throws(() => collection.evaluate(q), RangeError);
    }
  } finally {
    collection.free();
  }
  collection.free();
  assert.throws(() => collection.evaluate(0.5), /freed/);
  const empty = prepareCurves([]);
  try {
    assert.deepEqual(empty.evaluate(0.5), []);
    assert.throws(() => empty.evaluate(NaN), RangeError);
  } finally {
    empty.free();
  }
  assert.throws(
    () => prepareCurves([cases[0].properties, { curve_kind: "fitted" }]),
    TypeError,
  );
});

test("packed transport preserves missing/non-finite distinctions and strict element types", () => {
  const row = cases[0].properties;
  for (const value of [NaN, Infinity, -Infinity]) {
    assert.throws(
      () => evaluateCurve({ ...row, curve_atom_probability: value }, 0.5),
      TypeError,
    );
    assert.throws(
      () => prepareCurves([{ ...row, curve_location: value }]),
      TypeError,
    );
  }
  const tabulated = cases.find(
    (fixture) => fixture.name === "tabulated-narrow",
  ).properties;
  for (const knots of [
    [1, null],
    [1, "2"],
    [1, undefined],
    [1, NaN],
    [1, Infinity],
    [],
    [1],
  ]) {
    assert.throws(
      () => prepareCurves([{ ...tabulated, curve_values: knots }]),
      TypeError,
    );
  }
  const gumbel = cases.find(
    (fixture) => fixture.properties.curve_type === "gumbel_r",
  ).properties;
  assert.equal(
    evaluateCurve({ ...gumbel, curve_shape: NaN }, 0.5),
    evaluateCurve(gumbel, 0.5),
  );
});

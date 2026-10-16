import initialize, {
  initSync as initializeSync,
  PreparedCurve,
  PreparedCurves,
  curveKindNames,
  curveTypeNames,
  evaluateCurvePacked,
} from "../wasm/crc_framework.js";

let ready = false;
let pending;
let kindCodes;
let typeCodes;
function finishInit() {
  kindCodes = new Map(curveKindNames().map((name, code) => [name, code]));
  typeCodes = new Map(curveTypeNames().map((name, code) => [name, code]));
  ready = true;
}

/** Initialize once. Importing this module is safe during server rendering. */
export default function init(input) {
  if (ready) return Promise.resolve();
  if (!pending) {
    pending = initialize(input)
      .then(finishInit)
      .catch((error) => {
        pending = undefined;
        throw error;
      });
  }
  return pending;
}

export function initSync(input) {
  if (ready) return;
  initializeSync(input);
  finishInit();
}

function assertReady() {
  if (!ready)
    throw new Error(
      "CRC Framework is not initialized; await init() before evaluating curves",
    );
}

function checkProbability(q) {
  if (typeof q !== "number" || !Number.isFinite(q) || q < 0 || q > 1) {
    throw new RangeError("probability must be finite and in [0, 1]");
  }
}

/** Synchronous after init. Missing data and queries outside support return null. */
export function evaluateCurve(properties, q) {
  assertReady();
  checkProbability(q);
  const [rows, knots] = pack([properties]);
  return evaluateCurvePacked(rows, knots, q);
}

/** A malformed row aborts the batch; results preserve input order. */
export function evaluateCurves(properties, q) {
  assertReady();
  checkProbability(q);
  if (!Array.isArray(properties))
    throw new TypeError("curves must be an array");
  const curves = prepareCurves(properties);
  try {
    return curves.evaluate(q);
  } finally {
    curves.free();
  }
}

/** Explicit ownership for repeated queries: release the handle with free(). */
export function prepareCurve(properties) {
  assertReady();
  const [rows, knots] = pack([properties]);
  const handle = PreparedCurve.fromPacked(rows, knots);
  let freed = false;
  function checkAlive() {
    if (freed) throw new Error("prepared curve has been freed");
  }
  return {
    evaluate(q) {
      checkAlive();
      checkProbability(q);
      return handle.evaluate(q);
    },
    quantiles(probabilities) {
      checkAlive();
      if (
        !Array.isArray(probabilities) &&
        !(probabilities instanceof Float64Array)
      ) {
        throw new TypeError("probabilities must be an array or Float64Array");
      }
      for (const q of probabilities) checkProbability(q);
      return handle.quantiles(probabilities);
    },
    free() {
      if (!freed) {
        handle.free();
        freed = true;
      }
    },
  };
}

/** Prepare a collection once; each evaluate call crosses the WASM boundary once. */
export function prepareCurves(properties) {
  assertReady();
  if (!Array.isArray(properties))
    throw new TypeError("curves must be an array");
  const [rows, knots] = pack(properties);
  const handle = PreparedCurves.fromPacked(rows, knots);
  let freed = false;
  return {
    evaluate(q) {
      if (freed) throw new Error("prepared curves have been freed");
      checkProbability(q);
      const values = handle.evaluate(q);
      const result = new Array(values.length);
      for (let i = 0; i < values.length; i++)
        result[i] = Number.isNaN(values[i]) ? null : values[i];
      return result;
    },
    free() {
      if (!freed) {
        handle.free();
        freed = true;
      }
    },
  };
}

// Language conversion only: numbers stay float64 and missing fields get explicit
// presence bits, so a supplied NaN can never be mistaken for an omitted value.
const SCALARS = [
  "curve_shape",
  "curve_location",
  "curve_scale",
  "curve_atom_probability",
  "curve_atom_location",
];
const ARRAYS = ["curve_probabilities", "curve_values"];
// MVT repeats identical probability grids and knot strings across tile queries.
// Internal parsed arrays never escape this adapter. Bound both count and size.
const parsedArrays = new Map();
let parsedValues = 0;
function parseArray(text, field) {
  const cached = parsedArrays.get(text);
  if (cached) return cached;
  let values;
  try {
    values = JSON.parse(text);
  } catch {
    throw new TypeError(`${field} must be a JSON array`);
  }
  if (!Array.isArray(values)) throw new TypeError(`${field} must be an array`);
  if (values.length <= 65_536) {
    while (parsedArrays.size >= 2048 || parsedValues + values.length > 65_536) {
      const oldest = parsedArrays.keys().next().value;
      parsedValues -= parsedArrays.get(oldest).length;
      parsedArrays.delete(oldest);
    }
    parsedArrays.set(text, values);
    parsedValues += values.length;
  }
  return values;
}
function pack(properties) {
  const rows = new Float64Array(properties.length * 12);
  const knots = [];
  for (let i = 0; i < properties.length; i++) {
    const row = properties[i];
    if (!row || typeof row !== "object" || Array.isArray(row))
      throw new TypeError("curve properties must be an object");
    const offset = i * 12;
    // Unknown names use an invalid code and are rejected by the Rust decoder.
    rows[offset] = kindCodes.get(row.curve_kind) ?? -1;
    rows[offset + 1] = typeCodes.get(row.curve_type) ?? -1;
    let mask = 0;
    for (let j = 0; j < SCALARS.length; j++) {
      const value = row[SCALARS[j]];
      if (value == null) continue;
      if (typeof value !== "number")
        throw new TypeError(`${SCALARS[j]} must be a number`);
      mask |= 1 << j;
      rows[offset + 3 + j] = value;
    }
    rows[offset + 2] = mask;
    for (let j = 0; j < ARRAYS.length; j++) {
      const field = ARRAYS[j];
      let values = row[field];
      rows[offset + 8 + j * 2] = knots.length;
      rows[offset + 9 + j * 2] = -1;
      if (values == null) continue;
      if (typeof values === "string") {
        values = parseArray(values, field);
      }
      if (!Array.isArray(values))
        throw new TypeError(`${field} must be an array`);
      rows[offset + 9 + j * 2] = values.length;
      for (let k = 0; k < values.length; k++) {
        if (typeof values[k] !== "number")
          throw new TypeError(`${field} must contain numbers`);
        knots.push(values[k]);
      }
    }
  }
  return [rows, Float64Array.from(knots)];
}

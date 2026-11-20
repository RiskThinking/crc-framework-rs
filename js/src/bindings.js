import initWasm, {
  PreparedNumerical,
  PreparedNumericals,
  validateDocument as validateNative,
} from '../wasm/crc_framework.js';

export const DocumentSchema = Object.freeze({
  NUMERICAL: 'https://climateriskcommons.org/schemas/numerical/1.0',
  FITTING_RECORD: 'https://climateriskcommons.org/schemas/fitting/1.0',
  FITTING_SPECIFICATION: 'https://climateriskcommons.org/schemas/fitting/1.0#/$defs/specification',
  DATASET: 'https://climateriskcommons.org/schemas/dataset/2.0',
  CATALOG: 'https://climateriskcommons.org/schemas/taxonomy/1.0',
  MEASUREMENT: 'https://climateriskcommons.org/schemas/measurement-1.0.schema.json',
  IMPACT_CONTEXT: 'https://climateriskcommons.org/schemas/impact-context-1.0.schema.json',
  IMPACT: 'https://climateriskcommons.org/schemas/impact-2.0.schema.json',
  IMPACT_REFERENCE: 'https://climateriskcommons.org/schemas/impact-reference-2.0.schema.json',
});

let ready = false;
let pending;

export function initialize(loadInput) {
  pending ??= Promise.resolve().then(loadInput)
    .then(input => initWasm({ module_or_path: input }))
    .then(() => { ready = true; })
    .catch(error => {
      pending = undefined;
      throw error;
    });
  return pending;
}

function assertReady() {
  if (!ready) throw new Error('await init() before using the CRC Framework');
}

function assertText(text) {
  if (typeof text !== 'string') throw new TypeError('document must be JSON text');
}

function assertNumber(probability) {
  if (typeof probability !== 'number') throw new TypeError('probability must be a number');
}

/** Validate a current dataset, numerical, measurement, context, or impact JSON document. */
export function validateDocument(schema, text) {
  assertReady();
  if (typeof schema !== 'string') throw new TypeError('schema must be a DocumentSchema value');
  assertText(text);
  validateNative(schema, text);
}

/** Parse one current numerical JSON document, retaining its exact representation. */
export function prepareNumerical(text) {
  assertReady();
  assertText(text);
  const native = new PreparedNumerical(text);
  let freed = false;
  return {
    quantile(probabilities) {
      if (freed) throw new Error('prepared numerical has been freed');
      if (typeof probabilities === 'number') return native.quantile(probabilities);
      if (Array.isArray(probabilities)) {
        for (const probability of probabilities) assertNumber(probability);
      } else if (!(probabilities instanceof Float64Array)) {
        throw new TypeError('probabilities must be a number, array of numbers, or Float64Array');
      }
      return native.quantileBatch(probabilities);
    },
    free() {
      if (!freed) {
        native.free();
        freed = true;
      }
    },
  };
}

export function quantile(text, probabilities) {
  const prepared = prepareNumerical(text);
  try {
    return prepared.quantile(probabilities);
  } finally {
    prepared.free();
  }
}

/** Parse an array of numerical documents for repeated queries in one WASM call. */
export function prepareNumericals(text) {
  assertReady();
  assertText(text);
  const native = new PreparedNumericals(text);
  let freed = false;
  return {
    quantile(probability) {
      if (freed) throw new Error('prepared numericals have been freed');
      assertNumber(probability);
      return native.quantile(probability);
    },
    free() {
      if (!freed) {
        native.free();
        freed = true;
      }
    },
  };
}

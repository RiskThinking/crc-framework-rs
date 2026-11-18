import { readFile } from 'node:fs/promises';
import { initialize } from './bindings.js';

export { DocumentSchema, prepareNumerical, prepareNumericals, quantile, validateDocument } from './bindings.js';

export default function init(input) {
  return initialize(async () => {
    const provided = await input;
    const source = provided === undefined
      ? new URL('../wasm/crc_framework_bg.wasm', import.meta.url)
      : provided;
    return source instanceof URL && source.protocol === 'file:' ? readFile(source) : source;
  });
}

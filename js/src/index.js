import { initialize } from './bindings.js';

export { DocumentSchema, prepareNumerical, prepareNumericals, quantile, validateDocument } from './bindings.js';

/** Load the bundled WebAssembly asset, or initialize with a supplied module or source. */
export default function init(input) {
  return initialize(() => input);
}

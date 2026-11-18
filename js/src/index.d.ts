// Browser objects are usable when declared by the consumer's TypeScript environment.
type InstanceIfAvailable<Scope, Name extends PropertyKey> =
  Name extends keyof Scope
    ? Scope[Name] extends new (...args: never[]) => infer Instance ? Instance : never
    : never;
type WasmModule = typeof globalThis extends { WebAssembly: infer WebAssemblyAPI }
  ? InstanceIfAvailable<WebAssemblyAPI, 'Module'>
  : never;
export type InitInput =
  | string
  | ArrayBuffer
  | ArrayBufferView
  | InstanceIfAvailable<typeof globalThis, 'URL' | 'Request' | 'Response'>
  | WasmModule;
/** Initialize the bundled module, or supply your own WebAssembly module, bytes, or URL. */
export default function init(input?: InitInput | Promise<InitInput>): Promise<void>;
export interface PreparedNumerical {
  quantile(probability: number): number;
  quantile(probabilities: readonly number[] | Float64Array): Float64Array;
  free(): void;
}
export interface PreparedNumericals {
  quantile(probability: number): Float64Array;
  free(): void;
}
/** Prepare a numerical document for repeated queries. */
export function prepareNumerical(text: string): PreparedNumerical;
export function prepareNumericals(text: string): PreparedNumericals;
export function quantile(text: string, probability: number): number;
export function quantile(text: string, probabilities: readonly number[] | Float64Array): Float64Array;
/** Strict current schema and scientific-semantic validation. */
export function validateDocument(schema: DocumentSchema, text: string): void;

export const DocumentSchema: Readonly<{
 NUMERICAL:'https://climateriskcommons.org/schemas/numerical/1.0';
 FITTING_RECORD:'https://climateriskcommons.org/schemas/fitting/1.0';
 FITTING_SPECIFICATION:'https://climateriskcommons.org/schemas/fitting/1.0#/$defs/specification';
 DATASET:'https://climateriskcommons.org/schemas/dataset/2.0';
 CATALOG:'https://climateriskcommons.org/schemas/taxonomy/1.0';
 MEASUREMENT:'https://climateriskcommons.org/schemas/measurement-1.0.schema.json';
 IMPACT_CONTEXT:'https://climateriskcommons.org/schemas/impact-context-1.0.schema.json';
 IMPACT:'https://climateriskcommons.org/schemas/impact-2.0.schema.json';
 IMPACT_REFERENCE:'https://climateriskcommons.org/schemas/impact-reference-2.0.schema.json';
}>;

export type DocumentSchema = (typeof DocumentSchema)[keyof typeof DocumentSchema];

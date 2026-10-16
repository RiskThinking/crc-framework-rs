export type CurveKind =
  | "fitted"
  | "hurdle"
  | "point_mass"
  | "tabulated"
  | "no_data";
export type FittedFamily =
  | "genextreme"
  | "genpareto"
  | "weibull_min"
  | "weibull_max"
  | "gumbel_r"
  | "gumbel_l"
  | "skewnorm";
export type NoDataReason =
  | "below_effective_resolution"
  | "insufficient_informative_support"
  | "degenerate_effective_range"
  | "scientific_exclusion";
/** Canonical property subset; optional fields may be omitted or null. */
export interface CurveProperties {
  curve_kind: CurveKind;
  curve_type: FittedFamily | NoDataReason | "point_mass" | "linear_probability";
  curve_shape?: number | null;
  curve_location?: number | null;
  curve_scale?: number | null;
  curve_atom_probability?: number | null;
  curve_atom_location?: number | null;
  curve_probabilities?: readonly number[] | string | null;
  curve_values?: readonly number[] | string | null;
  [key: string]: unknown;
}
/** Untyped MVT properties can be passed directly; Rust validates the canonical subset. */
export type CurvePropertyRow = { [key: string]: unknown };
export type InitInput =
  | RequestInfo
  | URL
  | Response
  | BufferSource
  | WebAssembly.Module;
export default function init(input?: {
  module_or_path: InitInput | Promise<InitInput>;
}): Promise<void>;
export function initSync(input: {
  module: BufferSource | WebAssembly.Module;
}): void;
/** RangeError for invalid q; TypeError for malformed records; null for no data/outside support. */
export function evaluateCurve(
  properties: CurveProperties | CurvePropertyRow,
  q: number,
): number | null;
/** A malformed row aborts the batch. */
export function evaluateCurves(
  properties: readonly (CurveProperties | CurvePropertyRow)[],
  q: number,
): Array<number | null>;
export interface PreparedCurve {
  evaluate(q: number): number | null;
  quantiles(
    probabilities: readonly number[] | Float64Array,
  ): Array<number | null>;
  free(): void;
}
export function prepareCurve(
  properties: CurveProperties | CurvePropertyRow,
): PreparedCurve;

/** Reusable collection: one WASM call per probability, input order preserved. */
export interface PreparedCurves {
  evaluate(q: number): Array<number | null>;
  free(): void;
}
export function prepareCurves(
  properties: readonly (CurveProperties | CurvePropertyRow)[],
): PreparedCurves;

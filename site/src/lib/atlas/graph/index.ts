// Kind-A runtime entry point: executor, explanations, receptive fields, references, legacy specs.
export { fromParts, loadGraph, traceInputs } from './executor.ts';
export { checkExplain, pickUnit, unitCount } from './explain.ts';
export type { ExplainCheck } from './explain.ts';
export { receptiveField } from './receptive.ts';
export { readReference, parseReference } from './refs.ts';
export type { RefCase } from './refs.ts';
export { normalizeSpec } from './legacy.ts';

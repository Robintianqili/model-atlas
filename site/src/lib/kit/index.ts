/**
 * Pure TS utilities of the kit (safe to import anywhere, including workers).
 * Components are imported one by one so pages only ship what they use:
 *   import TensorGrid from '$lib/kit/TensorGrid.svelte';
 * (A barrel of .svelte files would pull side-effect CSS such as KaTeX into every page.)
 */
export * from './color.ts';
export * from './format.ts';
export * from './stats.ts';
export * from './draw.ts';
export * from './spec.ts';
export type * from './types.ts';

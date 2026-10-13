/**
 * The TinyCNN lesson's stage views, loaded lazily by Lesson.svelte: they render only once the model has run, so they
 * (and the formula renderer they bring) stay out of the page shell, which carries the cover, the input chip and the
 * lesson's text. Explore's zoom view imports the same components from its own chunk.
 */
export { default as InputDetail } from './InputDetail.svelte';
export { default as ConvDetail } from './ConvDetail.svelte';
export { default as KernelMaps } from './KernelMaps.svelte';
export { default as ReluDetail } from './ReluDetail.svelte';
export { default as PoolDetail } from './PoolDetail.svelte';
export { default as GrowStage } from './GrowStage.svelte';
export { default as FlattenDetail } from './FlattenDetail.svelte';
export { default as DenseDetail } from './DenseDetail.svelte';
export { default as SoftmaxDetail } from './SoftmaxDetail.svelte';

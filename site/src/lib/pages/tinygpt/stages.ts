/**
 * The lesson's stage views, loaded lazily by Lesson.svelte (they only render once the model has run, so they do not
 * need to be in the page shell; this keeps /m/tinygpt/ under its JS budget). The block player, the dot product, the
 * attention text and map and the logit lens are shared with Explore, which imports them from its own chunk.
 */
export { default as LessonTokens } from './LessonTokens.svelte';
export { default as EmbedStage } from './EmbedStage.svelte';
export { default as LnStage } from './LnStage.svelte';
export { default as SoftmaxRow } from './SoftmaxRow.svelte';
export { default as LogitsStage } from './LogitsStage.svelte';
export { default as GenStage } from './GenStage.svelte';
export { default as BlockView } from './BlockView.svelte';
export { default as DotProduct } from './DotProduct.svelte';
export { default as AttnText } from './AttnText.svelte';
export { default as LogitLens } from './LogitLens.svelte';
export { default as AttentionMatrix } from '$lib/kit/AttentionMatrix.svelte';

<!--
  Small line icon for an input kind (drawing, text, audio, …). Decorative: pair it with text.
  <InputIcon kind="drawing" />
-->
<script lang="ts">
	import type { InputKind } from '$lib/catalog/types';

	let { kind, size = 14 }: { kind: InputKind; size?: number } = $props();
</script>

<svg viewBox="0 0 16 16" width={size} height={size} aria-hidden="true" class="ii" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
	{#if kind === 'drawing'}
		<path d="M2.5 13.5l1.2-3.6L10.8 2.8a1.5 1.5 0 0 1 2.1 0l.3.3a1.5 1.5 0 0 1 0 2.1L6.1 12.3z" /><path d="M9.6 4l2.4 2.4" />
	{:else if kind === 'image' || kind === 'webcam'}
		<rect x="2" y="3" width="12" height="10" rx="1.5" /><circle cx="6" cy="6.5" r="1.2" /><path d="M2.5 12l3.8-3.6 2.6 2.4 2-1.8 2.6 2.6" />
	{:else if kind === 'text' || kind === 'tokens' || kind === 'conversation'}
		<path d="M3 4h10M3 7h10M3 10h6" />{#if kind === 'conversation'}<path d="M10.5 12.5l2-2" />{/if}
	{:else if kind === 'audio'}
		<path d="M2 8h1.5M4.5 5v6M7 2.5v11M9.5 5v6M12 6.5v3M14 8h0" />
	{:else if kind === 'series'}
		<path d="M1.5 12l3-4 3 2.5 3.5-6 3.5 3.5" /><path d="M1.5 14h13" />
	{:else if kind === 'tabular'}
		<rect x="2" y="3" width="12" height="10" rx="1" /><path d="M2 6.5h12M2 10h12M6.5 3v10" />
	{:else if kind === 'points2d' || kind === 'pointcloud'}
		<circle cx="4" cy="5" r="1" /><circle cx="8" cy="9" r="1" /><circle cx="12" cy="4" r="1" /><circle cx="5" cy="12" r="1" /><circle cx="12" cy="11" r="1" />
	{:else if kind === 'graph'}
		<circle cx="3.5" cy="8" r="1.6" /><circle cx="12" cy="3.5" r="1.6" /><circle cx="12" cy="12.5" r="1.6" /><path d="M5 7.2l5.5-3M5 8.8l5.5 3M12 5.1v5.8" />
	{:else if kind === 'game'}
		<rect x="2" y="2" width="12" height="12" rx="1" /><path d="M6 2v12M10 2v12M2 6h12M2 10h12" />
	{:else if kind === 'protein'}
		<path d="M2 11c2-6 4 2 6-4s4 2 6-4" /><circle cx="2" cy="11" r="1" /><circle cx="14" cy="3" r="1" />
	{:else if kind === 'ratings'}
		<path d="M8 2l1.8 3.7 4 .6-2.9 2.8.7 4L8 11.2 4.4 13.1l.7-4L2.2 6.3l4-.6z" />
	{:else if kind === 'video'}
		<rect x="2" y="4" width="9" height="8" rx="1.2" /><path d="M11 7l3-2v6l-3-2" />
	{:else}
		<circle cx="8" cy="8" r="5.5" /><path d="M5.5 8h5" />
	{/if}
</svg>

<style>
	.ii {
		display: inline-block;
		vertical-align: -2px;
		flex: 0 0 auto;
	}
</style>

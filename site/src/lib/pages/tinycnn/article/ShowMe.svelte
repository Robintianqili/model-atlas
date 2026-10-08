<!-- An (i)-style link from the article into the tool: selects a layer and scrolls to the zoom view. -->
<script lang="ts">
	import type { CnnStore } from '../store.svelte';

	let { S, onshow, layer, label }: { S: CnnStore; onshow: () => void; layer: string; label: string } = $props();

	function go() {
		const ch = S.sel.layer === layer ? S.sel.ch : 0;
		S.select(layer, ch);
		onshow();
	}
</script>

<button type="button" class="show" onclick={go}>
	<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" stroke-width="1.4" /><path d="M8 7v4.2M8 4.6v.1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" /></svg>
	{label}
</button>

<style>
	.show {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		padding: 0 8px;
		min-height: 26px;
		border: 1px solid color-mix(in srgb, var(--accent) 45%, transparent);
		border-radius: var(--r-pill);
		background: var(--accent-soft);
		color: var(--accent-strong);
		font-size: var(--fs-xs);
		font-weight: 650;
		vertical-align: baseline;
		white-space: nowrap;
	}
	.show:hover {
		border-color: var(--accent);
	}
</style>

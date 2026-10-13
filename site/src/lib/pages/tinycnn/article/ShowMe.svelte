<!-- An (i)-style link from the article into the tool: selects a layer, then opens Explore at the zoom view (#explore-zoom). -->
<script lang="ts">
	import type { CnnStore } from '../store.svelte';

	let { S, layer, label }: { S: CnnStore; layer: string; label: string } = $props();

	function go() {
		const ch = S.sel.layer === layer ? S.sel.ch : 0;
		S.select(layer, ch);
	}
</script>

<a class="show" href="#explore-zoom" onclick={go}>
	<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" stroke-width="1.4" /><path d="M8 7v4.2M8 4.6v.1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" /></svg>
	{label}
</a>

<style>
	.show {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		padding: 0 10px;
		min-height: 28px;
		border: 1px solid var(--line-control);
		border-radius: var(--r-pill);
		background: var(--panel);
		color: var(--ink);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
		text-decoration: none;
		vertical-align: baseline;
		white-space: nowrap;
	}
	.show:hover {
		border-color: var(--ink);
		background: var(--here-bg);
	}
</style>

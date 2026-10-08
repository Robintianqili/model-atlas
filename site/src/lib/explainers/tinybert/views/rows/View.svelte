<!-- tinybert 'rows' (meta.ts): token row + position row + segment row = the input row of one position; one feature outlined.
     On a narrow screen a strip scrolls sideways: every strip scrolls so that the outlined feature is in view. -->
<script lang="ts">
	import TensorStrip from '$lib/kit/TensorStrip.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { rowsOf } from './meta.ts';

	let P: StageProps = $props();
	const R = $derived(rowsOf(P));
	let box: HTMLDivElement | undefined = $state();
	$effect(() => {
		void R.f;
		void R.q;
		const el = box;
		if (!el) return;
		requestAnimationFrame(() => {
			for (const s of el.querySelectorAll<HTMLElement>('.strip')) {
				const c = s.querySelector<HTMLElement>('.sel');
				if (c && s.scrollWidth > s.clientWidth) s.scrollLeft = Math.max(0, s.scrollLeft + c.getBoundingClientRect().left - s.getBoundingClientRect().left - s.clientWidth / 2);
			}
		});
	});
</script>

<div class="rw" bind:this={box}>
	{#each R.parts as x (x.id)}
		{#if x.data}
			<div class="r" class:sum={x.id === 'embed'}>
				<span class="lab">{t(x.label)}</span>
				<TensorStrip data={x.data} name={x.label} shape={[R.d]} domain={R.domain} maxCells={R.d} cellSize={10} selected={R.f} />
			</div>
		{/if}
	{/each}
</div>

<style>
	.rw { display: grid; grid-template-columns: minmax(0, 1fr); gap: 10px; min-width: 0; }
	.r { display: grid; grid-template-columns: minmax(0, 1fr); gap: 2px; min-width: 0; }
	.r.sum { padding-top: 8px; border-top: 2px solid var(--rule-strong); }
	.lab { font-size: var(--fs-sm); }
</style>

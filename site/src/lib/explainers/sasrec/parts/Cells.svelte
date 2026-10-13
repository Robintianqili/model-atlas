<!-- sasrec (from two-tower): a vector as a row of cells colored on the kit's diverging scale (blue below 0, orange above), each with
     its number (or only the color when `values` is false); `names` label the cells for hover and screen readers. -->
<script lang="ts">
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3 } from '../../../atlas/core/format.ts';

	let { data, max = 0, values = true, label, names }: { data: ArrayLike<number>; max?: number; values?: boolean; label: string; names?: string[] } = $props();
	const xs = $derived(Array.from(data));
	const m = $derived(max || Math.max(1e-9, ...xs.map(Math.abs)));
	const sc = $derived(makeScale('diverging', [-m, m], themeState.resolved));
</script>

<span class="cells" role="img" aria-label="{label}: {xs.map((v, k) => (names ? `${names[k]} ` : '') + fmt3(v)).join(', ')}">
	{#each xs as v, k (k)}<span class="c" class:v={values} style:background={sc.css(v)} style:color={sc.ink(v)} title="{names ? `${names[k]}: ` : ''}{fmt3(v)}">{values ? fmt3(v) : ''}</span>{/each}
</span>

<style>
	.cells { display: inline-flex; flex-wrap: wrap; gap: 1px; max-width: 100%; }
	.c { display: inline-block; width: 9px; height: 14px; border-radius: 2px; }
	.c.v { width: auto; min-width: 38px; height: auto; padding: 1px 3px; font: 11px/1.4 var(--font-mono); text-align: center; }
</style>

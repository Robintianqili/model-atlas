<!--
  K4, promoted by the rule of two (two-tower's and sasrec's parts/Cells.svelte, with sasrec's fix: the scale is
  symmetric, ±max): a vector as a row of cells colored on the kit's diverging scale (blue below 0, orange above), each
  with its number, or only the color when `values` is false. `names` (strings or { en, zh }) label the cells for hover
  and screen readers; `max` fixes the scale (one domain for rows that are compared).

    <Cells data={vec} label={t({ en: 'the user vector', zh: '用户向量' })} names={dims} max={2} />
-->
<script lang="ts">
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { i18n } from '$lib/i18n/i18n.svelte';
	import type { I18n } from '../../../types/index.ts';
	import { fmt3 } from '../../../core/format.ts';
	import { nameText } from './stage.ts';

	let { data, max = 0, values = true, label, names, digits = 3 }: { data: ArrayLike<number>; max?: number; values?: boolean; label: string; names?: readonly (string | I18n)[]; digits?: number } = $props();
	const xs = $derived(Array.from(data));
	const m = $derived(max || Math.max(1e-9, ...xs.map(Math.abs)));
	const sc = $derived(makeScale('diverging', [-m, m], themeState.resolved));
	const nm = (k: number) => (names ? (nameText(names[k], i18n.lang) ?? '') : '');
	const num = (v: number) => fmt3(v, digits);
</script>

<span class="cells" role="img" aria-label="{label}: {xs.map((v, k) => (names ? `${nm(k)} ` : '') + num(v)).join(', ')}">
	{#each xs as v, k (k)}<span class="c" class:v={values} style:background={sc.css(v)} style:color={sc.ink(v)} title="{names ? `${nm(k)}: ` : ''}{num(v)}">{values ? num(v) : ''}</span>{/each}
</span>

<style>
	.cells {
		display: inline-flex;
		flex-wrap: wrap;
		gap: 1px;
		max-width: 100%;
	}
	.c {
		display: inline-block;
		width: 9px;
		height: 14px;
		border-radius: 2px;
	}
	.c.v {
		width: auto;
		min-width: 38px;
		height: auto;
		padding: 1px 3px;
		font: 11px/1.4 var(--font-mono);
		text-align: center;
	}
</style>

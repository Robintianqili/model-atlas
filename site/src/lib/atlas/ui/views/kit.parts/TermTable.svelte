<!--
  The terms of a weighted sum as a table: x, w and x·w per row, colored with the kit's scales, the sum in a footer
  that stays visible while the rows scroll (as TinyGPT's DotProduct). Long sums (more than `maxRows` terms) start as
  the largest contributions by |x·w| plus one line for the rest; a button shows every row.
-->
<script lang="ts">
	import Num from './Num.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3, fmtFull } from '../../../core/format.ts';
	import { t, type Text } from '$lib/i18n/i18n.svelte';

	type Role = 'in' | 'param' | 'q' | 'k' | 'v' | 'prob';
	interface Term {
		x: number;
		w: number;
		p: number;
	}
	interface Props {
		terms: readonly Term[];
		/** row label (the index of the term in words or symbols) */
		idx: (i: number) => string;
		head: { i: string; x: string; w: string; p: string };
		roles?: { x?: Role; w?: Role };
		/** scale kinds of the x and w columns (attention weights are probabilities: sequential) */
		kinds?: { x?: 'diverging' | 'sequential'; w?: 'diverging' | 'sequential' };
		sum: number;
		sumLabel?: Text;
		caption: Text;
		/** linked hover (bindable): the term index under the pointer or focus */
		hover?: number | null;
		maxRows?: number;
		top?: number;
	}
	let { terms, idx, head, roles = {}, kinds = {}, sum, sumLabel, caption, hover = $bindable(null), maxRows = 80, top = 12 }: Props = $props();

	let all = $state(false);
	const long = $derived(terms.length > maxRows);
	const order = $derived.by(() => {
		const ix = Array.from({ length: terms.length }, (_, i) => i);
		if (!long || all) return ix;
		return ix.sort((a, b) => Math.abs(terms[b].p) - Math.abs(terms[a].p)).slice(0, top);
	});
	const rest = $derived.by(() => {
		if (!long || all) return null;
		const shown = new Set(order);
		let s = 0;
		let n = 0;
		for (let i = 0; i < terms.length; i++) if (!shown.has(i)) (s += terms[i].p), n++;
		return { s, n };
	});
	const span = (k: 'x' | 'w' | 'p') => {
		let m = 0;
		for (const tm of terms) {
			const v = Math.abs(tm[k]);
			if (Number.isFinite(v) && v > m) m = v;
		}
		return m || 1;
	};
	const sx = $derived(makeScale(kinds.x ?? 'diverging', kinds.x === 'sequential' ? [0, span('x')] : [-span('x'), span('x')], themeState.resolved));
	const sw = $derived(makeScale(kinds.w ?? 'diverging', kinds.w === 'sequential' ? [0, span('w')] : [-span('w'), span('w')], themeState.resolved));
	const sp = $derived(makeScale('diverging', [-span('p'), span('p')], themeState.resolved));
</script>

<div class="tt">
	<div class="wrap" class:all use:keyScroll>
		<table class="num">
			<caption class="visually-hidden">{t(caption)}</caption>
			<thead>
				<tr>
					<th scope="col">{head.i}</th>
					<th scope="col" class="r-{roles.x ?? 'in'}">{head.x}</th>
					<th scope="col" class="r-{roles.w ?? 'param'}">{head.w}</th>
					<th scope="col">{head.p}</th>
				</tr>
			</thead>
			<tbody>
				{#each order as i (i)}
					{@const tm = terms[i]}
					<tr class:on={hover === i} onpointerenter={() => (hover = i)} onpointerleave={() => (hover = null)}>
						<th scope="row" class="c">{idx(i)}</th>
						<td style="background: {sx.css(tm.x)}; color: {sx.ink(tm.x)}" title={fmtFull(tm.x)}>{fmt3(tm.x)}</td>
						<td style="background: {sw.css(tm.w)}; color: {sw.ink(tm.w)}" title={fmtFull(tm.w)}>{fmt3(tm.w)}</td>
						<td style="background: {sp.css(tm.p)}; color: {sp.ink(tm.p)}" title={fmtFull(tm.p)}>{fmt3(tm.p)}</td>
					</tr>
				{/each}
				{#if rest}
					<tr class="rest">
						<th scope="row" colspan="3">{t({ en: `the other ${rest.n} products`, zh: `其余 ${rest.n} 个乘积` })}</th>
						<td><Num value={rest.s} /></td>
					</tr>
				{/if}
			</tbody>
			<tfoot>
				<tr>
					<th scope="row" colspan="3">{sumLabel ? t(sumLabel) : t({ en: `sum of ${terms.length}`, zh: `${terms.length} 项之和` })} Σ</th>
					<td><Num value={sum} /></td>
				</tr>
			</tfoot>
		</table>
	</div>
	{#if long}
		<button type="button" class="btn ghost more" aria-expanded={all} onclick={() => (all = !all)}>
			{all
				? t({ en: `Show the ${top} largest only`, zh: `只看最大的 ${top} 个` })
				: t({ en: `Show all ${terms.length} rows`, zh: `显示全部 ${terms.length} 行` })}
		</button>
	{/if}
</div>

<style>
	.tt {
		display: grid;
		gap: 6px;
		justify-items: start;
		min-width: 0;
	}
	.wrap {
		max-height: 340px;
		max-width: 100%;
		overflow: auto;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
	}
	table {
		border-collapse: collapse;
		font-size: 12px;
	}
	th,
	td {
		padding: 2px 8px;
		text-align: right;
		border-bottom: 1px solid var(--surface);
		white-space: nowrap;
	}
	thead th {
		position: sticky;
		top: 0;
		z-index: 1;
		background: var(--surface-2);
		font-weight: 600;
	}
	.r-in {
		box-shadow: inset 0 -2px 0 var(--ink-2);
	}
	.r-param {
		box-shadow: inset 0 -2px 0 var(--param);
	}
	.r-q {
		box-shadow: inset 0 -2px 0 var(--q);
	}
	.r-k {
		box-shadow: inset 0 -2px 0 var(--k);
	}
	.r-v {
		box-shadow: inset 0 -2px 0 var(--v);
	}
	.r-prob {
		box-shadow: inset 0 -2px 0 var(--seq-hi);
	}
	.c {
		color: var(--muted);
		font-weight: 400;
		text-align: left;
	}
	tr.on td,
	tr.on th {
		outline: 2px solid var(--highlight);
		outline-offset: -2px;
	}
	.rest th {
		font-weight: 400;
		color: var(--muted);
		text-align: left;
	}
	tfoot th,
	tfoot td {
		position: sticky;
		bottom: 0;
		border-top: 1.5px solid var(--ink-2);
		background: var(--surface);
	}
	tfoot th {
		text-align: left;
	}
	.more {
		font-size: var(--fs-xs);
	}
</style>

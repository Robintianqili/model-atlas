<!--
  zoom.softmax: one share of a softmax row. Modeled on TinyCNN's SoftmaxDetail and TinyGPT's SoftmaxRow: every entry of
  the row (classes, tokens, or the keys of an attention row) with its score x, e^(x − m) and its share, largest first,
  the unit's own entry outlined; masked entries (−∞) get exactly 0. Then the op's steps (largest score m, the
  exponential, the row sum, the share), the formula and the stored-value check. Click a row to pick that entry.
  K4: the entries are named by the classes or the vocabulary only when the softmax runs over a class, vocab or feature
  axis (a softmax over 4 experts, axis k, no longer took the characters of ids 0-3); props.labels names them otherwise
  (strings or { en, zh }), and props.axisWords / axisItems name an axis such as k.
-->
<script lang="ts">
	import Num from '../kit.parts/Num.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import Tex from '$lib/kit/Tex.svelte';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3, fmtPct } from '../../../core/format.ts';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../types/view.ts';
	import UnitHead from '../kit.parts/UnitHead.svelte';
	import OpFormula from '../kit.parts/OpFormula.svelte';
	import Notice from '../kit.parts/Notice.svelte';
	import { axesOfId, axisNames, keyTexts, layerLabel, nameText, outLabels, resolveUnit, tokenIds, tokenTexts, unitWhere } from '../kit.parts/stage.ts';

	let P: StageProps = $props();

	let pick = $derived.by<number[] | null>(() => {
		void P.at;
		void P.props.layer;
		return null;
	});
	let all = $state(false);
	const R = $derived(resolveUnit(P, pick));
	const e = $derived(R.e);
	const id = $derived(e?.layer ?? R.id ?? '');
	const out = $derived(R.trace && id ? R.trace.t[id] : null);
	const axes = $derived(out ? axesOfId(R.model, id, out.shape) : []);
	const ids = $derived(tokenIds(R.trace, R.model));
	const T = $derived(out ? (axes[0] === 't' ? out.shape[0] : axes[1] === 'tq' ? out.shape[1] : 0) : 0);
	const tokens = $derived(T ? tokenTexts(P, T, ids, i18n.lang) : []);
	const labels = $derived(outLabels(P, R.model));
	const where = $derived(e ? unitWhere(e.unit, axes, tokens, labels, axisNames(P)) : null);
	const ax = $derived(axes.length - 1);
	const kAt = $derived(e ? (e.values.k ?? e.unit[ax]) : 0);
	const lastAxis = $derived(axes[ax]);
	const own = $derived(Array.isArray(P.props.labels) ? (P.props.labels as unknown[]) : null);
	const names = $derived(axisNames(P));
	function itemName(j: number): string {
		if (own) return nameText(own[j], i18n.lang) ?? `#${j}`;
		if (lastAxis === 'tk') return `${t({ en: 'key', zh: '键' })} ${j}${keyNames[j] ? ` “${keyNames[j]}”` : ''}`;
		const item = names.items?.[lastAxis]?.[j];
		if (item !== undefined) return nameText(item, i18n.lang) ?? `#${j}`;
		const word = names.words?.[lastAxis];
		if (word) return `${t(word)} ${j}`;
		if (labels && (lastAxis === undefined || lastAxis === 'class' || lastAxis === 'vocab' || lastAxis === 'd' || lastAxis === 'f')) {
			const l = labels(j);
			return typeof l === 'string' ? l : t(l);
		}
		return `#${j}`;
	}
	const terms = $derived(e?.terms ?? []);
	// K4: the keys of an attention row by their own names (props.keys for cross-attention), not the queries' tokens
	const keyNames = $derived(lastAxis === 'tk' ? keyTexts(P, terms.length, ids, i18n.lang).map((x, j) => (x === String(j) ? '' : x)) : []);
	const order = $derived.by(() => {
		const ix = Array.from({ length: terms.length }, (_, i) => i).sort((a, b) => terms[b].p - terms[a].p || a - b);
		if (all || ix.length <= 12) return ix;
		const top = ix.slice(0, 10);
		if (!top.includes(kAt)) top.push(kAt);
		return top;
	});
	const rest = $derived.by(() => {
		if (all || terms.length <= 12) return null;
		const shown = new Set(order);
		let p = 0;
		let n = 0;
		for (let i = 0; i < terms.length; i++) if (!shown.has(i)) (p += terms[i].p), n++;
		return { p, n };
	});
	const maxP = $derived(Math.max(1e-12, ...terms.map((x) => x.p)));
	const sc = $derived(makeScale('sequential', [0, maxP], themeState.resolved));
	const masked = $derived(terms.filter((x) => x.x === -Infinity).length);
	function setK(j: number) {
		if (!e) return;
		const u = e.unit.slice();
		u[ax] = j;
		pick = u;
	}
	$effect(() => {
		if (P.autoplay && e) P.onend();
	});
</script>

{#if !e}
	<Notice error={R.error} />
{:else}
	<div class="sm">
		<UnitHead layer={layerLabel(R.model, id)} op={R.op?.def.label} where={where ?? undefined} value={e.result} />
		<p class="sub">
			{t({
				en: `The row has ${terms.length} scores${masked ? `; ${masked} are masked (−∞) and get exactly 0` : ''}. Each share is e^(score − largest) divided by the sum over the row.`,
				zh: `这一行有 ${terms.length} 个分数${masked ? `；其中 ${masked} 个被遮住（−∞），份额正好是 0` : ''}。每个份额等于 e^(分数 − 最大值) 除以整行的总和。`
			})}
		</p>
		<div class="wrap" use:keyScroll>
			<table class="num">
				<caption class="visually-hidden">{t({ en: 'Scores, exponentials and shares of the row', zh: '这一行的分数、指数和份额' })}</caption>
				<thead>
					<tr>
						<th scope="col" class="nm">{lastAxis === 'tk' ? t({ en: 'key', zh: '键' }) : t({ en: 'item', zh: '项' })}</th>
						<th scope="col">{t({ en: 'score x', zh: '分数 x' })}</th>
						<th scope="col">e<sup>x−m</sup></th>
						<th scope="col" class="sh">{t({ en: 'share', zh: '份额' })}</th>
					</tr>
				</thead>
				<tbody>
					{#each order as j (j)}
						{@const tm = terms[j]}
						<tr class:on={j === kAt}>
							<th scope="row" class="nm"><button type="button" class="lnk" aria-pressed={j === kAt} onclick={() => setK(j)}>{itemName(j)}</button></th>
							<td>{tm.x === -Infinity ? '−∞' : fmt3(tm.x)}</td>
							<td>{fmt3(tm.w)}</td>
							<td class="sh">
								<span class="cellbar"><span class="track"><span class="bar" style="width: {(tm.p / maxP) * 100}%; background: {sc.css(tm.p)}"></span></span><span class="pv">{fmtPct(tm.p)}</span></span>
							</td>
						</tr>
					{/each}
					{#if rest}
						<tr class="rest">
							<th scope="row" colspan="3" class="nm">{t({ en: `the other ${rest.n}`, zh: `其余 ${rest.n} 项` })}</th>
							<td class="sh"><span class="pv">{fmtPct(rest.p)}</span></td>
						</tr>
					{/if}
				</tbody>
				<tfoot>
					<tr>
						<th scope="row" class="nm">Σ</th>
						<td></td>
						<td><Num value={e.values.sum} /></td>
						<td class="sh"><span class="pv">100%</span></td>
					</tr>
				</tfoot>
			</table>
		</div>
		{#if terms.length > 12}
			<button type="button" class="btn ghost more" aria-expanded={all} onclick={() => (all = !all)}>
				{all ? t({ en: 'Show the 10 largest only', zh: '只看最大的 10 个' }) : t({ en: `Show all ${terms.length}`, zh: `显示全部 ${terms.length} 项` })}
			</button>
		{/if}
		{#if e.steps?.length}
			<ol class="steps" aria-label={t({ en: 'The steps for this share', zh: '这个份额的计算步骤' })}>
				{#each e.steps as s, i (i)}
					<li class:last={i === e.steps.length - 1}>
						<span class="tx">{#if s.tex}<Tex tex={s.tex} />{/if}</span>
						<span class="lb">{t(s.label)}</span>
						<span class="v num"><Num value={s.value} /></span>
					</li>
				{/each}
			</ol>
		{/if}
		<OpFormula tex={R.op?.def.formula} {e} caption={where ? { en: `${layerLabel(R.model, id).en}, ${where.en}`, zh: `${layerLabel(R.model, id).zh ?? id}，${where.zh}` } : undefined} provenance={P.provenance} digits={P.props.digits as number | undefined} />
	</div>
{/if}

<style>
	.sm {
		display: grid;
		gap: 12px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.sub {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		max-width: var(--measure);
	}
	.wrap {
		max-width: 100%;
		max-height: 420px;
		overflow: auto;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		justify-self: start;
	}
	table {
		border-collapse: collapse;
		font-size: 12px;
	}
	th,
	td {
		padding: 3px 8px;
		text-align: right;
		white-space: nowrap;
		border-bottom: 1px solid var(--surface-2);
	}
	thead th {
		position: sticky;
		top: 0;
		z-index: 1;
		background: var(--surface-2);
		font-weight: 600;
	}
	.nm {
		text-align: left;
		font-weight: 400;
	}
	.lnk {
		border: 0;
		background: none;
		padding: 0;
		font: inherit;
		color: var(--ink);
		cursor: pointer;
		white-space: pre;
		text-align: left;
	}
	tbody tr {
		cursor: pointer;
	}
	tr.on td,
	tr.on th {
		background: var(--accent-soft);
	}
	tr.on .lnk {
		font-weight: 700;
	}
	.sh {
		text-align: left;
	}
	.cellbar {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.track {
		display: inline-block;
		width: 80px;
		height: 10px;
		background: var(--surface-2);
		border-radius: 2px;
		overflow: hidden;
	}
	.bar {
		display: block;
		height: 100%;
		border-radius: 2px;
	}
	.pv {
		white-space: nowrap;
		min-width: 3.5em;
		text-align: right;
	}
	thead .sh,
	tfoot .sh {
		text-align: left;
	}
	.rest th,
	.rest td {
		color: var(--muted);
	}
	tfoot th,
	tfoot td {
		position: sticky;
		bottom: 0;
		border-top: 1.5px solid var(--ink-2);
		background: var(--surface);
	}
	.more {
		justify-self: start;
		font-size: var(--fs-xs);
	}
	.steps {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 2px;
		max-width: 560px;
	}
	.steps li {
		display: grid;
		grid-template-columns: 6.5em minmax(0, 1fr) auto;
		gap: 10px;
		align-items: baseline;
		padding: 3px 8px;
		border-radius: 3px;
		background: var(--surface-2);
		font-size: var(--fs-sm);
	}
	.steps li.last {
		box-shadow: inset 0 0 0 1.5px var(--accent);
		font-weight: 650;
	}
	.lb {
		color: var(--ink-2);
	}
	.v {
		text-align: right;
	}
</style>

<!--
  Softmax zoom: logits -> exp(logit - max) -> divide by the sum (model.softmaxDetail, live),
  as a 3-step player and a table of the live numbers.
-->
<script lang="ts">
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { sym } from '$lib/kit/tex';
	import { categorical, makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3, fmtPct } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import { onMount } from 'svelte';
	import type { CnnStore, SelView } from './store.svelte';

	type Part = 'player' | 'table' | 'formula' | 'note';
	interface Props {
		S: CnnStore;
		/** whose selection to show (default: the page store) */
		view?: SelView;
		/** blocks to show (default: all) */
		parts?: Part[];
		/** play logits → exp → normalize from the first step when shown */
		autoplay?: boolean;
		/** ms per step */
		interval?: number;
		onend?: () => void;
	}
	let { S, view, parts = ['player', 'table', 'formula', 'note'], autoplay = false, interval = 1400, onend }: Props = $props();
	const V = $derived(view ?? S);
	const show = (p: Part) => parts.includes(p);

	const sd = $derived(S.model && S.trace ? S.model.softmaxDetail(S.trace) : null);
	/** a softmax share is never exactly 100%: show ">99.9%" rather than a rounded "100.0%" */
	const share = (p: number, digits = 1) => (p >= 1 - 0.5 * 10 ** -(digits + 2) ? `>${(100 - 10 ** -digits).toFixed(digits)}%` : fmtPct(p, digits));
	const k = $derived(V.sel.layer === 'softmax' || V.sel.layer === 'dense' ? V.sel.ch : sd?.top ?? 0);
	let step = $state(2);
	onMount(() => {
		if (autoplay) step = 0;
	});
	const zMax = $derived(sd ? Math.max(...Array.from(sd.logits, Math.abs)) || 1 : 1);
	const div = $derived(makeScale('diverging', [-zMax, zMax], themeState.resolved));
	const seq = $derived(makeScale('sequential', [0, 1], themeState.resolved));
	const labels = [
		{ en: '1 · Logits: one raw score per class (any number, negative allowed)', zh: '1 · Logits：每个类别一个原始分数（可以是任何数，也可以是负数）' },
		{ en: '2 · exp(logit − max): all positive; the largest becomes exactly 1', zh: '2 · exp(logit − 最大值)：全都变成正数；最大的那个恰好是 1' },
		{ en: '3 · Divide by the sum: the 10 numbers now add up to 1', zh: '3 · 除以总和：这 10 个数加起来正好等于 1' }
	];
	const rows = $derived(
		sd
			? Array.from(sd.logits, (z, i) => ({ i, z, d: z - sd.max, e: sd.exps[i], p: sd.probs[i] })).sort((a, b) => b.z - a.z)
			: []
	);
</script>

{#if sd}
	{#if show('player')}
	<StepPlayer count={3} bind:step {labels} {interval} {autoplay} {onend} speeds={[0.5, 1]} label={t({ en: 'From scores to probabilities', zh: '从分数到概率' })} position={(i, n) => t({ en: `Part ${i} of ${n}`, zh: `第 ${i} / ${n} 部分` })}>
		{#snippet children(s)}
			<div class="bars">
				{#each rows as r (r.i)}
					{@const v = s === 0 ? r.z : s === 1 ? r.e : r.p}
					<button type="button" class="bar-row" class:on={r.i === k} onclick={() => V.select('softmax', r.i)}>
						<span class="cn"><span class="sw" style="background: {categorical(r.i, themeState.resolved)}"></span>{t(S.spec.classes[r.i])}</span>
						{#if s === 0}
							<span class="track div"><span class="b {v < 0 ? 'neg' : 'pos'}" style="--w: {(Math.abs(v) / zMax) * 50}%; background: {div.css(v)}"></span></span>
						{:else}
							<span class="track"><span class="b pos0" style="--w: {Math.max(0.4, v * 100)}%; background: {seq.css(0.25 + 0.75 * v)}"></span></span>
						{/if}
						<span class="v num">{s === 2 ? share(v) : fmt3(v)}</span>
					</button>
				{/each}
			</div>
		{/snippet}
	</StepPlayer>
	{/if}

	{#if show('table')}
	<div class="tbl-wrap" use:keyScroll>
		<table class="num">
			<thead>
				<tr>
					<th>{t({ en: 'class', zh: '类别' })}</th>
					<th class:hl={step === 0}>z</th>
					<th class:hl={step === 1}>z − m</th>
					<th class:hl={step === 1}>e<sup>z − m</sup></th>
					<th class:hl={step === 2}>p</th>
				</tr>
			</thead>
			<tbody>
				{#each rows as r (r.i)}
					<tr class:on={r.i === k}>
						<td><span class="cn"><span class="sw" style="background: {categorical(r.i, themeState.resolved)}"></span>{t(S.spec.classes[r.i])}</span></td>
						<td>{fmt3(r.z)}</td>
						<td>{fmt3(r.d)}</td>
						<td>{fmt3(r.e)}</td>
						<td>{share(r.p, 2)}</td>
					</tr>
				{/each}
				<tr class="sum">
					<td>Σ</td>
					<td></td>
					<td></td>
					<td>{fmt3(sd.sumExp)}</td>
					<td>100%</td>
				</tr>
			</tbody>
		</table>
	</div>
	{/if}
	{#if show('formula')}
	<FormulaCard
		tex={`\\begin{aligned} ${sym('prob', 'p_k')} &= \\frac{e^{${sym('out', 'z_k')} - m}}{\\sum_{j} e^{${sym('out', 'z_j')} - m}} = \\frac{{{e}}}{{{s}}} = {{p}} \\\\ m &= \\max_j ${sym('out', 'z_j')} = {{m}} \\end{aligned}`}
		values={{ e: sd.exps[k], s: sd.sumExp, p: sd.probs[k], m: sd.max }}
		provenance="live"
		caption={t({ en: `Probability of “${t(S.spec.classes[k])}”`, zh: `“${t(S.spec.classes[k])}”的概率` })}
	/>
	{/if}
	{#if show('note')}
	<p class="note">
		{t({
			en: 'Subtracting the max changes nothing in the result (it cancels in the fraction) but keeps exp from overflowing. A logit only matters relative to the others: a gap of about 4.6 makes one class 100 times more likely than another.',
			zh: '减去最大值不会改变结果（分子分母同时约掉），只是防止 exp 溢出。logit 只有相对大小才有意义：相差约 4.6，就意味着一个类别的可能性是另一个的 100 倍。'
		})}
	</p>
	{/if}
{/if}

<style>
	.bars {
		display: grid;
		gap: 2px;
		max-width: 560px;
	}
	.bar-row {
		display: grid;
		grid-template-columns: 7.5em 1fr 4.5em;
		align-items: center;
		gap: 8px;
		padding: 2px 6px;
		border: 1px solid transparent;
		border-radius: var(--r1);
		background: none;
		font-size: var(--fs-sm);
		text-align: left;
	}
	.bar-row.on {
		border-color: var(--accent);
		background: var(--accent-soft);
	}
	.cn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		white-space: nowrap;
	}
	.sw {
		width: 9px;
		height: 9px;
		border-radius: 2px;
		flex: 0 0 auto;
	}
	.track {
		position: relative;
		height: 12px;
		background: var(--surface-2);
		border-radius: 2px;
	}
	.track.div {
		background: linear-gradient(90deg, var(--surface-2) calc(50% - 0.5px), var(--rule-strong) calc(50% - 0.5px), var(--rule-strong) calc(50% + 0.5px), var(--surface-2) calc(50% + 0.5px));
	}
	.b {
		position: absolute;
		top: 1px;
		bottom: 1px;
		width: var(--w);
		border-radius: 2px;
		transition: width var(--dur-3) var(--ease);
	}
	.b.pos {
		left: 50%;
	}
	.b.neg {
		right: 50%;
	}
	.b.pos0 {
		left: 0;
	}
	.v {
		text-align: right;
		font-size: var(--fs-xs);
	}
	.tbl-wrap {
		overflow-x: auto;
		margin: 12px 0;
	}
	table {
		border-collapse: collapse;
		font-size: var(--fs-xs);
		min-width: 420px;
	}
	th,
	td {
		padding: 3px 10px;
		text-align: right;
		border-bottom: 1px solid var(--rule);
	}
	th:first-child,
	td:first-child {
		text-align: left;
		font-family: var(--font-body);
	}
	th.hl {
		background: var(--accent-soft);
	}
	tr.on td {
		background: var(--accent-soft);
		font-weight: 700;
	}
	tr.sum td {
		border-bottom: 0;
		font-weight: 700;
	}
	.note {
		margin: 8px 0 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		max-width: var(--measure);
	}
</style>

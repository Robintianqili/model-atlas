<!--
  zoom.normalize: one output number of a normalization (layernorm, and local ops such as l2norm). Modeled on TinyGPT's
  LayerNorm stage: the row it normalizes as bars (input x with its mean and spread, the normalized x̂ with mean 0 and
  spread 1, the output y after the learned scale and shift), all on one scale, the selected feature outlined; then the
  op's own steps (mean, variance, spread, x̂, ·γ, +β) with their values, the formula, and the stored-value check.
  Click, tap or arrow keys on any row pick another feature of the same row.
  K1: an L2 normalization (explain values x, len; no mean) is drawn as what it is: the row x, its length ‖x‖, and the
  row divided by that length, whose length is 1 (it used to show LayerNorm's mean/spread bands and "μ = 0, σ = 1").
-->
<script lang="ts">
	import Tex from '$lib/kit/Tex.svelte';
	import Num from '../kit.parts/Num.svelte';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../types/view.ts';
	import UnitHead from '../kit.parts/UnitHead.svelte';
	import OpFormula from '../kit.parts/OpFormula.svelte';
	import RowBars from '../kit.parts/RowBars.svelte';
	import Notice from '../kit.parts/Notice.svelte';
	import { axesOfId, axisNames, layerLabel, maxAbs, resolveUnit, tensorOf, tokenIds, tokenTexts, unitWhere } from '../kit.parts/stage.ts';

	let P: StageProps = $props();

	let pick = $derived.by<number[] | null>(() => {
		void P.at;
		void P.props.layer;
		return null;
	});
	const R = $derived(resolveUnit(P, pick));
	const e = $derived(R.e);
	const id = $derived(e?.layer ?? R.id ?? '');
	const out = $derived(R.trace && id ? R.trace.t[id] : null);
	const axes = $derived(out ? axesOfId(R.model, id, out.shape) : []);
	const T = $derived(out && axes[0] === 't' ? out.shape[0] : 0);
	const tokens = $derived(T ? tokenTexts(P, T, tokenIds(R.trace, R.model), i18n.lang) : []);
	const where = $derived(e ? unitWhere(e.unit, axes, tokens, undefined, axisNames(P)) : null);
	const cells = $derived(e?.reads[0]?.cells ?? []);
	const xT = $derived(e && R.trace ? tensorOf(R.trace, e.reads[0]?.tensor ?? '') : null);
	const rows = $derived.by(() => {
		if (!e || !xT || !out || !cells.length) return null;
		const x = Float64Array.from(cells, (c) => xT.data[c]);
		const y = Float64Array.from(cells, (c) => out.data[c]);
		const mean = e.values.mean ?? x.reduce((a, b) => a + b, 0) / x.length;
		const std = e.values.std ?? Math.sqrt(x.reduce((a, b) => a + (b - mean) ** 2, 0) / x.length);
		const xh = Float64Array.from(x, (v) => (v - mean) / std);
		const ym = y.reduce((a, b) => a + b, 0) / y.length;
		const ys = Math.sqrt(y.reduce((a, b) => a + (b - ym) ** 2, 0) / y.length);
		const sd = Math.sqrt(e.values.var ?? std * std);
		return { x, y, xh, mean, sd, ym, ys, max: maxAbs(1, x, xh, y) };
	});
	/** an L2 normalization: y = x / ‖x‖ (explain values len and x, no mean) */
	const l2 = $derived(!!e && e.values.len !== undefined && e.values.mean === undefined);
	const l2rows = $derived.by(() => {
		if (!l2 || !e || !xT || !out || !cells.length) return null;
		const x = Float64Array.from(cells, (c) => xT.data[c]);
		const y = Float64Array.from(cells, (c) => out.data[c]);
		const ylen = Math.sqrt(y.reduce((a, b) => a + b * b, 0));
		return { x, y, len: e.values.len, ylen, max: maxAbs(1e-9, x), ymax: maxAbs(1e-9, y) };
	});
	const sel = $derived(e ? e.unit[e.unit.length - 1] : 0);
	const lead = $derived(e ? e.unit.slice(0, -1) : []);
	const rowName = $derived(
		T && e ? { en: tokens[e.unit[0]] ? `token “${tokens[e.unit[0]]}”` : `position ${e.unit[0]}`, zh: tokens[e.unit[0]] ? `词元“${tokens[e.unit[0]]}”` : `位置 ${e.unit[0]}` } : { en: 'this row', zh: '这一行' }
	);
	$effect(() => {
		if (P.autoplay && e) P.onend();
	});
</script>

{#if !e}
	<Notice error={R.error} />
{:else}
	<div class="nz">
		<UnitHead layer={layerLabel(R.model, id)} op={R.op?.def.label} where={where ?? undefined} value={e.result} />
		{#if l2rows}
			<p class="sub">
				{t({ en: `The ${cells.length} numbers of ${rowName.en}, then the same numbers divided by the row's length ‖x‖ (the square root of the sum of their squares).`, zh: `${rowName.zh ?? rowName.en}的 ${cells.length} 个数，然后是同样的数除以这一行的长度 ‖x‖（各数平方和的平方根）。` })}
			</p>
			<div class="rows">
				<div class="r">
					<RowBars data={l2rows.x} max={l2rows.max} selected={sel} name="x" label={{ en: 'input row x', zh: '输入行 x' }} onselect={(i) => (pick = [...lead, i])} />
					<p class="st num">‖x‖ = <Num value={l2rows.len} /></p>
				</div>
				<div class="r">
					<RowBars data={l2rows.y} max={l2rows.ymax} selected={sel} name="y" label={{ en: 'the row divided by its length', zh: '除以长度后的行' }} onselect={(i) => (pick = [...lead, i])} />
					<p class="st num">‖y‖ = <Num value={l2rows.ylen} /> <span class="muted">· {t({ en: 'every row now has length 1: only its direction is left', zh: '现在每一行的长度都是 1：只剩下方向' })}</span></p>
				</div>
			</div>
		{:else if rows}
			<p class="sub">
				{t({ en: `The ${cells.length} numbers of ${rowName.en}, three times on one scale. The band is the mean ± the spread.`, zh: `${rowName.zh ?? rowName.en}的 ${cells.length} 个数，在同一个刻度上画三次。色带表示平均值 ± 离散程度。` })}
			</p>
			<div class="rows">
				<div class="r">
					<RowBars data={rows.x} max={rows.max} band={{ mean: rows.mean, sd: rows.sd }} selected={sel} name="x" label={{ en: 'input row x', zh: '输入行 x' }} onselect={(i) => (pick = [...lead, i])} />
					<p class="st num">μ = <Num value={rows.mean} />, σ = <Num value={rows.sd} /></p>
				</div>
				<div class="r">
					<RowBars data={rows.xh} max={rows.max} band={{ mean: 0, sd: 1 }} selected={sel} name="x̂" label={{ en: 'normalized row', zh: '归一化后的行' }} onselect={(i) => (pick = [...lead, i])} />
					<p class="st num">μ = 0, σ = 1 <span class="muted">· {t({ en: 'subtract the mean, divide by the spread', zh: '减去平均值，除以离散程度' })}</span></p>
				</div>
				<div class="r">
					<RowBars data={rows.y} max={rows.max} band={{ mean: rows.ym, sd: rows.ys }} selected={sel} name="y" label={{ en: 'output row y', zh: '输出行 y' }} onselect={(i) => (pick = [...lead, i])} />
					<p class="st num">μ = <Num value={rows.ym} />, σ = <Num value={rows.ys} /> <span class="muted">· {t({ en: 'a learned scale γ and shift β per feature', zh: '每个特征有学来的缩放 γ 和偏移 β' })}</span></p>
				</div>
			</div>
		{/if}
		{#if e.steps?.length}
			<ol class="steps" aria-label={t({ en: 'The steps for this feature', zh: '这个特征的计算步骤' })}>
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
	.nz {
		display: grid;
		gap: 12px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.sub {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.rows {
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	.st {
		margin: -2px 0 0;
		padding-left: calc(4.5em + 8px);
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	@media (max-width: 480px) {
		.st {
			padding-left: 0;
		}
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
		min-width: 0;
	}
	.v {
		text-align: right;
	}
	@media (max-width: 480px) {
		.steps li {
			grid-template-columns: 5em minmax(0, 1fr) auto;
			gap: 6px;
		}
	}
</style>

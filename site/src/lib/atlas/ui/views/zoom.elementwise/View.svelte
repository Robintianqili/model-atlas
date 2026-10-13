<!--
  zoom.elementwise: one number in, one number out (relu, tanh, sigmoid, silu, gelu and local activations), or a cell by
  cell sum (add). Modeled on TinyCNN's ReluDetail and TinyGPT's GeluPlot:
    - activations: the function as a curve, computed by the op's own forward() on a grid of x, with every input of
      the unit's channel (or token row) as a point on it and the unit circled; how many inputs fall in each part;
    - where the unit sits: input and output maps ([C,H,W]) or the input and output rows as bars ([T,D], [N]);
    - add: the two inputs and the sum as rows of bars on one scale, and a + b = y as bars;
    - the op's formula with live values, and the forward pass's stored value next to the recomputed one.
-->
<script lang="ts">
	import Num from '../kit.parts/Num.svelte';
	import { fmt3 } from '../../../core/format.ts';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../types/view.ts';
	import UnitHead from '../kit.parts/UnitHead.svelte';
	import OpFormula from '../kit.parts/OpFormula.svelte';
	import SpatialMaps from '../kit.parts/SpatialMaps.svelte';
	import CurvePlot from '../kit.parts/CurvePlot.svelte';
	import RowBars from '../kit.parts/RowBars.svelte';
	import SumBars from '../kit.parts/SumBars.svelte';
	import Notice from '../kit.parts/Notice.svelte';
	import { axesOfId, axisNames, domainOf, layerLabel, maxAbs, opCurve, outLabels, resolveUnit, tensorOf, tokenIds, tokenTexts, unitSlice, unitWhere } from '../kit.parts/stage.ts';

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
	const where = $derived(e ? unitWhere(e.unit, axes, tokens, outLabels(P, R.model), axisNames(P)) : null);
	const isAdd = $derived(!!e && !e.curve && (e.terms?.length ?? 0) === 2);
	const xT = $derived(e && R.trace ? tensorOf(R.trace, e.reads[0]?.tensor ?? '') : null);
	const bT = $derived(e && isAdd && R.trace ? tensorOf(R.trace, e.reads[1]?.tensor ?? '') : null);
	const spatial = $derived(axes.join() === 'c,y,x' && !!xT && xT.shape.length === 3);
	const sl = $derived(e && out ? unitSlice(out.shape, e.unit) : null);
	const xs = $derived(sl && xT && xT.data.length === out?.data.length ? xT.data.subarray(sl.from, sl.from + sl.n) : null);
	const ys = $derived(sl && out ? out.data.subarray(sl.from, sl.from + sl.n) : null);
	const opName = $derived(R.op?.def.op ?? R.layer?.op ?? '');
	const fnName = $derived(e?.curve?.fn ?? opName);
	const curve = $derived.by(() => {
		if (!e?.curve || !xs) return null;
		let lo = Math.min(-3, e.curve.x);
		let hi = Math.max(3, e.curve.x);
		for (let i = 0; i < xs.length; i++) if (Number.isFinite(xs[i])) (lo = Math.min(lo, xs[i])), (hi = Math.max(hi, xs[i]));
		const padX = (hi - lo) * 0.04;
		return opCurve(R, lo - padX, hi + padX);
	});
	const refCurve = $derived.by(() => {
		if (!curve || !/gelu|silu/.test(fnName)) return null;
		return { x: curve.x, y: Float64Array.from(curve.x, (v) => (v > 0 ? v : 0)), label: 'ReLU' };
	});
	const counts = $derived.by(() => {
		if (!xs || !ys) return null;
		let neg = 0;
		let zero = 0;
		for (let i = 0; i < xs.length; i++) {
			if (xs[i] < 0) neg++;
			if (ys[i] === 0) zero++;
		}
		return { neg, zero, n: xs.length };
	});
	const sliceName = $derived(
		spatial && e
			? { en: `channel ${e.unit[0]}`, zh: `通道 ${e.unit[0]}` }
			: T && e
				? { en: tokens[e.unit[0]] ? `“${tokens[e.unit[0]]}”` : `position ${e.unit[0]}`, zh: tokens[e.unit[0]] ? `“${tokens[e.unit[0]]}”` : `位置 ${e.unit[0]}` }
				: { en: 'the layer', zh: '这一层' }
	);
	const outDom = $derived(out ? domainOf('diverging', out.data) : ([-1, 1] as [number, number]));
	const inDom = $derived(xT ? domainOf('diverging', xT.data) : ([-1, 1] as [number, number]));
	const rowMax = $derived(maxAbs(1, xs, ys, bT && sl ? bT.data.subarray(sl.from, sl.from + sl.n) : null));
	$effect(() => {
		if (P.autoplay && e) P.onend();
	});
</script>

{#if !e}
	<Notice error={R.error} />
{:else}
	<div class="ew">
		<UnitHead layer={layerLabel(R.model, id)} op={R.op?.def.label} where={where ?? undefined} value={e.result} />

		{#if e.curve && curve && xs && ys && sl}
			<div class="two">
				<CurvePlot
					cx={curve.x}
					cy={curve.y}
					px={xs}
					py={ys}
					sel={sl.at}
					name={fnName.replace('_tanh', '')}
					label={{ en: `${fnName} on the ${sl.n} inputs of ${sliceName.en}`, zh: `${sliceName.zh ?? sliceName.en}的 ${sl.n} 个输入上的 ${fnName}` }}
					ref={refCurve}
					onpick={(i) => (pick = sl.unitAt(i))}
				/>
				<div class="side">
					<p class="xy num">
						x = <Num value={e.curve.x} signed /> <span class="arr">→</span> {fnName.replace('_tanh', '')}(x) = <strong><Num value={e.curve.y} signed /></strong>
					</p>
					{#if counts}
						<p class="cnt">
							{t({
								en: `Each dot is one of the ${counts.n} numbers of ${sliceName.en}: x is the input, the height is the output. ${counts.neg} of them are negative${counts.zero ? `; ${counts.zero} outputs are exactly 0` : ''}.`,
								zh: `每个点是${sliceName.zh ?? sliceName.en}的 ${counts.n} 个数之一：横坐标是输入，高度是输出。其中 ${counts.neg} 个是负数${counts.zero ? `；有 ${counts.zero} 个输出正好是 0` : ''}。`
							})}
						</p>
					{/if}
					<p class="muted hint">{t({ en: 'Hover, tap or use the arrow keys on the plot to pick another number.', zh: '在图上悬停、点按或用方向键，选另一个数。' })}</p>
				</div>
			</div>
		{/if}

		{#if isAdd && sl && xs && ys}
			{@const bs = bT ? bT.data.subarray(sl.from, sl.from + sl.n) : null}
			<div class="rows">
				<RowBars data={xs} max={rowMax} selected={sl.at} name="a" label={layerLabel(R.model, e.reads[0].tensor)} onselect={(i) => (pick = sl.unitAt(i))} />
				{#if bs}<RowBars data={bs} max={rowMax} selected={sl.at} name="b" label={layerLabel(R.model, e.reads[1].tensor)} onselect={(i) => (pick = sl.unitAt(i))} />{/if}
				<RowBars data={ys} max={rowMax} selected={sl.at} name="a + b" label={layerLabel(R.model, id)} onselect={(i) => (pick = sl.unitAt(i))} />
			</div>
			<p class="muted small">
				{t({
					en: `a: ${layerLabel(R.model, e.reads[0].tensor).en} · b: ${layerLabel(R.model, e.reads[1].tensor).en} · ${sliceName.en}, one scale for all three rows.`,
					zh: `a：${layerLabel(R.model, e.reads[0].tensor).zh ?? e.reads[0].tensor} · b：${layerLabel(R.model, e.reads[1].tensor).zh ?? e.reads[1].tensor} · ${sliceName.zh ?? sliceName.en}，三行使用同一个刻度。`
				})}
			</p>
			<SumBars
				rows={[
					{ label: 'a', value: e.values.a ?? e.terms![0].x },
					{ label: 'b', value: e.values.b ?? e.terms![1].x }
				]}
				total={e.result}
				totalLabel="= a + b"
				label={{ en: 'a + b for this cell', zh: '这个格子的 a + b' }}
			/>
		{:else if spatial && xT && out && sl}
			<SpatialMaps
				input={{ data: xT.data.subarray(sl.from, sl.from + sl.n), rows: out.shape[1], cols: out.shape[2], domain: inDom, label: { en: `Input: ${layerLabel(R.model, e.reads[0].tensor).en}`, zh: `输入：${layerLabel(R.model, e.reads[0].tensor).zh ?? e.reads[0].tensor}` } }}
				output={{ data: out.data.subarray(sl.from, sl.from + sl.n), rows: out.shape[1], cols: out.shape[2], domain: outDom, label: { en: `Output: ${sliceName.en}`, zh: `输出：${sliceName.zh ?? sliceName.en}` } }}
				box={{ y0: e.unit[1], x0: e.unit[2], y1: e.unit[1], x1: e.unit[2] }}
				cell={{ row: e.unit[1], col: e.unit[2] }}
				arrow={fnName.replace('_tanh', '')}
				onpick={(r, c) => (pick = [e!.unit[0], r, c])}
			/>
		{:else if xs && ys && sl}
			<div class="rows">
				<RowBars data={xs} max={rowMax} selected={sl.at} name="x" label={{ en: 'input', zh: '输入' }} onselect={(i) => (pick = sl.unitAt(i))} />
				<RowBars data={ys} max={rowMax} selected={sl.at} name={`${fnName.replace('_tanh', '')}(x)`} label={{ en: 'output', zh: '输出' }} onselect={(i) => (pick = sl.unitAt(i))} />
			</div>
			<p class="muted small">{t({ en: `${sliceName.en}: input and output on one scale; ${fmt3(rowMax)} is the top of the bars.`, zh: `${sliceName.zh ?? sliceName.en}：输入和输出使用同一个刻度；柱子顶端是 ${fmt3(rowMax)}。` })}</p>
		{/if}

		<OpFormula tex={R.op?.def.formula} {e} caption={where ? { en: `${layerLabel(R.model, id).en}, ${where.en}`, zh: `${layerLabel(R.model, id).zh ?? id}，${where.zh}` } : undefined} provenance={P.provenance} digits={P.props.digits as number | undefined} />
	</div>
{/if}

<style>
	.ew {
		display: grid;
		gap: 12px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.two {
		display: grid;
		grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr);
		gap: 12px 20px;
		align-items: start;
	}
	@media (max-width: 720px) {
		.two {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.side {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.side p {
		margin: 0;
	}
	.xy {
		font-size: var(--fs-md);
	}
	.arr {
		color: var(--muted);
		padding: 0 4px;
	}
	.cnt {
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.hint,
	.small {
		font-size: var(--fs-xs);
		margin: 0;
	}
	.rows {
		display: grid;
		gap: 4px;
		min-width: 0;
	}
</style>

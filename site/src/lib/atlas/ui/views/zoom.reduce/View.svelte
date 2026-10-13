<!--
  zoom.reduce: one output number = the largest (maxpool2d) or a weighted mean (mean_pool and other local reduces) of a
  window. Modeled on TinyCNN's PoolDetail: input map with the window → output map with the cell (slides on its own
  unless the host scrubs a scan tape), the window's numbers with the winner marked, then max → y, the op's formula
  and the forward pass's stored value next to the recomputed one.
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import Num from '../kit.parts/Num.svelte';
	import { fmt3 } from '../../../core/format.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import { motion } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../types/view.ts';
	import UnitHead from '../kit.parts/UnitHead.svelte';
	import OpFormula from '../kit.parts/OpFormula.svelte';
	import SpatialMaps from '../kit.parts/SpatialMaps.svelte';
	import Notice from '../kit.parts/Notice.svelte';
	import { axesOfId, axisNames, domainOf, gridRows, layerLabel, resolveUnit, scanDriven, tensorOf, unitWhere } from '../kit.parts/stage.ts';

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
	const where = $derived(e ? unitWhere(e.unit, axes, undefined, undefined, axisNames(P)) : null);
	const xId = $derived(e?.reads[0]?.tensor ?? '');
	const xT = $derived(R.trace && xId ? tensorOf(R.trace, xId) : null);
	const spatial = $derived(!!xT && xT.shape.length === 3 && axes.join() === 'c,y,x');
	const terms = $derived(e?.terms ?? []);
	// K4: without a layout the window is the most square grid that holds exactly its terms (3 terms: 1 × 3, never a 2 × 2
	// grid with an empty NaN cell)
	const k = $derived(e?.layout?.rows ?? gridRows(terms.length));
	const kc = $derived(e?.layout?.cols ?? (k ? Math.ceil(terms.length / k) : 0));
	// K4: a plain sum (every weight 1) or a mean (every weight 1/n) says so, instead of "each number is weighted"
	const same = $derived(terms.length > 1 && terms.every((x) => x.w === terms[0].w));
	const plainSum = $derived(same && terms[0].w === 1);
	const mean = $derived(same && Math.abs(terms[0].w * terms.length - 1) < 1e-9);
	const isMax = $derived(terms.length > 0 && terms.every((x) => x.w === 0 || x.w === 1) && terms.filter((x) => x.w === 1).length === 1);
	const win = $derived(terms.findIndex((x) => x.w === 1));
	const ch = $derived(spatial && e ? e.unit[0] : 0);
	const H = $derived(xT && spatial ? xT.shape[1] : 0);
	const W = $derived(xT && spatial ? xT.shape[2] : 0);
	const OH = $derived(out && spatial ? out.shape[1] : 0);
	const OW = $derived(out && spatial ? out.shape[2] : 0);
	const inDom = $derived(xT ? domainOf('diverging', xT.data) : ([-1, 1] as [number, number]));
	const outDom = $derived(out ? domainOf('diverging', out.data) : ([-1, 1] as [number, number]));
	const reduced = $derived(P.ctx?.reduced ?? motion.reduced);
	const canSlide = $derived(spatial && !scanDriven(P) && R.op?.def.unit?.scan === 'raster' && P.props.slide !== false);
	const posName = (i: number) => {
		if (k !== 2 || kc !== 2) return { en: `row ${Math.floor(i / kc)}, column ${i % kc} of the window`, zh: `窗口第 ${Math.floor(i / kc)} 行第 ${i % kc} 列` };
		return [
			{ en: 'top left', zh: '左上' },
			{ en: 'top right', zh: '右上' },
			{ en: 'bottom left', zh: '左下' },
			{ en: 'bottom right', zh: '右下' }
		][i];
	};
	$effect(() => {
		if (P.autoplay && !canSlide && e) P.onend();
	});
</script>

{#if !e}
	<Notice error={R.error} />
{:else}
	<div class="rd">
		<UnitHead layer={layerLabel(R.model, id)} op={R.op?.def.label} where={where ?? undefined} value={e.result} />

		{#if spatial && xT && out}
			<SpatialMaps
				input={{ data: xT.data.subarray(ch * H * W, (ch + 1) * H * W), rows: H, cols: W, domain: inDom, label: { en: `Input: ${layerLabel(R.model, xId).en} · channel ${ch}`, zh: `输入：${layerLabel(R.model, xId).zh ?? xId} · 通道 ${ch}` } }}
				output={{ data: out.data.subarray(ch * OH * OW, (ch + 1) * OH * OW), rows: OH, cols: OW, domain: outDom, label: { en: `Output: channel ${ch}`, zh: `输出：通道 ${ch}` } }}
				box={e.reads[0]?.box ?? null}
				cell={{ row: e.unit[1], col: e.unit[2] }}
				arrow={isMax ? `max ${k}×${kc}` : `${k}×${kc}`}
				onpick={(r, c) => (pick = [ch, r, c])}
				slide={canSlide ? { autoplay: P.autoplay && !reduced, interval: typeof P.props.interval === 'number' ? P.props.interval : 90, onend: P.onend, title: { en: 'Slide the window: press Play, or hover or tap the output map', zh: '滑动窗口：点“播放”，或在输出图上悬停、点按' } } : null}
			/>
		{/if}

		<div class="arith">
			<figure class="ng">
				<figcaption>{t({ en: `the ${terms.length} numbers in the window`, zh: `窗口里的 ${terms.length} 个数` })}</figcaption>
				<TensorGrid
					data={Float64Array.from(terms, (x) => x.x)}
					rows={k}
					cols={kc}
					domain={inDom}
					cell={44}
					readout={false}
					highlights={isMax && win >= 0 ? [{ row: Math.floor(win / kc), col: win % kc, color: 'var(--highlight)' }] : []}
					label={{ en: 'window', zh: '窗口' }}
				/>
			</figure>
			<span class="op" aria-hidden="true">{isMax ? '→ max →' : plainSum ? '→ Σ x →' : mean ? '→ mean →' : '→ Σ w·x →'}</span>
			<span class="res num"><Num value={e.result} signed /></span>
		</div>
		<p class="txt">
			{#if isMax && win >= 0}
				{t({
					en: `The largest of the ${terms.length} numbers is ${fmt3(terms[win].x)} (${posName(win).en}). It is copied to the output; the other ${terms.length - 1} are dropped.`,
					zh: `${terms.length} 个数中最大的是 ${fmt3(terms[win].x)}（${posName(win).zh}）。它被复制到输出，其余 ${terms.length - 1} 个被丢掉。`
				})}
			{:else if plainSum}
				{t({ en: `The ${terms.length} numbers are added: ${fmt3(e.result)}.`, zh: `这 ${terms.length} 个数相加：${fmt3(e.result)}。` })}
			{:else if mean}
				{t({ en: `The mean of the ${terms.length} numbers (their sum divided by ${terms.length}): ${fmt3(e.result)}.`, zh: `这 ${terms.length} 个数的平均值（总和除以 ${terms.length}）：${fmt3(e.result)}。` })}
			{:else}
				{t({ en: `Each number is weighted and the results are added: ${fmt3(e.result)}.`, zh: `每个数乘以权重后相加：${fmt3(e.result)}。` })}
			{/if}
		</p>

		<OpFormula tex={R.op?.def.formula} {e} caption={where ? { en: `${layerLabel(R.model, id).en}, ${where.en}`, zh: `${layerLabel(R.model, id).zh ?? id}，${where.zh}` } : undefined} provenance={P.provenance} digits={P.props.digits as number | undefined} />
	</div>
{/if}

<style>
	.rd {
		display: grid;
		gap: 12px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.arith {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 10px 14px;
	}
	.ng {
		margin: 0;
		display: grid;
		gap: 3px;
	}
	.ng figcaption {
		font-size: var(--t-xs);
		color: var(--muted);
	}
	.op {
		font-family: var(--font-mono);
		color: var(--muted);
	}
	.res {
		font-size: var(--fs-lg);
		font-weight: 650;
	}
	.txt {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		max-width: var(--measure);
	}
</style>

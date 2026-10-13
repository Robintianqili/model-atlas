<!--
  zoom.index-map: an output number that is an input number moved somewhere else, or replaced (flatten, concat,
  causal_mask, and local ops such as to_tokens or select_token). Modeled on TinyCNN's FlattenDetail: the input with the
  cell that is read, the output with the cell that is written, and the index arithmetic in words (flatten:
  index = c·H·W + row·W + col). For causal_mask, the cell either passes (key not after the query) or becomes −∞.
  Picking a cell in the output picks another unit.
-->
<script lang="ts">
	import Num from '../kit.parts/Num.svelte';
	import { fmt3 } from '../../../core/format.ts';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../types/view.ts';
	import UnitHead from '../kit.parts/UnitHead.svelte';
	import OpFormula from '../kit.parts/OpFormula.svelte';
	import TensorPanel from '../kit.parts/TensorPanel.svelte';
	import Notice from '../kit.parts/Notice.svelte';
	import { axesOfId, axisNames, layerLabel, outLabels, resolveUnit, tensorOf, tokenIds, tokenTexts, unflat, unitWhere } from '../kit.parts/stage.ts';

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
	const T = $derived(out ? (axes[0] === 't' ? out.shape[0] : axes[1] === 'tq' ? out.shape[1] : 0) : 0);
	const tokens = $derived(T ? tokenTexts(P, T, tokenIds(R.trace, R.model), i18n.lang) : []);
	const where = $derived(e ? unitWhere(e.unit, axes, tokens, outLabels(P, R.model), axisNames(P)) : null);
	const srcId = $derived(e?.reads[0]?.tensor ?? '');
	const src = $derived(R.trace && srcId ? tensorOf(R.trace, srcId) : null);
	const srcCell = $derived(src && e?.reads[0]?.cells.length ? unflat(src.shape, e.reads[0].cells[0]) : null);
	const op = $derived(R.op?.def.op ?? R.layer?.op ?? '');
	const isMask = $derived(e?.values.masked !== undefined && op !== 'attention_scores');
	const sliceName = (shape: readonly number[], axesOf: string[]) =>
		axesOf[0] === 'h' ? { en: 'head', zh: '头' } : shape.length === 3 ? { en: 'channel', zh: '通道' } : { en: 'slice', zh: '切片' };
	const srcAxes = $derived(src ? axesOfId(R.model, srcId, src.shape) : []);
	$effect(() => {
		if (P.autoplay && e) P.onend();
	});
</script>

{#if !e}
	<Notice error={R.error} />
{:else}
	<div class="im">
		<UnitHead layer={layerLabel(R.model, id)} op={R.op?.def.label} where={where ?? undefined} value={e.result} />
		<div class="pair">
			{#if src}
				<TensorPanel
					data={src.data}
					shape={src.shape}
					cell={srcCell}
					label={{ en: `Read from ${layerLabel(R.model, srcId).en}`, zh: `读自 ${layerLabel(R.model, srcId).zh ?? srcId}` }}
					sliceName={sliceName(src.shape, srcAxes)}
				/>
			{/if}
			<div class="arrow" aria-hidden="true">
				<span>{isMask ? (e.values.masked ? '−∞' : '=') : '→'}</span>
				<svg viewBox="0 0 40 16" width="40" height="16"><path d="M2 8h32m-6-5 6 5-6 5" fill="none" stroke="currentColor" stroke-width="1.8" /></svg>
			</div>
			{#if out}
				<TensorPanel
					data={out.data}
					shape={out.shape}
					cell={e.unit}
					label={{ en: `Written to ${layerLabel(R.model, id).en}`, zh: `写入 ${layerLabel(R.model, id).zh ?? id}` }}
					sliceName={sliceName(out.shape, axes)}
					onpick={(u) => (pick = u)}
				/>
			{/if}
		</div>
		<p class="line">
			{#if isMask}
				{#if e.values.masked}
					{t({ en: `Key ${e.values.j ?? 0} comes after query ${e.values.i ?? 0} (counting from 0): the score ${fmt3(e.values.x)} is replaced by −∞, so its weight after the softmax is exactly 0.`, zh: `键 ${e.values.j ?? 0} 在查询 ${e.values.i ?? 0} 之后（从 0 数起）：分数 ${fmt3(e.values.x)} 被换成 −∞，所以 softmax 之后它的权重正好是 0。` })}
				{:else}
					{t({ en: `Key ${e.values.j ?? 0} is not after query ${e.values.i ?? 0} (counting from 0): the score ${fmt3(e.values.x)} passes unchanged.`, zh: `键 ${e.values.j ?? 0} 不在查询 ${e.values.i ?? 0} 之后（从 0 数起）：分数 ${fmt3(e.values.x)} 原样通过。` })}
				{/if}
			{:else if src && src.shape.length === 3 && e.values.c !== undefined}
				{@const [, H, W] = src.shape}
				<span class="num">
					{t({ en: 'index', zh: '序号' })} = c·{H * W} + {t({ en: 'row', zh: '行' })}·{W} + {t({ en: 'col', zh: '列' })} = {e.values.c}·{H * W} + {e.values.row}·{W} + {e.values.col} = <strong>{e.values.index}</strong>
				</span>
				<span class="muted"> · {t({ en: 'counting from 0; the value moves, it does not change:', zh: '从 0 开始数；数值只是挪了位置，没有改变：' })} <Num value={e.result} /></span>
			{:else if e.values.part !== undefined}
				{t({ en: `This number comes unchanged from part ${e.values.part} of ${e.values.parts}.`, zh: `这个数原样来自 ${e.values.parts} 个部分中的第 ${e.values.part} 部分。` })}
			{:else}
				{t({ en: `The number ${fmt3(e.result)} is copied unchanged.`, zh: `数值 ${fmt3(e.result)} 原样复制。` })}
			{/if}
		</p>
		<OpFormula tex={R.op?.def.formula} {e} caption={where ? { en: `${layerLabel(R.model, id).en}, ${where.en}`, zh: `${layerLabel(R.model, id).zh ?? id}，${where.zh}` } : undefined} provenance={P.provenance} digits={P.props.digits as number | undefined} />
	</div>
{/if}

<style>
	.im {
		display: grid;
		gap: 12px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.pair {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 10px 14px;
		min-width: 0;
	}
	.pair > :global(figure) {
		flex: 1 1 240px;
		max-width: 420px;
	}
	.arrow {
		display: grid;
		justify-items: center;
		color: var(--muted);
		font-family: var(--font-mono);
		font-size: var(--t-xs);
	}
	@media (max-width: 560px) {
		.arrow {
			flex-basis: 100%;
		}
		.arrow svg {
			transform: rotate(90deg);
			margin: 8px 0;
		}
	}
	.line {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		max-width: var(--measure);
	}
</style>

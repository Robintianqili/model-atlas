<!--
  layer.tensor: one whole tensor, drawn the way its axes ask (design §5.1 "Displays by axes"), with the kit's grammar:
    c,y,x → small multiples (TileStack) + the selected channel enlarged · t,d → token chips over a [T×D] grid ·
    h,tq,tk → one head at a time (AttentionMatrix for weights, a grid with −∞ hatched for scores) ·
    class / vocab → probability bars (a distribution) or a logit strip · p,xy → points · t → a line · else TensorView.
  props: layer (id), field (+ key) for process/training state, axes (override), thumb (small, no controls).
  The step's focus (field = this id, or 'out') is outlined. A stats line under the tensor names its shape and range.
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import TensorStrip from '$lib/kit/TensorStrip.svelte';
	import TileStack from '$lib/kit/TileStack.svelte';
	import TensorView from '$lib/kit/TensorView.svelte';
	import AttentionMatrix from '$lib/kit/AttentionMatrix.svelte';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3 } from '../../../core/format.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../types/view.ts';
	import Notice from '../kit.parts/Notice.svelte';
	import { domainOf, outLabels, stageModel, stageTrace, tokenIds, tokenTexts } from '../kit.parts/stage.ts';
	import { pickTensor } from './pick.ts';
	import { outputId, outputRow } from '../../../lesson/facts.ts';

	let P: StageProps = $props();
	const X = $derived(pickTensor(P));
	const thumb = $derived(P.props.thumb === true);
	const shape = $derived(X?.t.shape ?? []);
	const data = $derived(X?.t.data ?? new Float32Array(0));
	const dom = $derived(X ? domainOf(X.kind, data) : ([-1, 1] as [number, number]));
	const model = $derived(stageModel(P));
	const ids = $derived(tokenIds(stageTrace(P), model));
	const T = $derived(X && (X.axes[0] === 't' || X.axes[1] === 'tq') ? (X.axes[0] === 't' ? shape[0] : shape[1]) : 0);
	const tokens = $derived(T ? tokenTexts(P, T, ids) : []);
	const labels = $derived(outLabels(P, model));
	/** the focused cell of this tensor (step focus, or the stage's unit) */
	const focusCell = $derived.by<number[] | null>(() => {
		if (!X) return null;
		for (const f of P.focus) {
			if ((f.field === X.id || f.field === 'out') && Array.isArray(f.index) && typeof f.index[0] === 'number') return f.index as number[];
		}
		if (P.at.unit && P.at.unit.layer === X.id) return P.at.unit.unit;
		if (P.sel.layer === X.id && Array.isArray(P.sel.unit)) return P.sel.unit;
		return null;
	});
	let chPick = $derived.by<number | null>(() => {
		void X?.id;
		return null;
	});
	const strongest = $derived.by(() => {
		if (!X || shape.length !== 3) return 0;
		const per = shape[1] * shape[2];
		let best = 0;
		let bs = -Infinity;
		for (let c = 0; c < shape[0]; c++) {
			let s = 0;
			for (let i = 0; i < per; i++) s += Math.abs(data[c * per + i]);
			if (s > bs) (bs = s), (best = c);
		}
		return best;
	});
	const ch = $derived(chPick ?? focusCell?.[0] ?? (X?.display === 'heads' ? 0 : strongest));
	let rowPick = $derived.by<number | null>(() => {
		void X?.id;
		return null;
	});
	// K2: the output layer opens at the package's answer row (manifest output.row: a masked model's [MASK] row), others at the last
	const lastRow = $derived(X && shape.length >= 2 && X.id === outputId(model?.spec ?? P.rt.spec, stageTrace(P)) ? outputRow(P.rt, X.t) : shape.length >= 2 ? shape[0] - 1 : 0);
	const row = $derived(rowPick ?? (X?.display === 'tokens' || X?.display === 'dist' || X?.display === 'logits' ? (focusCell?.[0] ?? lastRow) : 0));
	const per = $derived(shape.length === 3 ? shape[1] * shape[2] : 0);
	const slice = $derived(shape.length === 3 ? data.subarray(ch * per, (ch + 1) * per) : data);
	const lastN = $derived(shape[shape.length - 1] ?? 0);
	const rowData = $derived(shape.length >= 2 ? data.subarray(row * lastN, (row + 1) * lastN) : data);
	const stats = $derived.by(() => {
		let lo = Infinity;
		let hi = -Infinity;
		let s = 0;
		let n = 0;
		let masked = 0;
		for (let i = 0; i < data.length; i++) {
			const v = data[i];
			if (!Number.isFinite(v)) {
				masked++;
				continue;
			}
			lo = Math.min(lo, v);
			hi = Math.max(hi, v);
			s += v;
			n++;
		}
		return { lo, hi, mean: n ? s / n : NaN, masked };
	});
	const nameOf = (i: number) => {
		if (X?.axes[X.axes.length - 1] === 'vocab') {
			const l = labels?.(i);
			return l === undefined ? `#${i}` : typeof l === 'string' ? l : t(l);
		}
		const l = labels?.(i);
		return l === undefined ? `#${i}` : typeof l === 'string' ? l : t(l);
	};
	const items = $derived(
		X && (X.display === 'dist' || X.display === 'logits')
			? Array.from(rowData, (p, i) => ({ label: nameOf(i), p, id: i, color: X.axes[X.axes.length - 1] === 'class' ? categorical(i, themeState.resolved) : undefined }))
			: []
	);
	const top = $derived.by(() => {
		let b = 0;
		for (let i = 1; i < rowData.length; i++) if (rowData[i] > rowData[b]) b = i;
		return b;
	});
	const chips = $derived(tokens.map((s, i) => ({ text: s, id: ids ? ids[i] : undefined })));
	const causal = $derived.by(() => {
		if (X?.display !== 'heads' || shape.length !== 3) return false;
		const [, a, b] = shape;
		for (let i = 0; i < a; i++) for (let j = i + 1; j < b; j++) if (slice[i * b + j] !== 0 && Number.isFinite(slice[i * b + j])) return false;
		return a > 1;
	});
	const points = $derived.by(() => {
		if (X?.display !== 'points' || shape.length !== 2) return null;
		const n = shape[0];
		const xs = Array.from({ length: n }, (_, i) => data[i * 2]);
		const ys = Array.from({ length: n }, (_, i) => data[i * 2 + 1]);
		const m = Math.max(1e-9, ...xs.map(Math.abs), ...ys.map(Math.abs));
		return { xs, ys, m };
	});
	$effect(() => {
		if (P.autoplay && X) P.onend();
	});
</script>

{#if !X}
	<Notice text={{ en: 'No tensor to show: the step names no layer or field, or the source has not run yet.', zh: '没有可显示的张量：步骤没有指定层或字段，或者数据源还没有运行。' }} />
{:else}
	<div class="lt" class:thumb>
		{#if !thumb}
			<p class="hd"><strong>{t(X.label)}</strong> <span class="num muted">[{shape.join('×')}]</span> <span class="num muted ax">{X.axes.join(',')}</span></p>
		{/if}
		{#if X.display === 'tiles'}
			<div class="tiles">
				<TileStack
					{data}
					channels={shape[0]}
					height={shape[1]}
					width={shape[2]}
					kind={X.kind}
					scopeDomains={{ layer: dom }}
					tile={thumb ? 36 : shape[0] > 16 ? 44 : 60}
					channelLabels={Array.from({ length: shape[0] }, (_, c) => String(c))}
					bind:selected={() => ch, (c) => (chPick = c)}
					label={X.label}
				/>
				{#if !thumb}
					<figure class="big">
						<figcaption>{t({ en: `channel ${ch} of ${shape[0]} (counting from 0)`, zh: `通道 ${ch}（共 ${shape[0]} 个，从 0 数起）` })}</figcaption>
						<TensorGrid
							data={slice}
							rows={shape[1]}
							cols={shape[2]}
							kind={X.kind}
							domain={dom}
							maxCell={Math.max(8, Math.min(28, Math.floor(364 / Math.max(shape[1], shape[2]))))}
							highlights={focusCell && focusCell[0] === ch ? [{ row: focusCell[1], col: focusCell[2], color: 'var(--accent)' }] : []}
							label={X.label}
						/>
					</figure>
				{/if}
			</div>
		{:else if X.display === 'tokens'}
			{#if chips.length && !thumb}
				<TokenChips tokens={chips} selected={row} showIds={!!ids} label={{ en: 'Tokens: pick a row', zh: '词元：选一行' }} onselect={(i) => (rowPick = i)} />
			{/if}
			<TensorGrid
				{data}
				rows={shape[0]}
				cols={shape[1]}
				kind={X.kind}
				domain={dom}
				maxCell={thumb ? 6 : 14}
				interactive={!thumb}
				readout={!thumb}
				highlights={[{ row, col: 0, h: 1, w: shape[1], color: 'var(--q)' }, ...(focusCell && focusCell.length === 2 ? [{ row: focusCell[0], col: focusCell[1], color: 'var(--accent)' }] : [])]}
				label={X.label}
			/>
		{:else if X.display === 'heads'}
			{#if !thumb && shape[0] > 1}
				<div class="segmented" role="group" aria-label={t({ en: 'Head', zh: '头' })}>
					{#each Array.from({ length: shape[0] }, (_, h) => h) as h (h)}
						<button type="button" aria-pressed={h === ch} onclick={() => (chPick = h)}>{t({ en: `head ${h}`, zh: `头 ${h}` })}</button>
					{/each}
				</div>
			{/if}
			{#if X.kind === 'sequential'}
				<AttentionMatrix
					weights={slice}
					queries={tokens.length === shape[1] ? tokens : Array.from({ length: shape[1] }, (_, i) => String(i))}
					keys={tokens.length === shape[2] ? tokens : undefined}
					{causal}
					row={focusCell && focusCell[0] === ch ? focusCell[1] : null}
					col={focusCell && focusCell[0] === ch ? focusCell[2] : null}
					maxCell={thumb ? 10 : 30}
					label={X.label}
				/>
			{:else}
				<TensorGrid
					data={slice}
					rows={shape[1]}
					cols={shape[2]}
					domain={dom}
					maxCell={thumb ? 8 : 22}
					masked={(r, c) => !Number.isFinite(slice[r * shape[2] + c])}
					highlights={focusCell && focusCell[0] === ch ? [{ row: focusCell[1], col: focusCell[2], color: 'var(--accent)' }] : []}
					label={X.label}
				/>
			{/if}
		{:else if X.display === 'dist'}
			{#if chips.length > 1 && !thumb}
				<TokenChips tokens={chips} selected={row} label={{ en: 'Prediction after which token', zh: '在哪个词元之后的预测' }} onselect={(i) => (rowPick = i)} />
			{/if}
			<ProbBars {items} k={thumb ? 3 : 10} chosen={top} tokens={X.axes[X.axes.length - 1] === 'vocab'} label={X.label} />
		{:else if X.display === 'logits'}
			{#if chips.length > 1 && !thumb}
				<TokenChips tokens={chips} selected={row} label={{ en: 'Scores after which token', zh: '在哪个词元之后的分数' }} onselect={(i) => (rowPick = i)} />
			{/if}
			<TensorStrip data={rowData} domain={domainOf('diverging', rowData)} maxCells={thumb ? 16 : 40} shape={[rowData.length]} name={X.label} selected={top} />
			{#if !thumb}
				<p class="muted small">{t({ en: `Largest score: ${nameOf(top)} (${fmt3(rowData[top])}). These are scores, not yet shares: a softmax turns them into probabilities.`, zh: `最大的分数：${nameOf(top)}（${fmt3(rowData[top])}）。这些还是分数，不是份额：softmax 才把它们变成概率。` })}</p>
			{/if}
		{:else if X.display === 'points' && points}
			<svg class="pts" viewBox="-1.1 -1.1 2.2 2.2" role="img" aria-label="{t(X.label)}: {shape[0]} {t({ en: 'points', zh: '个点' })}">
				<line x1="-1.1" x2="1.1" y1="0" y2="0" class="axis" />
				<line x1="0" x2="0" y1="-1.1" y2="1.1" class="axis" />
				{#each points.xs as x, i (i)}<circle cx={x / points.m} cy={-points.ys[i] / points.m} r="0.025" class="pt" />{/each}
			</svg>
		{:else if X.display === 'series'}
			{@const n = data.length}
			{@const m = Math.max(1e-9, ...Array.from(data, (v) => (Number.isFinite(v) ? Math.abs(v) : 0)))}
			<svg class="ser" viewBox="0 0 400 120" preserveAspectRatio="none" role="img" aria-label="{t(X.label)}: {n}">
				<line x1="0" x2="400" y1="60" y2="60" class="axis" />
				<polyline points={Array.from(data, (v, i) => `${(i / Math.max(1, n - 1)) * 400},${60 - (v / m) * 55}`).join(' ')} class="line" />
			</svg>
		{:else if X.display === 'strip'}
			<TensorStrip {data} kind={X.kind} domain={dom} maxCells={thumb ? 16 : 48} shape={[...shape]} name={X.label} selected={focusCell?.[0] ?? null} />
		{:else if X.display === 'grid'}
			<TensorGrid
				{data}
				rows={shape[0]}
				cols={shape[1]}
				kind={X.kind}
				domain={dom}
				maxCell={thumb ? 8 : 24}
				interactive={!thumb}
				readout={!thumb}
				highlights={focusCell && focusCell.length === 2 ? [{ row: focusCell[0], col: focusCell[1], color: 'var(--accent)' }] : []}
				label={X.label}
			/>
		{:else}
			<TensorView tensor={X.t} label={X.label} kind={X.kind} />
		{/if}
		{#if !thumb}
			<div class="foot">
				<span class="num muted">
					min {fmt3(stats.lo)} · max {fmt3(stats.hi)} · {t({ en: 'mean', zh: '平均' })} {fmt3(stats.mean)}{#if stats.masked} · {stats.masked} {t({ en: 'masked (−∞)', zh: '个被遮住（−∞）' })}{/if}
				</span>
				{#if X.display !== 'dist' && X.display !== 'points' && X.display !== 'series'}<ColorLegend kind={X.kind} domain={dom} width={160} />{/if}
			</div>
		{/if}
	</div>
{/if}

<style>
	.lt {
		display: grid;
		gap: 10px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.lt.thumb {
		gap: 4px;
	}
	.hd {
		margin: 0;
		font-size: var(--fs-sm);
		display: flex;
		flex-wrap: wrap;
		gap: 2px 10px;
		align-items: baseline;
	}
	.ax {
		font-size: var(--fs-xs);
	}
	.tiles {
		display: flex;
		flex-wrap: wrap;
		gap: 14px 20px;
		align-items: flex-start;
		min-width: 0;
	}
	.tiles > :global(*) {
		min-width: 0;
		max-width: 100%;
	}
	.big {
		margin: 0;
		display: grid;
		gap: 4px;
		flex: 1 1 240px;
		max-width: 380px;
	}
	.big figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.segmented {
		flex-wrap: wrap;
		justify-self: start;
	}
	.foot {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 16px;
		align-items: center;
		font-size: var(--fs-xs);
	}
	.small {
		margin: 0;
		font-size: var(--fs-xs);
	}
	.pts {
		width: min(100%, 320px);
		height: auto;
		background: var(--surface-2);
		border-radius: var(--r2);
	}
	.ser {
		width: 100%;
		max-width: 560px;
		height: 120px;
	}
	.axis {
		stroke: var(--rule-strong);
		stroke-width: 0.01;
	}
	.ser .axis {
		stroke-width: 1;
		vector-effect: non-scaling-stroke;
	}
	.pt {
		fill: var(--accent);
	}
	.line {
		fill: none;
		stroke: var(--ink);
		stroke-width: 1.6;
		vector-effect: non-scaling-stroke;
	}
</style>

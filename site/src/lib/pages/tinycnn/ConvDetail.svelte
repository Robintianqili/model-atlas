<!--
  Convolution zoom: one output cell = for every input channel, a 3x3 window times that channel's
  3x3 kernel, summed; then all channel sums plus the bias. Uses model.convDetail (live).
  The StepPlayer slides the window over every output position; hovering the output map scrubs.

  Optional props (used by the guided lesson; defaults keep the zoom view unchanged):
  view (whose selection), parts (which blocks), autoplay / interval / fill / onend (the slide).
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import Num from '$lib/kit/Num.svelte';
	import MapCanvas from './MapCanvas.svelte';
	import NumGrid from './NumGrid.svelte';
	import { sym } from '$lib/kit/tex';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import { channelDomain } from './data';
	import type { CnnStore, SelView } from './store.svelte';

	type Part = 'slide' | 'channels' | 'arith' | 'formula';
	interface Props {
		S: CnnStore;
		/** whose selection to show (default: the page store) */
		view?: SelView;
		/** blocks to show (default: all) */
		parts?: Part[];
		/** start sliding the window when shown */
		autoplay?: boolean;
		/** ms per window position */
		interval?: number;
		/** while sliding, hatch the output cells the window has not reached yet */
		fill?: boolean;
		/** the slide reached the last position */
		onend?: () => void;
		/** heading of the slide player (default: "Slide the kernel: press Play, …") */
		title?: Text;
		/** keep the input and output maps side by side on phones (the guided lesson: watch both while it slides) */
		pair?: boolean;
	}
	let { S, view, parts = ['slide', 'channels', 'arith', 'formula'], autoplay = false, interval = 150, fill = false, onend, title, pair = false }: Props = $props();
	const V = $derived(view ?? S);
	const show = (p: Part) => parts.includes(p);
	let playing = $state(false);

	const L = $derived(S.layer(V.sel.layer));
	const inL = $derived(S.layer(L.in[0]));
	const O = $derived(L.outShape[0]);
	const OH = $derived(L.outShape[1]);
	const OW = $derived(L.outShape[2]);
	const C = $derived(inL.outShape[0]);
	const IH = $derived(inL.outShape[1]);
	const IW = $derived(inL.outShape[2]);
	const d = $derived(S.model && S.trace ? S.model.convDetail(S.trace, L.id, V.sel.ch, V.sel.y, V.sel.x) : null);

	/** the input channel shown large and in full numbers */
	let inChPicked = $state<number | null>(null);
	let lastKey = '';
	$effect(() => {
		// a new layer or output channel resets the pick to the channel that contributes most
		const k = `${L.id}:${V.sel.ch}`;
		if (k !== lastKey) {
			lastKey = k;
			inChPicked = null;
		}
	});
	const strongest = $derived.by(() => {
		if (!d) return 0;
		let b = 0;
		for (let c = 1; c < d.channelSums.length; c++) if (Math.abs(d.channelSums[c]) > Math.abs(d.channelSums[b])) b = c;
		return b;
	});
	const inCh = $derived(Math.min(C - 1, inChPicked ?? strongest));
	let hoverK = $state<number | null>(null);
	let hoverBar = $state<number | null>(null);

	// colors: input like its tensor, kernels on one range for the whole layer, products on their own range
	const inKind = $derived(inL.op === 'input' ? ('sequential' as const) : ('diverging' as const));
	const inData = $derived(S.trace?.t[inL.id].data ?? null);
	const inDomain = $derived(S.domainFor(inL.id, () => (inData ? channelDomain(inData, inCh, IH * IW) : [-1, 1])));
	const outData = $derived(S.trace ? S.trace.t[L.id].data.subarray(V.sel.ch * OH * OW, (V.sel.ch + 1) * OH * OW) : null);
	const outDomain = $derived(S.domainFor(L.id, () => (S.trace ? channelDomain(S.trace.t[L.id].data, V.sel.ch, OH * OW) : [-1, 1])));
	const kDomain = $derived.by<[number, number]>(() => {
		const w = S.model?.weights[`${L.id}.weight`]?.data;
		let m = 0;
		if (w) for (let i = 0; i < w.length; i++) m = Math.max(m, Math.abs(w[i]));
		return [-(m || 1), m || 1];
	});
	const pDomain = $derived.by<[number, number]>(() => {
		let m = 0;
		if (d) for (const p of d.products) for (let i = 0; i < p.length; i++) m = Math.max(m, Math.abs(p[i]));
		return [-(m || 1), m || 1];
	});
	const sumDomain = $derived.by<[number, number]>(() => {
		let m = Math.abs(d?.bias ?? 0);
		if (d) for (let c = 0; c < d.channelSums.length; c++) m = Math.max(m, Math.abs(d.channelSums[c]));
		m = Math.max(m, Math.abs(d?.total ?? 0));
		return [-(m || 1), m || 1];
	});
	const sumScale = $derived(makeScale('diverging', sumDomain, themeState.resolved));

	const inSlice = $derived(inData ? inData.subarray(inCh * IH * IW, (inCh + 1) * IH * IW) : null);
	const win = $derived(d ? { row: d.window.y0, col: d.window.x0, h: 3, w: 3 } : null);

	function onOutHover(c: { row: number; col: number } | null) {
		if (!c) {
			S.hoverRF = null;
			return;
		}
		V.setCell(c.row, c.col);
		S.hoverRF = S.receptiveField(L.id, c.row, c.col);
	}

	const sumAll = $derived(d ? d.channelSums.reduce((a, b) => a + b, 0) : 0);
	const tex = $derived(
		C === 1
			? `\\begin{aligned} ${sym('out', 'y_{ij}')} &= \\sum_{u,v=0}^{2} ${sym('param', 'K_{uv}')}\\,${sym('in', 'x_{i+u,\\,j+v}')} + ${sym('param', 'b')} \\\\ &= {{s}} + ({{b}}) = {{y}} \\\\ \\mathrm{ReLU}(${sym('out', 'y_{ij}')}) &= \\max(0, {{y}}) = {{r}} \\end{aligned}`
			: `\\begin{aligned} ${sym('out', 'y_{ij}')} &= \\sum_{c=0}^{${C - 1}} \\underbrace{\\sum_{u,v=0}^{2} ${sym('param', 'K^{(c)}_{uv}')}\\,${sym('in', 'x^{(c)}_{i+u,\\,j+v}')}}_{s_c} + ${sym('param', 'b')} \\\\ &= {{s}} + ({{b}}) = {{y}} \\\\ \\mathrm{ReLU}(${sym('out', 'y_{ij}')}) &= \\max(0, {{y}}) = {{r}} \\end{aligned}`
	);
	const values = $derived<Record<string, number>>(d ? { s: sumAll, b: d.bias, y: d.total, r: d.relu } : {});
	const labels = $derived(
		C === 1
			? { win: { en: 'input window x', zh: '输入窗口 x' }, ker: { en: 'kernel K', zh: '卷积核 K' }, prod: { en: 'K × x', zh: 'K × x' } }
			: {
					win: { en: `window in channel ${inCh}`, zh: `通道 ${inCh} 的窗口` },
					ker: { en: `kernel slice for channel ${inCh}`, zh: `对应通道 ${inCh} 的卷积核切片` },
					prod: { en: 'products', zh: '逐元素乘积' }
				}
	);
</script>

{#if d && inSlice && outData && win}
	{#if show('slide')}
	<StepPlayer
		count={OH * OW}
		bind:step={() => V.sel.y * OW + V.sel.x, (v) => V.setCell(Math.floor(v / OW), v % OW)}
		bind:playing
		{autoplay}
		{interval}
		{onend}
		speeds={[0.5, 1, 3, 8]}
		label={title ?? t({ en: 'Slide the kernel: press Play, or hover or tap the output map', zh: '滑动卷积核：点“播放”，或在输出图上悬停、点按' })}
		position={(i, n) => t({ en: `Position ${i} of ${n}`, zh: `位置 ${i} / ${n}` })}
	>
		{#snippet children()}
			<div class="maps" class:pair>
				<figure class="m">
					<figcaption>
						{t({ en: 'Input:', zh: '输入：' })} {inL.op === 'input' ? t({ en: 'your drawing', zh: '你的画' }) : t(inL.label)}{C > 1 ? ` · ${t({ en: 'channel', zh: '通道' })} ${inCh}` : ''}
						<span class="num muted">[{IH}×{IW}]</span>
					</figcaption>
					<TensorGrid
						data={inSlice}
						rows={IH}
						cols={IW}
						kind={inKind}
						domain={inDomain}
						maxCell={14}
						interactive={false}
						readout={false}
						highlights={[{ ...win, color: 'var(--accent)' }]}
						label="{t(inL.label)} {inCh}"
					/>
				</figure>
				<div class="arrow" aria-hidden="true">
					<span>3×3</span>
					<svg viewBox="0 0 40 16" width="40" height="16"><path d="M2 8h32m-6-5 6 5-6 5" fill="none" stroke="currentColor" stroke-width="1.8" /></svg>
				</div>
				<figure class="m">
					<figcaption>
						{t({ en: 'Output:', zh: '输出：' })} {t(L.label)} · {t({ en: 'channel', zh: '通道' })} {V.sel.ch}
						<span class="num muted">[{OH}×{OW}]</span>
					</figcaption>
					<TensorGrid
						data={outData}
						rows={OH}
						cols={OW}
						domain={outDomain}
						maxCell={14}
						highlights={[{ row: V.sel.y, col: V.sel.x, color: 'var(--accent)' }]}
						masked={fill && playing ? (r, c) => r * OW + c > V.sel.y * OW + V.sel.x : undefined}
						label="{t(L.label)} {V.sel.ch}"
						bind:cursor={() => ({ row: V.sel.y, col: V.sel.x }), (c) => { if (c) V.setCell(c.row, c.col); }}
						onhover={onOutHover}
					/>
				</figure>
			</div>
		{/snippet}
	</StepPlayer>
	{/if}

	{#if C > 1 && show('channels')}
		<div class="chans">
			<p class="sub">
				{t({
					en: `The same window position in all ${C} input channels. Each channel has its own 3×3 slice of the kernel. Click one to see its numbers.`,
					zh: `同一个窗口位置出现在全部 ${C} 个输入通道中。每个通道都有自己的一块 3×3 卷积核切片。点选一个查看具体数字。`
				})}
			</p>
			<div class="thumbs" role="group" aria-label={t({ en: 'Input channels', zh: '输入通道' })}>
				{#each Array.from({ length: C }, (_, c) => c) as c (c)}
					<button
						type="button"
						class="th"
						class:on={c === inCh}
						class:hl={hoverBar === c}
						aria-pressed={c === inCh}
						onclick={() => (inChPicked = c)}
						onpointerenter={() => (hoverBar = c)}
						onpointerleave={() => (hoverBar = null)}
					>
						<MapCanvas
							data={inData!.subarray(c * IH * IW, (c + 1) * IH * IW)}
							rows={IH}
							cols={IW}
							domain={S.domainFor(inL.id, () => channelDomain(inData!, c, IH * IW))}
							size={C > 8 ? 44 : 52}
							boxes={[{ y0: d.window.y0, x0: d.window.x0, y1: d.window.y1, x1: d.window.x1, color: 'var(--accent)' }]}
						/>
						<span class="mini">
							<MapCanvas data={d.products[c]} rows={3} cols={3} domain={pDomain} size={21} />
						</span>
						<span class="cs num" style="color: {d.channelSums[c] < 0 ? 'var(--div-neg)' : d.channelSums[c] > 0 ? 'var(--div-pos)' : 'var(--muted)'}">{fmt3(d.channelSums[c])}</span>
					</button>
				{/each}
			</div>
		</div>
	{/if}

	{#if show('arith')}
	<div class="arith-wrap">
	<div class="arith">
		<div class="mul">
			<NumGrid values={d.windows[inCh]} rows={3} cols={3} kind={inKind} domain={inDomain} label={labels.win} bind:hover={hoverK} />
			<span class="op" aria-hidden="true">×</span>
			<NumGrid values={d.kernels[inCh]} rows={3} cols={3} domain={kDomain} label={labels.ker} role="param" bind:hover={hoverK} />
			<span class="op" aria-hidden="true">=</span>
			<NumGrid values={d.products[inCh]} rows={3} cols={3} domain={pDomain} label={labels.prod} bind:hover={hoverK} />
			<span class="sumgrp">
				<span class="op" aria-hidden="true">→ Σ</span>
				<span class="sum num">
					{#if C > 1}<span class="muted">s<sub>{inCh}</sub> =</span>{/if}
					<Num value={d.channelSums[inCh]} signed />
				</span>
			</span>
		</div>
		{#if hoverK !== null}
			<p class="hov num">
				{fmt3(d.windows[inCh][hoverK])} × {fmt3(d.kernels[inCh][hoverK])} = {fmt3(d.products[inCh][hoverK])}
			</p>
		{:else}
			<p class="hov muted">{t({ en: 'Hover a cell to see one product.', zh: '把鼠标移到格子上查看一次乘法。' })}</p>
		{/if}

		<div class="sums" aria-label={t({ en: 'Sums per input channel, plus bias', zh: '各输入通道的和，加上偏置' })}>
			{#each Array.from({ length: C }, (_, c) => c) as c (c)}
				{@const v = d.channelSums[c]}
				<button
					type="button"
					class="srow"
					class:on={c === inCh}
					class:hl={hoverBar === c}
					onclick={() => (inChPicked = c)}
					onpointerenter={() => (hoverBar = c)}
					onpointerleave={() => (hoverBar = null)}
				>
					<span class="sl num">{C > 1 ? `s${c}` : 'Σ'}</span>
					<span class="track"><span class="b {v < 0 ? 'neg' : 'pos'}" style="--w: {(Math.abs(v) / sumDomain[1]) * 50}%; background: {sumScale.css(v)}"></span></span>
					<span class="sv num">{fmt3(v)}</span>
				</button>
			{/each}
			<div class="srow fixed">
				<span class="sl num">+ b</span>
				<span class="track"><span class="b {d.bias < 0 ? 'neg' : 'pos'}" style="--w: {(Math.abs(d.bias) / sumDomain[1]) * 50}%; background: {sumScale.css(d.bias)}"></span></span>
				<span class="sv num">{fmt3(d.bias)}</span>
			</div>
			<div class="srow fixed total">
				<span class="sl num">= y</span>
				<span class="track"><span class="b {d.total < 0 ? 'neg' : 'pos'}" style="--w: {(Math.abs(d.total) / sumDomain[1]) * 50}%; background: {sumScale.css(d.total)}"></span></span>
				<span class="sv num">{fmt3(d.total)}</span>
			</div>
		</div>
	</div>
	</div>
	{/if}

	{#if show('formula')}
	<FormulaCard
		{tex}
		{values}
		provenance="live"
		caption={t({ en: `Cell (${V.sel.y}, ${V.sel.x}) of ${t(L.label)}, channel ${V.sel.ch}`, zh: `${t(L.label)} 通道 ${V.sel.ch} 的格子 (${V.sel.y}, ${V.sel.x})` })}
		where={[
			{ tex: sym('param', 'K'), text: { en: 'kernel weights (learned)', zh: '卷积核权重（学到的）' } },
			{ tex: sym('in', 'x'), text: { en: 'input values in the window', zh: '窗口里的输入值' } },
			{ tex: sym('param', 'b'), text: { en: 'bias (learned)', zh: '偏置（学到的）' } },
			{ tex: sym('out', 'y'), text: { en: 'output cell', zh: '输出格' } }
		]}
	/>
	<p class="check muted">
		{t({ en: 'Stored by the forward pass for this cell:', zh: '前向传播为这个格子存下的值：' })}
		<Num value={d.traced} /> · {t({ en: `${C * 9} products + 1 bias`, zh: `${C * 9} 次乘法 + 1 个偏置` })}
	</p>
	{/if}
{/if}

<style>
	.maps {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 10px 14px;
	}
	.m {
		margin: 0;
		display: grid;
		gap: 4px;
		min-width: 0;
		flex: 1 1 220px;
		max-width: 400px;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.arrow {
		display: grid;
		justify-items: center;
		color: var(--muted);
		font-size: 11px;
		font-family: var(--font-mono);
	}
	@media (max-width: 560px) {
		.arrow {
			flex-basis: 100%;
		}
		.arrow svg {
			transform: rotate(90deg);
			margin: 12px 0;
		}
		/* pair: both maps stay in view side by side while the window slides */
		.maps.pair {
			display: grid;
			grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
			gap: 6px;
			align-items: end;
		}
		.maps.pair .m {
			max-width: none;
		}
		.maps.pair .arrow {
			align-self: center;
		}
		.maps.pair .arrow svg {
			transform: none;
			margin: 0;
			width: 22px;
		}
	}
	.chans {
		margin-top: 14px;
	}
	.sub {
		margin: 0 0 6px;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.thumbs {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.th {
		display: grid;
		grid-template-columns: auto auto;
		grid-template-areas: 'map mini' 'map cs';
		gap: 2px 4px;
		align-items: center;
		padding: 3px;
		border: 1px solid var(--rule);
		border-radius: var(--r1);
		background: var(--surface);
	}
	.th :global(.mc:first-child) {
		grid-area: map;
	}
	.mini {
		grid-area: mini;
	}
	.cs {
		grid-area: cs;
		font-size: 10px;
	}
	.th.on {
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent);
	}
	.th.hl,
	.srow.hl {
		background: var(--accent-soft);
	}
	/* the arithmetic stacks by its own width (it also sits in the narrower lesson stage) */
	.arith-wrap {
		container-type: inline-size;
	}
	.arith {
		display: grid;
		grid-template-columns: minmax(0, auto) minmax(0, 1fr);
		gap: 10px 28px;
		align-items: start;
		margin: 14px 0;
	}
	@container (max-width: 800px) {
		.arith {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.mul {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px;
	}
	.op {
		font-size: var(--fs-lg);
		color: var(--muted);
		padding-top: 14px;
	}
	.sumgrp {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		white-space: nowrap;
	}
	.sum {
		padding-top: 14px;
		font-size: var(--fs-md);
	}
	.hov {
		grid-column: 1;
		margin: 0;
		font-size: var(--fs-xs);
		min-height: 1.4em;
	}
	.sums {
		grid-column: 2;
		grid-row: 1 / span 2;
		display: grid;
		gap: 1px;
		max-width: 360px;
	}
	@container (max-width: 800px) {
		.sums {
			grid-column: 1;
			grid-row: auto;
		}
	}
	.srow {
		display: grid;
		grid-template-columns: 3em 1fr 5.4em;
		align-items: center;
		gap: 8px;
		padding: 1px 4px;
		border: 0;
		border-radius: 3px;
		background: none;
		font-size: 11px;
		text-align: left;
		min-height: 18px;
	}
	button.srow {
		cursor: pointer;
	}
	.srow.on {
		box-shadow: inset 0 0 0 1.5px var(--accent);
	}
	.srow.fixed {
		border-top: 1px solid var(--rule);
	}
	.srow.total {
		font-weight: 700;
	}
	.sl {
		color: var(--muted);
	}
	.track {
		position: relative;
		height: 10px;
		background: linear-gradient(90deg, transparent calc(50% - 0.5px), var(--rule-strong) calc(50% - 0.5px), var(--rule-strong) calc(50% + 0.5px), transparent calc(50% + 0.5px));
	}
	.b {
		position: absolute;
		top: 1px;
		bottom: 1px;
		width: var(--w);
		border-radius: 2px;
	}
	.b.pos {
		left: 50%;
	}
	.b.neg {
		right: 50%;
	}
	.sv {
		text-align: right;
	}
	.check {
		margin: 8px 0 0;
		font-size: var(--fs-xs);
	}
</style>

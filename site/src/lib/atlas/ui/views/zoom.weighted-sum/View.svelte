<!--
  zoom.weighted-sum: one output number = Σ x·w (+ b), from the op's UnitExplanation (conv2d, linear,
  attention_scores, attention_mix, and any local op that explains itself as a weighted sum).
  Modeled on TinyCNN's ConvDetail and DenseDetail and TinyGPT's DotProduct:
    - a kernel (explanation with a k×k layout): input map with the window → output map with the cell (slides on its
      own unless the host scrubs a scan tape), per-input-channel thumbnails, window × kernel = products as number
      grids, the per-channel sums plus the bias as bars;
    - a plain sum (linear, attention): what feeds it (the head's score table, the output row), the terms as a table
      (x, w, x·w; long sums start with the largest products), per-group sums and the bias as bars;
    - the op's formula with live values, and the value the forward pass stored next to the recomputed one.
  Every number is read from the UnitExplanation or the trace. Hover, tap or arrow keys on the output pick another unit
  (a local choice: the lesson's selection never changes).
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import TensorStrip from '$lib/kit/TensorStrip.svelte';
	import TileStack from '$lib/kit/TileStack.svelte';
	import RoleFrame from '$lib/kit/RoleFrame.svelte';
	import Num from '../kit.parts/Num.svelte';
	import { fmt3 } from '../../../core/format.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import { motion } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../types/view.ts';
	import UnitHead from '../kit.parts/UnitHead.svelte';
	import SumBars from '../kit.parts/SumBars.svelte';
	import TermTable from '../kit.parts/TermTable.svelte';
	import OpFormula from '../kit.parts/OpFormula.svelte';
	import SpatialMaps from '../kit.parts/SpatialMaps.svelte';
	import Notice from '../kit.parts/Notice.svelte';
	import { axesOfId, domainOf, layerLabel, outLabels, resolveUnit, scanDriven, tensorOf, tokenIds, tokenTexts, unitWhere } from '../kit.parts/stage.ts';

	let P: StageProps = $props();

	/** a unit the learner picked inside this view; reset whenever the host moves the stage */
	let pick = $derived.by<number[] | null>(() => {
		void P.at;
		void P.props.layer;
		return null;
	});
	const R = $derived(resolveUnit(P, pick));
	const e = $derived(R.e);
	const opName = $derived(R.layer?.op ?? '');
	const kindOp = $derived(R.op?.def.op ?? opName);
	const id = $derived(e?.layer ?? R.id ?? '');
	const out = $derived(R.trace && id ? R.trace.t[id] : null);
	const outAxes = $derived(out ? axesOfId(R.model, id, out.shape) : []);
	const xId = $derived(e?.reads[0]?.tensor ?? '');
	const wId = $derived(e?.reads[1]?.tensor ?? '');
	const xT = $derived(R.trace && xId ? tensorOf(R.trace, xId) : null);
	const isConv = $derived(!!e?.layout && !!xT && xT.shape.length === 3 && outAxes.join() === 'c,y,x');
	const ids = $derived(tokenIds(R.trace, R.model));
	const T = $derived(out && (outAxes[0] === 't' || outAxes[1] === 'tq') ? (outAxes[0] === 't' ? out.shape[0] : out.shape[1]) : 0);
	const tokens = $derived(T ? tokenTexts(P, T, ids) : []);
	const where = $derived(e ? unitWhere(e.unit, outAxes, tokens, outLabels(P, R.model)) : null);
	const reduced = $derived(P.ctx?.reduced ?? motion.reduced);

	// ------------------------------------------------ kernels (conv2d and any k×k layout)
	const k = $derived(e?.layout?.rows ?? 0);
	const kc = $derived(e?.layout?.cols ?? 0);
	const groups = $derived(e?.groups ?? []);
	const strongest = $derived.by(() => {
		let b = 0;
		for (let c = 1; c < groups.length; c++) if (Math.abs(groups[c].sum) > Math.abs(groups[b].sum)) b = c;
		return b;
	});
	let gPick = $derived.by<number | null>(() => {
		void id;
		void e?.unit[0];
		return null;
	});
	const g = $derived(Math.min(Math.max(0, groups.length - 1), gPick ?? strongest));
	const gTerms = $derived(e?.terms && groups[g] ? e.terms.slice(groups[g].from, groups[g].to) : (e?.terms ?? []));
	const C = $derived(xT && xT.shape.length === 3 ? xT.shape[0] : 1);
	const H = $derived(xT && xT.shape.length === 3 ? xT.shape[1] : 0);
	const W = $derived(xT && xT.shape.length === 3 ? xT.shape[2] : 0);
	const inKind = $derived<'diverging' | 'sequential'>(R.model && xId && opOfId(xId) === 'input' ? 'sequential' : 'diverging');
	const inDomain = $derived(xT ? domainOf(inKind, xT.data) : ([-1, 1] as [number, number]));
	const outDomain = $derived(out ? domainOf('diverging', out.data) : ([-1, 1] as [number, number]));
	const wT = $derived(R.model && wId ? (R.model.weights[wId] ?? tensorOf(R.trace, wId)) : null);
	const wDomain = $derived(wT ? domainOf('diverging', wT.data) : domainOf('diverging', Float64Array.from(gTerms, (x) => x.w)));
	const pDomain = $derived(domainOf('diverging', Float64Array.from(e?.terms ?? [], (x) => x.p)));
	const ch = $derived(isConv && e ? e.unit[0] : 0);
	const OH = $derived(out && isConv ? out.shape[1] : 0);
	const OW = $derived(out && isConv ? out.shape[2] : 0);
	let hoverK = $state<number | null>(null);
	const hk = $derived(hoverK !== null && hoverK < gTerms.length ? hoverK : null);
	const kHl = $derived(hk === null ? [] : [{ row: Math.floor(hk / kc), col: hk % kc, color: 'var(--highlight)' }]);
	/** number cells: 44 px for 3×3 kernels (labels need the room), smaller for bigger kernels */
	const numCell = $derived(kc <= 3 ? 44 : kc <= 5 ? 40 : 32);
	const thumbCell = $derived(W ? Math.max(1, Math.floor(46 / Math.max(H, W))) : 2);

	function opOfId(tid: string): string | undefined {
		try {
			return R.model?.layer(tid.split('#')[0]).op;
		} catch {
			return undefined;
		}
	}
	const canSlide = $derived(isConv && !scanDriven(P) && R.op?.def.unit?.scan === 'raster' && P.props.slide !== false);
	$effect(() => {
		// a 'play: view' step on a stage that has nothing to animate is done at once
		if (P.autoplay && !canSlide && e) P.onend();
	});

	// ------------------------------------------------ plain sums (linear, attention)
	const roles = $derived<{ x: 'in' | 'q' | 'v'; w: 'param' | 'k' | 'prob' }>(
		kindOp === 'attention_scores' ? { x: 'q', w: 'k' } : kindOp === 'attention_mix' ? { x: 'v', w: 'prob' } : { x: 'in', w: 'param' }
	);
	// K1: props.names renames the columns ({ x: 'h', w: 'W', p: 'h·W' }): x and y clash with a lesson's own notation (flow
	// matching's x0 and x1)
	const head = $derived({
		...(kindOp === 'attention_scores'
			? { i: 'c', x: 'q', w: 'k', p: 'q·k' }
			: kindOp === 'attention_mix'
				? { i: t({ en: 'key', zh: '键' }), x: 'v', w: 'a', p: 'a·v' }
				: { i: 'i', x: 'x', w: 'w', p: 'x·w' }),
		...((P.props.names ?? {}) as Record<string, string>)
	});
	const keyTokens = $derived.by(() => {
		if (kindOp !== 'attention_mix' || !e?.terms) return [];
		return tokenTexts(P, e.terms.length, ids);
	});
	const termIdx = $derived((i: number) => {
		if (kindOp === 'attention_mix') return `${i}${keyTokens[i] ? ` ${keyTokens[i]}` : ''}`;
		if (groups.length > 1 && !isConv) {
			const gi = groups.findIndex((x) => i >= x.from && i < x.to);
			return gi >= 0 ? `c${gi}·${i - groups[gi].from}` : String(i);
		}
		return String(i);
	});
	// a flattened [C, H, W] input: the products as maps, one per channel (TinyCNN dense)
	const flatMaps = $derived.by(() => {
		if (!e?.terms || isConv || groups.length < 2 || !R.layer || !R.model) return null;
		const src = R.layer.in[0];
		let pre: string | undefined;
		try {
			const l = R.model.layer(src);
			if (l.op === 'flatten') pre = l.in[0];
		} catch {
			pre = undefined;
		}
		const pt = pre && R.trace ? R.trace.t[pre] : null;
		if (!pt || pt.shape.length !== 3 || pt.shape[0] !== groups.length) return null;
		return { c: pt.shape[0], h: pt.shape[1], w: pt.shape[2], data: Float64Array.from(e.terms, (x) => x.p), label: layerLabel(R.model, pre!) };
	});
	let mapPick = $state(0);
	// the score table of one head (attention_scores) and the output row (linear on [T, X])
	const scoreHead = $derived(kindOp === 'attention_scores' && out && e ? { h: e.unit[0], tq: out.shape[1], tk: out.shape[2] } : null);
	const scoreSlice = $derived(scoreHead && out ? out.data.subarray(scoreHead.h * scoreHead.tq * scoreHead.tk, (scoreHead.h + 1) * scoreHead.tq * scoreHead.tk) : null);
	const scoreDomain = $derived(scoreSlice ? domainOf('diverging', scoreSlice) : ([-1, 1] as [number, number]));
	const rowOut = $derived.by(() => {
		if (!out || !e || isConv || scoreHead) return null;
		const n = out.shape[out.shape.length - 1];
		const r = out.shape.length >= 2 ? e.unit[0] : 0;
		const from = out.shape.length >= 2 ? r * n : 0;
		return { data: out.data.subarray(from, from + n), n, r, rows: out.shape.length >= 2 ? out.shape[0] : 1 };
	});
	const sumRows = $derived.by(() => {
		if (!e) return [];
		if (groups.length > 1) return groups.map((x, i) => ({ label: `c${i}`, value: x.sum, on: i === g, title: x.label }));
		return [{ label: 'Σ', value: e.values.s ?? (e.terms ?? []).reduce((a, x) => a + x.p, 0) }];
	});
	const sumTotal = $derived(e ? (kindOp === 'attention_scores' ? (e.values.s ?? e.result) : e.result) : 0);
</script>

{#if !e}
	<Notice error={R.error} />
{:else}
	<div class="ws">
		<UnitHead layer={layerLabel(R.model, id)} op={R.op?.def.label} where={where ?? undefined} value={e.result} />

		{#if isConv && out}
			<SpatialMaps
				input={{
					data: xT!.data.subarray(g * H * W, (g + 1) * H * W),
					rows: H,
					cols: W,
					kind: inKind,
					domain: inDomain,
					label: { en: `Input: ${layerLabel(R.model, xId).en}${C > 1 ? ` · channel ${g}` : ''}`, zh: `输入：${layerLabel(R.model, xId).zh ?? xId}${C > 1 ? ` · 通道 ${g}` : ''}` }
				}}
				output={{
					data: out.data.subarray(ch * OH * OW, (ch + 1) * OH * OW),
					rows: OH,
					cols: OW,
					domain: outDomain,
					label: { en: `Output: channel ${ch}`, zh: `输出：通道 ${ch}` }
				}}
				box={e.reads[0]?.box ?? null}
				cell={{ row: e.unit[1], col: e.unit[2] }}
				arrow={`${k}×${kc}`}
				onpick={(r, c) => (pick = [ch, r, c])}
				slide={canSlide ? { autoplay: P.autoplay && !reduced, interval: typeof P.props.interval === 'number' ? P.props.interval : 60, onend: P.onend, title: { en: 'Slide the kernel: press Play, or hover or tap the output map', zh: '滑动卷积核：点“播放”，或在输出图上悬停、点按' } } : null}
			/>

			{#if C > 1}
				<div class="chans">
					<p class="sub">
						{t({
							en: `The same window in all ${C} input channels; each has its own ${k}×${kc} slice of the kernel. Pick one to see its numbers.`,
							zh: `同一个窗口出现在全部 ${C} 个输入通道中；每个通道都有自己的一块 ${k}×${kc} 卷积核切片。点选一个查看具体数字。`
						})}
					</p>
					<div class="thumbs" role="group" aria-label={t({ en: 'Input channels', zh: '输入通道' })}>
						{#each groups as gr, c (c)}
							<button type="button" class="th" class:on={c === g} aria-pressed={c === g} title={t(gr.label)} onclick={() => (gPick = c)}>
								<TensorGrid
									data={xT!.data.subarray(c * H * W, (c + 1) * H * W)}
									rows={H}
									cols={W}
									kind={inKind}
									domain={inDomain}
									cell={thumbCell}
									gridLines={false}
									interactive={false}
									readout={false}
									highlights={e.reads[0]?.box ? [{ row: e.reads[0].box.y0, col: e.reads[0].box.x0, h: e.reads[0].box.y1 - e.reads[0].box.y0 + 1, w: e.reads[0].box.x1 - e.reads[0].box.x0 + 1, color: 'var(--accent)' }] : []}
									label={gr.label}
								/>
								<span class="cs num" class:neg={gr.sum < 0} class:pos={gr.sum > 0}>{fmt3(gr.sum)}</span>
							</button>
						{/each}
					</div>
				</div>
			{/if}

			<div class="arith-wrap">
				<div class="arith">
					<div class="mul">
						<figure class="ng">
							<figcaption>{C > 1 ? t({ en: `window, channel ${g}`, zh: `窗口（通道 ${g}）` }) : t({ en: 'window x', zh: '窗口 x' })}</figcaption>
							<TensorGrid
								data={Float64Array.from(gTerms, (x) => x.x)}
								rows={k}
								cols={kc}
								kind={inKind}
								domain={inDomain}
								cell={numCell}
								readout={false}
								highlights={kHl}
								masked={(r, c) => gTerms[r * kc + c]?.xAt === undefined}
								label={{ en: 'window', zh: '窗口' }}
								onhover={(c) => (hoverK = c ? c.row * kc + c.col : null)}
							/>
						</figure>
						<span class="op" aria-hidden="true">×</span>
						<RoleFrame role="param" label={C > 1 ? { en: `kernel slice ${g}`, zh: `卷积核切片 ${g}` } : { en: 'kernel w', zh: '卷积核 w' }}>
							<TensorGrid
								data={Float64Array.from(gTerms, (x) => x.w)}
								rows={k}
								cols={kc}
								domain={wDomain}
								cell={numCell}
								readout={false}
								highlights={kHl}
								label={{ en: 'kernel', zh: '卷积核' }}
								onhover={(c) => (hoverK = c ? c.row * kc + c.col : null)}
							/>
						</RoleFrame>
						<span class="op" aria-hidden="true">=</span>
						<figure class="ng">
							<figcaption>{t({ en: 'products x·w', zh: '乘积 x·w' })}</figcaption>
							<TensorGrid
								data={Float64Array.from(gTerms, (x) => x.p)}
								rows={k}
								cols={kc}
								domain={pDomain}
								cell={numCell}
								readout={false}
								highlights={kHl}
								label={{ en: 'products', zh: '乘积' }}
								onhover={(c) => (hoverK = c ? c.row * kc + c.col : null)}
							/>
						</figure>
						<span class="sumgrp">
							<span class="op" aria-hidden="true">→ Σ</span>
							<span class="sum num">{#if C > 1}<span class="muted">s<sub>{g}</sub> =</span>{/if} <Num value={groups[g]?.sum ?? 0} signed /></span>
						</span>
					</div>
					<p class="hov num" aria-live="polite">
						{#if hk !== null}
							{fmt3(gTerms[hk].x)} × {fmt3(gTerms[hk].w)} = {fmt3(gTerms[hk].p)}{#if gTerms[hk].xAt === undefined}<span class="muted"> · {t({ en: 'padding: reads 0', zh: '填充：读到 0' })}</span>{/if}
						{:else}
							<span class="muted">{t({ en: 'Hover, tap or use the arrow keys on a cell to see one product.', zh: '悬停、点按或用方向键选一个格子，查看一次乘法。' })}</span>
						{/if}
					</p>
					<div class="sums">
						<SumBars
							rows={sumRows}
							bias={e.bias ?? null}
							total={e.result}
							label={{ en: 'Sums per input channel, plus the bias', zh: '各输入通道的和，加上偏置' }}
							onpick={groups.length > 1 ? (i) => (gPick = i) : undefined}
						/>
					</div>
				</div>
			</div>
		{:else}
			{#if scoreHead && scoreSlice}
				<figure class="m">
					<figcaption>
						{t({ en: `Scores of head ${scoreHead.h} (counting from 0): queries are rows, keys are columns`, zh: `头 ${scoreHead.h}（从 0 数起）的分数：行是查询，列是键` })}
						<span class="num muted">[{scoreHead.tq}×{scoreHead.tk}]</span>
					</figcaption>
					<TensorGrid
						data={scoreSlice}
						rows={scoreHead.tq}
						cols={scoreHead.tk}
						domain={scoreDomain}
						maxCell={22}
						masked={(r, c) => !Number.isFinite(scoreSlice[r * scoreHead.tk + c])}
						highlights={[{ row: e.unit[1], col: e.unit[2], color: 'var(--accent)' }]}
						label={{ en: 'attention scores', zh: '注意力分数' }}
						bind:cursor={() => ({ row: e!.unit[1], col: e!.unit[2] }), (c) => { if (c) pick = [scoreHead.h, c.row, c.col]; }}
						onhover={(c) => { if (c) pick = [scoreHead.h, c.row, c.col]; }}
					/>
				</figure>
			{:else if rowOut}
				<div class="rowout" use:keyScroll>
					<p class="cap">
						{rowOut.n === 1
							? t({ en: 'The output', zh: '这个输出' })
							: rowOut.rows > 1
								? t({ en: `The ${rowOut.n} outputs at ${tokens[rowOut.r] ? `“${tokens[rowOut.r]}”` : `position ${rowOut.r}`}: pick one to see its sum`, zh: `${tokens[rowOut.r] ? `“${tokens[rowOut.r]}”` : `位置 ${rowOut.r}`}的 ${rowOut.n} 个输出：选一个查看它的求和` })
								: t({ en: `The ${rowOut.n} outputs: pick one to see its sum`, zh: `${rowOut.n} 个输出：选一个查看它的求和` })}
					</p>
					<TensorStrip
						data={rowOut.data}
						shape={rowOut.rows > 1 ? [rowOut.n] : [rowOut.n]}
						name={rowOut.rows > 1
							? { en: `Output at ${tokens[rowOut.r] ? `“${tokens[rowOut.r]}”` : `position ${rowOut.r}`}: pick a number`, zh: `${tokens[rowOut.r] ? `“${tokens[rowOut.r]}”` : `位置 ${rowOut.r}`}的输出：选一个数` }
							: { en: 'Output: pick a number', zh: '输出：选一个数' }}
						maxCells={32}
						domain={domainOf('diverging', rowOut.data)}
						selected={e.unit[e.unit.length - 1]}
						onselect={(i) => (pick = rowOut.rows > 1 ? [rowOut.r, i] : [i])}
					/>
				</div>
			{/if}

			{#if flatMaps}
				<figure class="m">
					<figcaption>
						{t({ en: `Each product x·w at its place in ${flatMaps.label.en}: orange pushes this output up, blue pushes it down`, zh: `每个乘积 x·w 放回它在“${flatMaps.label.zh ?? flatMaps.label.en}”中的位置：橙色把输出推高，蓝色把它压低` })}
					</figcaption>
					<TileStack
						data={flatMaps.data}
						channels={flatMaps.c}
						height={flatMaps.h}
						width={flatMaps.w}
						tile={56}
						scopeDomains={{ layer: pDomain }}
						channelLabels={Array.from({ length: flatMaps.c }, (_, c) => String(c))}
						bind:selected={mapPick}
						label={{ en: 'products per channel', zh: '每个通道的乘积' }}
					/>
				</figure>
			{/if}

			<div class="plain">
				<TermTable
					terms={e.terms ?? []}
					idx={termIdx}
					{head}
					{roles}
					kinds={{ w: roles.w === 'prob' ? 'sequential' : 'diverging' }}
					sum={e.values.s ?? (e.terms ?? []).reduce((a, x) => a + x.p, 0)}
					caption={{ en: 'The products that add up to this output', zh: '加起来得到这个输出的各个乘积' }}
				/>
				<div class="side">
					{#if kindOp === 'attention_scores'}
						<p class="scal num">
							Σ q·k = <Num value={e.values.s} /> <span class="muted">→</span> × {fmt3(e.values.scale)} = <strong><Num value={e.result} /></strong>
							{#if e.values.masked}<span class="muted"> · {t({ en: 'masked: this key is in the future', zh: '被遮住：这个键在未来' })}</span>{/if}
						</p>
					{:else}
						<SumBars
							rows={sumRows}
							bias={e.bias ?? null}
							total={sumTotal}
							label={e.bias !== undefined && e.bias !== null ? { en: 'The sum, plus the bias', zh: '求和，再加上偏置' } : { en: 'The sum', zh: '求和' }}
						/>
					{/if}
				</div>
			</div>
		{/if}

		<OpFormula
			tex={R.op?.def.formula}
			{e}
			caption={where ? { en: `${layerLabel(R.model, id).en}, ${where.en}`, zh: `${layerLabel(R.model, id).zh ?? id}，${where.zh}` } : undefined}
			provenance={P.provenance}
		/>
	</div>
{/if}

<style>
	.ws {
		display: grid;
		gap: 12px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.m {
		margin: 0;
		display: grid;
		gap: 4px;
		min-width: 0;
		max-width: 520px;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
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
		justify-items: center;
		gap: 2px;
		padding: 3px;
		border: 1px solid var(--rule);
		border-radius: var(--r1);
		background: var(--surface);
		cursor: pointer;
	}
	.th.on {
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent);
	}
	.cs {
		font-size: 10px;
		color: var(--muted);
	}
	.cs.neg {
		color: var(--div-neg-text);
	}
	.cs.pos {
		color: var(--div-pos-text);
	}
	.arith-wrap {
		container-type: inline-size;
	}
	.arith {
		display: grid;
		grid-template-columns: minmax(0, auto) minmax(0, 1fr);
		gap: 10px 28px;
		align-items: start;
	}
	.mul {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px;
		min-width: 0;
	}
	.ng {
		margin: 0;
		display: grid;
		gap: 3px;
	}
	.ng figcaption {
		font-size: 11px;
		color: var(--muted);
	}
	.op {
		font-size: var(--fs-lg);
		color: var(--muted);
	}
	.sumgrp {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		white-space: nowrap;
	}
	.sum {
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
	}
	/* narrow stages (phones, the lesson's stage column): one column, the sums under the products */
	@container (max-width: 760px) {
		.arith {
			grid-template-columns: minmax(0, 1fr);
		}
		.sums {
			grid-column: 1;
			grid-row: auto;
		}
	}
	.cap {
		margin: 0 0 4px;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.rowout {
		min-width: 0;
		max-width: 100%;
		overflow-x: auto;
	}
	.plain {
		display: grid;
		grid-template-columns: minmax(0, auto) minmax(0, 1fr);
		gap: 12px 24px;
		align-items: start;
		container-type: inline-size;
	}
	@media (max-width: 640px) {
		.plain {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.side {
		min-width: 0;
	}
	.scal {
		margin: 0;
		font-size: var(--fs-sm);
	}
</style>

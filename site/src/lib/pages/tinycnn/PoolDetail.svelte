<!--
  Max-pool zoom: each output cell keeps the largest of a 2x2 block (model.poolDetail, live).
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import NumGrid from './NumGrid.svelte';
	import { sym } from '$lib/kit/tex';
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import { channelDomain } from './data';
	import type { CnnStore, SelView } from './store.svelte';

	type Part = 'slide' | 'formula' | 'note';
	interface Props {
		S: CnnStore;
		/** whose selection to show (default: the page store) */
		view?: SelView;
		/** blocks to show (default: all) */
		parts?: Part[];
		/** start sliding the 2×2 window when shown */
		autoplay?: boolean;
		/** ms per window position */
		interval?: number;
		/** while sliding, hatch the output cells the window has not reached yet */
		fill?: boolean;
		onend?: () => void;
		/** heading of the slide player (default: "Slide the 2×2 window: press Play, …") */
		title?: Text;
		/** keep the input and output maps side by side on phones (the guided lesson) */
		pair?: boolean;
	}
	let { S, view, parts = ['slide', 'formula', 'note'], autoplay = false, interval = 260, fill = false, onend, title, pair = false }: Props = $props();
	const V = $derived(view ?? S);
	const show = (p: Part) => parts.includes(p);
	let playing = $state(false);

	const L = $derived(S.layer(V.sel.layer));
	const inL = $derived(S.layer(L.in[0]));
	const OH = $derived(L.outShape[1]);
	const OW = $derived(L.outShape[2]);
	const IH = $derived(inL.outShape[1]);
	const IW = $derived(inL.outShape[2]);
	const d = $derived(S.model && S.trace ? S.model.poolDetail(S.trace, L.id, V.sel.ch, V.sel.y, V.sel.x) : null);
	const inSlice = $derived(S.trace ? S.trace.t[inL.id].data.subarray(V.sel.ch * IH * IW, (V.sel.ch + 1) * IH * IW) : null);
	const outSlice = $derived(S.trace ? S.trace.t[L.id].data.subarray(V.sel.ch * OH * OW, (V.sel.ch + 1) * OH * OW) : null);
	const dom = $derived(S.domainFor(inL.id, () => (S.trace ? channelDomain(S.trace.t[inL.id].data, V.sel.ch, IH * IW) : [-1, 1])));
	const outDom = $derived(S.scope === 'unit' ? dom : S.domainFor(L.id, () => dom));
	let hoverK = $state<number | null>(null);

	function hover(c: { row: number; col: number } | null) {
		if (!c) {
			S.hoverRF = null;
			return;
		}
		V.setCell(c.row, c.col);
		S.hoverRF = S.receptiveField(L.id, c.row, c.col);
	}
</script>

{#if d && inSlice && outSlice}
	{#if show('slide')}
	<StepPlayer
		count={OH * OW}
		bind:step={() => V.sel.y * OW + V.sel.x, (v) => V.setCell(Math.floor(v / OW), v % OW)}
		bind:playing
		{autoplay}
		{interval}
		{onend}
		speeds={[0.5, 1, 3]}
		label={title ?? t({ en: 'Slide the 2×2 window: press Play, or hover or tap the output map', zh: '滑动 2×2 窗口：点“播放”，或在输出图上悬停、点按' })}
		position={(i, n) => t({ en: `Position ${i} of ${n}`, zh: `位置 ${i} / ${n}` })}
	>
		{#snippet children()}
			<div class="maps" class:pair>
				<figure>
					<figcaption>{t({ en: 'Input:', zh: '输入：' })} {t(inL.label)} · {t({ en: 'channel', zh: '通道' })} {V.sel.ch} <span class="num muted">[{IH}×{IW}]</span></figcaption>
					<TensorGrid
						data={inSlice}
						rows={IH}
						cols={IW}
						domain={dom}
						maxCell={18}
						interactive={false}
						readout={false}
						highlights={[{ row: d.window.y0, col: d.window.x0, h: 2, w: 2, color: 'var(--accent)' }, { row: d.argmaxAt[0], col: d.argmaxAt[1], style: 'dashed' }]}
						label={t(inL.label)}
					/>
				</figure>
				<div class="mid">
					<NumGrid values={d.values} rows={2} cols={2} domain={dom} label={t({ en: '2×2 window', zh: '2×2 窗口' })} mark={d.argmax} bind:hover={hoverK} />
					<span class="arrow" aria-hidden="true">max →</span>
				</div>
				<figure>
					<figcaption>{t({ en: 'Output:', zh: '输出：' })} {t(L.label)} <span class="num muted">[{OH}×{OW}]</span></figcaption>
					<TensorGrid
						data={outSlice}
						rows={OH}
						cols={OW}
						domain={outDom}
						maxCell={OH <= 4 ? 40 : 22}
						highlights={[{ row: V.sel.y, col: V.sel.x, color: 'var(--accent)' }]}
						masked={fill && playing ? (r, c) => r * OW + c > V.sel.y * OW + V.sel.x : undefined}
						label={t(L.label)}
						bind:cursor={() => ({ row: V.sel.y, col: V.sel.x }), (c) => { if (c) V.setCell(c.row, c.col); }}
						onhover={hover}
					/>
				</figure>
			</div>
		{/snippet}
	</StepPlayer>
	{/if}
	{#if show('formula')}
	<FormulaCard
		tex={`\\begin{aligned} ${sym('out', 'p_{ij}')} &= \\max_{u,v \\in \\{0,1\\}} ${sym('in', 'a_{2i+u,\\,2j+v}')} \\\\ &= \\max({{a}}, {{b}}, {{c}}, {{d}}) \\\\ &= {{p}} \\end{aligned}`}
		values={{ a: d.values[0], b: d.values[1], c: d.values[2], d: d.values[3], p: d.value }}
		provenance="live"
		caption={t({ en: `Cell (${V.sel.y}, ${V.sel.x}): no weights, nothing learned`, zh: `格子 (${V.sel.y}, ${V.sel.x})：没有权重，不需要学习` })}
	/>
	{/if}
	{#if show('note')}
	<p class="note">
		{t({
			en: 'Pooling halves the width and height. It keeps "was the pattern somewhere in this 2×2 block?" and drops exactly where, so a stroke can shift by a pixel without changing much downstream.',
			zh: '池化把宽和高都减半。它保留“这个 2×2 区块里是否出现过这种模式”，丢掉具体位置，所以笔画挪动一个像素，后面的层几乎不受影响。'
		})}
	</p>
	{/if}
{/if}

<style>
	.maps {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 12px 18px;
	}
	figure {
		margin: 0;
		display: grid;
		gap: 4px;
		flex: 1 1 200px;
		max-width: 440px;
		min-width: 0;
	}
	/* the lesson (pair): inside the lesson app's stage pane (a size container) the maps leave room for the formula under
	   them; elsewhere 100cqh is the viewport height */
	.maps.pair figure {
		max-width: clamp(190px, calc(100cqh - 400px), 440px);
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.mid {
		display: grid;
		justify-items: center;
		gap: 4px;
	}
	/* pair (phones): input and output side by side, the 2×2 window below them */
	@media (max-width: 560px) {
		.maps.pair {
			display: grid;
			grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
			gap: 8px 10px;
			align-items: end;
		}
		.maps.pair figure {
			max-width: none;
		}
		.maps.pair .mid {
			grid-column: 1 / -1;
			grid-row: 2;
			display: flex;
			align-items: center;
			justify-content: center;
			gap: 10px;
		}
	}
	.arrow {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.note {
		margin: 8px 0 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		max-width: var(--measure);
	}
</style>

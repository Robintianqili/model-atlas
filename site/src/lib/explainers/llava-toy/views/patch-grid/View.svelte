<!--
  llava-toy/patch-grid (adapted from the tinyvit package's patch-grid): the answer row's attention in one LM block. Its
  weight on each image token is a shade over the patch that token came from, with the share printed in the patch; its
  weights on the question's words are bars beside the drawing. Buttons pick the mean of the heads or one head (and, with
  props.pickBlock, the block). Nothing animates.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmtPct } from '../../../../atlas/core/format.ts';
	import { makeScale } from '$lib/kit/color';
	import Notice from '../../../../atlas/ui/views/kit.parts/Notice.svelte';
	import { GRID, N_IMG, PATCH, SIDE, gridOf } from './grid.ts';

	let P: StageProps = $props();
	let headPick = $derived.by<number | null>(() => {
		void P.at;
		return null;
	});
	let blockPick = $derived.by<number | null>(() => {
		void P.at;
		return null;
	});
	const G = $derived(gridOf(P, { head: headPick, block: blockPick }));
	const scale = $derived(makeScale('sequential', [0, G ? Math.max(...G.w, ...G.words.map((x) => x.w), 1e-9) : 1]));
	const S = 10; // svg units per pixel
	const pr = (k: number) => Math.floor(k / GRID);
	const pc = (k: number) => k % GRID;
	$effect(() => {
		if (P.autoplay && G) P.onend();
	});
</script>

{#if !G}
	<Notice text={{ en: 'The answer row appears once the model has run on your input.', zh: '模型在你的输入上运行之后，这里会显示写答案的那一行。' }} />
{:else}
	<div class="pg">
		<div class="ctl">
			{#if P.props.pickBlock && G.blocks > 1}
				<div class="segmented" role="group" aria-label={t({ en: 'LM block', zh: '语言模型的块' })}>
					{#each Array.from({ length: G.blocks }, (_, b) => b + 1) as b (b)}
						<button type="button" aria-pressed={b === G.block} onclick={() => (blockPick = b)}>{t({ en: `block ${b}`, zh: `第 ${b} 块` })}</button>
					{/each}
				</div>
			{/if}
			<div class="segmented" role="group" aria-label={t({ en: 'Head', zh: '头' })}>
				<button type="button" aria-pressed={G.head < 0} onclick={() => (headPick = -1)}>{t({ en: 'mean of heads', zh: '各头平均' })}</button>
				{#each Array.from({ length: G.heads }, (_, h) => h) as h (h)}
					<button type="button" aria-pressed={h === G.head} onclick={() => (headPick = h)}>{t({ en: `head ${h}`, zh: `头 ${h}` })}</button>
				{/each}
			</div>
		</div>
		<div class="row">
			<svg viewBox="0 0 {SIDE * S} {SIDE * S}" class="img" role="img" aria-label={t({ en: 'Your drawing, each patch shaded by the weight of its image token', zh: '你的画，每个图块按它的图像词元的权重上色' })}>
				<rect width={SIDE * S} height={SIDE * S} class="paper" />
				{#each G.inked as _, k (k)}
					<rect x={pc(k) * PATCH * S} y={pr(k) * PATCH * S} width={PATCH * S} height={PATCH * S} fill={scale.css(G.w[k])} />
				{/each}
				{#each Array.from(G.img) as v, i (i)}
					{#if v > 0.02}<rect x={(i % SIDE) * S} y={Math.floor(i / SIDE) * S} width={S} height={S} class="px" fill-opacity={0.75 * v} />{/if}
				{/each}
				{#each G.inked as ink, k (k)}
					{@const x = pc(k) * PATCH * S}
					{@const y = pr(k) * PATCH * S}
					<rect {x} {y} width={PATCH * S} height={PATCH * S} class="cell" />
					<text x={x + PATCH * S - 6} y={y + PATCH * S - 8} class="pct" text-anchor="end">{fmtPct(G.w[k], 0)}</text>
					<text x={x + 6} y={y + 18} class="tok">▦{k}</text>
					{#if ink}<circle cx={x + PATCH * S - 10} cy={y + 12} r="5" class="mark" />{/if}
					{#if k === G.top}<rect x={x + 2} y={y + 2} width={PATCH * S - 4} height={PATCH * S - 4} class="top" />{/if}
				{/each}
			</svg>
			<div class="side">
				<dl class="facts">
					<div><dt>{t({ en: `${N_IMG} image tokens`, zh: `${N_IMG} 个图像词元` })}</dt><dd class="num">{fmtPct(G.imgShare)}</dd></div>
					<div><dt>{t({ en: '… on patches with ink (●)', zh: '…… 其中有墨的图块（●）' })}</dt><dd class="num">{fmtPct(G.inkShare)}</dd></div>
				</dl>
				<ul class="words" aria-label={t({ en: 'Weights on the words', zh: '各个词得到的权重' })}>
					{#each G.words as x, j (j)}
						<li><span class="w">{x.name}</span><span class="bar"><span style:width="{(100 * x.w) / Math.max(...G.words.map((u) => u.w), G.w[G.top], 1e-9)}%" style:background={scale.css(x.w)}></span></span><span class="num">{fmtPct(x.w)}</span></li>
					{/each}
				</ul>
			</div>
		</div>
		<p class="cap muted">
			{t({
				en: `LM block ${G.block}, ${G.head < 0 ? 'mean of the heads' : `head ${G.head} (counting from 0)`}: the weights of the last row, which writes the answer. A stronger color means more; the dashed patch gets the most; all ${N_IMG + G.words.length} weights add up to 100%.`,
				zh: `语言模型第 ${G.block} 块，${G.head < 0 ? '各头平均' : `头 ${G.head}（从 0 数起）`}：最后一行的权重，答案由这一行写出。颜色越浓，得到的越多；虚线框的图块得到最多；全部 ${N_IMG + G.words.length} 个权重加起来是 100%。`
			})}
		</p>
	</div>
{/if}

<style>
	.pg { display: grid; gap: 10px; grid-template-columns: minmax(0, 1fr); min-width: 0; }
	.ctl { display: flex; flex-wrap: wrap; gap: 8px; }
	.row { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr)); gap: 14px; align-items: start; }
	.img { width: 100%; max-width: 300px; }
	.side { display: grid; gap: 10px; min-width: 0; }
	.cap { font-size: var(--fs-xs); margin: 4px 0 0; }
	.paper { fill: var(--surface-2); }
	.px { fill: var(--ink); }
	.cell { fill: none; stroke: var(--ink-2); stroke-width: 1.5; }
	.mark { fill: var(--accent); stroke: var(--surface); stroke-width: 1.5; }
	.top { fill: none; stroke: var(--accent); stroke-width: 4; stroke-dasharray: 8 5; }
	.pct, .tok { font-size: 16px; font-family: var(--font-mono); fill: var(--ink); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; }
	.tok { font-size: 13px; }
	.facts { display: grid; gap: 4px; margin: 0; font-size: var(--fs-sm); }
	.facts div { display: flex; justify-content: space-between; gap: 12px; }
	.facts dd { margin: 0; }
	.words { list-style: none; margin: 0; padding: 0; display: grid; gap: 4px; font-size: var(--fs-sm); }
	.words li { display: grid; grid-template-columns: 5em minmax(0, 1fr) 4em; gap: 8px; align-items: center; }
	.w { font-family: var(--font-mono); }
	.bar { height: 12px; background: var(--surface-2); border-radius: 2px; overflow: hidden; }
	.bar span { display: block; height: 100%; }
	.num { text-align: right; font-variant-numeric: tabular-nums; }
</style>

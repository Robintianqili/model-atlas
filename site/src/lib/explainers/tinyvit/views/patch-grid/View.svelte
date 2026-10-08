<!--
  tinyvit/patch-grid: the 28×28 drawing cut into 4×4 patches of 7×7 pixels.
  props.mode 'patches': the current patch (a scan over the patch convolution, or props.patch) outlined, its 49 pixels
  beside it. props.mode 'attention': one attention row (block props.block, head props.head, query props.query = 0, the
  [CLS] row) as a shade over each patch, with its share printed in the patch; head (and, with props.pickBlock, block)
  buttons switch the row. Nothing animates by itself; a scan step's scrubber moves the patch.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmtPct } from '../../../../atlas/core/format.ts';
	import Notice from '../../../../atlas/ui/views/kit.parts/Notice.svelte';
	import { GRID, PATCH, SIDE, gridOf } from './grid.ts';

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
	const max = $derived(G && G.w.length ? Math.max(...G.w, 1e-9) : 1);
	const S = 10; // svg units per pixel
	const pr = (k: number) => Math.floor(k / GRID);
	const pc = (k: number) => k % GRID;
	$effect(() => {
		if (P.autoplay && G) P.onend();
	});
</script>

{#if !G}
	<Notice text={{ en: 'Draw something first: the patches come from your drawing.', zh: '先画点什么：图块来自你的画。' }} />
{:else}
	<div class="pg">
		{#if G.mode === 'attention'}
			<div class="ctl">
				{#if P.props.pickBlock && G.blocks > 1}
					<div class="segmented" role="group" aria-label={t({ en: 'Block', zh: '块' })}>
						{#each Array.from({ length: G.blocks }, (_, b) => b + 1) as b (b)}
							<button type="button" aria-pressed={b === G.block} onclick={() => (blockPick = b)}>{t({ en: `block ${b}`, zh: `第 ${b} 块` })}</button>
						{/each}
					</div>
				{/if}
				<div class="segmented" role="group" aria-label={t({ en: 'Head', zh: '头' })}>
					{#each Array.from({ length: G.heads }, (_, h) => h) as h (h)}
						<button type="button" aria-pressed={h === G.head} onclick={() => (headPick = h)}>{t({ en: `head ${h}`, zh: `头 ${h}` })}</button>
					{/each}
				</div>
			</div>
		{/if}
		<div class="row">
			<svg viewBox="0 0 {SIDE * S} {SIDE * S}" class="img" role="img" aria-label={t({ en: 'Your drawing cut into 16 patches', zh: '你的画被切成 16 个图块' })}>
				<rect width={SIDE * S} height={SIDE * S} class="paper" />
				{#each Array.from(G.img) as v, i (i)}
					{#if v > 0.02}<rect x={(i % SIDE) * S} y={Math.floor(i / SIDE) * S} width={S} height={S} class="px" fill-opacity={v} />{/if}
				{/each}
				{#each G.inked as ink, k (k)}
					{@const x = pc(k) * PATCH * S}
					{@const y = pr(k) * PATCH * S}
					{#if G.mode === 'attention'}
						<rect {x} {y} width={PATCH * S} height={PATCH * S} class="att" fill-opacity={(0.8 * G.w[k]) / max} />
						<text x={x + PATCH * S - 6} y={y + PATCH * S - 8} class="pct" text-anchor="end">{fmtPct(G.w[k], 0)}</text>
						{#if k === G.top}<rect x={x + 2} y={y + 2} width={PATCH * S - 4} height={PATCH * S - 4} class="top" />{/if}
					{:else if k !== G.patch}
						<rect {x} {y} width={PATCH * S} height={PATCH * S} class="dim" />
					{/if}
					<rect {x} {y} width={PATCH * S} height={PATCH * S} class="cell" />
					{#if ink}<circle cx={x + 9} cy={y + 9} r="5" class="mark" />{/if}
				{/each}
				{#if G.mode === 'patches'}
					<rect x={pc(G.patch) * PATCH * S + 2} y={pr(G.patch) * PATCH * S + 2} width={PATCH * S - 4} height={PATCH * S - 4} class="cur" />
				{/if}
			</svg>
			{#if G.mode === 'patches'}
				<figure class="zoom">
					<svg viewBox="0 0 {PATCH * 30} {PATCH * 30}" role="img" aria-label={t({ en: 'The 49 pixels of this patch', zh: '这个图块的 49 个像素' })}>
						<rect width={PATCH * 30} height={PATCH * 30} class="paper" />
						{#each G.pixels as v, i (i)}
							<rect x={(i % PATCH) * 30} y={Math.floor(i / PATCH) * 30} width="30" height="30" class="px" fill-opacity={v} />
							<text x={(i % PATCH) * 30 + 15} y={Math.floor(i / PATCH) * 30 + 19} class="v" class:light={v > 0.55} text-anchor="middle">{v.toFixed(1)}</text>
						{/each}
					</svg>
					<figcaption>
						{t({
							en: `Patch ${G.patch} (row ${pr(G.patch)}, column ${pc(G.patch)}, counting from 0): 7 × 7 = 49 numbers`,
							zh: `第 ${G.patch} 个图块（第 ${pr(G.patch)} 行、第 ${pc(G.patch)} 列，从 0 数起）：7 × 7 = 49 个数`
						})}
					</figcaption>
				</figure>
			{:else}
				<dl class="facts">
					<dt>{t({ en: '[CLS] row → itself', zh: '[CLS] 行 → 它自己' })}</dt>
					<dd class="num">{fmtPct(G.self)}</dd>
					<dt>{t({ en: 'on patches with ink (●)', zh: '落在有墨的图块上（●）' })}</dt>
					<dd class="num">{fmtPct(G.inkShare)}</dd>
					<dt>{t({ en: 'largest share: patch', zh: '份额最大的图块' })}</dt>
					<dd class="num">{G.top} · {fmtPct(G.w[G.top])}</dd>
				</dl>
			{/if}
		</div>
		<p class="cap muted">
			{#if G.mode === 'attention'}
				{t({
					en: `Block ${G.block}, head ${G.head} (counting from 0): the weights of the [CLS] row. Darker patches get more; all 17 weights add up to 100%.`,
					zh: `第 ${G.block} 块、头 ${G.head}（从 0 数起）：[CLS] 行的权重。颜色越深，得到的越多；17 个权重加起来是 100%。`
				})}
			{:else}
				{t({ en: '● marks a patch with ink.', zh: '● 表示有墨的图块。' })}
			{/if}
		</p>
	</div>
{/if}

<style>
	.pg {
		display: grid;
		gap: 10px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.ctl {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.row {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr));
		gap: 14px;
		align-items: start;
	}
	.img {
		width: 100%;
		max-width: 300px;
	}
	.zoom {
		margin: 0;
		max-width: 220px;
	}
	.zoom figcaption,
	.cap {
		font-size: var(--fs-xs);
		margin: 4px 0 0;
	}
	.paper {
		fill: var(--surface-2);
	}
	.px {
		fill: var(--ink);
	}
	.att {
		fill: var(--q);
	}
	.dim {
		fill: var(--surface);
		fill-opacity: 0.5;
	}
	.cell {
		fill: none;
		stroke: var(--ink-2);
		stroke-width: 1.5;
	}
	.mark {
		fill: var(--accent);
		stroke: var(--surface);
		stroke-width: 1.5;
	}
	.cur,
	.top {
		fill: none;
		stroke: var(--accent);
		stroke-width: 4;
	}
	.top {
		stroke-dasharray: 8 5;
	}
	.pct {
		font-size: 16px;
		font-family: var(--font-mono);
		fill: var(--ink);
		paint-order: stroke;
		stroke: var(--surface);
		stroke-width: 3px;
	}
	.v {
		font-size: 10px;
		font-family: var(--font-mono);
		fill: var(--ink);
	}
	.v.light {
		fill: var(--surface);
	}
	.facts {
		display: grid;
		grid-template-columns: auto auto;
		gap: 4px 12px;
		margin: 0;
		font-size: var(--fs-sm);
		align-content: start;
	}
	.facts dd {
		margin: 0;
	}
</style>

<!--
  Live training curves of the new head: cross-entropy loss, and accuracy on the training examples
  and (for the preset pair) on the 200 held-out test drawings. Two charts, one y-axis each.
  Hover/drag scrubs the training step (linked to the StepPlayer, the network view and the map).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { linear, ticks, linePath, markerPath, clamp } from './chart';

	interface Props {
		loss: number[];
		trainAcc: number[];
		testAcc: number[] | null;
		/** planned number of steps (x-axis length) */
		total: number;
		epoch?: number;
		busy?: boolean;
	}
	let { loss, trainAcc, testAcc, total, epoch = $bindable(0), busy = false }: Props = $props();

	let w = $state(320);
	const H = 150;
	const M = { l: 40, r: 64, t: 12, b: 28 };
	const x = $derived(linear([0, Math.max(1, total)], [M.l, Math.max(M.l + 40, w - M.r)]));
	const lossMax = $derived(Math.max(0.75, ...loss.slice(0, 1)));
	const yL = $derived(linear([0, lossMax], [H - M.b, M.t]));
	const yA = $derived(linear([0.4, 1], [H - M.b, M.t]));
	const xt = $derived(ticks(0, total, w < 360 ? 2 : 4));

	// Mouse hover previews a step and leaving restores the pinned one; a click or a touch drag pins it.
	let pinned: number | null = null;
	function scrub(e: PointerEvent, svg: SVGSVGElement) {
		if (busy || loss.length < 2) return;
		const r = svg.getBoundingClientRect();
		const px = ((e.clientX - r.left) / r.width) * w;
		if (pinned === null) pinned = epoch;
		epoch = clamp(Math.round(x.invert(px)), 0, loss.length - 1);
		if (e.type === 'pointerdown' || e.pointerType !== 'mouse') pinned = epoch;
	}
	function leave() {
		if (pinned !== null && !busy) epoch = pinned;
		pinned = null;
	}
	const pts = (arr: number[], y: (v: number) => number) => arr.map((v, i) => [x(i), y(v)] as [number, number]);
	const ep = $derived(clamp(epoch, 0, Math.max(0, loss.length - 1)));
</script>

<div class="charts" bind:clientWidth={w}>
	{#snippet frame(kind: 'loss' | 'acc')}
		<svg
			viewBox="0 0 {w} {H}"
			width={w}
			height={H}
			role="img"
			aria-label={kind === 'loss'
				? t({ en: `Training loss by step; now ${loss[ep]?.toFixed(3) ?? '–'} at step ${ep}`, zh: `各步的训练损失；第 ${ep} 步为 ${loss[ep]?.toFixed(3) ?? '–'}` })
				: t({ en: `Accuracy by step; at step ${ep}: training ${((trainAcc[ep] ?? 0) * 100).toFixed(0)}%${testAcc ? `, test ${((testAcc[ep] ?? 0) * 100).toFixed(1)}%` : ''}`, zh: `各步的准确率；第 ${ep} 步：训练集 ${((trainAcc[ep] ?? 0) * 100).toFixed(0)}%${testAcc ? `，测试集 ${((testAcc[ep] ?? 0) * 100).toFixed(1)}%` : ''}` })}
			onpointermove={(e) => scrub(e, e.currentTarget)}
			onpointerdown={(e) => scrub(e, e.currentTarget)}
			onpointerleave={leave}
		>
			{#if kind === 'loss'}
				{#each ticks(0, lossMax, 3) as v (v)}
					<line class="grid" x1={M.l} x2={w - M.r} y1={yL(v)} y2={yL(v)} />
					<text class="tick" x={M.l - 6} y={yL(v) + 4} text-anchor="end">{v.toFixed(1)}</text>
				{/each}
			{:else}
				{#each [0.4, 0.6, 0.8, 1] as v (v)}
					<line class="grid" x1={M.l} x2={w - M.r} y1={yA(v)} y2={yA(v)} />
					<text class="tick" x={M.l - 6} y={yA(v) + 4} text-anchor="end">{(v * 100).toFixed(0)}%</text>
				{/each}
			{/if}
			{#each xt as v (v)}
				<text class="tick" x={x(v)} y={H - 8} text-anchor="middle">{v}</text>
			{/each}
			<line class="axis" x1={M.l} x2={w - M.r} y1={H - M.b} y2={H - M.b} />
			{#if loss.length > 1}
				<line class="cursor" x1={x(ep)} x2={x(ep)} y1={M.t} y2={H - M.b} />
			{/if}
			{#if kind === 'loss'}
				<path class="ln loss" d={linePath(pts(loss, yL))} />
				{#if loss.length}
					<path class="mk loss" d={markerPath('circle', 4)} transform="translate({x(ep)},{yL(loss[ep])})" />
					<text class="lab" x={x(loss.length - 1) + 8} y={yL(loss[loss.length - 1]) + 4}>{loss[loss.length - 1].toFixed(3)}</text>
				{/if}
			{:else}
				<path class="ln tr" d={linePath(pts(trainAcc, yA))} />
				{#if testAcc}<path class="ln te" d={linePath(pts(testAcc, yA))} />{/if}
				{#if trainAcc.length}
					<path class="mk tr" d={markerPath('circle', 4)} transform="translate({x(ep)},{yA(trainAcc[ep])})" />
					{#if trainAcc.length > 1}<text class="lab" x={x(trainAcc.length - 1) + 8} y={yA(trainAcc[trainAcc.length - 1]) + (testAcc ? -4 : 4)}>{t({ en: 'train', zh: '训练' })}</text>{/if}
				{/if}
				{#if testAcc && testAcc.length}
					<path class="mk te" d={markerPath('square', 4)} transform="translate({x(ep)},{yA(testAcc[ep])})" />
					{#if testAcc.length > 1}<text class="lab" x={x(testAcc.length - 1) + 8} y={yA(testAcc[testAcc.length - 1]) + 12}>{t({ en: 'test', zh: '测试' })}</text>{/if}
				{/if}
			{/if}
		</svg>
	{/snippet}
	<figure>
		<figcaption>{t({ en: 'Training loss (cross-entropy)', zh: '训练损失（交叉熵）' })}</figcaption>
		{@render frame('loss')}
	</figure>
	<figure>
		<figcaption>
			{t({ en: 'Accuracy', zh: '准确率' })}
			<span class="lg"><svg width="14" height="10" aria-hidden="true"><path class="mk tr" d={markerPath('circle', 4)} transform="translate(7,5)" /></svg>{t({ en: 'training examples', zh: '训练样本' })}</span>
			{#if testAcc}<span class="lg"><svg width="14" height="10" aria-hidden="true"><path class="mk te" d={markerPath('square', 4)} transform="translate(7,5)" /></svg>{t({ en: '200 held-out test drawings', zh: '200 张留出的测试图' })}</span>{/if}
		</figcaption>
		{@render frame('acc')}
	</figure>
	<p class="xl muted">{t({ en: 'x-axis: training step (one step = one pass over all the training examples). Hover or drag to scrub.', zh: '横轴：训练步数（一步 = 把所有训练样本过一遍）。悬停或拖动可以回看。' })}</p>
</div>

<style>
	.charts {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 6px;
		min-width: 0;
	}
	figure {
		margin: 0;
	}
	figcaption {
		font-size: var(--fs-sm);
		font-weight: 600;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 12px;
		align-items: center;
	}
	.lg {
		font-weight: 400;
		color: var(--ink-2);
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: var(--fs-xs);
	}
	svg {
		display: block;
		max-width: 100%;
		height: auto;
		touch-action: pan-y;
		cursor: ew-resize;
	}
	.grid {
		stroke: var(--rule);
		stroke-width: 1;
	}
	.axis {
		stroke: var(--rule-strong);
		stroke-width: 1;
	}
	.tick {
		font-size: 11px;
		fill: var(--muted);
		font-family: var(--font-mono);
	}
	.lab {
		font-size: 11px;
		fill: var(--ink-2);
		font-family: var(--font-mono);
	}
	.cursor {
		stroke: var(--ink-2);
		stroke-width: 1;
		stroke-dasharray: 3 3;
	}
	.ln {
		fill: none;
		stroke-width: 2;
		stroke-linejoin: round;
		stroke-linecap: round;
	}
	.ln.loss {
		stroke: var(--grad);
	}
	.ln.tr {
		stroke: var(--muted);
	}
	.ln.te {
		stroke: var(--accent);
	}
	.mk {
		stroke: var(--surface);
		stroke-width: 2;
		paint-order: stroke;
	}
	.mk.loss {
		fill: var(--grad);
	}
	.mk.tr {
		fill: var(--muted);
	}
	.mk.te {
		fill: var(--accent);
	}
	.xl {
		margin: 0;
		font-size: var(--fs-xs);
	}
</style>

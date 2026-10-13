<!--
  Training loss curve (curve.json: train + validation loss over 72 evaluation points), the 10 saved
  checkpoints as dots on the validation line, the selected checkpoint as a marker, and the loss of
  the learner's own text at every checkpoint that has been loaded (live). Hover shows a crosshair
  and the values; clicking a checkpoint dot selects it.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtCount } from '$lib/kit/format';

	type Pt = { step: number; tokens: number; train: number; val: number };
	interface Props {
		points: Pt[];
		ckpts: { step: number; val: number }[];
		selected: number;
		yours: { step: number; loss: number }[];
		xMode: 'log' | 'linear';
		uniform: number;
		onselect: (step: number) => void;
	}
	let { points, ckpts, selected, yours, xMode, uniform, onselect }: Props = $props();

	let W = $state(640);
	const H = $derived(W < 480 ? 230 : 280);
	const m = $derived({ l: 40, r: W < 480 ? 12 : 18, t: 12, b: 40 });
	const lastStep = $derived(points.length ? points[points.length - 1].step : 30000);
	const L0 = Math.log10(10);
	const L1 = $derived(Math.log10(lastStep + 10));

	const yLo = $derived(Math.max(0, Math.floor(Math.min(1.5, ...yours.map((y) => y.loss - 0.2)))));
	const yHi = $derived(Math.min(16, Math.ceil(Math.max(7.2, ...yours.map((y) => y.loss + 0.3)))));

	const xOf = (s: number) => {
		const w = W - m.l - m.r;
		const u = xMode === 'log' ? (Math.log10(s + 10) - L0) / (L1 - L0) : s / lastStep;
		return m.l + u * w;
	};
	const yOf = (L: number) => m.t + (1 - (Math.min(Math.max(L, yLo), yHi) - yLo) / (yHi - yLo)) * (H - m.t - m.b);

	const path = (key: 'train' | 'val') => 'M' + points.map((p) => `${xOf(p.step).toFixed(1)},${yOf(p[key]).toFixed(1)}`).join('L');
	const trainPath = $derived(points.length ? path('train') : '');
	const valPath = $derived(points.length ? path('val') : '');
	const xTicks = $derived(
		xMode === 'log'
			? [0, 10, 100, 1000, 10000, lastStep]
			: [0, 5000, 10000, 15000, 20000, 25000, lastStep].filter((s) => s <= lastStep)
	);
	const yTicks = $derived.by(() => {
		const stepY = yHi - yLo > 9 ? 2 : 1;
		const out: number[] = [];
		for (let v = Math.ceil(yLo); v <= yHi; v += stepY) out.push(v);
		return out;
	});
	const tickLabel = (s: number) => (s >= 1000 ? `${s / 1000}k` : String(s));

	// hover crosshair
	let hover = $state<Pt | null>(null);
	function onMove(e: PointerEvent) {
		if (!points.length) return;
		const svg = e.currentTarget as SVGSVGElement;
		const r = svg.getBoundingClientRect();
		const x = ((e.clientX - r.left) / r.width) * W;
		let best = points[0];
		let bd = Infinity;
		for (const p of points) {
			const d = Math.abs(xOf(p.step) - x);
			if (d < bd) {
				bd = d;
				best = p;
			}
		}
		hover = best;
	}
</script>

<figure class="lc" bind:clientWidth={W}>
	<div class="legend small">
		<span><i class="ln c0"></i>{t({ en: 'training loss', zh: '训练损失' })}</span>
		<span><i class="ln c1"></i>{t({ en: 'validation loss', zh: '验证损失' })}</span>
		<span><i class="dt c1"></i>{t({ en: 'saved checkpoints', zh: '保存的检查点' })}</span>
		<span><i class="dt c2"></i>{t({ en: 'your text (live)', zh: '你的文本（实时）' })}</span>
	</div>
	<div class="wrap">
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<svg
			width={W}
			height={H}
			viewBox="0 0 {W} {H}"
			onpointermove={onMove}
			onpointerleave={() => (hover = null)}
			role="img"
			aria-label={t({
				en: 'Loss curve: training and validation loss fall from 6.9 nats at step 0 to 1.9 at step 30,000.',
				zh: '损失曲线：训练与验证损失从第 0 步的 6.9 奈特降到第 30,000 步的 1.9。'
			})}
		>
			{#each yTicks as v (v)}
				<line class="grid" x1={m.l} x2={W - m.r} y1={yOf(v)} y2={yOf(v)} />
				<text class="tick" x={m.l - 7} y={yOf(v) + 4} text-anchor="end">{v}</text>
			{/each}
			{#each xTicks as s (s)}
				<text class="tick" x={xOf(s)} y={H - m.b + 16} text-anchor="middle">{tickLabel(s)}</text>
			{/each}
			<text class="axis" x={m.l + (W - m.l - m.r) / 2} y={H - 6} text-anchor="middle">
				{t({ en: xMode === 'log' ? 'training step (log scale)' : 'training step', zh: xMode === 'log' ? '训练步数（对数刻度）' : '训练步数' })}
			</text>
			<text class="axis" x={11} y={m.t + (H - m.t - m.b) / 2} text-anchor="middle" transform="rotate(-90 11 {m.t + (H - m.t - m.b) / 2})">
				{t({ en: 'loss (nats/token)', zh: '损失（奈特/词元）' })}
			</text>

			<line class="uni" x1={m.l} x2={W - m.r} y1={yOf(uniform)} y2={yOf(uniform)} />
			<text class="unilbl" x={W - m.r - 4} y={yOf(uniform) - 5} text-anchor="end">
				{t({ en: 'uniform guess', zh: '均匀乱猜' })} {fmt3(uniform)}
			</text>

			<line class="sel" x1={xOf(selected)} x2={xOf(selected)} y1={m.t} y2={H - m.b} />
			<path class="line c0" d={trainPath} />
			<path class="line c1" d={valPath} />

			{#each ckpts as c (c.step)}
				<!-- svelte-ignore a11y_click_events_have_key_events -->
				<g class="ck" class:on={c.step === selected} onclick={() => onselect(c.step)} role="presentation">
					<circle class="hit" cx={xOf(c.step)} cy={yOf(c.val)} r="12" />
					<circle class="dot c1" cx={xOf(c.step)} cy={yOf(c.val)} r={c.step === selected ? 6.5 : 4.5} />
				</g>
			{/each}
			{#each yours as y (y.step)}
				<circle class="dot c2" class:on={y.step === selected} cx={xOf(y.step)} cy={yOf(y.loss)} r={y.step === selected ? 6.5 : 4.5} />
			{/each}
			{#if hover}
				<line class="cross" x1={xOf(hover.step)} x2={xOf(hover.step)} y1={m.t} y2={H - m.b} />
			{/if}
		</svg>
		{#if hover}
			<div class="tip small" style:left="{Math.min(Math.max(xOf(hover.step), 90), W - 90)}px">
				<strong class="num">{t({ en: 'step', zh: '第' })} {hover.step.toLocaleString('en-US')}{t({ en: '', zh: ' 步' })}</strong>
				<span class="num">{fmtCount(hover.tokens)} {t({ en: 'tokens seen', zh: '个词元已读' })}</span>
				<span><i class="ln c0"></i>{t({ en: 'train', zh: '训练' })} <span class="num">{fmt3(hover.train)}</span></span>
				<span><i class="ln c1"></i>{t({ en: 'val', zh: '验证' })} <span class="num">{fmt3(hover.val)}</span></span>
			</div>
		{/if}
	</div>
</figure>

<style>
	.lc {
		margin: 0;
		display: grid;
		gap: 6px;
		min-width: 0;
	}
	.wrap {
		position: relative;
	}
	svg {
		display: block;
		width: 100%;
		height: auto;
		background: var(--surface);
		border: 1px solid var(--rule);
		border-radius: var(--r2);
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 16px;
		color: var(--ink-2);
	}
	.legend span,
	.tip span {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.ln {
		display: inline-block;
		width: 16px;
		height: 2px;
		border-radius: 1px;
	}
	.dt {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 50%;
	}
	.ln.c0 {
		background: var(--cat-0);
	}
	.ln.c1,
	.dt.c1 {
		background: var(--cat-1);
	}
	.dt.c2 {
		background: var(--cat-2);
	}
	.grid {
		stroke: var(--rule);
		stroke-width: 1;
	}
	.tick {
		font-family: var(--font-mono);
		font-size: var(--t-xs);
		fill: var(--muted);
	}
	.axis {
		font-size: 12px;
		fill: var(--ink-2);
	}
	.line {
		fill: none;
		stroke-width: 2;
		stroke-linejoin: round;
		stroke-linecap: round;
	}
	.line.c0 {
		stroke: var(--cat-0);
	}
	.line.c1 {
		stroke: var(--cat-1);
		stroke-width: 1.5;
	}
	.uni {
		stroke: var(--rule-strong);
		stroke-width: 1;
	}
	.unilbl {
		font-size: var(--t-xs);
		fill: var(--muted);
	}
	.sel {
		stroke: var(--accent);
		stroke-width: 2;
		opacity: 0.7;
	}
	.cross {
		stroke: var(--ink-2);
		stroke-width: 1;
		pointer-events: none;
	}
	.ck {
		cursor: pointer;
	}
	.hit {
		fill: transparent;
	}
	.dot {
		stroke: var(--surface);
		stroke-width: 2;
	}
	.dot.c1 {
		fill: var(--cat-1);
	}
	.dot.c2 {
		fill: var(--cat-2);
		pointer-events: none;
	}
	.ck.on .dot,
	.dot.c2.on {
		stroke: var(--accent);
		stroke-width: 2.5;
	}
	.tip {
		position: absolute;
		top: 8px;
		transform: translateX(-50%);
		display: grid;
		gap: 2px;
		padding: 6px 10px;
		background: var(--surface);
		border: 1px solid var(--rule-strong);
		border-radius: var(--r2);
		box-shadow: var(--shadow-1);
		pointer-events: none;
		white-space: nowrap;
	}
	.small {
		font-size: var(--fs-xs);
	}
</style>

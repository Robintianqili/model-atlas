<!--
  'optimizers/range' (meta.ts): the loss against the learning rate, both axes on log scales (decade ticks), the line in
  the accent color, and at the end frame three labelled marks: the steepest fall, the lowest loss and the blow-up.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { MARK_NAMES, rangePic } from './meta.ts';

	let stage: StageProps = $props();
	const pic = $derived(rangePic(stage));
	const [W, H] = [420, 250];
	const m = { l: 44, r: 14, t: 14, b: 40 };
	const log = Math.log10;
	const dom = $derived.by(() => {
		if (!pic) return null;
		const ys = pic.loss.filter((v) => v > 0);
		return { x0: -3, x1: 1, y0: Math.floor(log(Math.min(...ys)) * 2) / 2, y1: Math.ceil(log(Math.max(...ys)) * 2) / 2 };
	});
	const X = (v: number) => (dom ? m.l + ((log(v) - dom.x0) / (dom.x1 - dom.x0)) * (W - m.l - m.r) : 0);
	const Y = (v: number) => (dom ? m.t + (1 - (log(v) - dom.y0) / (dom.y1 - dom.y0 || 1)) * (H - m.t - m.b) : 0);
	const xt = [0.001, 0.01, 0.1, 1, 10];
	const yt = $derived(dom ? [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50].filter((v) => log(v) >= dom.y0 - 1e-9 && log(v) <= dom.y1 + 1e-9) : []);
	const line = $derived(pic ? pic.lr.map((v, i) => `${X(v).toFixed(1)},${Y(pic.loss[i]).toFixed(1)}`).join(' ') : '');
	const L = {
		fig: { en: 'Loss against learning rate in the range test', zh: '区间测试中损失随学习率的变化' },
		x: { en: 'learning rate η (log scale)', zh: '学习率 η（对数刻度）' },
		y: { en: 'loss (log scale)', zh: '损失（对数刻度）' }
	};
</script>

{#if pic && dom}
	<figure class="range">
		<svg viewBox="0 0 {W} {H}" role="img" aria-label={t(L.fig)}>
			<rect class="frame" x={m.l} y={m.t} width={W - m.l - m.r} height={H - m.t - m.b} />
			{#each xt as v (v)}
				<line class="grid" x1={X(v)} x2={X(v)} y1={m.t} y2={H - m.b} />
				<text class="tick" x={X(v)} y={H - m.b + 14} text-anchor="middle">{v}</text>
			{/each}
			{#each yt as v (v)}
				<line class="grid" x1={m.l} x2={W - m.r} y1={Y(v)} y2={Y(v)} />
				<text class="tick" x={m.l - 5} y={Y(v) + 4} text-anchor="end">{v}</text>
			{/each}
			<text class="axl" x={m.l + (W - m.l - m.r) / 2} y={H - 6} text-anchor="middle">{t(L.x)}</text>
			<text class="axl" transform="translate(11 {m.t + (H - m.t - m.b) / 2}) rotate(-90)" text-anchor="middle">{t(L.y)}</text>
			<polyline class="line" points={line} />
			{#each pic.marks as k, i (k.key)}
				<line class="mark" x1={X(k.lr)} x2={X(k.lr)} y1={m.t} y2={H - m.b} />
				<circle class="dot" cx={X(k.lr)} cy={Y(k.loss)} r="4.5" />
				<text class="ml" x={X(k.lr) - 6} y={m.t + 12 + 14 * i} text-anchor="end">{t(MARK_NAMES[k.key])}: η = {fmt3(k.lr)}</text>
			{/each}
		</svg>
	</figure>
{/if}

<style>
	.range {
		margin: 0;
		max-width: 560px;
		min-width: 0;
	}
	svg {
		width: 100%;
		height: auto;
		display: block;
	}
	.frame {
		fill: none;
		stroke: var(--rule-strong);
	}
	.grid {
		stroke: var(--rule);
		stroke-width: 1;
	}
	.tick {
		font-size: 10px;
		fill: var(--ink-2);
		font-family: var(--font-mono);
	}
	.axl {
		font-size: 11px;
		fill: var(--ink-2);
	}
	.line {
		fill: none;
		stroke: var(--accent);
		stroke-width: 2.2;
		stroke-linejoin: round;
	}
	.mark {
		stroke: var(--ink-2);
		stroke-dasharray: 3 3;
	}
	.dot {
		fill: var(--ink);
		stroke: var(--surface);
		stroke-width: 1.5;
	}
	.ml {
		font-size: 10.5px;
		fill: var(--ink);
		font-family: var(--font-mono);
		paint-order: stroke;
		stroke: var(--surface);
		stroke-width: 3px;
	}
</style>

<!--
  compare: two sources side by side (StepDef.compare: TinyViT and TinyCNN on the same drawing, lr 0.03 vs lr 3, Euler with
  4 steps vs 50, plain vs PINN). Traces: each side's output (or props.layer) drawn the same way (probability bars for
  class / vocab outputs, else the tensor), and one line on how they differ (their top answers, the largest |a − b|).
  Tapes: one metric (props.metric, else a loss, else the first shared metric) of both over the frames, the stage's
  frame marked, with both values at that frame.
-->
<script lang="ts">
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import Num from '../kit.parts/Num.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3, fmtPct } from '../../../core/format.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../types/view.ts';
	import { outputLabel } from '../../../lesson/facts.ts';
	import Notice from '../kit.parts/Notice.svelte';
	import TensorPanel from '../kit.parts/TensorPanel.svelte';
	import { vocabText } from '../kit.parts/stage.ts';
	import { pickComparison, sideName, type Side } from './pick.ts';

	let P: StageProps = $props();
	const C = $derived(pickComparison(P));
	const nameOf = (s: Side, i: number) => {
		const last = (s.spec?.layers.find((l) => l.id === s.layer)?.axes ?? []).slice(-1)[0];
		if (last === 'vocab') return vocabText(P, i);
		return t(outputLabel(s.spec ?? null, i));
	};
	const items = (s: Side) =>
		s.probs ? Array.from(s.probs, (p, i) => ({ label: nameOf(s, i), p, id: i, color: (s.spec?.layers.find((l) => l.id === s.layer)?.axes ?? []).slice(-1)[0] === 'class' ? categorical(i, themeState.resolved) : undefined })) : [];
	const W = 520;
	const Hh = 170;
	const pad = { l: 44, r: 10, t: 10, b: 22 };
	const chart = $derived.by(() => {
		if (!C?.a.series || !C.b.series) return null;
		const a = C.a.series;
		const b = C.b.series;
		const n = Math.max(a.length, b.length);
		let lo = Infinity;
		let hi = -Infinity;
		for (const s of [a, b]) for (const v of s) if (Number.isFinite(v)) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
		if (!Number.isFinite(lo)) return null;
		if (hi - lo < 1e-12) hi = lo + 1;
		const sx = (i: number) => pad.l + (n > 1 ? (i / (n - 1)) * (W - pad.l - pad.r) : 0);
		const sy = (v: number) => pad.t + (1 - (v - lo) / (hi - lo)) * (Hh - pad.t - pad.b);
		const line = (s: Float64Array) => {
			const pts: string[] = [];
			s.forEach((v, i) => {
				if (Number.isFinite(v)) pts.push(`${sx(i).toFixed(1)},${sy(v).toFixed(1)}`);
			});
			return pts.join(' ');
		};
		const at = Math.max(0, Math.min(n - 1, P.at.index));
		return { la: line(a), lb: line(b), lo, hi, n, x: sx(at), at, va: a[Math.min(at, a.length - 1)], vb: b[Math.min(at, b.length - 1)] };
	});
	$effect(() => {
		if (P.autoplay && C) P.onend();
	});
</script>

{#if !C}
	<Notice text={{ en: 'Nothing to compare: the step needs two sources (compare: { a, b }) that have run.', zh: '没有可比较的内容：这一步需要两个已经运行过的数据源（compare: { a, b }）。' }} />
{:else if C.a.kind === 'trace' && C.b.kind === 'trace'}
	<div class="cmp">
		<div class="sides">
			{#each [C.a, C.b] as s, i (i)}
				<section class="side" aria-label={t(sideName(P, s))}>
					<h4><span class="tag" class:b={i === 1}>{i === 0 ? 'A' : 'B'}</span> {t(sideName(P, s))}</h4>
					{#if s.probs}
						<ProbBars items={items(s)} k={5} chosen={i === 0 ? C.topA : C.topB} />
					{:else if s.t}
						<TensorPanel data={s.t.data} shape={s.t.shape} label={sideName(P, s)} />
					{/if}
				</section>
			{/each}
		</div>
		<p class="sum">
			{#if C.topA !== undefined && C.topB !== undefined && C.a.probs && C.b.probs}
				{#if C.topA === C.topB}
					{t({ en: `Both put “${nameOf(C.a, C.topA)}” first: A with ${fmtPct(C.a.probs[C.topA])}, B with ${fmtPct(C.b.probs[C.topB])}.`, zh: `两者都把“${nameOf(C.a, C.topA)}”排第一：A 是 ${fmtPct(C.a.probs[C.topA])}，B 是 ${fmtPct(C.b.probs[C.topB])}。` })}
				{:else}
					{t({ en: `They disagree: A says “${nameOf(C.a, C.topA)}” (${fmtPct(C.a.probs[C.topA])}), B says “${nameOf(C.b, C.topB)}” (${fmtPct(C.b.probs[C.topB])}).`, zh: `两者不一致：A 认为是“${nameOf(C.a, C.topA)}”（${fmtPct(C.a.probs[C.topA])}），B 认为是“${nameOf(C.b, C.topB)}”（${fmtPct(C.b.probs[C.topB])}）。` })}
				{/if}
			{/if}
			{#if C.maxDiff !== undefined}
				<span class="muted">{t({ en: 'Largest difference |A − B|:', zh: '最大差值 |A − B|：' })} <Num value={C.maxDiff} /></span>
			{/if}
		</p>
	</div>
{:else if chart}
	<div class="cmp">
		<figure class="chart">
			<figcaption>
				<span class="key"><span class="sw a"></span>{t(C.a.label)}</span>
				<span class="key"><span class="sw b"></span>{t(C.b.label)}</span>
				<span class="muted num">{C.metric}</span>
			</figcaption>
			<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="{C.metric}: {t(C.a.label)} {t({ en: 'and', zh: '和' })} {t(C.b.label)}, {chart.n} {t({ en: 'frames', zh: '帧' })}">
				<line x1={pad.l} x2={W - pad.r} y1={Hh - pad.b} y2={Hh - pad.b} class="axis" />
				<text x={pad.l - 4} y={pad.t + 8} class="tick" text-anchor="end">{fmt3(chart.hi)}</text>
				<text x={pad.l - 4} y={Hh - pad.b} class="tick" text-anchor="end">{fmt3(chart.lo)}</text>
				<line x1={chart.x} x2={chart.x} y1={pad.t} y2={Hh - pad.b} class="now" />
				<polyline points={chart.la} class="la" />
				<polyline points={chart.lb} class="lb" />
			</svg>
		</figure>
		<p class="sum num">
			{t({ en: `frame ${chart.at + 1}:`, zh: `第 ${chart.at + 1} 帧：` })}
			A = <Num value={chart.va} /> · B = <Num value={chart.vb} />
		</p>
	</div>
{:else}
	<Notice text={{ en: 'These two sources have no metric in common to compare.', zh: '这两个数据源没有可以比较的共同指标。' }} />
{/if}

<style>
	.cmp {
		display: grid;
		gap: 12px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.sides {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr));
		gap: 14px 24px;
	}
	.side {
		display: grid;
		gap: 6px;
		align-content: start;
		min-width: 0;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-sm);
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.tag {
		display: inline-grid;
		place-items: center;
		width: 1.5em;
		height: 1.5em;
		border-radius: 50%;
		background: var(--accent);
		color: var(--accent-ink);
		font-size: var(--fs-xs);
	}
	.tag.b {
		background: var(--ink-2);
		color: var(--bg);
	}
	.sum {
		margin: 0;
		font-size: var(--fs-sm);
		display: flex;
		flex-wrap: wrap;
		gap: 4px 12px;
	}
	.chart {
		margin: 0;
		max-width: 560px;
		display: grid;
		gap: 4px;
	}
	figcaption {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		font-size: var(--fs-xs);
		align-items: center;
	}
	.key {
		display: inline-flex;
		gap: 6px;
		align-items: center;
	}
	.sw {
		width: 18px;
		height: 0;
		border-top: 2px solid var(--accent);
	}
	.sw.b {
		border-top: 2px dashed var(--ink-2);
	}
	svg {
		width: 100%;
		height: auto;
	}
	.axis {
		stroke: var(--rule-strong);
	}
	.now {
		stroke: var(--highlight);
		stroke-dasharray: 3 2;
	}
	.la {
		fill: none;
		stroke: var(--accent);
		stroke-width: 1.8;
	}
	.lb {
		fill: none;
		stroke: var(--ink-2);
		stroke-width: 1.6;
		stroke-dasharray: 5 3;
	}
	.tick {
		fill: var(--muted);
		font-size: var(--t-xs);
		font-family: var(--font-mono);
	}
</style>

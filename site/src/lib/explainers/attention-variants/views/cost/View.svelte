<!--
  'attention-variants/cost' (meta.ts): 'convert' draws three perplexity bars from 0 on one linear axis (cat-0 MHA, cat-1
  GQA, cat-2 MQA, the same colors as the presets view) with KL, top-1 agreement and cache per token as text; 'uptrain'
  draws three lines on a log perplexity axis over the five evaluation points, each line named at its end.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
	import { bytesText } from '../../data.ts';
	import { curvesOf, evalsOf, KV, LABEL, STEPS } from './meta.ts';

	let P: StageProps = $props();
	const up = $derived(P.props?.mode === 'uptrain');
	const ev = $derived(evalsOf(P));
	const cv = $derived(curvesOf(P));
	$effect(() => void (P.autoplay && P.onend()));
	const CLS = ['k0', 'k1', 'k2'];
	const fx = (k: string) => P.rt.num(`fact:${k}`).toLocaleString('en-US');
	// bars
	const [W, ML, MR, RH] = [360, 4, 8, 40];
	const top = $derived(Math.ceil(Math.max(1, ...ev.map((m) => m?.ppl ?? 0)) / 10) * 10);
	const bx = (v: number) => ML + (v / top) * (W - ML - MR);
	// lines (log y)
	const [LW, LH, pl, pr, pt, pb] = [360, 220, 40, 92, 10, 34];
	const all = $derived(cv.flat().filter(Number.isFinite));
	const lo = $derived(Math.log10(Math.max(1, Math.min(...all) * 0.9)));
	const hi = $derived(Math.log10(Math.max(...all) * 1.1));
	const lx = (i: number) => pl + (i / (STEPS.length - 1)) * (LW - pl - pr);
	const ly = (v: number) => pt + (1 - (Math.log10(v) - lo) / (hi - lo || 1)) * (LH - pt - pb);
	const yt = $derived([5, 7, 10, 20, 30, 50, 70].filter((v) => Math.log10(v) >= lo && Math.log10(v) <= hi));
	/** the end labels' y, pushed apart to at least 11 px (lowest line first) */
	const ends = $derived.by(() => {
		const ys = KV.map((_, k) => ({ k, y: ly(cv[k][STEPS.length - 1]) })).sort((a, b) => b.y - a.y);
		for (let i = 1; i < ys.length; i++) ys[i].y = Math.min(ys[i].y, ys[i - 1].y - 11);
		return Object.fromEntries(ys.map((e) => [e.k, e.y + 3]));
	});
</script>

{#if up}
	<figure class="wrap">
		<svg viewBox="0 0 {LW} {LH}" role="img" aria-label={t({ en: 'Validation perplexity while uptraining, log axis', zh: '继续训练过程中的验证集困惑度（对数轴）' })}>
			{#each yt as v (v)}
				<line class="grid" x1={pl} x2={LW - pr} y1={ly(v)} y2={ly(v)} />
				<text class="tick" x={pl - 4} y={ly(v) + 3} text-anchor="end">{v}</text>
			{/each}
			{#each STEPS as s, i (s)}<text class="tick" x={lx(i)} y={LH - pb + 13} text-anchor="middle">{s.toLocaleString('en-US')}</text>{/each}
			{#each KV as kv, k (kv)}
				{#if cv[k].every(Number.isFinite)}
					<polyline class="line {CLS[k]}" points={cv[k].map((v, i) => `${lx(i)},${ly(v)}`).join(' ')} />
					{#each cv[k] as v, i (i)}<circle class="dot {CLS[k]}" cx={lx(i)} cy={ly(v)} r="2.5" />{/each}
					<text class="end" x={lx(STEPS.length - 1) + 6} y={ends[k]}>{KV[k] === 4 ? 'MHA' : KV[k] === 2 ? 'GQA' : 'MQA'} {fmt3(cv[k][STEPS.length - 1])}</text>
				{/if}
			{/each}
			<text class="axis" x={(pl + LW - pr) / 2} y={LH - 4} text-anchor="middle">{t({ en: 'uptraining steps (evaluation points, evenly spaced)', zh: '继续训练的步数（评测点，等距排列）' })}</text>
			<text class="axis" x="10" y={LH / 2} transform="rotate(-90 10 {LH / 2})" text-anchor="middle">{t({ en: 'perplexity (log axis)', zh: '困惑度（对数轴）' })}</text>
		</svg>
		<figcaption>{t({ en: `PyTorch, ${fx('bigTokens')} validation tokens; ${fx('upSteps')} steps = ${fmtPct(P.rt.num('fact:upSteps') / P.rt.num('fact:pretrainSteps'), 0)} of TinyGPT’s ${fx('pretrainSteps')} training steps.`, zh: `PyTorch，${fx('bigTokens')} 个验证集词元；${fx('upSteps')} 步 = TinyGPT 训练步数 ${fx('pretrainSteps')} 的 ${fmtPct(P.rt.num('fact:upSteps') / P.rt.num('fact:pretrainSteps'), 0)}。` })}</figcaption>
	</figure>
{:else}
	<figure class="wrap">
		<svg viewBox="0 0 {W} {KV.length * RH + 30}" role="img" aria-label={t({ en: 'Perplexity of TinyGPT with shared keys and values, no training', zh: '共用键和值、不再训练的 TinyGPT 的困惑度' })}>
			{#each KV as kv, k (kv)}
				{@const m = ev[k]}
				<text class="name" x={ML} y={k * RH + 11}>{t(LABEL[kv])}{#if m} · <tspan class="num">{fmt3(m.ppl)}</tspan>{/if}</text>
				{#if m}
					<rect class="bar {CLS[k]}" x={bx(0)} y={k * RH + 15} width={Math.max(1.5, bx(m.ppl) - bx(0))} height="10" />
					<text class="sub" x={ML} y={k * RH + 37}>KL {fmt3(m.kl)} · top-1 {fmtPct(m.agree)} · {bytesText(m.cacheBytes)} {t({ en: 'per token', zh: '每词元' })}</text>
				{/if}
			{/each}
			<text class="axis" x={W / 2} y={KV.length * RH + 26} text-anchor="middle">{t({ en: 'perplexity on 256 tokens of four stories (axis from 0)', zh: '四个故事 256 个词元上的困惑度（坐标轴从 0 开始）' })}</text>
		</svg>
	</figure>
{/if}

<style>
	.wrap {
		margin: 0;
		max-width: 560px;
	}
	svg {
		width: 100%;
		height: auto;
		display: block;
	}
	.grid {
		stroke: var(--rule);
	}
	.tick {
		font-size: 10px;
		fill: var(--muted);
		font-family: var(--font-mono);
	}
	.axis,
	.name,
	.sub,
	.end {
		font-size: 11px;
		fill: var(--ink-2);
	}
	.sub {
		font-size: 10px;
	}
	.num {
		font-family: var(--font-mono);
	}
	.line {
		fill: none;
		stroke-width: 2;
	}
	.line.k0 {
		stroke: var(--cat-0);
	}
	.line.k1 {
		stroke: var(--cat-1);
	}
	.line.k2 {
		stroke: var(--cat-2);
	}
	.bar.k0,
	.dot.k0 {
		fill: var(--cat-0);
	}
	.bar.k1,
	.dot.k1 {
		fill: var(--cat-1);
	}
	.bar.k2,
	.dot.k2 {
		fill: var(--cat-2);
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
</style>

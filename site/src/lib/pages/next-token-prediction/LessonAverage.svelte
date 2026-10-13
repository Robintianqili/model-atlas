<!--
  Lesson step 5: the loss of every window, averaged. The sentence colored by surprisal (−ln p,
  darker = more surprising), one bar per window, and a running sum that adds the windows one by
  one into the average loss and the perplexity (live, final fp32). Compared with TinyGPT's
  validation numbers (precomputed) and a blind guess.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { sym } from '$lib/kit/tex';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtInt, visibleToken } from '$lib/kit/format';
	import { pct, sci } from './lessonMath';
	import type { WalkContext } from '$lib/kit/types';
	import type { NtpLesson } from './lessonState.svelte';
	import { FACTS } from './facts';

	let { L, ctx }: { L: NtpLesson; ctx: WalkContext } = $props();
	const S = $derived(L.S);
	const sc = $derived(S.fresh ? S.score : null);
	const N = $derived(sc ? sc.scored : 0);
	const nll = $derived(sc ? sc.p.map((p, i) => (i === 0 ? 0 : -Math.log(p))) : []);

	const VMAX = 8;
	const scale = $derived(makeScale('sequential', [0, VMAX], themeState.resolved));

	// ---------------------------------------------------------------- running sum (k windows added so far)
	let k = $state(untrack(() => (ctx.reduced ? Number.MAX_SAFE_INTEGER : 0)));
	let running = $state(false);
	const kk = $derived(Math.min(k, N));
	$effect(() => {
		// start once the lesson is on screen and the scores are in
		if (ctx.shown && !ctx.reduced && N > 0 && untrack(() => k) === 0) running = true;
	});
	$effect(() => {
		if (!running) return;
		const n = N;
		const ms = Math.max(110, Math.min(380, Math.round(6500 / Math.max(1, n))));
		const h = setInterval(() => {
			if (k >= n) {
				running = false;
				ctx.done();
				return;
			}
			k++;
		}, ms);
		return () => clearInterval(h);
	});
	// a new sentence after the sum has been shown: show the new total at once
	let lastKey = untrack(() => L.idsKey);
	$effect(() => {
		const key = L.idsKey;
		if (key === lastKey) return;
		lastKey = key;
		if (!untrack(() => running) && untrack(() => k) > 0) k = Number.MAX_SAFE_INTEGER;
	});
	// Play lesson pressed after the sum was shown: the step is done
	$effect(() => {
		if (ctx.playing && untrack(() => !running && N > 0 && k >= N)) ctx.done();
	});
	function replay() {
		k = 0;
		running = true;
	}
	function showAll() {
		running = false;
		k = Number.MAX_SAFE_INTEGER;
	}
	const sum = $derived.by(() => {
		let s = 0;
		for (let i = 1; i <= kk; i++) s += nll[i] ?? 0;
		return s;
	});
	const mean = $derived(kk > 0 ? sum / kk : 0);
	const done = $derived(N > 0 && kk >= N);

	// ---------------------------------------------------------------- linked token (hover / focus / tap)
	let hov = $state<number | null>(null);
	const shown = $derived(hov ?? (L.pos >= 1 && L.pos <= kk ? L.pos : null));
	let spans: HTMLElement[] = $state([]);
	function onKey(e: KeyboardEvent, i: number) {
		let j = i;
		if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = Math.min(S.ids.length - 1, i + 1);
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = Math.max(1, i - 1);
		else if (e.key === 'Home') j = 1;
		else if (e.key === 'End') j = S.ids.length - 1;
		else return;
		e.preventDefault();
		hov = j;
		spans[j]?.focus();
	}

	// ---------------------------------------------------------------- bars
	let W = $state(560);
	const H = 150;
	const m = { l: 34, r: 10, t: 14, b: 22 };
	const yTop = $derived(Math.max(VMAX, ...nll.slice(1).map((v) => Math.ceil(v))));
	const bw = $derived(N > 0 ? (W - m.l - m.r) / N : 0);
	const yOf = (v: number) => m.t + (1 - Math.min(v, yTop) / yTop) * (H - m.t - m.b);
	const yTicks = $derived(yTop > 10 ? [0, 5, 10, 15, 20].filter((v) => v <= yTop) : [0, 2, 4, 6, 8].filter((v) => v <= yTop));

	const lossTex = `\\mathcal{L} = \\frac{1}{N}\\sum_{i=1}^{N} -\\ln ${sym('prob', 'p_i')} = \\frac{ {{sum}} }{ {{n}} } = ${sym('out', '{{L}}')}`;
	const pplTex = `\\text{PPL} = e^{\\mathcal{L}} = e^{ {{L}} } = ${sym('out', '{{ppl}}')}`;
</script>

<div class="av">
	<div class="top">
		<p class="eyebrow">{t({ en: 'Every window of your sentence', zh: '你的句子的每一个窗口' })}</p>
	</div>
	{#if S.ids.length < 2}
		<p class="muted">{t({ en: 'Type at least two tokens.', zh: '至少输入两个词元。' })}</p>
	{:else if !sc}
		<p class="muted">{t({ en: 'TinyGPT is scoring your sentence…', zh: 'TinyGPT 正在给你的句子打分…' })}</p>
	{:else}
		<div class="text" role="listbox" aria-label={t({ en: 'Your sentence, colored by surprise', zh: '按意外程度着色的句子' })} aria-orientation="horizontal">
			{#each S.ids as id, i (i)}
				{@const s = S.disp(id).replace(/\n/g, '↵\n')}
				{#if i === 0}
					<span class="t first" title={t({ en: 'The first token has no context, so it is not scored.', zh: '第一个词元没有上下文，因此不计分。' })}>{s}</span>
				{:else}
					<span
						bind:this={spans[i]}
						class="t"
						class:on={i <= kk}
						class:sel={i === L.pos}
						class:hov={shown === i}
						style:background={i <= kk ? scale.css(nll[i]) : null}
						style:color={i <= kk ? scale.ink(nll[i]) : null}
						role="option"
						aria-selected={i === L.pos}
						tabindex={i === Math.max(1, L.pos) ? 0 : -1}
						aria-label={`${visibleToken(S.disp(id))}: p ${fmt3(sc.p[i])}, ${fmt3(nll[i])} nats`}
						onpointerenter={() => (hov = i)}
						onpointerleave={() => (hov = null)}
						onclick={() => (hov = i)}
						onfocus={() => (hov = i)}
						onkeydown={(e) => onKey(e, i)}>{s}</span
					>
				{/if}
			{/each}
		</div>
		<div class="legend small">
			<span>0 {t({ en: 'nats (no surprise)', zh: '奈特（毫不意外）' })}</span>
			<span class="grad" style:background="linear-gradient(90deg, {scale.css(0)}, {scale.css(VMAX / 2)}, {scale.css(VMAX)})"></span>
			<span>≥ {VMAX}</span>
			<span class="read">
				{#if shown !== null}
					<span class="tok">{visibleToken(S.disp(S.ids[shown]))}</span>
					p = <span class="num">{pct(sc.p[shown])}</span>, −ln p = <span class="num">{fmt3(nll[shown])}</span>
				{/if}
			</span>
		</div>

		<figure class="bars" bind:clientWidth={W}>
			<svg width={W} height={H} viewBox="0 0 {W} {H}" role="img" aria-label={t({ en: `One bar per window: its loss −ln p. Average ${fmt3(mean)} nats over ${kk} windows.`, zh: `每个窗口一根条形：它的损失 −ln p。${kk} 个窗口的平均是 ${fmt3(mean)} 奈特。` })}>
				{#each yTicks as v (v)}
					<line class="grid" x1={m.l} x2={W - m.r} y1={yOf(v)} y2={yOf(v)} />
					<text class="tick" x={m.l - 6} y={yOf(v) + 4} text-anchor="end">{v}</text>
				{/each}
				{#each nll as v, i (i)}
					{#if i >= 1 && i <= kk}
						<rect
							class="bar"
							class:sel={i === L.pos}
							class:hov={shown === i}
							x={m.l + (i - 1) * bw + Math.min(1, bw * 0.15)}
							y={yOf(v)}
							width={Math.max(1, bw - Math.min(2, bw * 0.3))}
							height={Math.max(0.5, yOf(0) - yOf(v))}
							fill={scale.css(v)}
						/>
					{/if}
				{/each}
				{#if kk > 0}
					<line class="mean" x1={m.l} x2={W - m.r} y1={yOf(mean)} y2={yOf(mean)} />
					<text class="meanlbl" x={W - m.r - 4} y={yOf(mean) - 6} text-anchor="end">
						{done ? t({ en: 'average', zh: '平均' }) : t({ en: 'average so far', zh: '目前的平均' })} {fmt3(mean)}
					</text>
				{/if}
				<text class="tick" x={m.l} y={H - 6}>{t({ en: 'window 1', zh: '窗口 1' })}</text>
				<text class="tick" x={W - m.r} y={H - 6} text-anchor="end">{N}</text>
			</svg>
		</figure>

		<div class="ctl">
			<span class="small num">{t({ en: `${kk} of ${N} windows added`, zh: `已加入 ${kk} / ${N} 个窗口` })}</span>
			{#if done}
				<button type="button" class="btn small-b" onclick={replay}>{t({ en: 'Add them up again', zh: '再加一遍' })}</button>
			{:else}
				<button type="button" class="btn small-b" onclick={showAll}>{t({ en: 'Show the total', zh: '直接看总数' })}</button>
			{/if}
		</div>

		<div class="cols">
			<FormulaCard
				tex={lossTex}
				values={{ sum, n: String(kk), L: mean }}
				caption={{ en: 'Loss: the average −ln p (nats per token)', zh: '损失：−ln p 的平均值（奈特/词元）' }}
			/>
			<FormulaCard tex={pplTex} values={{ L: mean, ppl: Math.exp(mean) }} caption={{ en: 'Perplexity undoes the log', zh: '困惑度把对数还原回来' }} />
		</div>
		<div class="cmp small">
			<div class="row head" aria-hidden="true">
				<span></span><span>{t({ en: 'loss (nats)', zh: '损失（奈特）' })}</span><span>{t({ en: 'perplexity', zh: '困惑度' })}</span><span></span>
			</div>
			<div class="row you">
				<span class="k">{t({ en: 'your sentence', zh: '你的句子' })}</span>
				<span class="num">{done ? fmt3(sc.meanNats) : '…'}</span>
				<span class="num">{done ? sci(sc.perplexity) : '…'}</span>
				<ProvenanceBadge mode="live" compact />
			</div>
			<div class="row">
				<span class="k">{t({ en: 'TinyGPT on stories it never saw', zh: 'TinyGPT 在没见过的故事上' })}</span>
				<span class="num">{FACTS.valLoss.toFixed(2)}</span>
				<span class="num">{FACTS.valPpl.toFixed(2)}</span>
				<ProvenanceBadge mode="precomputed" compact note={{ en: `Measured offline on all ${fmtInt(FACTS.valTokens)} validation tokens.`, zh: `离线在全部 ${fmtInt(FACTS.valTokens)} 个验证词元上测得。` }} />
			</div>
			<div class="row">
				<span class="k">{t({ en: 'blind guess (1/1,024 each)', zh: '盲猜（每个 1/1,024）' })}</span>
				<span class="num">6.93</span>
				<span class="num">1,024</span>
				<span></span>
			</div>
		</div>
	{/if}
</div>

<style>
	.av {
		container: av / inline-size;
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.top {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
		justify-content: space-between;
	}
	.eyebrow {
		margin: 0;
	}
	.text {
		padding: 10px 12px;
		background: var(--surface);
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		font-family: var(--font-mono);
		font-size: var(--fs-md);
		line-height: 2;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.t {
		border-radius: 3px;
		padding: 2px 0;
		cursor: pointer;
		box-decoration-break: clone;
		-webkit-box-decoration-break: clone;
		transition: background-color var(--dur-2) var(--ease);
	}
	.t:not(.on) {
		color: var(--muted);
	}
	.t.first {
		cursor: help;
		outline: 1px dashed var(--rule-strong);
		outline-offset: -1px;
		color: var(--muted);
	}
	.t.sel {
		box-shadow: inset 0 -3px 0 var(--real);
	}
	.t.hov {
		outline: 2px solid var(--highlight);
		outline-offset: 1px;
	}
	.t:focus-visible {
		outline: 3px solid var(--accent);
		outline-offset: 2px;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 10px;
		align-items: center;
		color: var(--ink-2);
	}
	.grad {
		display: inline-block;
		width: 110px;
		height: 10px;
		border-radius: 3px;
		border: 1px solid var(--rule);
	}
	.read {
		min-height: 1.4em;
		margin-left: auto;
	}
	.tok {
		font-family: var(--font-mono);
		background: var(--surface-2);
		padding: 0 4px;
		border-radius: 3px;
		white-space: pre;
	}
	.bars {
		margin: 0;
		min-width: 0;
	}
	svg {
		display: block;
		width: 100%;
		height: auto;
		background: var(--surface);
		border: 1px solid var(--rule);
		border-radius: var(--r2);
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
	.bar {
		stroke: var(--rule-strong);
		stroke-width: 0.5;
	}
	.bar.sel {
		stroke: var(--real);
		stroke-width: 2;
	}
	.bar.hov {
		stroke: var(--highlight);
		stroke-width: 1.5;
	}
	.mean {
		stroke: var(--ink);
		stroke-width: 1.5;
		stroke-dasharray: 5 4;
	}
	.meanlbl {
		font-size: var(--t-xs);
		font-weight: 700;
		fill: var(--ink);
		paint-order: stroke;
		stroke: var(--surface);
		stroke-width: 3px;
	}
	.ctl {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 12px;
		align-items: center;
	}
	.small-b {
		min-height: 32px;
		padding: 0 12px;
		font-size: var(--fs-xs);
	}
	.cols {
		display: grid;
		grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr);
		gap: 10px;
		align-items: start;
	}
	@container av (max-width: 560px) {
		.cols {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.small {
		font-size: var(--fs-xs);
	}
	.cmp {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto auto auto;
		align-items: center;
		gap: 4px 12px;
		padding: 8px 10px;
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	.row {
		display: contents;
	}
	.row.head > span {
		color: var(--muted);
		text-align: right;
	}
	.row.you > span {
		font-weight: 700;
	}
	.row .num {
		text-align: right;
	}
	.k {
		min-width: 0;
	}
</style>

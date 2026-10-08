<!--
  All 1,024 probabilities of one window as bars, most likely first: one bar per token, rank 1 on
  the left. The rank axis is logarithmic so the first bars are wide enough to see while every one
  of the 1,024 bars still has its place; the y axis is linear (true heights) or log (the tail).
  The token that really came next is marked in green with its rank. Hover, tap, or focus the chart
  and use the arrow keys to read any bar.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, visibleToken } from '$lib/kit/format';
	import { pct, type Dist } from './lessonMath';

	interface Props {
		dist: Dist;
		disp: (id: number) => string;
		log?: boolean;
		height?: number;
		/** dashed line at 1/1024 (a blind guess) */
		uniform?: boolean;
	}
	let { dist, disp, log = $bindable(false), height = 190, uniform = true }: Props = $props();

	const V = $derived(dist.probs.length);
	let W = $state(560);
	const m = { l: 42, r: 12, t: 28, b: 30 };
	const H = $derived(height);
	const iw = $derived(Math.max(10, W - m.l - m.r));
	const ih = $derived(H - m.t - m.b);

	/** left edge of the bar at 0-based rank r (r = V is the right end) */
	const xOf = (r: number) => m.l + (Math.log(1 + r) / Math.log(1 + V)) * iw;
	const rankAt = (x: number) => Math.max(0, Math.min(V - 1, Math.floor(Math.exp(((x - m.l) / iw) * Math.log(1 + V)) - 1)));

	/** log axis: from the smallest power of ten under the smallest p (at most 1e-12) up to 1 */
	const logLo = $derived.by(() => {
		let mn = 1;
		for (let i = 0; i < dist.probs.length; i++) if (dist.probs[i] > 0 && dist.probs[i] < mn) mn = dist.probs[i];
		return Math.max(-12, Math.floor(Math.log10(mn)));
	});
	const yOf = (p: number) => {
		if (!log) return m.t + (1 - p) * ih;
		const v = Math.max(logLo, Math.log10(Math.max(p, 1e-300)));
		return m.t + (v / logLo) * ih;
	};
	const yTicks = $derived.by(() => {
		if (!log) return [0, 0.25, 0.5, 0.75, 1].map((v) => ({ y: yOf(v), label: String(v) }));
		const out: { y: number; label: string }[] = [];
		const stepE = logLo <= -9 ? 3 : 2;
		for (let e = 0; e >= logLo; e -= stepE) out.push({ y: yOf(10 ** e), label: e === 0 ? '1' : `1e${e}` });
		return out;
	});
	const xTicks = $derived([1, 10, 100].filter((r) => r <= V));

	// all bars as one path
	const path = $derived.by(() => {
		const base = m.t + ih;
		let d = '';
		for (let r = 0; r < V; r++) {
			const y = yOf(dist.probs[dist.order[r]]);
			if (base - y < 0.05) continue;
			const x0 = xOf(r);
			const w = xOf(r + 1) - x0;
			const gap = w > 3 ? 1 : 0;
			d += `M${x0.toFixed(2)} ${base.toFixed(2)}V${y.toFixed(2)}h${Math.max(0.5, w - gap).toFixed(2)}V${base.toFixed(2)}z`;
		}
		return d;
	});
	const tRank = $derived(dist.rank - 1); // 0-based position of the target in the sorted order
	const mid = (r: number) => (xOf(r) + xOf(r + 1)) / 2;
	const tx = $derived(mid(tRank));
	const above = $derived.by(() => {
		let c = 0;
		for (let i = 0; i < dist.probs.length; i++) if (dist.probs[i] > 1 / V) c++;
		return c;
	});

	// cursor (hover / tap / keyboard), as a 0-based rank
	let cur = $state<number | null>(null);
	let focused = $state(false);
	const shown = $derived(cur ?? (focused ? tRank : null));
	function fromEvent(e: PointerEvent) {
		const svg = e.currentTarget as SVGSVGElement;
		const r = svg.getBoundingClientRect();
		cur = rankAt(((e.clientX - r.left) / r.width) * W);
	}
	function onkey(e: KeyboardEvent) {
		const c = shown ?? tRank;
		let j = c;
		if (e.key === 'ArrowRight' || e.key === 'ArrowUp') j = c + 1;
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') j = c - 1;
		else if (e.key === 'PageUp') j = c + 32;
		else if (e.key === 'PageDown') j = c - 32;
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = V - 1;
		else return;
		e.preventDefault();
		cur = Math.max(0, Math.min(V - 1, j));
	}
	const label = (id: number) => visibleToken(disp(id));
	const at = $derived(shown ?? tRank);
	const flip = $derived(tx > m.l + iw * 0.6);
</script>

<figure class="dc" bind:clientWidth={W}>
	<div class="head">
		<span class="cap small">{t({ en: `All ${V.toLocaleString('en-US')} tokens, most likely first`, zh: `全部 ${V.toLocaleString('en-US')} 个词元，按概率从高到低` })}</span>
		<span class="segmented" role="group" aria-label={t({ en: 'y-axis scale', zh: 'y 轴刻度' })}>
			<button type="button" aria-pressed={!log} onclick={() => (log = false)}>{t({ en: 'linear p', zh: '线性 p' })}</button>
			<button type="button" aria-pressed={log} onclick={() => (log = true)}>{t({ en: 'log p', zh: '对数 p' })}</button>
		</span>
	</div>
	<svg
		width={W}
		{height}
		viewBox="0 0 {W} {H}"
		tabindex="0"
		role="slider"
		aria-valuemin={1}
		aria-valuemax={V}
		aria-valuenow={at + 1}
		aria-valuetext={`${t({ en: 'rank', zh: '第' })} ${at + 1}${t({ en: '', zh: ' 名' })}: ${label(dist.order[at])}, p = ${fmt3(dist.probs[dist.order[at]])}`}
		aria-label={t({
			en: `${V} probabilities, sorted. The real next token is at rank ${dist.rank} with p = ${fmt3(dist.p)}. Use the arrow keys to read the bars.`,
			zh: `${V} 个概率，已排序。真实的下一个词元排第 ${dist.rank}，p = ${fmt3(dist.p)}。用方向键逐个查看。`
		})}
		onpointermove={fromEvent}
		onpointerdown={fromEvent}
		onpointerleave={(e) => e.pointerType === 'mouse' && (cur = null)}
		onfocus={() => (focused = true)}
		onblur={() => {
			focused = false;
			cur = null;
		}}
		onkeydown={onkey}
	>
		{#each yTicks as tk (tk.label)}
			<line class="grid" x1={m.l} x2={W - m.r} y1={tk.y} y2={tk.y} />
			<text class="tick" x={m.l - 6} y={tk.y + 4} text-anchor="end">{tk.label}</text>
		{/each}
		{#each xTicks as r (r)}
			<line class="xt" x1={xOf(r - 1)} x2={xOf(r - 1)} y1={m.t + ih} y2={m.t + ih + 4} />
			<text class="tick" x={xOf(r - 1)} y={H - 14} text-anchor={r === 1 ? 'start' : 'middle'}>{r.toLocaleString('en-US')}</text>
		{/each}
		<text class="tick" x={W - m.r} y={H - 14} text-anchor="end">{V.toLocaleString('en-US')}</text>
		<text class="axis" x={m.l + iw / 2} y={H - 2} text-anchor="middle">{t({ en: 'rank (log scale)', zh: '排名（对数刻度）' })}</text>

		{#if uniform}
			<line class="uni" x1={m.l} x2={W - m.r} y1={yOf(1 / V)} y2={yOf(1 / V)} />
		{/if}
		<path class="bars" d={path} />

		<!-- the real next token -->
		<line class="tline" x1={tx} x2={tx} y1={m.t - 6} y2={m.t + ih} />
		<circle class="tdot" cx={tx} cy={yOf(dist.p)} r="4.5" />
		<text class="tlbl" x={flip ? tx - 6 : tx + 6} y={m.t - 12} text-anchor={flip ? 'end' : 'start'}>
			“{label(dist.target)}” · {t({ en: 'rank', zh: '第' })} {dist.rank}{t({ en: '', zh: ' 名' })} · {pct(dist.p)}
		</text>

		{#if shown !== null}
			<line class="cur" x1={mid(shown)} x2={mid(shown)} y1={m.t} y2={m.t + ih} />
		{/if}
	</svg>
	<p class="read small" aria-hidden="true">
		{#if shown !== null}
			{t({ en: 'Rank', zh: '第' })} <span class="num">{shown + 1}</span>{t({ en: '', zh: ' 名' })}{t({ en: ': ', zh: '：' })}<span class="chip-t" class:real={dist.order[shown] === dist.target}>{label(dist.order[shown])}</span>
			<span class="num muted">#{dist.order[shown]}</span> · p = <span class="num">{pct(dist.probs[dist.order[shown]])}</span>
			{#if dist.order[shown] === dist.target}<strong>· {t({ en: 'the real next token', zh: '真实的下一个词元' })}</strong>{/if}
		{:else}
			<!-- one string per language: no stray space between the two Chinese sentences -->
			<span class="muted"
				>{t({
					en: `Hover, tap, or focus the chart and use the arrow keys to read any bar.${uniform ? ` Dashed line: 1/1,024, a blind guess. ${above} ${above === 1 ? 'token is' : 'tokens are'} above it.` : ''}`,
					zh: `把鼠标移到图上、点一下，或聚焦后用方向键，读出任意一根条形。${uniform ? `虚线：1/1,024，即盲猜。有 ${above} 个词元高于它。` : ''}`
				})}</span
			>
		{/if}
	</p>
</figure>

<style>
	.dc {
		margin: 0;
		display: grid;
		gap: 6px;
		min-width: 0;
	}
	.head {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 12px;
		align-items: center;
		justify-content: space-between;
	}
	.cap {
		color: var(--ink-2);
		font-weight: 650;
	}
	svg {
		display: block;
		width: 100%;
		height: auto;
		background: var(--surface);
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		touch-action: pan-y;
		cursor: crosshair;
	}
	svg:focus-visible {
		outline: 3px solid var(--accent);
		outline-offset: 2px;
	}
	.grid {
		stroke: var(--rule);
		stroke-width: 1;
	}
	.xt {
		stroke: var(--rule-strong);
	}
	.tick {
		font-family: var(--font-mono);
		font-size: 10.5px;
		fill: var(--muted);
	}
	.axis {
		font-size: 11px;
		fill: var(--ink-2);
	}
	.uni {
		stroke: var(--ink-2);
		stroke-width: 1;
		stroke-dasharray: 4 4;
		opacity: 0.7;
	}
	.bars {
		fill: var(--seq-mid);
	}
	.tline {
		stroke: var(--real);
		stroke-width: 1.5;
		stroke-dasharray: 3 3;
	}
	.tdot {
		fill: var(--real);
		stroke: var(--surface);
		stroke-width: 2;
	}
	.tlbl {
		font-family: var(--font-mono);
		font-size: 12px;
		font-weight: 700;
		fill: var(--ink);
		paint-order: stroke;
		stroke: var(--surface);
		stroke-width: 3px;
	}
	.cur {
		stroke: var(--highlight);
		stroke-width: 1;
		pointer-events: none;
	}
	.read {
		margin: 0;
		min-height: 2.6em;
	}
	.small {
		font-size: var(--fs-xs);
	}
	.chip-t {
		font-family: var(--font-mono);
		background: var(--surface-2);
		padding: 0 4px;
		border-radius: 3px;
		white-space: pre;
	}
	.chip-t.real {
		outline: 1.5px solid var(--real);
	}
</style>

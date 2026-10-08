<!--
  char-rnn/timeline: the LSTM's memory under the learner's characters (props.mode, see data.ts).
  memory / unroll: h of all 64 units as a heatmap (rows = units, columns = characters; the followed unit outlined); unroll
  fills it one character at a time and writes each top guess under its column (green when the next character matches).
  unit: one unit's cell value as bars under the characters, then its top contexts on held-out stories (precomputed).
  fade: the share of a memory left after d characters. sample: the drawn text and the greedy text, and the current draw.
  next: the chances of the next character after the character picked above the bars (default: the last one); unlike
  kit.probbars it marks no bar "chosen" (nothing is drawn here) and prints no ids that read like ranks.
  Wide content scrolls inside its own box; nothing animates by itself (the lesson's scrubber moves unroll and sample).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
	import Notice from '../../../../atlas/ui/views/kit.parts/Notice.svelte';
	import { REF_GATE, show, timelineOf } from './data.ts';

	let P: StageProps = $props();
	const G = $derived(timelineOf(P));
	const CW = 11;
	const RH = 4;
	const n = $derived(G?.chars.length ?? 0);
	const div = $derived(makeScale('diverging', [-1, 1], themeState.resolved));
	let canvas = $state<HTMLCanvasElement | null>(null);
	const cols = $derived.by<(ArrayLike<number> | null)[]>(() => {
		if (!G) return [];
		if ('cols' in G) return G.cols;
		if ('h' in G && G.h) return Array.from({ length: n }, (_, s) => G.h.data.subarray(s * 64, s * 64 + 64));
		return [];
	});
	$effect(() => {
		const cv = canvas;
		if (!cv || !cols.length) return;
		const ctx = cv.getContext('2d');
		if (!ctx) return;
		const img = ctx.createImageData(n, 64);
		cols.forEach((c, s) => {
			for (let j = 0; j < 64; j++) {
				const [r, g, b] = c ? div.rgb(c[j]) : [0, 0, 0];
				const o = (j * n + s) * 4;
				img.data.set([r, g, b, c ? 255 : 0], o);
			}
		});
		ctx.putImageData(img, 0, 0);
	});
	const cur = $derived(G && 'cols' in G ? G.t : n - 1);
	// keep the column the narration talks about in view when the strip scrolls (phones): the current character, else the
	// longest word (unit mode), else the last character
	let scroller = $state<HTMLDivElement | null>(null);
	$effect(() => {
		const el = scroller;
		const w = G && 'word' in G ? G.word : null;
		const [a, b] = w ? [w.from, w.to] : [cur, cur];
		if (!el || a < 0) return;
		if (b * CW + CW > el.scrollLeft + el.clientWidth || a * CW < el.scrollLeft) el.scrollLeft = Math.max(0, (b + 2) * CW - el.clientWidth);
	});
	// unit mode: bars of c, the zero line placed so positive and negative bars share 82 px
	const cu = $derived(G && 'cu' in G && G.cu ? Array.from(G.cu) : []);
	const hp = $derived(Math.max(0.5, ...cu));
	const hn = $derived(Math.max(0, ...cu.map((v) => -v)));
	const cmax = $derived(Math.max(hp, hn));
	const y0 = $derived(4 + (82 * hp) / (hp + hn));
	const bar = (v: number) => (82 * Math.abs(v)) / (hp + hn);
	// fade mode: chart
	const W = 520;
	const Hh = 200;
	const pad = { l: 50, r: 14, t: 10, b: 28 };
	const fx = (d: number, D: number) => pad.l + (D > 0 ? d / D : 0) * (W - pad.l - pad.r);
	const fy = (v: number) => pad.t + (1 - v) * (Hh - pad.t - pad.b);
	const line = (a: number[], D: number) => a.map((v, d) => `${fx(d, D).toFixed(1)},${fy(v).toFixed(1)}`).join(' ');
	const text = (ids: number[], from: number) => ids.slice(from).map((i) => show(G?.alphabet[i])).join('');
	// next mode: the character the chances follow (null: the last one); a new step or input starts from the last again
	let row = $derived.by<number | null>(() => {
		void P.at;
		return null;
	});
	$effect(() => {
		if (P.autoplay && G) P.onend();
	});
</script>

{#if !G || !n}
	<Notice text={{ en: 'Type some text first: every column of this picture is one of your characters.', zh: '先输入一些文字：这幅图的每一列都是你的一个字符。' }} />
{:else if G.mode === 'memory' || G.mode === 'unroll'}
	<div class="tl">
		<div class="scroll" bind:this={scroller} use:keyScroll>
			<div class="strip" style="--n: {'guesses' in G ? n + 1 : n}; --cw: {CW}px">
				<div class="chars" aria-hidden="true">
					{#each G.chars as ch, s (s)}<span class:cur={s === cur}>{show(ch)}</span>{/each}
				</div>
				<div class="heat" style="height: {64 * RH}px" role="img" aria-label={t({ en: 'Hidden state h: rows are the 64 units, columns are your characters', zh: '隐藏状态 h：行是 64 个单元，列是你的字符' })}>
					<canvas bind:this={canvas} width={n} height="64" style="width: {n * CW}px; height: {64 * RH}px" aria-hidden="true"></canvas>
					{#if Number.isFinite(G.unit)}<span class="row" style="top: {G.unit * RH - 1}px"></span>{/if}
					{#if cur >= 0}<span class="col" style="left: {cur * CW - 1}px"></span>{/if}
				</div>
				{#if 'guesses' in G}
					<div class="chars guess" aria-hidden="true">
						<span></span>{#each G.guesses as g, s (s)}<span class:hit={g.next !== undefined && g.top === g.next} class:miss={g.next !== undefined && g.top !== g.next}>{show(G.alphabet[g.top])}</span>{/each}
					</div>
				{/if}
			</div>
		</div>
		<div class="foot">
			<ColorLegend kind="diverging" domain={[-1, 1]} label={{ en: 'h', zh: 'h' }} width={140} />
			<span class="note">{t({ en: `Rows: the 64 units (outlined: unit ${G.unit}). Columns: your characters, in reading order.`, zh: `行：64 个单元（加框的是单元 ${G.unit}）。列：你的字符，按读的顺序。` })}</span>
			{#if 'guesses' in G && G.guesses[G.t]}
				{@const g = G.guesses[G.t]}
				<span class="note">{t({ en: `Character ${G.t} (counting from 0) is “${show(G.chars[G.t])}”; the top guess for the next one is “${show(G.alphabet[g.top])}” with ${fmtPct(g.pTop)}. Bottom row: each top guess, under the character it tried to predict, green when it was right.`, zh: `第 ${G.t} 个字符（从 0 数起）是“${show(G.chars[G.t])}”；对下一个字符的第一猜测是“${show(G.alphabet[g.top])}”，概率 ${fmtPct(g.pTop)}。最下面一行：每一次的第一猜测，写在它想猜的那个字符下面，猜对的是绿色。` })}</span>
			{/if}
		</div>
	</div>
{:else if G.mode === 'unit' && 'cu' in G && G.cu}
	<div class="tl">
		<div class="scroll" bind:this={scroller} use:keyScroll>
			<div class="strip" style="--n: {n}; --cw: {CW}px">
				<div class="chars" aria-hidden="true">
					{#each G.chars as ch, s (s)}<span class:word={!!G.word && s >= G.word.from && s <= G.word.to}>{show(ch)}</span>{/each}
				</div>
				<svg class="bars" width={n * CW} height="90" role="img" aria-label={t({ en: `Cell value of unit ${G.unit} under each character`, zh: `每个字符下单元 ${G.unit} 的细胞值` })}>
					<line x1="0" x2={n * CW} y1={y0} y2={y0} class="axis" />
					{#each cu as v, s (s)}
						<rect x={s * CW + 1} width={CW - 2} y={v >= 0 ? y0 - bar(v) : y0} height={bar(v)} fill={makeScale('diverging', [-cmax, cmax], themeState.resolved).css(v)}><title>{show(G.chars[s])}: {fmt3(v)}</title></rect>
					{/each}
				</svg>
			</div>
		</div>
		<p class="note">{t({ en: `Bars: the cell value c of unit ${G.unit} after each character (up to ${fmt3(cmax)}). Underlined: your longest word.`, zh: `条形：读完每个字符后单元 ${G.unit} 的细胞值 c（最大到 ${fmt3(cmax)}）。下划线：你最长的单词。` })}</p>
		{#if G.top.length}
			<figure class="tops">
				<figcaption>{t({ en: `Precomputed: where unit ${G.unit}'s cell value was highest in ${G.topChars.toLocaleString('en')} characters of stories it never trained on`, zh: `预先算好的：在 ${G.topChars.toLocaleString('en')} 个没训练过的故事字符里，单元 ${G.unit} 的细胞值最高的地方` })}</figcaption>
				<ol>
					{#each G.top as e, k (k)}<li><span class="mono">{e.before.replaceAll('\n', '↵')}<mark>{show(e.at)}</mark><span class="after">{e.after.replaceAll('\n', '↵')}</span></span> <span class="num">c = {fmt3(e.c)}</span></li>{/each}
				</ol>
			</figure>
		{/if}
	</div>
{:else if G.mode === 'next' && 'probs' in G && G.probs}
	{@const r = row ?? n - 1}
	{@const V = G.probs.shape[1]}
	<div class="tl">
		<TokenChips tokens={G.chars.map((text) => ({ text }))} selected={r} label={{ en: 'The guess after which character', zh: '在哪个字符之后的猜测' }} onselect={(i) => (row = i)} />
		<ProbBars items={Array.from(G.probs.data.subarray(r * V, r * V + V), (p, k) => ({ label: G.alphabet[k], p }))} k={10} tokens label={{ en: `Chances of the character after “${show(G.chars[r])}” (character ${r}, counting from 0)`, zh: `“${show(G.chars[r])}”（第 ${r} 个字符，从 0 数起）后面那个字符的概率` }} />
	</div>
{:else if G.mode === 'fade' && 'fade' in G && G.fade}
	{@const f = G.fade}
	<figure class="tl">
		<svg viewBox="0 0 {W} {Hh}" role="img" aria-label={t({ en: 'Share of a memory left after d characters', zh: '记忆在 d 个字符之后还剩的份额' })}>
			{#each [0, 0.25, 0.5, 0.75, 1] as v (v)}
				<line x1={pad.l} x2={W - pad.r} y1={fy(v)} y2={fy(v)} class="grid" />
				<text x={pad.l - 6} y={fy(v) + 4} class="tick" text-anchor="end">{fmtPct(v, 0)}</text>
			{/each}
			{#each Array.from({ length: f.D + 1 }, (_, d) => d).filter((d) => d % 10 === 0 || d === f.D) as d (d)}
				<text x={fx(d, f.D)} y={Hh - 8} class="tick" text-anchor="middle">{d}</text>
			{/each}
			{#if f.D > 0}
				<polyline points={line(Array.from({ length: f.D + 1 }, (_, d) => REF_GATE ** d), f.D)} class="ref" />
				<polyline points={line(f.patientCurve, f.D)} class="pat" />
				<polyline points={line(f.keep, f.D)} class="med" />
			{/if}
		</svg>
		<figcaption class="note">
			{t({ en: `Up: the share of a memory still there. Across: d, the number of characters later (0 to ${f.D}). Solid: the median over all units and starting points. Dashed: unit ${f.patient}, the one that keeps the most. Thin grey: a forget gate of ${REF_GATE} at every character.`, zh: `纵轴：记忆还剩的份额。横轴：d，往后过了几个字符（0 到 ${f.D}）。实线：所有单元、所有起点的中位数。虚线：单元 ${f.patient}，保留得最多的那个。细灰线：每个字符的遗忘门都是 ${REF_GATE} 的情况。` })}
		</figcaption>
	</figure>
{:else if G.mode === 'sample' && 'a' in G && G.a}
	{@const a = G.a}
	<div class="tl">
		<p class="mono txt"><span class="prompt">{text(a.ids, Math.max(0, a.prompt - 24)).slice(0, Math.min(24, a.prompt))}</span><strong>{text(a.ids, a.prompt)}</strong></p>
		{#if a.token !== undefined && !Number.isFinite(a.u)}
			<p class="note">{t({ en: `It took the top character, “${show(G.alphabet[a.token])}”, whose chance was ${fmtPct(a.pModel ?? NaN)}: no random number is drawn.`, zh: `它选了第一名“${show(G.alphabet[a.token])}”，概率 ${fmtPct(a.pModel ?? NaN)}：不抽随机数。` })}</p>
		{:else if 'draw' in G && G.draw}
			{@const k0 = a.token ?? 0}
			<figure class="draw">
				<svg viewBox="0 0 {W} 54" role="img" aria-label={t({ en: 'The chances laid end to end from 0 to 1, and where u lands', zh: '把各个概率从 0 到 1 首尾相接，以及 u 落在哪里' })}>
					{#each G.draw as p, k (k)}
						<rect x={10 + G.from[k] * (W - 20)} y="8" width={p * (W - 20)} height="26" class="seg" />
						{#if p > 0.04}<text x={10 + (G.from[k] + p / 2) * (W - 20)} y="26" class="segl" text-anchor="middle">{show(G.alphabet[k])}</text>{/if}
					{/each}
					<rect x={10 + (G.from[k0] + G.draw[k0] / 2) * (W - 20) - Math.max(1.5, (G.draw[k0] * (W - 20)) / 2)} y="8" width={Math.max(3, G.draw[k0] * (W - 20))} height="26" class="seg on" />
					<line x1={10 + (a.u ?? 0) * (W - 20)} x2={10 + (a.u ?? 0) * (W - 20)} y1="2" y2="40" class="u" />
					<text x={10 + (a.u ?? 0) * (W - 20)} y="52" class="tick" text-anchor="middle">u</text>
				</svg>
				<figcaption>{t({ en: `u = ${fmt3(a.u ?? NaN)} lands on “${show(G.alphabet[a.token ?? 0])}” (chance ${fmtPct(a.p ?? NaN)}). Every character’s chance is laid end to end in the order of the alphabet (␣ ↵ a–z A–Z, punctuation, digits), the order the draw walks; the highlighted piece is the one u falls in.`, zh: `u = ${fmt3(a.u ?? NaN)} 落在“${show(G.alphabet[a.token ?? 0])}”上（概率 ${fmtPct(a.p ?? NaN)}）。所有字符的概率按字母表的顺序（␣ ↵ a–z A–Z、标点、数字）首尾相接，抽取就是按这个顺序走的；高亮的那段就是 u 落进去的那个字符。` })}</figcaption>
			</figure>
		{/if}
	</div>
{:else}
	<Notice text={{ en: 'Nothing to show for this text yet.', zh: '这段文字还没有可显示的内容。' }} />
{/if}

<style>
	figure {
		margin: 0;
	}
	.tl {
		display: grid;
		gap: 10px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
		margin: 0;
	}
	.scroll {
		overflow-x: auto;
		padding-bottom: 4px;
	}
	.strip {
		width: calc(var(--n) * var(--cw));
	}
	.chars {
		display: flex;
		font-family: var(--font-mono);
		font-size: 11px;
		line-height: 16px;
	}
	.chars span {
		width: var(--cw);
		flex: none;
		text-align: center;
		color: var(--ink-2);
	}
	.chars span.cur {
		color: var(--ink);
		font-weight: 700;
		background: var(--accent-soft);
	}
	.chars span.word {
		color: var(--ink);
		text-decoration: underline;
	}
	.chars.guess span {
		color: var(--muted);
	}
	.chars.guess span.hit {
		color: var(--real);
		font-weight: 700;
	}
	.heat {
		position: relative;
	}
	canvas {
		display: block;
		image-rendering: pixelated;
	}
	.row,
	.col {
		position: absolute;
		border: 1.5px solid var(--ink);
		pointer-events: none;
	}
	.row {
		left: 0;
		right: 0;
		height: 5px;
	}
	.col {
		top: -2px;
		bottom: -2px;
		width: 13px;
	}
	.foot {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 16px;
		align-items: center;
	}
	.note,
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		margin: 0;
	}
	.axis,
	.grid {
		stroke: var(--rule);
	}
	.tick {
		font-size: 14px;
		fill: var(--ink-2);
	}
	.med {
		fill: none;
		stroke: var(--accent);
		stroke-width: 2.5;
	}
	.pat {
		fill: none;
		stroke: var(--ink);
		stroke-width: 1.5;
		stroke-dasharray: 5 4;
	}
	.ref {
		fill: none;
		stroke: var(--muted);
		stroke-width: 1;
	}
	.tops ol {
		margin: 6px 0 0;
		padding-left: 1.6em;
		font-size: var(--fs-sm);
	}
	.mono {
		font-family: var(--font-mono);
		white-space: pre-wrap;
		word-break: break-all;
	}
	mark {
		background: var(--accent-soft);
		color: var(--ink);
		font-weight: 700;
	}
	.after,
	.prompt {
		color: var(--muted);
	}
	.txt {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.seg {
		fill: var(--surface-2);
		stroke: var(--rule-strong);
		stroke-width: 0.5;
	}
	.seg.on {
		fill: var(--accent-soft);
		fill-opacity: 0.55;
		stroke: var(--accent);
		stroke-width: 2;
	}
	.segl {
		font-family: var(--font-mono);
		font-size: 12px;
		fill: var(--ink);
	}
	.u {
		stroke: var(--ink);
		stroke-width: 2;
	}
</style>

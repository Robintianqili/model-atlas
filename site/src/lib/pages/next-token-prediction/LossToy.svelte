<!--
  Part 2a "The loss": drag p (the probability given to the correct token) along the -ln p curve.
  The learner's own tokens (live TinyGPT) sit on the same curve as dots; their average height is
  the loss of the text.
-->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Num from '$lib/kit/Num.svelte';
	import Term from '$lib/kit/Term.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import { sym } from '$lib/kit/tex';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtPct, visibleToken } from '$lib/kit/format';
	import type { NtpState } from './state.svelte';
	import { FACTS } from './facts';

	let { S }: { S: NtpState } = $props();

	const P_MIN = 0.0005;
	const L_MAX = -Math.log(P_MIN); // 7.6 nats
	const P_UNIFORM = 1 / FACTS.vocab;
	const P_TINYGPT = Math.exp(-FACTS.valLoss); // geometric-mean probability of the correct token on validation text

	let p = $state(0.25);
	let xLog = $state(false);
	let W = $state(640);
	const H = $derived(W < 480 ? 230 : 270);
	const m = { l: 46, r: 14, t: 14, b: 40 };

	const loss = $derived(-Math.log(p));
	const bits = $derived(-Math.log2(p));

	const xOf = (q: number) => {
		const w = W - m.l - m.r;
		if (xLog) return m.l + ((Math.log10(Math.max(q, P_MIN)) - Math.log10(P_MIN)) / -Math.log10(P_MIN)) * w;
		return m.l + q * w;
	};
	const pOfX = (x: number) => {
		const w = W - m.l - m.r;
		const u = Math.min(1, Math.max(0, (x - m.l) / w));
		const q = xLog ? 10 ** (Math.log10(P_MIN) * (1 - u)) : u;
		return Math.min(1, Math.max(0.001, q));
	};
	const yOf = (L: number) => m.t + (1 - Math.min(L, L_MAX) / L_MAX) * (H - m.t - m.b);

	const curve = $derived.by(() => {
		const pts: string[] = [];
		const N = 160;
		for (let i = 0; i <= N; i++) {
			const q = P_MIN ** (1 - i / N); // log-spaced: dense near 0 where the curve is steep
			pts.push(`${xOf(q).toFixed(1)},${yOf(-Math.log(q)).toFixed(1)}`);
		}
		return 'M' + pts.join('L');
	});
	const xTicks = $derived(xLog ? [0.001, 0.01, 0.1, 1] : [0, 0.25, 0.5, 0.75, 1]);
	const yTicks = [0, 1, 2, 3, 4, 5, 6, 7];

	// learner's tokens (live) on the same curve
	const dots = $derived.by(() => {
		if (!S.fresh || !S.score) return [];
		const out: { i: number; p: number; L: number }[] = [];
		for (let i = 1; i < S.score.p.length; i++) out.push({ i, p: S.score.p[i], L: -Math.log(S.score.p[i]) });
		return out;
	});
	const meanL = $derived(S.fresh && S.score && S.score.scored > 0 ? S.score.meanNats : null);

	let dragging = false;
	function onDown(e: PointerEvent) {
		const svg = e.currentTarget as SVGSVGElement;
		dragging = true;
		svg.setPointerCapture(e.pointerId);
		setFromEvent(e, svg);
	}
	function onMove(e: PointerEvent) {
		if (!dragging) return;
		setFromEvent(e, e.currentTarget as SVGSVGElement);
	}
	function onUp() {
		dragging = false;
	}
	function setFromEvent(e: PointerEvent, svg: SVGSVGElement) {
		const r = svg.getBoundingClientRect();
		p = round(pOfX(((e.clientX - r.left) / r.width) * W));
	}
	const round = (q: number) => (q >= 0.01 ? Math.round(q * 1000) / 1000 : Number(q.toPrecision(2)));

	function onKey(e: KeyboardEvent) {
		let q = p;
		if (e.key === 'ArrowRight' || e.key === 'ArrowUp') q = p < 0.01 ? p + 0.001 : p + 0.01;
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') q = p <= 0.01 ? p - 0.001 : p - 0.01;
		else if (e.key === 'PageUp') q = p * 2;
		else if (e.key === 'PageDown') q = p / 2;
		else if (e.key === 'Home') q = P_UNIFORM;
		else if (e.key === 'End') q = 1;
		else return;
		e.preventDefault();
		// the lowest p the keys reach is the random guess 1/1024 (Home), as in the preset
		if (q <= P_UNIFORM * 1.0001) {
			p = P_UNIFORM;
			return;
		}
		p = Math.min(1, Math.max(0.001, q < 0.01 ? Number(q.toPrecision(2)) : Math.round(q * 1000) / 1000));
	}

	const presets = [
		{ p: P_UNIFORM, label: { en: 'random guess 1/1024', zh: '随机猜 1/1024' } },
		{ p: Number(P_TINYGPT.toFixed(3)), label: { en: "TinyGPT's typical p", zh: 'TinyGPT 的典型 p' } },
		{ p: 0.5, label: { en: 'coin flip 0.5', zh: '抛硬币 0.5' } },
		{ p: 0.99, label: { en: 'almost sure 0.99', zh: '几乎确定 0.99' } }
	];

	const tex = $derived(
		`\\begin{aligned} ${sym('out', '\\mathcal{L}')} &= -\\ln ${sym('prob', 'p')} = -\\ln({{p}}) \\\\ &= {{L}}\\ \\text{${t({ en: 'nats', zh: '奈特' })}} = {{b}}\\ \\text{${t({ en: 'bits', zh: '比特' })}} \\end{aligned}`
	);
	let hoverDot = $state<{ i: number; p: number; L: number } | null>(null);
</script>

<div class="toy">
	<div class="prose">
		<Lang>
			{#snippet en()}
				<p>
					For every example, the model does not give one answer. It gives a probability to <em>every</em> token in its
					<Term id="vocabulary" /> (1,024 for TinyGPT), and they add up to 1. Training looks at only one of those numbers:
					<strong>p</strong>, the probability it gave to the token that really came next. The
					<Term id="loss" /> for that example is <strong>−ln p</strong>, the <Term id="cross-entropy" />.
				</p>
				<p>
					Drag the point. When the model is sure and right (p near 1) the loss is near 0. When it gives the right token a
					tiny probability, the loss shoots up. A model that knows nothing spreads its bets evenly, p = 1/1024, which
					costs ln 1024 = 6.93 nats every time.
				</p>
			{/snippet}
			{#snippet zh()}
				<p>
					对每个样本，模型给出的不是一个答案，而是给<Term id="vocabulary" />里的<em>每一个</em>词元一个概率（TinyGPT 有 1,024 个），加起来等于 1。训练只看其中一个数：<strong>p</strong>，也就是模型给真正出现的下一个词元的概率。这个样本的<Term id="loss" />就是 <strong>−ln p</strong>，即<Term id="cross-entropy" />。
				</p>
				<p>
					拖动这个点。模型很有把握且猜对了（p 接近 1），损失接近 0；它给正确词元的概率很小，损失就猛涨。什么都不懂的模型只能平均押注，p = 1/1024，每次都要付出 ln 1024 = 6.93 奈特的损失。
				</p>
			{/snippet}
		</Lang>
	</div>

	<div class="panel-row">
		<figure class="chart" bind:clientWidth={W}>
			<div class="chart-head">
				<span class="segmented" role="group" aria-label={t({ en: 'x-axis scale', zh: 'x 轴刻度' })}>
					<button type="button" aria-pressed={!xLog} onclick={() => (xLog = false)}>{t({ en: 'linear p', zh: '线性 p' })}</button>
					<button type="button" aria-pressed={xLog} onclick={() => (xLog = true)}>{t({ en: 'log p', zh: '对数 p' })}</button>
				</span>
				{#if dots.length}
					<span class="dots-key small">
						<i class="dotkey"></i>{t({ en: 'your tokens (TinyGPT)', zh: '你的词元（TinyGPT）' })}
						<ProvenanceBadge mode="live" compact />
					</span>
				{/if}
			</div>
			<!-- svelte-ignore a11y_no_static_element_interactions -->
			<svg
				width={W}
				height={H}
				viewBox="0 0 {W} {H}"
				class="plot"
				onpointerdown={onDown}
				onpointermove={onMove}
				onpointerup={onUp}
				onpointercancel={onUp}
				aria-label={t({ en: 'The curve loss = −ln p', zh: '曲线：损失 = −ln p' })}
			>
				{#each yTicks as L (L)}
					<line class="grid" x1={m.l} x2={W - m.r} y1={yOf(L)} y2={yOf(L)} />
					<text class="tick" x={m.l - 8} y={yOf(L) + 4} text-anchor="end">{L}</text>
				{/each}
				{#each xTicks as q (q)}
					<text class="tick" x={xOf(q)} y={H - m.b + 16} text-anchor="middle">{q}</text>
				{/each}
				<text class="axis" x={m.l + (W - m.l - m.r) / 2} y={H - 6} text-anchor="middle">
					{t({ en: 'p = probability given to the correct token', zh: 'p = 给正确词元的概率' })}
				</text>
				<text class="axis" x={12} y={m.t + (H - m.t - m.b) / 2} text-anchor="middle" transform="rotate(-90 12 {m.t + (H - m.t - m.b) / 2})">
					{t({ en: 'loss −ln p (nats)', zh: '损失 −ln p（奈特）' })}
				</text>

				<path class="curve" d={curve} />

				{#if meanL !== null}
					<line class="mean" x1={m.l} x2={W - m.r} y1={yOf(meanL)} y2={yOf(meanL)} />
					<text class="meanlbl" x={xOf(xLog ? 0.002 : 0.06)} y={yOf(meanL) - 6} text-anchor="start">
						{t({ en: 'your text: average', zh: '你的文本：平均' })} {fmt3(meanL)}
					</text>
				{/if}

				<!-- reference points -->
				<circle class="ref" cx={xOf(P_UNIFORM)} cy={yOf(-Math.log(P_UNIFORM))} r="4" />
				<text class="reflbl" x={xOf(P_UNIFORM) + 8} y={yOf(-Math.log(P_UNIFORM)) + 4}>
					{t({ en: 'random guess', zh: '随机猜' })} 6.93
				</text>

				{#each dots as d (d.i)}
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<circle
						class="dot"
						class:on={S.active === d.i}
						cx={xOf(d.p)}
						cy={yOf(d.L)}
						r={S.active === d.i ? 6 : 4}
						onpointerenter={() => {
							hoverDot = d;
							S.hover = d.i;
						}}
						onpointerleave={() => {
							hoverDot = null;
							S.hover = null;
						}}
						onpointerdown={(e) => {
							e.stopPropagation();
							S.focus = d.i;
						}}
					/>
				{/each}

				<line class="drop" x1={xOf(p)} x2={xOf(p)} y1={yOf(loss)} y2={H - m.b} />
				<line class="drop" x1={m.l} x2={xOf(p)} y1={yOf(loss)} y2={yOf(loss)} />
				<circle
					class="handle"
					cx={xOf(p)}
					cy={yOf(loss)}
					r="9"
					tabindex="0"
					role="slider"
					aria-label={t({ en: 'p, the probability of the correct token', zh: 'p：正确词元的概率' })}
					aria-valuemin={P_UNIFORM}
					aria-valuemax={1}
					aria-valuenow={p}
					aria-valuetext={`p = ${fmt3(p)}, loss ${fmt3(loss)} nats`}
					onkeydown={onKey}
				/>
			</svg>
			{#if hoverDot && S.score}
				<p class="dot-read small">
					{t({ en: 'Token', zh: '词元' })}
					<span class="tokchip">{visibleToken(S.disp(S.score.ids[hoverDot.i]))}</span>:
					p = <Num value={hoverDot.p} plain />, −ln p = <Num value={hoverDot.L} plain />
				</p>
			{:else}
				<p class="dot-read small muted">
					{t({
						en: 'Drag anywhere on the chart, or focus the point and use the arrow keys (PageUp/PageDown double or halve p).',
						zh: '在图上任意位置拖动，或聚焦这个点后用方向键调整（PageUp/PageDown 让 p 加倍或减半）。'
					})}
				</p>
			{/if}
		</figure>

		<div class="side">
			<div class="presets" role="group" aria-label={t({ en: 'Set p', zh: '设置 p' })}>
				{#each presets as pr (pr.p)}
					<button type="button" class="chip" onclick={() => (p = pr.p)} aria-pressed={Math.abs(p - pr.p) < 1e-9}>
						{t(pr.label)}
					</button>
				{/each}
			</div>
			<FormulaCard {tex} values={{ p, L: loss, b: bits }} caption={{ en: 'Loss for one example', zh: '一个样本的损失' }} />
			<div class="split" aria-label={t({ en: 'How the probability is split', zh: '概率如何分配' })}>
				<div class="bar">
					<span class="seg right" style="width: {Math.max(0.6, p * 100)}%"></span>
					<span class="seg rest" style="width: {Math.max(0, 100 - Math.max(0.6, p * 100))}%"></span>
				</div>
				<div class="splitlbl small">
					<span><i class="sw right"></i>{t({ en: 'correct token', zh: '正确词元' })} {fmtPct(p)}</span>
					<span><i class="sw rest"></i>{t({ en: 'the other 1,023 tokens', zh: '其余 1,023 个词元' })} {fmtPct(1 - p)}</span>
				</div>
			</div>
			<p class="note small">
				{#if p < 0.002}
					{t({ en: 'This is as bad as guessing blindly among all 1,024 tokens.', zh: '这和在全部 1,024 个词元里盲猜一样糟。' })}
				{:else if p < 0.1}
					{t({ en: 'Halving p always adds the same amount: ln 2 = 0.693 nats (exactly 1 bit). Switch to the log axis and the curve becomes a straight line.', zh: '把 p 减半，损失总是增加同样多：ln 2 = 0.693 奈特（正好 1 比特）。切换到对数坐标，曲线就变成一条直线。' })}
				{:else if p < 0.9}
					{t({ en: 'Raising p for the right token means taking probability away from all the others: they must still add up to 1.', zh: '提高正确词元的 p，就意味着要从其他所有词元那里拿走概率：它们的总和仍然必须是 1。' })}
				{:else}
					{t({ en: 'Near p = 1 the curve is flat: being even more sure barely lowers the loss.', zh: '在 p = 1 附近曲线很平：再更有把握一点，损失也几乎不再下降。' })}
				{/if}
			</p>
		</div>
	</div>

	<div class="prose">
		<Lang>
			{#snippet en()}
				<p>
					Two units appear on this page. With the natural log, loss is in <strong>nats</strong>; with log base 2, it is in
					<strong>bits</strong> (1 nat = 1.443 bits). Bits have a handy reading: −log₂ p is how many yes/no questions'
					worth of surprise the token was. Training lowers the average −ln p over billions of examples by
					<Term id="gradient-descent" />: each step nudges the weights so the tokens that really came next get a little more
					probability.
				</p>
				<p>
					The grey dots are your own tokens, scored by TinyGPT. They all lie on the curve, because each one is just a p and
					its −ln p. The loss of your whole text is their average height (the dashed line). The next section shows where
					each dot comes from.
				</p>
			{/snippet}
			{#snippet zh()}
				<p>
					本页会出现两种单位。用自然对数，损失的单位是<strong>奈特（nat）</strong>；用以 2 为底的对数，单位是<strong>比特（bit）</strong>（1 奈特 = 1.443 比特）。比特有个好懂的解释：−log₂ p 表示这个词元带来的“意外”，相当于多少个是/否问题。训练就是用<Term id="gradient-descent" />在几十亿个样本上降低 −ln p 的平均值：每一步都轻轻调整权重，让真正出现的下一个词元多得到一点概率。
				</p>
				<p>
					图上的灰点是你自己的词元，由 TinyGPT 打分。它们都落在曲线上，因为每个点不过是一个 p 和它的 −ln p。整段文本的损失就是这些点的平均高度（虚线）。下一节会展示每个点是怎么来的。
				</p>
			{/snippet}
		</Lang>
	</div>
</div>

<style>
	.toy {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
	}
	.prose {
		max-width: var(--measure);
	}
	.prose p {
		margin: 0 0 0.8em;
	}
	.panel-row {
		display: grid;
		grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr);
		gap: var(--s5);
		align-items: start;
	}
	@media (max-width: 860px) {
		.panel-row {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.chart {
		margin: 0;
		min-width: 0;
		display: grid;
		gap: 6px;
	}
	.chart-head {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 16px;
		align-items: center;
		justify-content: space-between;
	}
	.dots-key {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		color: var(--ink-2);
	}
	.dotkey {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 50%;
		background: var(--muted);
		box-shadow: 0 0 0 2px var(--surface);
	}
	.plot {
		display: block;
		width: 100%;
		height: auto;
		touch-action: pan-y;
		user-select: none;
		background: var(--surface);
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		cursor: crosshair;
	}
	.grid {
		stroke: var(--rule);
		stroke-width: 1;
	}
	.tick {
		font-family: var(--font-mono);
		font-size: 11px;
		fill: var(--muted);
	}
	.axis {
		font-size: 12px;
		fill: var(--ink-2);
	}
	.curve {
		fill: none;
		stroke: var(--ink);
		stroke-width: 2;
		stroke-linejoin: round;
	}
	.mean {
		stroke: var(--ink-2);
		stroke-width: 1.5;
		stroke-dasharray: 5 4;
	}
	.meanlbl,
	.reflbl {
		font-size: 11px;
		fill: var(--ink-2);
		paint-order: stroke;
		stroke: var(--surface);
		stroke-width: 3px;
	}
	.ref {
		fill: var(--muted);
		stroke: var(--surface);
		stroke-width: 2;
	}
	.dot {
		fill: var(--muted);
		stroke: var(--surface);
		stroke-width: 2;
		cursor: pointer;
	}
	.dot.on {
		fill: var(--highlight);
	}
	.drop {
		stroke: var(--seq-mid);
		stroke-width: 1;
	}
	.handle {
		fill: var(--seq-hi);
		stroke: var(--surface);
		stroke-width: 3;
		cursor: grab;
	}
	.handle:focus-visible {
		outline: none;
		stroke: var(--accent);
		stroke-width: 4;
	}
	.dot-read {
		margin: 0;
		min-height: 1.5em;
	}
	.tokchip {
		font-family: var(--font-mono);
		background: var(--surface-2);
		padding: 0 4px;
		border-radius: 3px;
		white-space: pre;
	}
	.side {
		display: grid;
		gap: 12px;
		min-width: 0;
	}
	.presets {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.presets .chip {
		cursor: pointer;
		font: inherit;
		font-size: var(--fs-xs);
	}
	.presets .chip[aria-pressed='true'] {
		background: var(--accent-soft);
		border-color: var(--accent);
	}
	.split {
		display: grid;
		gap: 6px;
	}
	.bar {
		display: flex;
		height: 16px;
		gap: 2px;
		border-radius: 4px;
		overflow: hidden;
	}
	.seg {
		display: block;
		height: 100%;
		transition: width var(--dur-1) linear;
	}
	.seg.right,
	.sw.right {
		background: var(--seq-hi);
	}
	.seg.rest,
	.sw.rest {
		background: var(--surface-3);
	}
	.splitlbl {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		color: var(--ink-2);
	}
	.sw {
		display: inline-block;
		width: 10px;
		height: 10px;
		border-radius: 2px;
		margin-right: 5px;
	}
	.note {
		margin: 0;
		color: var(--ink-2);
	}
	.small {
		font-size: var(--fs-xs);
	}
</style>

<!--
  Lesson step 4: the loss of the followed window is −ln p(the real next token). The −ln p curve,
  the window's point on it (live, final fp32), what the loss would have been for the model's other
  top guesses, the blind-guess reference, and the formula with the real value.
-->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { sym } from '$lib/kit/tex';
	import { t, colon } from '$lib/i18n/i18n.svelte';
	import { fmt3, visibleToken } from '$lib/kit/format';
	import { pct } from './lessonMath';
	import LessonWindow from './LessonWindow.svelte';
	import type { NtpLesson } from './lessonState.svelte';
	import { FACTS } from './facts';

	let { L }: { L: NtpLesson } = $props();
	const S = $derived(L.S);
	const d = $derived(L.dist('final'));

	let W = $state(560);
	const H = $derived(W < 420 ? 220 : 250);
	const m = { l: 40, r: 14, t: 16, b: 38 };
	const loss = $derived(d ? -Math.log(d.p) : 0);
	const yMax = $derived(Math.min(24, Math.max(7.6, loss + 0.8)));
	const xOf = (p: number) => m.l + p * (W - m.l - m.r);
	const yOf = (v: number) => m.t + (1 - Math.min(v, yMax) / yMax) * (H - m.t - m.b);
	const curve = $derived.by(() => {
		const pMin = Math.exp(-yMax);
		const pts: string[] = [];
		const N = 180;
		for (let i = 0; i <= N; i++) {
			const q = pMin ** (1 - i / N);
			pts.push(`${xOf(q).toFixed(1)},${yOf(-Math.log(q)).toFixed(1)}`);
		}
		return 'M' + pts.join('L');
	});
	const yTicks = $derived.by(() => {
		const st = yMax > 12 ? 4 : yMax > 9 ? 2 : 1;
		const out: number[] = [];
		for (let v = 0; v <= yMax; v += st) out.push(v);
		return out;
	});
	/** the model's other top guesses: what the loss would have been had the sentence continued with them */
	const alts = $derived(d ? d.top.filter((x) => x.id !== d.target).slice(0, 3) : []);
	const best = $derived(alts[0]);
	const uni = 1 / FACTS.vocab;

	const tex = `\\begin{aligned} ${sym('out', '\\mathcal{L}')} &= -\\ln ${sym('prob', 'p')} = -\\ln({{p}}) \\\\ &= {{L}}\\ \\text{${'nats'}} \\end{aligned}`;
	const texZh = `\\begin{aligned} ${sym('out', '\\mathcal{L}')} &= -\\ln ${sym('prob', 'p')} = -\\ln({{p}}) \\\\ &= {{L}}\\ \\text{奈特} \\end{aligned}`;
	const lab = (id: number) => visibleToken(S.disp(id));
	// keep the window's label and the blind-guess label apart when the two points are close
	const near = $derived(Math.abs(loss - FACTS.uniformLoss) < 1.2);
	const tLabelY = $derived(near && loss < FACTS.uniformLoss ? yOf(loss) + 20 : yOf(loss) - 10);
	const uLabelY = $derived(near && loss >= FACTS.uniformLoss ? yOf(FACTS.uniformLoss) + 18 : yOf(FACTS.uniformLoss) + 4);
	const lx = (p: number) => (xOf(p) > W * 0.62 ? 'end' : 'start');

	// ---- label layout: estimated text widths (mono: 0.6 em per Latin character, 1 em per CJK one)
	type Lab = { x: number; y: number; anchor: 'start' | 'end'; w: number };
	const textW = (s: string, px: number) => [...s].reduce((a, c) => a + (/[⺀-鿿＀-￯]/.test(c) ? px : 0.6 * px), 0);
	const span = (l: Lab) => (l.anchor === 'end' ? [l.x - l.w, l.x] : [l.x, l.x + l.w]);
	const clash = (a: Lab, b: Lab) => Math.abs(a.y - b.y) < 13 && span(a)[0] < span(b)[1] + 4 && span(b)[0] < span(a)[1] + 4;
	const iw = $derived(W - m.l - m.r);
	const mainText = $derived(d ? `“${lab(d.target)}”${colon()} p = ${fmt3(d.p)} → ${fmt3(loss)}` : '');
	/** this window's label: next to its point, and lifted above the curve when it sits left of the point */
	const mainLab = $derived.by((): Lab | null => {
		if (!d) return null;
		const anchor = lx(d.p);
		const w = textW(mainText, 12.5);
		const x = xOf(d.p) + (anchor === 'end' ? -12 : 12);
		let y = tLabelY;
		if (anchor === 'end') {
			// the curve is highest at the label's left end: keep the text's baseline above it there
			const p0 = Math.min(1, Math.max(0.02, (x - w - m.l) / iw));
			y = Math.min(y, yOf(-Math.log(p0)) - 5);
		}
		return { x, y: Math.max(m.t + 11, y), anchor, w };
	});
	const blindText = $derived(`${t({ en: 'blind guess 1/1,024', zh: '盲猜 1/1,024' })} → 6.93`);
	/** the model's other top guesses: a label only where it does not land on another label */
	const altLabs = $derived.by(() => {
		const placed: Lab[] = [{ x: xOf(uni) + 8, y: uLabelY, anchor: 'start', w: textW(blindText, 11.5) }];
		if (mainLab) placed.push(mainLab);
		return alts.map((a) => {
			const anchor = lx(a.p);
			const l: Lab = { x: xOf(a.p) + (anchor === 'end' ? -8 : 8), y: yOf(-Math.log(a.p)) - 6, anchor, w: textW(`“${lab(a.id)}”`, 11.5) };
			const show = !placed.some((o) => clash(o, l));
			if (show) placed.push(l);
			return { ...a, l, show };
		});
	});
</script>

<div class="ls">
	<div class="top">
		<LessonWindow {L} />
		<ProvenanceBadge mode="live" compact precision="fp32" tolerance={FACTS.tolerance} against="PyTorch fp32" />
	</div>
	{#if L.pos < 1}
		<p class="muted">{t({ en: 'Type at least two tokens.', zh: '至少输入两个词元。' })}</p>
	{:else if !d}
		<p class="muted">{t({ en: 'TinyGPT is reading this window…', zh: 'TinyGPT 正在读这个窗口…' })}</p>
	{:else}
		<figure class="chart" bind:clientWidth={W}>
			<svg width={W} height={H} viewBox="0 0 {W} {H}" role="img" aria-label={t({ en: `The curve loss = −ln p. This window: p = ${fmt3(d.p)}, loss ${fmt3(loss)} nats.`, zh: `曲线：损失 = −ln p。这个窗口：p = ${fmt3(d.p)}，损失 ${fmt3(loss)} 奈特。` })}>
				{#each yTicks as v (v)}
					<line class="grid" x1={m.l} x2={W - m.r} y1={yOf(v)} y2={yOf(v)} />
					<text class="tick" x={m.l - 7} y={yOf(v) + 4} text-anchor="end">{v}</text>
				{/each}
				{#each [0, 0.25, 0.5, 0.75, 1] as q (q)}
					<text class="tick" x={xOf(q)} y={H - m.b + 15} text-anchor="middle">{q}</text>
				{/each}
				<text class="axis" x={m.l + (W - m.l - m.r) / 2} y={H - 5} text-anchor="middle">
					{t({ en: 'p = probability given to the real next token', zh: 'p = 给真实下一个词元的概率' })}
				</text>
				<text class="axis" x={11} y={m.t + (H - m.t - m.b) / 2} text-anchor="middle" transform="rotate(-90 11 {m.t + (H - m.t - m.b) / 2})">
					{t({ en: 'loss −ln p (nats)', zh: '损失 −ln p（奈特）' })}
				</text>
				<path class="curve" d={curve} />

				<!-- blind guess -->
				<circle class="ref" cx={xOf(uni)} cy={yOf(-Math.log(uni))} r="4" />
				<text class="lbl muted-l" x={xOf(uni) + 8} y={uLabelY}>{blindText}</text>

				<!-- the model's other top guesses (a label only where it does not land on another) -->
				{#each altLabs as a (a.id)}
					<circle class="alt" cx={xOf(a.p)} cy={yOf(-Math.log(a.p))} r="4" />
					{#if a.show}<text class="lbl muted-l" x={a.l.x} y={a.l.y} text-anchor={a.l.anchor}>“{lab(a.id)}”</text>{/if}
				{/each}

				<!-- this window -->
				<line class="drop" x1={xOf(d.p)} x2={xOf(d.p)} y1={yOf(loss)} y2={H - m.b} />
				<line class="drop" x1={m.l} x2={xOf(d.p)} y1={yOf(loss)} y2={yOf(loss)} />
				<circle class="pt" cx={xOf(d.p)} cy={yOf(loss)} r="7" />
				{#if mainLab}
					<text class="lbl strong" x={mainLab.x} y={mainLab.y} text-anchor={mainLab.anchor}>{mainText}</text>
				{/if}
			</svg>
		</figure>
		<div class="cols">
			<FormulaCard
				tex={t({ en: tex, zh: texZh })}
				values={{ p: d.p, L: loss }}
				caption={{ en: 'The loss of this one window', zh: '这一个窗口的损失' }}
				provenance="live"
			/>
			<div class="notes small">
				<p>
					{t({ en: 'In bits (log base 2):', zh: '换成比特（以 2 为底）：' })} −log₂ p = <Num value={-Math.log2(d.p)} />
					{t({ en: 'bits.', zh: '比特。' })}
				</p>
				{#if best}
					<p>
						{t({ en: 'Had your sentence continued with', zh: '如果你的句子接下来是' })}
						<span class="tok">{lab(best.id)}</span>
						{t({ en: `instead (p = ${pct(best.p)}), the loss would have been`, zh: `（p = ${pct(best.p)}），损失就会是` })}
						<strong><Num value={-Math.log(best.p)} /></strong>{t({ en: ' nats.', zh: ' 奈特。' })}
					</p>
				{/if}
				<p class="muted">
					{t({
						en: 'Only the real next token counts. The other 1,023 probabilities matter only through it: they must leave room for it, since all add up to 1.',
						zh: '只有真实的下一个词元算数。其余 1,023 个概率只通过它起作用：它们加起来必须是 1，所以得给它留出位置。'
					})}
				</p>
			</div>
		</div>
	{/if}
</div>

<style>
	.ls {
		container: ls / inline-size;
		display: grid;
		gap: 12px;
		min-width: 0;
	}
	.top {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 8px;
		align-items: flex-start;
		justify-content: flex-end;
	}
	.top > :global(.wp) {
		flex: 1 1 280px;
	}
	.chart {
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
		font-size: 10.5px;
		fill: var(--muted);
	}
	.axis {
		font-size: 11.5px;
		fill: var(--ink-2);
	}
	.curve {
		fill: none;
		stroke: var(--ink);
		stroke-width: 2;
		stroke-linejoin: round;
	}
	.ref {
		fill: none;
		stroke: var(--muted);
		stroke-width: 2;
	}
	.alt {
		fill: var(--muted);
		stroke: var(--surface);
		stroke-width: 2;
	}
	.drop {
		stroke: var(--real);
		stroke-width: 1.2;
		stroke-dasharray: 3 3;
	}
	.pt {
		fill: var(--real);
		stroke: var(--surface);
		stroke-width: 3;
	}
	.lbl {
		font-size: 11.5px;
		paint-order: stroke;
		stroke: var(--surface);
		stroke-width: 3px;
		font-family: var(--font-mono);
	}
	.muted-l {
		fill: var(--ink-2);
	}
	.strong {
		font-weight: 700;
		fill: var(--ink);
		font-size: 12.5px;
	}
	.cols {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		gap: 14px;
		align-items: start;
	}
	@container ls (max-width: 520px) {
		.cols {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.notes {
		display: grid;
		gap: 6px;
	}
	.notes p {
		margin: 0;
	}
	.small {
		font-size: var(--fs-xs);
	}
	.tok {
		font-family: var(--font-mono);
		background: var(--surface-2);
		padding: 0 4px;
		border-radius: 3px;
		white-space: pre;
	}
</style>

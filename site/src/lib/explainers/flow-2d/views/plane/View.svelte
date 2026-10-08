<!-- flow-2d/plane (props: meta.ts). Canvas from −2.5 to 2.5; a time slider in modes 'slider' and 'field'. Nothing moves
     by itself: the picture follows the step's frame (Play steps through the frames) and the slider. -->
<script lang="ts">
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import type { I18n } from '../../../../atlas/types/core.ts';
	import { t as tr } from '$lib/i18n/i18n.svelte';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3, fmtInt } from '../../../../atlas/core/format.ts';
	import { dataOf, SPAN, velocity } from '../../flow.ts';
	import { LABELS, presetOf } from '../../adapters/points2d/adapter.ts';
	import { describePlane, middleOf, modeOf, pairIndex, pairsOf, paramsOf, predOf, runEnd, samplesOf, spreadAround } from './meta.ts';

	let { rt, at, props }: StageProps = $props();
	const p = $derived({ rt, at, props });
	const mode = $derived(modeOf(p));
	const data = $derived(dataOf(rt.input));
	const pairs = $derived(['noise', 'pairs', 'slider', 'pair'].includes(mode) ? pairsOf(p) : null);
	const k = $derived(pairIndex(p));
	const pred = $derived(mode === 'pair' ? predOf(p, k) : null);
	const arrows = $derived((props?.arrows as string[] | undefined) ?? ['target']);
	let tNow = $state<number | null>(null);
	const tv = $derived(tNow ?? (typeof props?.t === 'number' ? (props.t as number) : 0.5));
	const G = 20;
	const field = $derived.by(() => {
		const pr = mode === 'field' ? paramsOf(p) : null;
		if (!pr) return null;
		const xy = new Float64Array(2 * G * G);
		for (let i = 0; i < G * G; i++) (xy[2 * i] = -SPAN + ((i % G) + 0.5) * ((2 * SPAN) / G)), (xy[2 * i + 1] = -SPAN + (Math.floor(i / G) + 0.5) * ((2 * SPAN) / G));
		return { xy, v: velocity(pr, xy, tv, G * G) };
	});
	const samples = $derived(mode === 'euler' ? samplesOf(p) : null);
	const few = $derived(mode === 'few' ? [runEnd(p, 'moveFew'), runEnd(p, 'move')] : null);
	const mid = $derived(middleOf(data));
	const spreads = $derived(few ? few.map((e) => (e ? spreadAround(e.x.data, e.x.shape[0], mid) : NaN)) : []);
	const description = $derived(tr(describePlane(p)));

	/** half the width of the plane drawn: ±2.5 (all the noise), zoomed onto the pair in mode 'pair' and onto the data in 'few' */
	const span = $derived.by(() => {
		let m = 0;
		if (mode === 'pair' && pairs && k < pairs.n) for (const v of [pairs.x0[2 * k], pairs.x0[2 * k + 1], pairs.x1[2 * k], pairs.x1[2 * k + 1], ...(pred ?? [])]) m = Math.max(m, Math.abs(v));
		else if (mode === 'few') for (const v of data.xy) m = Math.max(m, Math.abs(v));
		return m > 0 ? Math.min(SPAN, Math.max(1.2, 1.25 * m)) : SPAN;
	});
	let W = $state(360);
	const side = $derived(mode === 'few' ? Math.max(140, Math.min(300, (W - 12) / 2)) : Math.max(220, Math.min(W, 440)));
	let c0 = $state<HTMLCanvasElement | null>(null);
	let c1 = $state<HTMLCanvasElement | null>(null);
	const theme = $derived(themeState.resolved);

	type Ctx = { g: CanvasRenderingContext2D; s: number; r: number; col: Record<string, string> };
	const X = (c: Ctx, v: number) => ((v + c.r) / (2 * c.r)) * c.s;
	const Y = (c: Ctx, v: number) => ((c.r - v) / (2 * c.r)) * c.s;
	function dots(c: Ctx, xy: ArrayLike<number>, n: number, color: string, r: number, alpha = 1) {
		Object.assign(c.g, { fillStyle: color, globalAlpha: alpha });
		for (let i = 0; i < n; i++) c.g.beginPath(), c.g.arc(X(c, xy[2 * i]), Y(c, xy[2 * i + 1]), r, 0, 2 * Math.PI), c.g.fill();
		c.g.globalAlpha = 1;
	}
	/** stroke one path in one colour and width; `draw` adds its segments */
	function stroke(c: Ctx, color: string, w: number, draw: (g: CanvasRenderingContext2D) => void, alpha = 1) {
		Object.assign(c.g, { strokeStyle: color, lineWidth: w, globalAlpha: alpha, lineJoin: 'round' });
		c.g.beginPath();
		draw(c.g);
		c.g.stroke();
		c.g.globalAlpha = 1;
	}
	const ring = (c: Ctx, x: number, y: number) => stroke(c, c.col.accent, 2.5, (g) => g.arc(X(c, x), Y(c, y), 7, 0, 2 * Math.PI));
	/** an arrow from (x, y) by (dx, dy). With a halo (mode 'pair') it is outlined in the background colour and a short
	 *  arrow keeps a head of at least 6 px, so the untrained network's tiny arrow stays visible over the dots */
	function arrow(c: Ctx, x: number, y: number, dx: number, dy: number, color: string, w: number, halo = '') {
		const [ax, ay, bx, by] = [X(c, x), Y(c, y), X(c, x + dx), Y(c, y + dy)];
		const len = Math.hypot(bx - ax, by - ay);
		if (len < 0.5) return;
		const [ux, uy, h] = [(bx - ax) / len, (by - ay) / len, Math.min(4 + 2 * w, halo ? Math.max(0.45 * len, Math.min(6, len)) : 0.45 * len)];
		for (const [col, lw] of (halo ? [[halo, w + 3], [color, w]] : [[color, w]]) as [string, number][]) {
			stroke(c, col, lw, (g) => (g.moveTo(ax, ay), g.lineTo(bx - ux * h * 0.6, by - uy * h * 0.6)));
			c.g.fillStyle = col;
			c.g.beginPath();
			c.g.moveTo(bx, by);
			c.g.lineTo(bx - h * ux - 0.55 * h * uy, by - h * uy + 0.55 * h * ux);
			c.g.lineTo(bx - h * ux + 0.55 * h * uy, by - h * uy - 0.55 * h * ux);
			c.g.closePath();
			c.g.fill();
			if (lw > w) c.g.stroke();
		}
	}
	function setup(cv: HTMLCanvasElement, s: number): Ctx | null {
		const dpr = Math.min(3, globalThis.devicePixelRatio || 1);
		cv.width = Math.round(s * dpr);
		cv.height = Math.round(s * dpr);
		const g = cv.getContext('2d');
		if (!g) return null;
		g.setTransform(dpr, 0, 0, dpr, 0, 0);
		const css = getComputedStyle(cv);
		const v = (k: string, d: string) => css.getPropertyValue(k).trim() || d;
		const col = { surface: v('--surface', '#fffdf8'), rule: v('--rule', '#ccc'), ink: v('--ink', '#222'), ink2: v('--ink-2', '#555'), real: v('--real', '#1a8a4a'), gen: v('--generated', '#c2338a'), accent: v('--accent', '#0b6b70') };
		const c: Ctx = { g, s, r: span, col };
		g.fillStyle = col.surface;
		g.fillRect(0, 0, s, s);
		stroke(c, col.rule, 1, () => (g.moveTo(s / 2, 0), g.lineTo(s / 2, s), g.moveTo(0, s / 2), g.lineTo(s, s / 2)));
		g.font = `600 13px ${css.fontFamily}`;
		return c;
	}
	/** a label beside a point, pushed away from (ox, oy) in screen space */
	function label(c: Ctx, text: string, x: number, y: number, ox: number, oy: number) {
		const [sx, sy] = [X(c, x), Y(c, y)];
		const [dx, dy] = [sx - X(c, ox), sy - Y(c, oy)];
		const d = Math.hypot(dx, dy) || 1;
		const w = c.g.measureText(text).width;
		const [tx, ty] = [Math.max(2, Math.min(c.s - w - 2, sx + (dx / d) * 14 - w / 2)), Math.max(12, Math.min(c.s - 3, sy + (dy / d) * 14 + 4))];
		// a halo in the background colour keeps the label readable over the dots and lines
		Object.assign(c.g, { strokeStyle: c.col.surface, lineWidth: 4, lineJoin: 'round', fillStyle: c.col.ink });
		c.g.strokeText(text, tx, ty);
		c.g.fillText(text, tx, ty);
	}
	function paint(c: Ctx, panel: number) {
		const faint = ['field', 'euler', 'few'].includes(mode);
		dots(c, data.xy, data.n, c.col.real, data.n > 600 ? 1.6 : 2.2, faint ? 0.35 : 0.9);
		if (pairs) {
			const n = pairs.n;
			if (mode !== 'noise')
				stroke(c, c.col.ink2, 0.8, (g) => { for (let i = 0; i < n; i++) g.moveTo(X(c, pairs.x0[2 * i]), Y(c, pairs.x0[2 * i + 1])), g.lineTo(X(c, pairs.x1[2 * i]), Y(c, pairs.x1[2 * i + 1])); }, mode === 'pair' ? 0.12 : 0.25);
			dots(c, pairs.x0, n, c.col.ink2, 2, mode === 'pair' ? 0.35 : 0.8);
			const tAt = (i: number) => (mode === 'slider' ? tv : pairs.t[i]);
			const xt = (i: number, d: number) => (1 - tAt(i)) * pairs.x0[2 * i + d] + tAt(i) * pairs.x1[2 * i + d];
			if (mode === 'slider') dots(c, Float64Array.from({ length: 2 * n }, (_, j) => xt(j >> 1, j & 1)), n, c.col.accent, 2.4);
			if (mode !== 'noise' && k < n) {
				// the followed pair: its line (thick in mode 'pairs', before x_t is introduced) and its ends; from mode
				// 'slider' on also its point x_t, ringed, with the arrows drawn on top of the ring
				const [a0, b0, a1, b1] = [pairs.x0[2 * k], pairs.x0[2 * k + 1], pairs.x1[2 * k], pairs.x1[2 * k + 1]];
				stroke(c, c.col.accent, mode === 'pairs' ? 3.5 : 2, (g) => (g.moveTo(X(c, a0), Y(c, b0)), g.lineTo(X(c, a1), Y(c, b1))));
				dots(c, [a0, b0], 1, c.col.ink2, 4.5);
				dots(c, [a1, b1], 1, c.col.real, 4.5);
				const [px, py] = [xt(k, 0), xt(k, 1)];
				if (mode !== 'pairs') {
					ring(c, px, py);
					if (mode === 'pair') {
						if (arrows.includes('target')) arrow(c, px, py, pairs.target[2 * k], pairs.target[2 * k + 1], c.col.real, 2.5, c.col.surface);
						if (arrows.includes('pred') && pred) arrow(c, px, py, pred[0], pred[1], c.col.gen, 2.5, c.col.surface);
					}
					label(c, 'x_t', px, py, px + (b1 - b0), py - (a1 - a0));
				}
				// labels last, so no arrow covers them; x₁ sits beside the line (the target arrow runs on past x₁)
				label(c, 'x₀', a0, b0, a1, b1);
				label(c, 'x₁', a1, b1, a1 - (b1 - b0), b1 + (a1 - a0));
			}
		}
		if (field) {
			let m = 0;
			for (let i = 0; i < G * G; i++) m = Math.max(m, Math.hypot(field.v[2 * i], field.v[2 * i + 1]));
			const sc = m > 0 ? (0.9 * ((2 * SPAN) / G)) / m : 0;
			for (let i = 0; i < G * G; i++) arrow(c, field.xy[2 * i], field.xy[2 * i + 1], sc * field.v[2 * i], sc * field.v[2 * i + 1], c.col.ink2, 1.1);
		}
		if (samples) {
			// every point's path (point i's positions in the frames so far), then point 0's, the one the narration follows
			const trace = (g: CanvasRenderingContext2D, i: number) => samples.path.forEach((q, j) => (j ? g.lineTo : g.moveTo).call(g, X(c, q[2 * i]), Y(c, q[2 * i + 1])));
			stroke(c, c.col.gen, 0.8, (g) => { for (let i = 0; i < samples.x.shape[0]; i++) trace(g, i); }, 0.22);
			dots(c, samples.x.data, samples.x.shape[0], c.col.gen, 2.2);
			stroke(c, c.col.accent, 2, (g) => trace(g, 0));
			ring(c, samples.x.data[0], samples.x.data[1]);
		}
		const end = few?.[panel];
		if (end) {
			dots(c, end.x.data, end.x.shape[0], c.col.gen, 2);
			// the middle of the data (the mean of its points), which the arrows at t = 0 point to
			const [mx, my] = [X(c, mid[0]), Y(c, mid[1])];
			for (const [col, lw] of [[c.col.surface, 5], [c.col.ink, 2]] as [string, number][]) stroke(c, col, lw, (g) => (g.moveTo(mx - 7, my), g.lineTo(mx + 7, my), g.moveTo(mx, my - 7), g.lineTo(mx, my + 7)));
		}
	}
	$effect(() => {
		void [p, side, theme, tv, field, samples, span];
		for (const [cv, panel] of [[c0, 0], [c1, 1]] as const) {
			const c = cv ? setup(cv, side) : null;
			if (c) paint(c, panel);
		}
	});

	const L = (en: string, zh: string): I18n => ({ en, zh });
	const keys = $derived.by(() => {
		const ks: [string, I18n][] = [['real', L('your points (data)', '你的点（数据）')]];
		if (pairs) ks.push(['ink2', L('noise', '噪声')]);
		if (pairs && mode === 'pairs') ks.push(['accent', L(`pair ${k} (the thick line)`, `第 ${k} 对（粗线）`)]);
		else if (pairs && mode !== 'noise') ks.push(['accent', L(mode === 'slider' ? `x_t at the slider’s t; pair ${k} ringed` : `pair ${k} and its point x_t`, mode === 'slider' ? `滑块时刻 t 的 x_t；圈出的是第 ${k} 对` : `第 ${k} 对和它的点 x_t`)]);
		if (mode === 'pair') ks.push(['real', L('target arrow x₁ − x₀', '目标箭头 x₁ − x₀')]);
		if (mode === 'pair' && arrows.includes('pred')) ks.push(['gen', L('the network’s arrow', '网络给出的箭头')]);
		if (field) ks.push(['ink2', L('the network’s arrows (lengths scaled to fit)', '网络的箭头（长度按比例缩放）')]);
		if (samples || few) ks.push(['gen', L('sampled points', '采样的点')]);
		if (samples) ks.push(['accent', L('point 0 and its path', '0 号点和它的路径')]);
		return ks;
	});
	const sw: Record<string, string> = { real: 'var(--real)', ink2: 'var(--ink-2)', accent: 'var(--accent)', gen: 'var(--generated)' };
</script>

<figure class="fp" bind:clientWidth={W} style:min-height="380px">
	<div class="row" role="img" aria-label={description}>
		<canvas bind:this={c0} style:width="{side}px" style:height="{side}px"></canvas>
		{#if mode === 'few'}<canvas bind:this={c1} style:width="{side}px" style:height="{side}px"></canvas>{/if}
	</div>
	{#if few}
		<div class="row small">
			{#each few as e, i (i)}
				<span style:width="{side}px"
					>{#if e}<b class="num">{fmtInt(e.K)}</b> {tr(L('steps', '步'))} · {tr(L('mean distance to the data', '到数据的平均距离'))} <span class="num">{fmt3(e.dist)}</span> · {tr(L('to the middle', '到中间'))}
						<span class="num">{fmt3(spreads[i])}</span>{/if}</span
				>
			{/each}
		</div>
	{/if}
	{#if mode === 'euler'}
		<!-- the task in the stage: a press here keeps Play going (keepPlay), unlike one in the input box (kit issue 18) -->
		<p class="row small">{tr(L('Try another shape:', '换个形状试试：'))} {#each ['moons', 'ring'] as id (id)}<button type="button" class="btn" onclick={() => rt.setInput(presetOf(id))}>{tr(LABELS[id])}</button>{/each}</p>
	{/if}
	{#if mode === 'slider' || mode === 'field'}
		<label class="time small">
			<span>t = <span class="num">{fmt3(tv)}</span></span>
			<input type="range" min="0" max="1" step="0.01" value={tv} oninput={(e) => (tNow = Number((e.currentTarget as HTMLInputElement).value))} aria-label={tr(L('time t', '时间 t'))} />
		</label>
	{/if}
	<figcaption class="small">
		{#each keys as [c, label], i (i)}<span><i style:background={sw[c]}></i>{tr(label)}</span>{/each}
		{#if samples}<span class="muted">{tr(L('Euler step', '欧拉步'))} <span class="num">{fmtInt(samples.path.length - 1)}</span>, t = <span class="num">{fmt3(Number(at?.frame?.metrics?.t ?? 0))}</span></span>{/if}
		{#if few}<span class="muted"><b class="num">+</b> {tr(L('the middle of your points, which lie on average', '你的点的中间，你的点平均离它'))} <span class="num">{fmt3(spreadAround(data.xy, data.n, mid))}</span> {tr(L('from it', ''))}</span>{/if}
		{#if span < SPAN}<span class="muted">{tr(L('zoomed in: x and y from', '放大显示：x 和 y 从'))} <span class="num">{fmt3(-span)}</span> {tr(L('to', '到'))} <span class="num">{fmt3(span)}</span></span>{/if}
		{#if field}<span class="muted">{tr(L('after', '训练'))} <span class="num">{fmtInt(Number((at?.state as { step?: number } | undefined)?.step ?? 0))}</span> {tr(L('training steps', '步之后'))}</span>{/if}
	</figcaption>
</figure>

<style>
	.fp {
		margin: 0;
		display: grid;
		gap: 6px;
		min-width: 0;
		align-content: start;
	}
	.row {
		display: flex;
		gap: 12px;
		flex-wrap: wrap;
	}
	canvas {
		display: block;
		border: 1px solid var(--rule);
		border-radius: var(--r1);
		max-width: 100%;
	}
	.time {
		display: flex;
		gap: 8px;
		align-items: center;
		max-width: 440px;
	}
	.time input {
		flex: 1;
	}
	figcaption {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		color: var(--ink-2);
	}
	figcaption i {
		display: inline-block;
		width: 10px;
		height: 10px;
		border-radius: 50%;
		margin-right: 5px;
		vertical-align: -1px;
	}
</style>

<!-- realnvp-2d/plane (props: meta.ts). One canvas: the data plane (±1.5) or the noise plane (±3). Nothing moves by
     itself: the picture follows the step's frame (Play steps through the layers or the training snapshots). -->
<script lang="ts">
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import type { I18n, Tensor } from '../../../../atlas/types/core.ts';
	import { t as tr } from '$lib/i18n/i18n.svelte';
	import { themeState } from '$lib/kit/theme.svelte';
	import { makeScale } from '$lib/kit/color';
	import { fmt3, fmtInt, fmtPct, paren } from '../../../../atlas/core/format.ts';
	import { DG, DSPAN, grid, LINES, logp, PER, type Params } from '../../nvp.ts';
	import { GAP } from '../../process.ts';
	import { cellOf, SIDE, dataFor, describePlane, isSample, layersDone, metricsOf, modeOf, namesOf, paramsFor, pointsOf, spanOf } from './meta.ts';

	let { rt, at, props, source }: StageProps = $props();
	const p = $derived({ rt, at, props, source });
	const mode = $derived(modeOf(p));
	const data = $derived(dataFor(p));
	const span = $derived(spanOf(p));
	const m = $derived(metricsOf(p));
	const description = $derived(tr(describePlane(p)));
	const L = (en: string, zh: string): I18n => ({ en, zh });
	const fr = $derived((at?.state ?? {}) as { pts?: Tensor; x?: Tensor; grid?: Tensor; logp?: Tensor; params?: Params; step?: number; loss?: number });
	const before = $derived(mode === 'couple' && at?.tape && at.index > 0 ? (at.tape.state(at.index - 1) as { pts?: Tensor; grid?: Tensor }) : null);
	/** mode 'learn': log p on a 28 × 28 grid from the frame's weights (a few ms) */
	const learnLp = $derived(mode === 'learn' && fr.params ? logp(fr.params, grid(28, DSPAN), 28 * 28) : null);
	const cell = $derived.by(() => {
		if (mode !== 'warp' || !props?.cell || !data.n || isSample(p)) return null;
		const pr = paramsFor(p);
		return pr ? cellOf(pr, data.xy[0], data.xy[1], layersDone(p)) : null;
	});

	let W = $state(360);
	const side = $derived(Math.max(220, Math.min(W, 440)));
	let canvas = $state<HTMLCanvasElement | null>(null);
	const theme = $derived(themeState.resolved);
	type Ctx = { g: CanvasRenderingContext2D; s: number; col: Record<string, string> };
	const X = (c: Ctx, v: number) => ((v + span) / (2 * span)) * c.s;
	const Y = (c: Ctx, v: number) => ((span - v) / (2 * span)) * c.s;
	function dots(c: Ctx, xy: ArrayLike<number>, n: number, color: string, r: number, alpha = 1) {
		Object.assign(c.g, { fillStyle: color, globalAlpha: alpha });
		for (let i = 0; i < n; i++) c.g.beginPath(), c.g.arc(X(c, xy[2 * i]), Y(c, xy[2 * i + 1]), r, 0, 2 * Math.PI), c.g.fill();
		c.g.globalAlpha = 1;
	}
	function stroke(c: Ctx, color: string, w: number, draw: (g: CanvasRenderingContext2D) => void, alpha = 1, dash: number[] = []) {
		Object.assign(c.g, { strokeStyle: color, lineWidth: w, globalAlpha: alpha, lineJoin: 'round' });
		c.g.setLineDash(dash);
		c.g.beginPath();
		draw(c.g);
		c.g.stroke();
		c.g.setLineDash([]);
		c.g.globalAlpha = 1;
	}
	const ring = (c: Ctx, x: number, y: number) => stroke(c, c.col.accent, 2.5, (g) => g.arc(X(c, x), Y(c, y), 7, 0, 2 * Math.PI));
	const circle = (c: Ctx, r: number, color: string) => stroke(c, color, 1.2, (g) => g.arc(X(c, 0), Y(c, 0), (r / (2 * span)) * c.s, 0, 2 * Math.PI), 0.9, [5, 4]);
	/** the grid lines (LINES vertical, then LINES horizontal, PER points each) */
	function lines(c: Ctx, gr: Tensor, color: string, alpha: number) {
		const d = gr.data;
		stroke(c, color, 0.8, (g) => {
			for (let l = 0; l < 2 * LINES; l++)
				for (let q = 0; q < PER; q++) {
					const i = 2 * (l * PER + q);
					(q ? g.lineTo : g.moveTo).call(g, X(c, d[i]), Y(c, d[i + 1]));
				}
		}, alpha);
	}
	function arrow(c: Ctx, x0: number, y0: number, x1: number, y1: number, color: string) {
		const [ax, ay, bx, by] = [X(c, x0), Y(c, y0), X(c, x1), Y(c, y1)];
		const len = Math.hypot(bx - ax, by - ay);
		if (len < 2) return;
		const [ux, uy, h] = [(bx - ax) / len, (by - ay) / len, Math.min(10, 0.45 * len)];
		for (const [col, lw] of [[c.col.surface, 5], [color, 2.2]] as [string, number][]) {
			stroke(c, col, lw, (g) => (g.moveTo(ax, ay), g.lineTo(bx - ux * h * 0.6, by - uy * h * 0.6)));
			Object.assign(c.g, { fillStyle: col });
			c.g.beginPath();
			c.g.moveTo(bx, by);
			c.g.lineTo(bx - h * ux - 0.55 * h * uy, by - h * uy + 0.55 * h * ux);
			c.g.lineTo(bx - h * ux + 0.55 * h * uy, by - h * uy - 0.55 * h * ux);
			c.g.closePath();
			c.g.fill();
		}
	}
	/** p = e^(log p) on a G × G grid over the data plane, on the sequential scale */
	function heat(c: Ctx, lp: ArrayLike<number>, G: number) {
		let top = -Infinity;
		for (let i = 0; i < G * G; i++) top = Math.max(top, lp[i]);
		const sc = makeScale('sequential', [0, Math.exp(top)], theme);
		const img = new ImageData(G, G);
		for (let i = 0; i < G * G; i++) img.data.set([...sc.rgb(Math.exp(lp[i])), 255], 4 * ((G - 1 - Math.floor(i / G)) * G + (i % G)));
		const off = document.createElement('canvas');
		(off.width = G), (off.height = G);
		off.getContext('2d')!.putImageData(img, 0, 0);
		c.g.imageSmoothingEnabled = true;
		c.g.drawImage(off, 0, 0, c.s, c.s);
	}
	/** the inset (top left): the square around point 0 before (dashed) and after the layers (filled), at one scale */
	function inset(c: Ctx, poly: Float64Array) {
		const B = Math.round(Math.min(130, c.s * 0.32));
		c.g.fillStyle = c.col.surface;
		c.g.fillRect(6, 6, B, B);
		stroke(c, c.col.rule, 1, (g) => g.rect(6, 6, B, B));
		let [cx, cy, ext] = [0, 0, SIDE];
		for (let i = 0; i < 32; i++) (cx += poly[2 * i] / 32), (cy += poly[2 * i + 1] / 32);
		for (let i = 0; i < 32; i++) ext = Math.max(ext, 2 * Math.abs(poly[2 * i] - cx), 2 * Math.abs(poly[2 * i + 1] - cy));
		const [k, ox, oy] = [(0.8 * B) / ext, 6 + B / 2, 6 + B / 2];
		const path = (g: CanvasRenderingContext2D) => { for (let i = 0; i <= 32; i++) (i ? g.lineTo : g.moveTo).call(g, ox + (poly[2 * (i % 32)] - cx) * k, oy - (poly[2 * (i % 32) + 1] - cy) * k); };
		Object.assign(c.g, { fillStyle: c.col.accent, globalAlpha: 0.45 });
		c.g.beginPath();
		path(c.g);
		c.g.fill();
		c.g.globalAlpha = 1;
		stroke(c, c.col.accent, 2, path);
		const h = (SIDE * k) / 2;
		stroke(c, c.col.ink, 1.5, (g) => g.rect(ox - h, oy - h, 2 * h, 2 * h), 1, [4, 3]);
	}
	function paint(c: Ctx) {
		if (mode === 'density' && fr.logp) heat(c, fr.logp.data, DG);
		if (mode === 'learn' && learnLp) heat(c, learnLp, 28);
		if (mode === 'warp') (circle(c, 1, c.col.ink2), circle(c, 2, c.col.ink2));
		if (mode === 'samples' && props?.gap) (circle(c, GAP[0], c.col.ink2), circle(c, GAP[1], c.col.ink2));
		if (before?.grid) lines(c, before.grid, c.col.ink2, 0.3);
		if ((mode === 'couple' || mode === 'warp') && fr.grid) lines(c, fr.grid, c.col.ink2, mode === 'couple' ? 0.75 : 0.55);
		const pts = pointsOf(p);
		const n = data.n;
		if (mode === 'data' || mode === 'density' || mode === 'learn' || mode === 'samples') dots(c, data.xy, n, c.col.real, n > 400 ? 1.6 : 2.2, mode === 'data' ? 0.9 : 0.55);
		if (mode === 'couple' && before?.pts) dots(c, before.pts.data, n, c.col.real, 1.6, 0.3);
		if (mode === 'warp' && isSample(p)) dots(c, data.xy, n, c.col.real, 1.6, 0.35);
		if (pts && mode !== 'density') {
			const k = pts.shape[0];
			dots(c, pts.data, k, isSample(p) ? c.col.gen : c.col.real, k > 400 ? 1.7 : 2.2, 0.9);
		}
		if (cell) {
			Object.assign(c.g, { fillStyle: c.col.accent, globalAlpha: 0.35 });
			c.g.beginPath();
			for (let i = 0; i < 32; i++) (i ? c.g.lineTo : c.g.moveTo).call(c.g, X(c, cell.poly[2 * i]), Y(c, cell.poly[2 * i + 1]));
			c.g.closePath();
			c.g.fill();
			c.g.globalAlpha = 1;
			stroke(c, c.col.accent, 2, (g) => { for (let i = 0; i <= 32; i++) (i ? g.lineTo : g.moveTo).call(g, X(c, cell.poly[2 * (i % 32)]), Y(c, cell.poly[2 * (i % 32) + 1])); });
			inset(c, cell.poly);
		}
		if (mode === 'couple' && pts && before?.pts) {
			const [x0, y0, x1, y1] = [before.pts.data[0], before.pts.data[1], pts.data[0], pts.data[1]];
			if (props?.back) arrow(c, x1, y1, x0, y0, c.col.accent), ring(c, x0, y0);
			else arrow(c, x0, y0, x1, y1, c.col.accent), ring(c, x1, y1);
		} else if (pts && mode === 'warp') ring(c, pts.data[0], pts.data[1]);
		if ((mode === 'density' || mode === 'data') && n) ring(c, data.xy[0], data.xy[1]);
	}
	$effect(() => {
		void [p, side, theme, learnLp, cell];
		const cv = canvas;
		if (!cv) return;
		const dpr = Math.min(3, globalThis.devicePixelRatio || 1);
		(cv.width = Math.round(side * dpr)), (cv.height = Math.round(side * dpr));
		const g = cv.getContext('2d');
		if (!g) return;
		g.setTransform(dpr, 0, 0, dpr, 0, 0);
		const css = getComputedStyle(cv);
		const v = (k: string, d: string) => css.getPropertyValue(k).trim() || d;
		const c: Ctx = { g, s: side, col: { surface: v('--surface', '#fffdf8'), rule: v('--rule', '#ccc'), ink: v('--ink', '#222'), ink2: v('--ink-2', '#555'), real: v('--real', '#1a8a4a'), gen: v('--generated', '#c2338a'), accent: v('--accent', '#0b6b70') } };
		g.fillStyle = c.col.surface;
		g.fillRect(0, 0, side, side);
		stroke(c, c.col.rule, 1, () => (g.moveTo(side / 2, 0), g.lineTo(side / 2, side), g.moveTo(0, side / 2), g.lineTo(side, side / 2)));
		paint(c);
	});

	const sw: Record<string, string> = { real: 'var(--real)', ink2: 'var(--ink-2)', accent: 'var(--accent)', gen: 'var(--generated)', seq: 'var(--seq-mid)' };
	const keys = $derived.by(() => {
		const ks: [string, I18n][] = [];
		const sample = isSample(p);
		if (mode !== 'warp' || !sample || data.n) ks.push(['real', sample || mode === 'samples' ? L('the points it trained on', '它训练用的点') : L('your points', '你的点')]);
		if (sample || mode === 'samples') ks.push(['gen', L('samples of the flow', '流模型的样本')]);
		if (mode === 'couple') ks.push(['ink2', L('grid before the layer (faint) and after it', '这一层之前（淡）和之后的网格')], ['accent', L('point 0 and its move', '0 号点和它的移动')]);
		if (mode === 'warp') ks.push(['ink2', L(sample ? 'the noise plane’s grid, pulled back' : 'a grid over your points, pushed along; dashed: |z| = 1 and 2', sample ? '噪声平面上的网格，被倒推回来' : '覆盖你的点的网格，跟着被推动；虚线：|z| = 1 和 2')]);
		if (mode === 'warp' && !sample) ks.push(['accent', L(cell ? 'point 0; inset: the small square around it before (dashed) and after (filled), at one scale' : 'point 0', cell ? '0 号点；小插图：它周围的小方块变换前（虚线）和变换后（填色），比例相同' : '0 号点')]);
		if (mode === 'density' || mode === 'learn') ks.push(['seq', L('density p(x) of the flow (stronger colour: higher)', '流模型的密度 p(x)（颜色越浓越高）')]);
		if (mode === 'density' || mode === 'data') ks.push(['accent', L('point 0', '0 号点')]);
		if (mode === 'samples' && props?.gap) ks.push(['ink2', L('dashed: the empty gap between the rings', '虚线：两个圆环之间的空隙')]);
		return ks;
	});
</script>

<figure class="sp" bind:clientWidth={W} style:min-height="380px">
	<div class="pic" role="img" aria-label={description}><canvas bind:this={canvas} style:width="{side}px" style:height="{side}px" aria-hidden="true"></canvas></div>
	<figcaption class="small">
		{#each keys as [c, label], i (i)}<span><i style:background={sw[c]}></i>{tr(label)}</span>{/each}
		{#if mode === 'couple' && typeof m.s === 'number'}
			{@const [u, v] = namesOf(m.keep)}
			<span class="eq"
				>{tr(L('layer', '第'))} <span class="num">{m.k}</span>{tr(L(` keeps ${u}:`, ` 层保持 ${u} 不变：`))} s = <span class="num">{fmt3(m.s)}</span>, t = <span class="num">{fmt3(m.t)}</span>;
				{#if props?.back}{v} = ({v}′ − t)·e<sup>−s</sup> = (<span class="num">{fmt3(m.v2)}</span> − <span class="num">{paren(m.t)}</span>) × <span class="num">{fmt3(Math.exp(-m.s))}</span> = <span class="num">{fmt3((m.v2 - m.t) * Math.exp(-m.s))}</span>{:else}{v}′ = {v}·e<sup>s</sup> + t = <span class="num">{paren(m.v)}</span> × <span class="num">{fmt3(m.es)}</span> + <span class="num">{paren(m.t)}</span> = <span class="num">{fmt3(m.v2)}</span>{/if}</span
			>
		{/if}
		{#if cell}<span class="eq"
				>{tr(L('the small square’s area: ×', '小方块的面积：×'))} <span class="num">{fmt3(cell.ratio)}</span>{tr(L(' after ', '，穿过 '))}<span class="num">{fmtInt(layersDone(p))}</span>{tr(L(' layers', ' 层后'))}</span
			>{/if}
		{#if mode === 'density' && typeof m.logp === 'number'}<span class="eq"
				>log p({tr(L('point 0', '0 号点'))}) = <span class="num">{fmt3(m.logN)}</span> + <span class="num">{fmt3(m.ld)}</span> = <span class="num">{fmt3(m.logp)}</span></span
			>{/if}
		{#if mode === 'learn'}<span class="muted">{tr(L('training step', '训练步'))} <span class="num">{fmtInt(Number(fr.step ?? 0))}</span></span>{/if}
		{#if mode === 'samples' && props?.gap && typeof m.gap === 'number'}<span class="eq">{tr(L('samples in the gap', '落在空隙里的样本'))}: <span class="num">{fmtPct(m.gap)}</span></span>{/if}
	</figcaption>
</figure>

<style>
	.sp {
		margin: 0;
		display: grid;
		gap: 6px;
		min-width: 0;
		align-content: start;
	}
	.pic {
		min-width: 0;
	}
	canvas {
		display: block;
		border: 1px solid var(--rule);
		border-radius: var(--r1);
		max-width: 100%;
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
	.eq {
		color: var(--ink);
	}
</style>

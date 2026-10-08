<!-- score-2d/plane (props: meta.ts). Canvas from −2.5 to 2.5. Nothing moves by itself: the picture follows the step's
     frame (Play steps through the frames) and the sliders. The chains sliders re-run the Langevin chains here, from the
     trained weights and the same seed, so at the default values they draw exactly the source's frames. -->
<script lang="ts">
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import type { I18n, Tensor } from '../../../../atlas/types/core.ts';
	import { t as tr } from '$lib/i18n/i18n.svelte';
	import { themeState } from '$lib/kit/theme.svelte';
	import { makeScale } from '$lib/kit/color';
	import { fmt3, fmtInt, fmtPct } from '../../../../atlas/core/format.ts';
	import { createRng } from '../../../../atlas/core/rng.ts';
	import { chains, coverage, dataOf, densityNear, exact, grid, meanNearest, probeOf, score, SHOW, SIGMAS, SMIN, SPAN } from '../../score.ts';
	import { trained } from '../../explainer.ts';
	import { chainsOf, describePlane, followed, modeOf, pairsOf, paramsOf, sigmaOf } from './meta.ts';

	let { rt, at, props }: StageProps = $props();
	const p = $derived({ rt, at, props });
	const mode = $derived(modeOf(p));
	const data = $derived(dataOf(rt.input));
	const sig = $derived(sigmaOf(p));
	const probe = $derived(probeOf(data));
	const description = $derived(tr(describePlane(p)));
	const L = (en: string, zh: string): I18n => ({ en, zh });

	// mode 'field': the σ slider (an index into SIGMAS)
	let lvl = $state<number | null>(null);
	const level = $derived(lvl ?? Math.max(0, Math.min(SIGMAS.length - 1, Number(props?.level ?? SIGMAS.indexOf(SHOW)))));
	// mode 'chains': the step size α (at σ = 0.16) and the noise scale; null = the source's own run
	const STEPS = [0.0002, 0.0008, 0.0032, 0.0128, 0.0512];
	const NOISE = [0, 0.25, 0.5, 1, 1.5];
	let ai = $state(2);
	let ni = $state(3);
	let local = $state.raw<{ path: Float64Array[]; dist: number; cover: number } | null>(null);
	let job = 0;
	function rerun() {
		const src = trained(rt);
		const id = ++job;
		if (!src || (ai === 2 && ni === 3)) return void (local = null);
		const it = chains(src.params, { sigmas: [SHOW], T: 100, alpha: STEPS[ai] * (SMIN / SHOW) ** 2, noise: NOISE[ni], chains: 300 }, createRng(1));
		const path: Float64Array[] = [];
		const slice = () => {
			if (id !== job) return;
			for (let k = 0; k < 10; k++) {
				const r = it.next();
				if (r.done) break;
				path.push(r.value.x);
			}
			const x = path[path.length - 1];
			const done = path.length > 100;
			local = { path: path.slice(), dist: done ? meanNearest(x, x.length / 2, data) : NaN, cover: done ? coverage(x, x.length / 2, data) : NaN };
			if (!done) requestAnimationFrame(slice);
		};
		slice();
	}
	const run = $derived(mode === 'chains' ? (local ?? chainsOf(p)) : null);
	const metrics = $derived(at?.frame?.metrics ?? {});

	let W = $state(360);
	const side = $derived(Math.max(220, Math.min(W, 440)));
	let canvas = $state<HTMLCanvasElement | null>(null);
	const theme = $derived(themeState.resolved);
	type Ctx = { g: CanvasRenderingContext2D; s: number; col: Record<string, string> };
	const X = (c: Ctx, v: number) => ((v + SPAN) / (2 * SPAN)) * c.s;
	const Y = (c: Ctx, v: number) => ((SPAN - v) / (2 * SPAN)) * c.s;
	function dots(c: Ctx, xy: ArrayLike<number>, n: number, color: string, r: number, alpha = 1) {
		Object.assign(c.g, { fillStyle: color, globalAlpha: alpha });
		for (let i = 0; i < n; i++) c.g.beginPath(), c.g.arc(X(c, xy[2 * i]), Y(c, xy[2 * i + 1]), r, 0, 2 * Math.PI), c.g.fill();
		c.g.globalAlpha = 1;
	}
	function stroke(c: Ctx, color: string, w: number, draw: (g: CanvasRenderingContext2D) => void, alpha = 1) {
		Object.assign(c.g, { strokeStyle: color, lineWidth: w, globalAlpha: alpha, lineJoin: 'round' });
		c.g.beginPath();
		draw(c.g);
		c.g.stroke();
		c.g.globalAlpha = 1;
	}
	const ring = (c: Ctx, x: number, y: number) => stroke(c, c.col.accent, 2.5, (g) => g.arc(X(c, x), Y(c, y), 7, 0, 2 * Math.PI));
	/** an arrow from (x, y) by (dx, dy) in plane units; with a halo it is outlined in the background colour */
	function arrow(c: Ctx, x: number, y: number, dx: number, dy: number, color: string, w: number, halo = '') {
		const [ax, ay, bx, by] = [X(c, x), Y(c, y), X(c, x + dx), Y(c, y + dy)];
		const len = Math.hypot(bx - ax, by - ay);
		if (len < 0.5) return;
		const [ux, uy, h] = [(bx - ax) / len, (by - ay) / len, Math.min(4 + 2 * w, halo ? Math.max(0.45 * len, Math.min(6, len)) : 0.45 * len)];
		for (const [col, lw] of (halo ? [[halo, w + 3], [color, w]] : [[color, w]]) as [string, number][]) {
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
	function label(c: Ctx, text: string, x: number, y: number, dx = 10, dy = -10) {
		Object.assign(c.g, { strokeStyle: c.col.surface, lineWidth: 4, lineJoin: 'round', fillStyle: c.col.ink });
		c.g.strokeText(text, X(c, x) + dx, Y(c, y) + dy);
		c.g.fillText(text, X(c, x) + dx, Y(c, y) + dy);
	}
	/** arrows on a G × G grid; lengths ∝ |v| / ref, at most 0.9 of a cell (ref: the 75th percentile of |v|); with `log`,
	 *  ∝ log(1 + |v|) / log(1 + log), so arrows that differ 30-fold both stay visible */
	function field(c: Ctx, v: Float64Array, G: number, color: string, log = 0) {
		const xy = grid(G, SPAN);
		const len = Array.from({ length: G * G }, (_, i) => Math.hypot(v[2 * i], v[2 * i + 1]));
		const r = len.slice().sort((a, b) => a - b)[Math.floor(0.75 * len.length)] || 1;
		const cell = (2 * SPAN) / G;
		for (let i = 0; i < G * G; i++) {
			const f = (0.9 * cell * (log ? Math.log1p(len[i]) / Math.log1p(log) : Math.min(1, len[i] / r))) / len[i];
			if (len[i] > 0) arrow(c, xy[2 * i], xy[2 * i + 1], v[2 * i] * f, v[2 * i + 1] * f, color, 1.1);
		}
	}
	/** the density on a 48 × 48 grid (modes 'density' and 'score'), computed once per input and σ */
	const density = $derived.by(() => {
		if (mode !== 'density' && mode !== 'score') return new Float64Array(0);
		const xy = grid(48, SPAN);
		return Float64Array.from({ length: 48 * 48 }, (_, i) => densityNear(data, xy[2 * i], xy[2 * i + 1], sig));
	});
	const exactField = (G: number, s: number) => {
		const xy = grid(G, SPAN);
		return Float64Array.from({ length: 2 * G * G }, (_, k) => { const e = exact(data, xy[k & ~1], xy[k | 1], s); return k & 1 ? e.sy : e.sx; });
	};
	function paint(c: Ctx) {
		const pr = paramsOf(p) ?? (mode === 'limits' ? (trained(rt)?.params ?? null) : null);
		if (mode === 'density' || mode === 'score') {
			const G = 48;
			const v = density;
			const sc = makeScale('sequential', [0, Math.max(...v)], theme);
			const img = new ImageData(G, G);
			for (let i = 0; i < G * G; i++) {
				const [r, g, b] = sc.rgb(v[i]);
				const o = 4 * ((G - 1 - Math.floor(i / G)) * G + (i % G));
				img.data.set([r, g, b, mode === 'score' ? 110 : 255], o);
			}
			const off = document.createElement('canvas');
			(off.width = G), (off.height = G);
			off.getContext('2d')!.putImageData(img, 0, 0);
			c.g.imageSmoothingEnabled = true;
			c.g.drawImage(off, 0, 0, c.s, c.s);
		}
		const faint = ['score', 'field', 'chains', 'limits'].includes(mode);
		dots(c, data.xy, data.n, c.col.real, data.n > 400 ? 1.6 : 2.2, faint ? 0.4 : 0.9);
		if (mode === 'score') field(c, exactField(20, sig), 20, c.col.ink2);
		if (mode === 'density' || mode === 'score') {
			const e = exact(data, probe[0], probe[1], sig);
			if (mode === 'score') arrow(c, probe[0], probe[1], (0.5 * e.sx) / Math.hypot(e.sx, e.sy), (0.5 * e.sy) / Math.hypot(e.sx, e.sy), c.col.accent, 2.5, c.col.surface);
			ring(c, probe[0], probe[1]);
		}
		const q = mode === 'pairs' ? pairsOf(p) : null;
		if (q) {
			stroke(c, c.col.ink2, 0.7, (g) => { for (let i = 0; i < q.n; i++) g.moveTo(X(c, q.x[2 * i]), Y(c, q.x[2 * i + 1])), g.lineTo(X(c, q.xt[2 * i]), Y(c, q.xt[2 * i + 1])); }, 0.25);
			dots(c, q.xt, q.n, c.col.ink2, 2, 0.8);
			const i = followed(q);
			const [x, y, xt, yt] = [q.x[2 * i], q.x[2 * i + 1], q.xt[2 * i], q.xt[2 * i + 1]];
			dots(c, [x, y], 1, c.col.real, 4.5);
			arrow(c, xt, yt, x - xt, y - yt, c.col.real, 2.5, c.col.surface);
			ring(c, xt, yt);
			label(c, 'x̃', xt, yt);
			label(c, 'x', x, y, -16, 16);
		}
		if (mode === 'field' && pr) field(c, score(pr, grid(20, SPAN), SIGMAS[level], 400), 20, c.col.ink2);
		if (mode === 'limits' && pr) {
			const G = 14;
			const ex = exactField(G, sig);
			const net = score(pr, grid(G, SPAN), sig, G * G);
			const ref = Math.max(...Array.from({ length: G * G }, (_, i) => Math.hypot(ex[2 * i], ex[2 * i + 1]))) || 1;
			field(c, ex, G, c.col.real, ref);
			field(c, net, G, c.col.gen, ref);
		}
		if (run) {
			// the paths of the last 30 steps (the big early steps of an annealed run would cover the plane)
			const N = run.path[0].length / 2;
			const from = Math.max(0, run.path.length - 31);
			const trace = (g: CanvasRenderingContext2D, i: number) => run.path.slice(from).forEach((q, j) => (j ? g.lineTo : g.moveTo).call(g, X(c, q[2 * i]), Y(c, q[2 * i + 1])));
			stroke(c, c.col.gen, 0.7, (g) => { for (let i = 0; i < N; i++) trace(g, i); }, 0.16);
			const x = run.path[run.path.length - 1];
			dots(c, x, N, c.col.gen, 2.2);
			stroke(c, c.col.accent, 2, (g) => trace(g, 0));
			ring(c, x[0], x[1]);
		}
	}
	$effect(() => {
		void [p, side, theme, level, run];
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
		g.font = `600 14px ${css.fontFamily}`;
		paint(c);
	});

	const keys = $derived.by(() => {
		const ks: [string, I18n][] = [['real', L('your points', '你的点')]];
		if (mode === 'density') ks.push(['seq', L(`density at σ = ${fmt3(sig)} (stronger colour: higher)`, `σ = ${fmt3(sig)} 时的密度（颜色越浓越高）`)]);
		if (mode === 'density' || mode === 'score') ks.push(['accent', L('the ringed point', '圈出的点')]);
		if (mode === 'score') ks.push(['ink2', L(`the score at σ = ${fmt3(sig)} (long arrows shortened to fit)`, `σ = ${fmt3(sig)} 时的分数（长箭头按比例缩短）`)]);
		if (mode === 'pairs') ks.push(['ink2', L('noisy copies x̃ = x + σ·ε', '加噪副本 x̃ = x + σ·ε')], ['real', L('the way back, x − x̃', '回去的路 x − x̃')]);
		if (mode === 'field') ks.push(['ink2', L('the network’s score (long arrows shortened to fit)', '网络给出的分数（长箭头按比例缩短）')]);
		if (mode === 'limits') ks.push(['real', L('true score', '真实的分数')], ['gen', L('the network’s score (both lengths on a log scale)', '网络给出的分数（两者长度都按对数比例）')]);
		if (run) ks.push(['gen', L('chains and their last 30 steps', '链和它们最后 30 步的路径')], ['accent', L('chain 0', '0 号链')]);
		return ks;
	});
	const at0 = $derived(mode === 'score' ? exact(data, probe[0], probe[1], sig) : null);
	const sw: Record<string, string> = { real: 'var(--real)', ink2: 'var(--ink-2)', accent: 'var(--accent)', gen: 'var(--generated)', seq: 'var(--seq-mid)' };
</script>

<figure class="sp" bind:clientWidth={W} style:min-height="380px">
	<div class="pic" role="img" aria-label={description}><canvas bind:this={canvas} style:width="{side}px" style:height="{side}px" aria-hidden="true"></canvas></div>
	{#if mode === 'field'}
		<label class="small ctl"
			><span>σ = <span class="num">{fmt3(SIGMAS[level])}</span></span>
			<input type="range" min="0" max={SIGMAS.length - 1} step="1" value={level} oninput={(e) => (lvl = Number((e.currentTarget as HTMLInputElement).value))} aria-label={tr(L('noise level σ', '噪声水平 σ'))} /></label
		>
	{/if}
	{#if mode === 'chains' && props?.sliders}
		<label class="small ctl"
			><span>{tr(L('step size α', '步长 α'))} = <span class="num">{STEPS[ai]}</span></span>
			<input type="range" min="0" max={STEPS.length - 1} step="1" value={ai} onchange={(e) => ((ai = Number((e.currentTarget as HTMLInputElement).value)), rerun())} aria-label={tr(L('step size α', '步长 α'))} /></label
		>
		<label class="small ctl"
			><span>{tr(L('noise', '噪声'))} × <span class="num">{NOISE[ni]}</span></span>
			<input type="range" min="0" max={NOISE.length - 1} step="1" value={ni} onchange={(e) => ((ni = Number((e.currentTarget as HTMLInputElement).value)), rerun())} aria-label={tr(L('noise scale', '噪声倍数'))} /></label
		>
	{/if}
	<figcaption class="small">
		{#each keys as [c, label], i (i)}<span><i style:background={sw[c]}></i>{tr(label)}</span>{/each}
		{#if at0}<span
				>{tr(L('At the ringed point: score', '圈出的点处：分数'))} (<span class="num">{fmt3(at0.sx)}</span>, <span class="num">{fmt3(at0.sy)}</span>), {tr(L('length', '长度'))} <span class="num">{fmt3(Math.hypot(at0.sx, at0.sy))}</span>; {tr(L('your nearest point carries', '你最近的点占权重的'))} <span class="num">{fmtPct(at0.w)}</span>{tr(L(' of the weight', ''))}</span
			>{/if}
		{#if run}<span class="muted"
				>{tr(L('step', '步'))} <span class="num">{fmtInt(run.path.length - 1)}</span>{#if metrics.sigma && !local}, σ = <span class="num">{fmt3(metrics.sigma)}</span>{/if}{#if local && Number.isFinite(local.dist)}; {tr(L('your settings: mean distance to your points', '你的设置：到你的点的平均距离'))} <span class="num">{fmt3(local.dist)}</span>, {tr(L('points covered', '覆盖的点'))} <span class="num">{fmtPct(local.cover)}</span>{/if}</span
			>{/if}
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
	.ctl {
		display: flex;
		gap: 8px;
		align-items: center;
		max-width: 440px;
	}
	.ctl span {
		min-width: 9em;
	}
	.ctl input {
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

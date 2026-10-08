<!-- gan-2d/arena (props: meta.ts). Canvas; nothing animates by itself: the picture follows the frame (Play steps it). -->
<script lang="ts">
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import { hexToRgb } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3, fmtInt, fmtPct } from '../../../../atlas/core/format.ts';
	import { boxAround, dGrid, MODE_MIN, MODE_R, NEAR, nearCount, PLANE, RES, sheet, uphill, type Box } from '../../net.ts';
	import { alongArrows, lateRange } from '../../scene.ts';
	import { untrack } from 'svelte';
	import meta, { describeScene, scene, shownSource, type Choice } from './meta.ts';

	let { rt, source, at, props }: StageProps = $props();
	const show = $derived((props?.show as string[] | undefined) ?? ['map', 'real', 'fake']);
	const has = (k: string) => show.includes(k);
	const choices = $derived((props?.choices as Choice[] | undefined) ?? []);
	const one = $derived(typeof props?.one === 'number' ? (props.one as number) : -1);
	let picked = $state<string | null>(null);
	const src = $derived(shownSource({ source, props }, picked));

	let version = $state(0);
	$effect(() => {
		const tp = rt.sources[src]?.tape as { subscribe?: (fn: () => void) => () => void } | null | undefined;
		if (tp && typeof tp.subscribe === 'function') return tp.subscribe(() => version++);
	});
	const atState = $derived(at?.state);
	const sc = $derived.by(() => {
		void version;
		void atState;
		const s = src;
		return untrack(() => scene({ rt, source, at, props }, s === source ? null : s));
	});
	const description = $derived(t(describeScene(sc, show)));
	/** the magnified window of the slowed-down steps (around the generated points, before and after) */
	const zoom = $derived(sc && has('zoom') ? boxAround(sc.fake, sc.fakeBefore ?? new Float64Array(0)) : null);
	const along = $derived(sc?.fakeBefore && has('move') ? alongArrows(sc) : NaN);
	/** before → after on the slowed-down steps, else the value; 4 digits where the narration quotes 4 (steps 5-7) */
	const f3 = (v: number, d = 3) => (Number.isFinite(v) ? fmt3(v, d) : '—'); // no real points: a dash, not NaN
	const pair = (a: number, b: number) => (sc?.before ? `${f3(a, 4)} → ${f3(b, 4)}` : f3(b, props?.reach ? 4 : 3));
	/** the blobs each finished run covered in the second half of its steps (the collapse step's comparison) */
	const late = $derived.by(() => {
		void version;
		return choices.filter((c) => rt.sources[c.source]?.status === 'ready').map((c) => ({ label: c.label, ...lateRange(rt.sources[c.source].tape, 'modes') }));
	});

	let W = $state(360);
	const theme = $derived(themeState.resolved);
	const extra = $derived(has('noise') || zoom !== null);
	const narrow = $derived(W < 560);
	const small = $derived(extra ? Math.round(narrow ? Math.min(W, 260) : Math.min(200, W * 0.4)) : 0);
	const side = $derived(Math.round(Math.max(180, Math.min(440, narrow || !extra ? W - 4 : W - small - 18))));
	let plane = $state<HTMLCanvasElement | null>(null);
	let aside = $state<HTMLCanvasElement | null>(null);

	type Pal = { surface: string; ink: string; rule: string; real: string; gen: string; accent: string };
	function setup(c: HTMLCanvasElement, s: number): [CanvasRenderingContext2D, Pal] {
		const dpr = typeof devicePixelRatio === 'number' ? Math.min(3, devicePixelRatio) : 1;
		c.width = Math.round(s * dpr);
		c.height = Math.round(s * dpr);
		const g = c.getContext('2d')!;
		g.setTransform(dpr, 0, 0, dpr, 0, 0);
		const css = getComputedStyle(c);
		const v = (k: string, d: string) => css.getPropertyValue(k).trim() || d;
		return [g, { surface: v('--surface', theme === 'dark' ? '#1a1d21' : '#fffdf8'), ink: v('--ink-2', '#555555'), rule: v('--rule', '#cccccc'), real: v('--real', '#1a8a4a'), gen: v('--generated', '#c2338a'), accent: v('--accent', '#0b6b70') }];
	}
	const dot = (g: CanvasRenderingContext2D, x: number, y: number, r: number) => (g.beginPath(), g.arc(x, y, r, 0, Math.PI * 2));

	/** one panel: the plane, a window of it (`box`, magnified), or the noise square, with the layers asked for */
	function drawPanel(c: HTMLCanvasElement | null, s: number, box: Box, kind: 'plane' | 'zoom' | 'noise') {
		if (!c || !sc) return;
		const [magnified, noise] = [kind === 'zoom', kind === 'noise'];
		const [g, p] = setup(c, s);
		const [x0, x1, y0, y1] = box;
		const X = (x: number) => ((x - x0) / (x1 - x0)) * s;
		const Y = (y: number) => ((y1 - y) / (y1 - y0)) * s;
		g.fillStyle = p.surface;
		g.fillRect(0, 0, s, s);
		if (has('map') && !noise) {
			const grid = dGrid(sc.params, RES, box).data;
			const img = new ImageData(RES, RES);
			const base = hexToRgb(p.surface.startsWith('#') ? p.surface : '#fffdf8');
			const [cr, cg] = [hexToRgb(p.real), hexToRgb(p.gen)];
			grid.forEach((v, i) => {
				const tgt = v >= 0.5 ? cr : cg;
				const a = 0.08 + 0.5 * Math.min(1, Math.abs(2 * v - 1));
				img.data.set([0, 1, 2].map((k) => base[k] + (tgt[k] - base[k]) * a).concat(255), i * 4);
			});
			const off = Object.assign(document.createElement('canvas'), { width: RES, height: RES });
			off.getContext('2d')!.putImageData(img, 0, 0);
			g.imageSmoothingEnabled = true;
			g.drawImage(off, 0, 0, s, s);
			// the line where D(x) = ½ (marching squares); on a D step also the line before the step, dashed
			const cw = s / RES;
			const half = (grid: ArrayLike<number>, dash: number[]) => {
				const v = (r: number, cc: number) => grid[r * RES + cc] - 0.5;
				(g.strokeStyle = p.ink), (g.lineWidth = 1.2), g.setLineDash(dash);
				g.beginPath();
				for (let r = 0; r + 1 < RES; r++)
					for (let cc = 0; cc + 1 < RES; cc++) {
						const [a, b, e, f] = [v(r, cc), v(r, cc + 1), v(r + 1, cc + 1), v(r + 1, cc)];
						const pts: number[][] = [];
						const [px, py] = [(cc + 0.5) * cw, (r + 0.5) * cw];
						if (a * b < 0) pts.push([px + (a / (a - b)) * cw, py]);
						if (b * e < 0) pts.push([px + cw, py + (b / (b - e)) * cw]);
						if (f * e < 0) pts.push([px + (f / (f - e)) * cw, py + cw]);
						if (a * f < 0) pts.push([px, py + (a / (a - f)) * cw]);
						for (let k = 0; k + 1 < pts.length; k += 2) (g.moveTo(pts[k][0], pts[k][1]), g.lineTo(pts[k + 1][0], pts[k + 1][1]));
					}
				g.stroke();
			};
			half(grid, []);
			if (sc.before && sc.turn === 0) half(dGrid(sc.before, RES, box).data, [5, 4]);
			g.setLineDash([]);
		}
		g.strokeStyle = p.rule;
		g.lineWidth = 1;
		g.beginPath();
		if (noise) g.rect(X(-1), Y(1), X(1) - X(-1), Y(-1) - Y(1));
		else g.moveTo(X(0), 0), g.lineTo(X(0), s), g.moveTo(0, Y(0)), g.lineTo(s, Y(0));
		g.stroke();
		if (has('blobs') && sc.centres && !noise)
			for (const [cx, cy] of sc.centres) {
				const hit = nearCount(cx, cy, sc.fake) >= MODE_MIN;
				g.setLineDash(hit ? [] : [4, 3]);
				g.strokeStyle = p.ink;
				g.lineWidth = hit ? 2 : 1.2;
				dot(g, X(cx), Y(cy), (MODE_R / (x1 - x0)) * s);
				g.stroke();
			}
		g.setLineDash([]);
		if (has('sheet')) {
			const sh = sheet(noise ? null : sc.params);
			const [Ln, Pn] = sh.shape;
			g.strokeStyle = p.gen;
			g.globalAlpha = 0.45;
			for (let l = 0; l < Ln; l++) {
				g.beginPath();
				for (let k = 0; k < Pn; k++) g[k ? 'lineTo' : 'moveTo'](X(sh.data[(l * Pn + k) * 2]), Y(sh.data[(l * Pn + k) * 2 + 1]));
				g.stroke();
			}
			g.globalAlpha = 1;
		}
		if (noise) {
			g.fillStyle = p.ink;
			for (let j = 0; j < sc.z.length / 2; j++) (dot(g, X(sc.z[2 * j]), Y(sc.z[2 * j + 1]), 1.8), g.fill());
			if (one >= 0 && one < sc.z.length / 2) (g.strokeStyle = p.accent), (g.lineWidth = 2.5), dot(g, X(sc.z[2 * one]), Y(sc.z[2 * one + 1]), 7), g.stroke();
			return;
		}
		if (has('real')) {
			const n = sc.realShown.length / 2;
			g.fillStyle = p.real;
			for (let i = 0; i < n; i++) (dot(g, X(sc.realShown[2 * i]), Y(sc.realShown[2 * i + 1]), n > 300 ? 2 : 2.6), g.fill());
		}
		const P = sc.fake.length / 2;
		const move = has('move') && sc.fakeBefore && (magnified || !zoom);
		if (move) {
			// where each point was (a small dot) and the line to where it is now
			g.strokeStyle = p.accent;
			g.fillStyle = p.accent;
			g.lineWidth = 1.6;
			g.globalAlpha = 0.6;
			for (let j = 0; j < P; j++) {
				const [bx, by] = [X(sc.fakeBefore![2 * j]), Y(sc.fakeBefore![2 * j + 1])];
				(g.beginPath(), g.moveTo(bx, by), g.lineTo(X(sc.fake[2 * j]), Y(sc.fake[2 * j + 1])), g.stroke());
				(dot(g, bx, by, 1.6), g.fill());
			}
			g.globalAlpha = 1;
		}
		if (has('fake')) {
			g.strokeStyle = p.gen;
			g.lineWidth = 1.3;
			for (let j = 0; j < P; j++) (dot(g, X(sc.fake[2 * j]), Y(sc.fake[2 * j + 1]), 3), g.stroke());
		}
		if (has('arrows') && (magnified || !zoom)) {
			const from = sc.fakeBefore ?? sc.fake;
			const u = uphill(sc.params, from);
			let max = 0;
			for (let j = 0; j < P; j++) max = Math.max(max, Math.hypot(u[2 * j], u[2 * j + 1]));
			g.strokeStyle = p.ink;
			g.fillStyle = p.ink;
			g.lineWidth = 1.1;
			g.globalAlpha = 0.9;
			const L = magnified ? Math.min(16, s / 14) : 24;
			for (let j = 0; max > 0 && j < P; j++) {
				const [x, y] = [X(from[2 * j]), Y(from[2 * j + 1])];
				const [dx, dy] = [(u[2 * j] / max) * L, (-u[2 * j + 1] / max) * L];
				const len = Math.hypot(dx, dy);
				if (len < 2) continue;
				const [ux, uy, h] = [dx / len, dy / len, Math.min(5, 0.45 * len)];
				g.beginPath();
				g.moveTo(x, y), g.lineTo(x + dx, y + dy);
				g.stroke();
				g.beginPath();
				g.moveTo(x + dx, y + dy);
				g.lineTo(x + dx - h * ux - 0.6 * h * uy, y + dy - h * uy + 0.6 * h * ux);
				g.lineTo(x + dx - h * ux + 0.6 * h * uy, y + dy - h * uy - 0.6 * h * ux);
				g.fill();
			}
			g.globalAlpha = 1;
		}
		if (one >= 0 && one < P) (g.strokeStyle = p.accent), (g.lineWidth = 2.5), dot(g, X(sc.fake[2 * one]), Y(sc.fake[2 * one + 1]), 8), g.stroke();
		if (zoom && !magnified) (g.strokeStyle = p.accent), (g.lineWidth = 1.5), g.strokeRect(X(zoom[0]), Y(zoom[3]), X(zoom[1]) - X(zoom[0]), Y(zoom[2]) - Y(zoom[3]));
	}

	const NOISE: Box = [-1.1, 1.1, -1.1, 1.1];
	$effect(() => {
		void [sc, side, small, theme, show, zoom];
		drawPanel(plane, side, PLANE, 'plane');
		if (zoom || has('noise')) drawPanel(aside, small, zoom ?? NOISE, zoom ? 'zoom' : 'noise');
	});
	const magnification = $derived(zoom ? (PLANE[1] - PLANE[0]) / (zoom[1] - zoom[0]) : NaN);
</script>

<figure class="arena" bind:clientWidth={W} style:min-height="{meta.minHeight}px">
	{#if choices.length}
		<div class="choices" role="group" aria-label={t({ en: 'Which run', zh: '哪一次运行' })}>
			{#each choices as c (c.source)}
				<button type="button" class="btn" aria-pressed={src === c.source} onclick={() => ((picked = c.source), void rt.ensure(c.source, 'last').catch(() => -1))}>{t(c.label)}</button>
			{/each}
		</div>
	{/if}
	{#if sc}
		<div class="pics" class:narrow role="img" aria-label={description}>
			{#if has('noise') && !zoom}
				<div class="pane">
					<canvas bind:this={aside} style:width="{small}px" style:height="{small}px" aria-hidden="true"></canvas>
					<span class="small muted">{t({ en: 'noise z (z₁ across, z₂ up)', zh: '噪声 z（z₁ 横向，z₂ 纵向）' })}</span>
				</div>
			{/if}
			<div class="pane">
				<canvas bind:this={plane} style:width="{side}px" style:height="{side}px" aria-hidden="true"></canvas>
				<span class="small muted">{t({ en: 'the plane (x across, y up)', zh: '平面（x 横向，y 纵向）' })}</span>
			</div>
			{#if zoom}
				<div class="pane">
					<canvas bind:this={aside} style:width="{small}px" style:height="{small}px" aria-hidden="true"></canvas>
					<span class="small muted">{t({ en: `the framed box, magnified ${fmt3(magnification)}×`, zh: `框出的区域，放大 ${fmt3(magnification)} 倍` })}</span>
				</div>
			{/if}
		</div>
		<figcaption class="small">
			{#if has('real')}<span><i class="fill"></i>{t({ en: 'real points', zh: '真实的点' })} <span class="num muted">({fmtInt(sc.realShown.length / 2)})</span></span>{/if}
			{#if has('fake')}<span><i class="ring"></i>{t({ en: 'generated points', zh: '生成的点' })} <span class="num muted">({fmtInt(sc.fake.length / 2)})</span>{#if sc.off}<span class="muted">{t({ en: `, ${fmtInt(sc.off)} of them off the picture`, zh: `，其中 ${fmtInt(sc.off)} 个在图外` })}</span>{/if}</span>{/if}
			{#if has('map')}<span class="key"><i class="m0"></i>{t({ en: 'D says fake', zh: '判别器认为是假' })} <i class="m1"></i>{t(sc.turn === 0 ? { en: 'D says real (line: ½; dashed: before this step)', zh: '判别器认为是真（线：½；虚线：这一步之前）' } : { en: 'D says real (line: ½)', zh: '判别器认为是真（线：½）' })}</span>{/if}
			<span>{t({ en: 'step', zh: '第' })} <span class="num">{fmtInt(sc.step)}</span>{t({ en: '', zh: ' 步' })}</span>
			{#if has('map')}<span>{#if sc.turn !== 1}{t({ en: 'mean D: real', zh: '判别器平均分：真' })} <span class="num">{pair(sc.dRealBefore, sc.dReal)}</span>{t({ en: ', generated', zh: '，生成' })}{:else}{t({ en: 'mean D on the generated points:', zh: '判别器给生成点的平均分：' })}{/if} <span class="num">{pair(sc.dFakeBefore, sc.dFake)}</span></span>{/if}
			{#if Number.isFinite(along)}<span>{t({ en: 'moved along their arrow:', zh: '顺着箭头移动的点：' })} <span class="num">{fmtInt(along)} / {fmtInt(sc.fake.length / 2)}</span></span>{/if}
			{#if props?.reach && Number.isFinite(sc.recall)}<span>{t({ en: `real points with a generated point within ${NEAR}:`, zh: `${NEAR} 以内有生成点的真实点：` })} <span class="num">{fmtPct(sc.recall)}</span>{t({ en: '; generated points that close to a real one:', zh: '；离真实点这么近的生成点：' })} <span class="num">{fmtPct(sc.precision)}</span></span>{/if}
			{#if sc.centres}<span>{t({ en: 'blobs covered', zh: '覆盖的团' })} <span class="num">{fmtInt(sc.modes)} / {sc.centres.length}</span></span>{/if}
			{#if late.length}<span>{t({ en: 'blobs covered from step', zh: '从第' })} <span class="num">{fmtInt(late[0].from)}</span>{t({ en: ' on:', zh: ' 步起覆盖的团：' })} {#each late as l, i (i)}{i ? t({ en: '; ', zh: '；' }) : ''}{t(l.label)} <span class="num">{l.lo}–{l.hi}</span>{/each}</span>{/if}
		</figcaption>
	{:else}
		<p class="muted small">{t({ en: 'Training these networks… (it starts when you pick it)', zh: '正在训练这两个网络……（选中后开始）' })}</p>
	{/if}
</figure>

<style>
	.arena { margin: 0; display: grid; gap: 8px; min-width: 0; align-content: start; }
	.pics { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }
	.pics.narrow { flex-direction: column; }
	.pane { display: grid; gap: 4px; }
	canvas { display: block; max-width: 100%; border: 1px solid var(--rule); border-radius: var(--r1); }
	.choices, figcaption { display: flex; flex-wrap: wrap; gap: 6px 14px; }
	.btn[aria-pressed='true'] { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
	figcaption { color: var(--ink-2); }
	i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }
	.fill { background: var(--real); }
	.ring { border: 1.5px solid var(--generated); }
	.key i { border-radius: 2px; margin-left: 2px; }
	.m0 { background: color-mix(in srgb, var(--generated) 45%, var(--surface)); }
	.m1 { background: color-mix(in srgb, var(--real) 45%, var(--surface)); }
</style>

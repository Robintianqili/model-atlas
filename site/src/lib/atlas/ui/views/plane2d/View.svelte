<!--
  plane2d: points on a plane (class colors), with optional layers: the network's answer over the plane (a [R, R] grid,
  with the 0.5 boundary for two-class probabilities), arrows (a velocity field, optionally at a time t picked with a
  slider), trajectories and generated samples. Canvas, crisp at any devicePixelRatio. Keyboard: focus the plane, ←/→
  step through the points, Escape clears; the readout under the plane says the point's numbers and the network's answer.
-->
<script lang="ts">
	import type { StageProps } from '../../../types/view.ts';
	import type { I18n } from '../../../types/core.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '../../../core/format.ts';
	import { categorical, makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { isLive, tapeOf } from '../../train/refs.ts';
	import meta, { planeData, pointValue } from './meta.ts';

	// props one by one (no rest/spread props: they would grow the Svelte runtime chunk every page preloads)
	let { rt, source, at, focus, sel, ctx, provenance, autoplay, onend, props }: StageProps = $props();
	const p: StageProps = $derived({ rt, source, at, focus, sel, ctx, provenance, autoplay, onend, props });

	let version = $state(0);
	$effect(() => {
		const tp = tapeOf(p.rt, p.source);
		if (isLive(tp)) return tp.subscribe(() => version++);
	});

	const time = $derived(p.props?.time as { min: number; max: number; value?: number; step?: number; label?: I18n } | undefined);
	let tNow = $state<number | null>(null);
	const tVal = $derived(tNow ?? time?.value ?? time?.min ?? 0);

	const d = $derived.by(() => {
		void version;
		return planeData(p, tVal);
	});

	let W = $state(360);
	const side = $derived(Math.max(200, Math.min(W, 440)));
	let canvas = $state<HTMLCanvasElement | null>(null);
	const theme = $derived(themeState.resolved);

	const sx = (x: number) => ((x - d.bounds.x0) / (d.bounds.x1 - d.bounds.x0)) * side;
	const sy = (y: number) => ((d.bounds.y1 - y) / (d.bounds.y1 - d.bounds.y0)) * side;
	const cls = (i: number) => (d.labels && i < d.labels.data.length ? Math.max(0, Math.round(d.labels.data[i])) : 0);

	let colors: { theme: string; surface: string; rule: string; ink2: string; real: string; generated: string; accent: string } | null = null;
	const rgbOf = (hex: string): number[] => {
		const h = hex.replace('#', '');
		const f = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
		return [parseInt(f.slice(0, 2), 16), parseInt(f.slice(2, 4), 16), parseInt(f.slice(4, 6), 16)];
	};

	function mix(a: number[], b: number[], u: number): number[] {
		return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
	}

	function draw() {
		const c = canvas;
		if (!c) return;
		const dpr = typeof devicePixelRatio === 'number' ? Math.min(3, devicePixelRatio) : 1;
		c.width = Math.round(side * dpr);
		c.height = Math.round(side * dpr);
		const g = c.getContext('2d');
		if (!g) return;
		g.setTransform(dpr, 0, 0, dpr, 0, 0);
		if (!colors || colors.theme !== theme) {
			// read the theme tokens once per color scheme (not on every repaint)
			const css = getComputedStyle(c);
			colors = {
				theme,
				surface: css.getPropertyValue('--surface').trim() || (theme === 'dark' ? '#1a1d21' : '#fffdf8'),
				rule: css.getPropertyValue('--rule').trim() || '#cccccc',
				ink2: css.getPropertyValue('--ink-2').trim() || '#555555',
				real: css.getPropertyValue('--real').trim() || '#1a8a4a',
				generated: css.getPropertyValue('--generated').trim() || '#c2338a',
				accent: css.getPropertyValue('--accent').trim() || '#0b6b70'
			};
		}
		const { surface, rule, ink2, real, generated, accent } = colors;
		g.fillStyle = surface;
		g.fillRect(0, 0, side, side);

		// 1. the network's answer over the plane
		if (d.grid) {
			const [R, C] = d.grid.shape;
			const img = new ImageData(C, R);
			const base = rgbOf(surface.startsWith('#') ? surface : theme === 'dark' ? '#1a1d21' : '#fffdf8');
			const c0 = rgbOf(categorical(0, theme));
			const c1 = rgbOf(categorical(1, theme));
			let lo = Infinity,
				hi = -Infinity;
			for (const v of d.grid.data) {
				lo = Math.min(lo, v);
				hi = Math.max(hi, v);
			}
			const sc = d.gridKind === 'prob' ? null : makeScale(d.gridKind, d.gridKind === 'diverging' ? [-Math.max(Math.abs(lo), Math.abs(hi)), Math.max(Math.abs(lo), Math.abs(hi))] : [Math.min(0, lo), hi || 1], theme);
			for (let i = 0; i < R * C; i++) {
				const v = d.grid.data[i];
				let rgb: number[];
				if (sc) rgb = sc.rgb(v);
				else {
					const u = Math.min(1, Math.abs(v - 0.5) * 2);
					rgb = mix(base, v >= 0.5 ? c1 : c0, 0.12 + 0.43 * u);
				}
				img.data[i * 4] = rgb[0];
				img.data[i * 4 + 1] = rgb[1];
				img.data[i * 4 + 2] = rgb[2];
				img.data[i * 4 + 3] = 255;
			}
			const off = document.createElement('canvas');
			off.width = C;
			off.height = R;
			off.getContext('2d')!.putImageData(img, 0, 0);
			g.imageSmoothingEnabled = true;
			g.drawImage(off, 0, 0, side, side);
			// the 0.5 boundary (marching squares, linear interpolation along cell edges)
			if (d.gridKind === 'prob') {
				g.strokeStyle = ink2;
				g.lineWidth = 1.5;
				g.beginPath();
				const cw = side / C;
				const ch = side / R;
				const val = (r: number, cc: number) => d.grid!.data[r * C + cc] - 0.5;
				for (let r = 0; r + 1 < R; r++)
					for (let cc = 0; cc + 1 < C; cc++) {
						const a = val(r, cc),
							b = val(r, cc + 1),
							e = val(r + 1, cc + 1),
							f = val(r + 1, cc);
						const pts: [number, number][] = [];
						const cx = (cc + 0.5) * cw,
							cy = (r + 0.5) * ch;
						if (a * b < 0) pts.push([cx + (a / (a - b)) * cw, cy]);
						if (b * e < 0) pts.push([cx + cw, cy + (b / (b - e)) * ch]);
						if (f * e < 0) pts.push([cx + (f / (f - e)) * cw, cy + ch]);
						if (a * f < 0) pts.push([cx, cy + (a / (a - f)) * ch]);
						if (pts.length >= 2) {
							g.moveTo(pts[0][0], pts[0][1]);
							g.lineTo(pts[1][0], pts[1][1]);
							if (pts.length === 4) {
								g.moveTo(pts[2][0], pts[2][1]);
								g.lineTo(pts[3][0], pts[3][1]);
							}
						}
					}
				g.stroke();
			}
		}

		// 2. axes
		g.strokeStyle = rule;
		g.lineWidth = 1;
		g.beginPath();
		if (d.bounds.x0 < 0 && d.bounds.x1 > 0) {
			g.moveTo(sx(0), 0);
			g.lineTo(sx(0), side);
		}
		if (d.bounds.y0 < 0 && d.bounds.y1 > 0) {
			g.moveTo(0, sy(0));
			g.lineTo(side, sy(0));
		}
		g.stroke();

		// 3. arrows
		if (d.vectors) {
			const K = d.vectors.shape[0];
			const v = d.vectors.data;
			let maxLen = 0;
			for (let k = 0; k < K; k++) maxLen = Math.max(maxLen, Math.hypot(v[4 * k + 2], v[4 * k + 3]));
			const cell = side / Math.max(4, Math.sqrt(K));
			const scale = maxLen > 0 ? (0.9 * cell) / maxLen : 0;
			g.strokeStyle = ink2;
			g.fillStyle = ink2;
			g.lineWidth = 1.2;
			for (let k = 0; k < K; k++) {
				const x = sx(v[4 * k]),
					y = sy(v[4 * k + 1]);
				const dx = v[4 * k + 2] * scale,
					dy = -v[4 * k + 3] * scale;
				const len = Math.hypot(dx, dy);
				if (len < 0.5) continue;
				g.beginPath();
				g.moveTo(x, y);
				g.lineTo(x + dx, y + dy);
				g.stroke();
				const ux = dx / len,
					uy = dy / len,
					h = Math.min(5, 0.4 * len);
				g.beginPath();
				g.moveTo(x + dx, y + dy);
				g.lineTo(x + dx - h * ux - 0.6 * h * uy, y + dy - h * uy + 0.6 * h * ux);
				g.lineTo(x + dx - h * ux + 0.6 * h * uy, y + dy - h * uy - 0.6 * h * ux);
				g.closePath();
				g.fill();
			}
		}

		// 4. paths
		if (d.paths) {
			const [N, K] = d.paths.shape;
			const a = d.paths.data;
			g.strokeStyle = generated;
			g.globalAlpha = Math.max(0.15, Math.min(0.6, 30 / N));
			g.lineWidth = 1;
			for (let i = 0; i < N; i++) {
				g.beginPath();
				for (let k = 0; k < K; k++) {
					const o = (i * K + k) * 2;
					if (k) g.lineTo(sx(a[o]), sy(a[o + 1]));
					else g.moveTo(sx(a[o]), sy(a[o + 1]));
				}
				g.stroke();
			}
			g.globalAlpha = 1;
		}

		// 5. data points
		if (d.points) {
			const n = d.points.shape[0];
			const r = n > 600 ? 2 : n > 200 ? 3 : 4;
			const xy = d.points.data;
			for (let i = 0; i < n; i++) {
				g.beginPath();
				g.arc(sx(xy[2 * i]), sy(xy[2 * i + 1]), r, 0, Math.PI * 2);
				g.fillStyle = d.labelled ? categorical(cls(i), theme) : real;
				g.fill();
				if (n <= 600) {
					g.lineWidth = 1;
					g.strokeStyle = surface;
					g.stroke();
				}
			}
		}

		// 6. generated samples
		if (d.samples) {
			const s = d.samples.data;
			g.strokeStyle = generated;
			g.lineWidth = 1.2;
			for (let i = 0; i < d.samples.shape[0]; i++) {
				g.beginPath();
				g.arc(sx(s[2 * i]), sy(s[2 * i + 1]), 3, 0, Math.PI * 2);
				g.stroke();
			}
		}

		// 7. highlighted / probed point
		const hi = probe ?? d.highlight;
		if (hi !== null && d.points && hi < d.points.shape[0]) {
			g.beginPath();
			g.arc(sx(d.points.data[2 * hi]), sy(d.points.data[2 * hi + 1]), 9, 0, Math.PI * 2);
			g.lineWidth = 2.5;
			g.strokeStyle = accent;
			g.stroke();
		}
	}

	let probe = $state<number | null>(null);
	$effect(() => {
		void d;
		void side;
		void theme;
		void probe;
		draw();
	});

	function nearest(px: number, py: number): number | null {
		if (!d.points) return null;
		let best: number | null = null;
		let bd = 14 * 14;
		const xy = d.points.data;
		for (let i = 0; i < d.points.shape[0]; i++) {
			const dx = sx(xy[2 * i]) - px,
				dy = sy(xy[2 * i + 1]) - py;
			const dd = dx * dx + dy * dy;
			if (dd < bd) {
				bd = dd;
				best = i;
			}
		}
		return best;
	}
	function move(e: PointerEvent) {
		const r = (e.currentTarget as HTMLCanvasElement).getBoundingClientRect();
		probe = nearest(((e.clientX - r.left) / r.width) * side, ((e.clientY - r.top) / r.height) * side);
	}
	function key(e: KeyboardEvent) {
		const n = d.points?.shape[0] ?? 0;
		if (!n) return;
		const cur = probe ?? d.highlight ?? -1;
		let next: number | null = cur;
		if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = (cur + (e.key === 'ArrowDown' ? 10 : 1)) % n;
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = (cur - (e.key === 'ArrowUp' ? 10 : 1) + n * 10) % n;
		else if (e.key === 'Home') next = 0;
		else if (e.key === 'End') next = n - 1;
		else if (e.key === 'Escape') next = null;
		else return;
		e.preventDefault();
		probe = next;
	}

	const probeInfo = $derived.by(() => {
		const i = probe ?? d.highlight;
		if (i === null || !d.points || i >= d.points.shape[0]) return null;
		const x = d.points.data[2 * i],
			y = d.points.data[2 * i + 1];
		const pv = pointValue(d, i);
		return { i, x, y, c: cls(i), g: pv.v, exact: pv.exact };
	});
	const className = (k: number) => (d.classes?.[k] ? t(d.classes[k]) : t({ en: `class ${k}`, zh: `第 ${k} 类` }));
	const description = $derived(t(meta.describe({ ...p, props: { ...(p.props ?? {}) } })));
</script>

<figure class="pl" bind:clientWidth={W} style:min-height="{meta.minHeight}px">
	<canvas
		bind:this={canvas}
		style:width="{side}px"
		style:height="{side}px"
		tabindex="0"
		role="slider"
		aria-label={description}
		aria-valuemin={0}
		aria-valuemax={Math.max(0, (d.points?.shape[0] ?? 1) - 1)}
		aria-valuenow={probe ?? d.highlight ?? 0}
		aria-valuetext={probeInfo ? `${t({ en: 'point', zh: '点' })} ${probeInfo.i}: x ${fmt3(probeInfo.x)}, y ${fmt3(probeInfo.y)}` : t({ en: 'no point selected', zh: '未选中点' })}
		onpointermove={move}
		onpointerdown={move}
		onpointerleave={() => (probe = null)}
		onkeydown={key}
		onblur={() => (probe = null)}
	></canvas>
	{#if time}
		<label class="time small">
			<span>{time.label ? t(time.label) : 't'} = <span class="num">{fmt3(tVal)}</span></span>
			<input type="range" min={time.min} max={time.max} step={time.step ?? (time.max - time.min) / 50} value={tVal} oninput={(e) => (tNow = Number((e.currentTarget as HTMLInputElement).value))} />
		</label>
	{/if}
	<figcaption class="small">
		{#if d.points && d.labelled}
			<span class="legend">
				{#each d.classCounts as c, k (k)}<span><i style:background={categorical(k, theme)}></i>{className(k)} <span class="muted num">({c})</span></span>{/each}
			</span>
		{/if}
		{#if d.grid && d.gridKind === 'prob'}
			<span class="muted">{t({ en: 'Background: the network’s answer at each place (line: 50/50).', zh: '背景：网络在每个位置的回答（线：五五开）。' })}</span>
		{/if}
		{#if probeInfo}
			<span class="readout">
				{t({ en: 'Point', zh: '点' })} <span class="num">{probeInfo.i}</span>: x = <span class="num">{fmt3(probeInfo.x)}</span>, y = <span class="num">{fmt3(probeInfo.y)}</span>{#if d.labelled}, {className(probeInfo.c)}{/if}{#if Number.isFinite(probeInfo.g)}; {d.gridKind === 'prob' ? t({ en: 'share for class 1', zh: '属于第 1 类的比例' }) : t({ en: 'value', zh: '值' })} {probeInfo.exact ? '' : '≈ '}<span class="num">{fmt3(probeInfo.g)}</span>{/if}
			</span>
		{/if}
		{#if !d.points && !d.samples && !d.paths && !d.grid && !d.vectors}<span class="muted">{t({ en: 'No points yet.', zh: '还没有点。' })}</span>{/if}
		{#if d.samples}<span class="muted"><i class="ring"></i>{t({ en: 'generated points', zh: '生成的点' })}</span>{/if}
	</figcaption>
</figure>

<style>
	.pl {
		margin: 0;
		display: grid;
		gap: 6px;
		justify-items: start;
		min-width: 0;
		align-content: start;
	}
	canvas {
		display: block;
		border: 1px solid var(--rule);
		border-radius: var(--r1);
		touch-action: none;
		max-width: 100%;
	}
	canvas:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
	figcaption {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		color: var(--ink-2);
	}
	.legend {
		display: inline-flex;
		flex-wrap: wrap;
		gap: 4px 12px;
	}
	.legend i {
		display: inline-block;
		width: 10px;
		height: 10px;
		border-radius: 50%;
		margin-right: 5px;
		vertical-align: -1px;
	}
	.ring {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 50%;
		border: 1.5px solid var(--generated);
		margin-right: 5px;
		vertical-align: -1px;
	}
	.time {
		display: flex;
		gap: 8px;
		align-items: center;
		width: 100%;
		max-width: 440px;
	}
	.time input {
		flex: 1;
	}
</style>

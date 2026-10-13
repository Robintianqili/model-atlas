<!--
  gaussian-process Capture (local adapter 'gp-points'): pick a preset, or click the plot to add a point (click a point
  to remove it). Keyboard: the arrow keys move a cross (Shift: bigger steps), Enter adds a point there, Delete removes
  the last one. The kernel's settings and the samples' seed are kept; "New samples" draws other sample functions.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { MAXP, PRESETS, pointsOf, rawOf, XHI, XLO, YHI, YLO, type GpRaw, type Pt } from '../../gp.ts';

	interface Props { raw: unknown; onchange: (raw: GpRaw) => void; compact?: boolean }
	let { raw, onchange, compact = false }: Props = $props();
	const r = $derived(rawOf(raw));
	const pts = $derived(pointsOf(raw));
	const keep = $derived({ ...(r.ls !== undefined ? { ls: r.ls } : {}), ...(r.sf2 !== undefined ? { sf2: r.sf2 } : {}), ...(r.seed !== undefined ? { seed: r.seed } : {}) });
	let W = $state(480);
	const H = $derived(compact ? 130 : 180);
	const X = (x: number) => 6 + ((x - XLO) / (XHI - XLO)) * (W - 12);
	const Y = (y: number) => 6 + ((YHI - y) / (YHI - YLO)) * (H - 12);
	const r1 = (v: number) => Math.round(v * 10) / 10;
	let cur = $state({ x: 0, y: 0 });
	const put = (p: Pt[]) => onchange({ points: p, ...keep });

	function click(e: MouseEvent) {
		const b = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
		const px = ((e.clientX - b.left) / b.width) * W, py = ((e.clientY - b.top) / b.height) * H;
		const hit = pts.findIndex((p) => Math.hypot(X(p.x) - px, Y(p.y) - py) < 9);
		if (hit >= 0) return put(pts.filter((_, i) => i !== hit));
		if (pts.length >= MAXP) return;
		const x = r1(XLO + ((px - 6) / (W - 12)) * (XHI - XLO)), y = r1(YHI - ((py - 6) / (H - 12)) * (YHI - YLO));
		cur = { x: Math.max(XLO, Math.min(XHI, x)), y: Math.max(YLO, Math.min(YHI, y)) };
		put([...pts, cur]);
	}
	function key(e: KeyboardEvent) {
		const s = e.shiftKey ? 1 : 0.1, mv: Record<string, [number, number]> = { ArrowLeft: [-s, 0], ArrowRight: [s, 0], ArrowUp: [0, s], ArrowDown: [0, -s] };
		if (mv[e.key]) cur = { x: r1(Math.max(XLO, Math.min(XHI, cur.x + mv[e.key][0]))), y: r1(Math.max(YLO, Math.min(YHI, cur.y + mv[e.key][1]))) };
		else if ((e.key === 'Enter' || e.key === ' ') && pts.length < MAXP) put([...pts, { ...cur }]);
		else if ((e.key === 'Delete' || e.key === 'Backspace') && pts.length) put(pts.slice(0, -1));
		else return;
		e.preventDefault();
	}
	const name = $derived(r.points ? null : (r.preset ?? 'gap'));
	const f1 = (v: number) => v.toFixed(1).replace('-', '−');
</script>

<div class="cap">
	<div class="row" role="group" aria-label={t({ en: 'Point sets', zh: '数据点预设' })}>
		{#each PRESETS as p (p.id)}<button type="button" class="btn" aria-pressed={name === p.id} onclick={() => onchange({ preset: p.id, ...keep })}>{t(p.label)}</button>{/each}
		<button type="button" class="btn" onclick={() => put([])}>{t({ en: 'Clear', zh: '清空' })}</button>
		<button type="button" class="btn" onclick={() => onchange({ ...r, seed: (r.seed ?? 1) + 1 })}>{t({ en: 'New samples', zh: '换一组样本' })}</button>
	</div>
	<div bind:clientWidth={W}>
		<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions (role application: a custom 2-D input with its own keys) -->
		<svg width={W} height={H} viewBox="0 0 {W} {H}" tabindex="0" role="application" onclick={click} onkeydown={key}
			aria-label={t({ en: `The points: click to add one, click a point to remove it; arrow keys move the cross to x ${f1(cur.x)}, y ${f1(cur.y)}, Enter adds, Delete removes the last. ${pts.length} points.`, zh: `数据点：点击添加，点击已有的点删除；方向键移动十字（现在 x ${f1(cur.x)}，y ${f1(cur.y)}），回车添加，Delete 删除最后一个。共 ${pts.length} 个点。` })}>
			<rect x="0" y="0" width={W} height={H} rx="4" class="bg" />
			<line x1="6" x2={W - 6} y1={Y(0)} y2={Y(0)} class="axis" /><line x1={X(0)} x2={X(0)} y1="6" y2={H - 6} class="axis" />
			<path d="M{X(cur.x) - 5},{Y(cur.y)}h10M{X(cur.x)},{Y(cur.y) - 5}v10" class="cross" />
			{#each pts as p, i (i)}<circle cx={X(p.x)} cy={Y(p.y)} r="4.5" class="pt" />{/each}
		</svg>
	</div>
	<p class="note">
		{#if !compact}{t({ en: `x from ${f1(XLO)} to ${f1(XHI)}, y from ${f1(YLO)} to ${f1(YHI)}; at most ${MAXP} points. The model adds them in the order you click.`, zh: `x 从 ${f1(XLO)} 到 ${f1(XHI)}，y 从 ${f1(YLO)} 到 ${f1(YHI)}；最多 ${MAXP} 个点。模型按你点击的顺序逐个加入。` })}{:else}{t({ en: 'Click to add a point, click a point to remove it.', zh: '点击添加一个点，点击已有的点删除它。' })}{/if}
	</p>
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	svg { display: block; max-width: 100%; touch-action: manipulation; cursor: crosshair; }
	svg:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
	.bg { fill: var(--surface); stroke: var(--rule-strong); }
	.axis { stroke: var(--rule); stroke-width: 1; }
	.cross { stroke: var(--muted); stroke-width: 1.4; }
	.pt { fill: var(--cat-1); stroke: var(--surface); stroke-width: 1.4; }
	p { margin: 0; }
	.note { font-size: var(--fs-xs); color: var(--muted); }
</style>

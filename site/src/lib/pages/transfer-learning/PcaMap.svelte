<!--
  Where drawings land in the frozen feature space: a 2-D PCA map (basis and context points
  precomputed in pca.json) with the learner's examples and pad drawing projected live, and the
  new head's decision boundary on this plane (exact on the plane, because the head is linear).
-->
<script lang="ts">
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { t, i18n, type Text } from '$lib/i18n/i18n.svelte';
	import { projectPCA, predictHead, type Head } from '$lib/models/tinycnn_finetune';
	import { markerPath } from './chart';
	import type { PcaJson } from './data';

	interface MapItem {
		id: string;
		cls: 0 | 1;
		feat: Float32Array | null;
		drawn: boolean;
		flipped: boolean;
	}
	interface Props {
		pca: PcaJson;
		items: MapItem[];
		pad: Float32Array | null;
		head: Head | null;
		trained: boolean;
		preset: boolean;
		names: [string, string];
		testFeats: Float32Array[] | null;
		testLabels: number[] | null;
		hover?: string | null;
		basisKey?: 'all' | 'heldOut';
		onselect?: (id: string) => void;
		/** the learner's earlier drawings (guided lesson), drawn as small numbered marks */
		trail?: { feat: Float32Array; n: number }[];
		/** show the basis switch / boundary checkbox row (default true) */
		showBar?: boolean;
		/** label of the big pad mark (default “your drawing”; the lesson says “this drawing” for a preset test drawing) */
		padLabel?: Text;
		/** the legend's own provenance label (the guided lesson's stage header carries one instead) */
		badge?: boolean;
	}
	let {
		pca,
		items,
		pad,
		head,
		trained,
		preset,
		names,
		testFeats,
		testLabels,
		hover = $bindable(null),
		basisKey = $bindable('all'),
		onselect,
		trail = [],
		showBar = true,
		padLabel = { en: 'your drawing', zh: '你的画' },
		badge = true
	}: Props = $props();

	let showLine = $state(true);
	let w = $state(360);
	const basis = $derived(pca.bases[basisKey]);
	const H = $derived(Math.round(Math.min(440, Math.max(280, w * 0.72))));
	const PAD = 14;

	const proj = $derived(
		items.filter((e) => e.feat).map((e) => ({ ...e, p: projectPCA(basis, e.feat!) as [number, number] }))
	);
	const padP = $derived(pad ? (projectPCA(basis, pad) as [number, number]) : null);
	const trailP = $derived(trail.map((d) => ({ n: d.n, p: projectPCA(basis, d.feat) as [number, number] })));

	// equal-aspect scale fitted to all points (precomputed + live)
	const view = $derived.by(() => {
		let x0 = Infinity,
			x1 = -Infinity,
			y0 = Infinity,
			y1 = -Infinity;
		const add = (x: number, y: number) => {
			if (x < x0) x0 = x;
			if (x > x1) x1 = x;
			if (y < y0) y0 = y;
			if (y > y1) y1 = y;
		};
		for (const p of basis.points) add(p.x, p.y);
		for (const e of proj) add(e.p[0], e.p[1]);
		if (padP) add(padP[0], padP[1]);
		for (const d of trailP) add(d.p[0], d.p[1]);
		const s = Math.min((w - 2 * PAD) / (x1 - x0 || 1), (H - 2 * PAD) / (y1 - y0 || 1));
		const cx = (x0 + x1) / 2;
		const cy = (y0 + y1) / 2;
		return {
			s,
			X: (x: number) => w / 2 + (x - cx) * s,
			// screen y grows downward; PCA y up
			Y: (y: number) => H / 2 - (y - cy) * s,
			inv: (px: number, py: number) => [cx + (px - w / 2) / s, cy - (py - H / 2) / s] as [number, number]
		};
	});

	// The head's margin z_A − z_B on the PCA plane through the feature mean is a·x + b·y + c (exact on the plane).
	const line = $derived.by(() => {
		if (!head) return null;
		const [c0, c1] = basis.components;
		let a = 0,
			b = 0,
			c = head.b[0] - head.b[1];
		for (let d = 0; d < head.D; d++) {
			const dw = (head.W[d] - head.W[head.D + d]) / head.scale;
			a += dw * c0[d];
			b += dw * c1[d];
			c += dw * basis.mean[d];
		}
		return { a, b, c };
	});

	// clip the view rectangle by the half-plane margin >= 0 (side A) and < 0 (side B)
	function clipRect(sign: 1 | -1): [number, number][] {
		if (!line) return [];
		const rect: [number, number][] = [
			[0, 0],
			[w, 0],
			[w, H],
			[0, H]
		];
		const m = (px: number, py: number) => {
			const [x, y] = view.inv(px, py);
			return sign * (line.a * x + line.b * y + line.c);
		};
		const out: [number, number][] = [];
		for (let i = 0; i < rect.length; i++) {
			const P = rect[i];
			const Q = rect[(i + 1) % rect.length];
			const mp = m(P[0], P[1]);
			const mq = m(Q[0], Q[1]);
			if (mp >= 0) out.push(P);
			if (mp >= 0 !== mq >= 0) {
				const tt = mp / (mp - mq);
				out.push([P[0] + (Q[0] - P[0]) * tt, P[1] + (Q[1] - P[1]) * tt]);
			}
		}
		return out;
	}
	const polyA = $derived(clipRect(1));
	const polyB = $derived(clipRect(-1));
	const seg = $derived.by(() => {
		// where the boundary crosses the edges of the view rectangle
		if (!line) return null;
		const rect: [number, number][] = [
			[0, 0],
			[w, 0],
			[w, H],
			[0, H]
		];
		const m = (px: number, py: number) => {
			const [x, y] = view.inv(px, py);
			return line.a * x + line.b * y + line.c;
		};
		const pts: [number, number][] = [];
		for (let i = 0; i < 4; i++) {
			const P = rect[i];
			const Q = rect[(i + 1) % 4];
			const mp = m(P[0], P[1]);
			const mq = m(Q[0], Q[1]);
			if (mp >= 0 !== mq >= 0) {
				const tt = mp / (mp - mq);
				pts.push([P[0] + (Q[0] - P[0]) * tt, P[1] + (Q[1] - P[1]) * tt]);
			}
		}
		return pts.length >= 2 ? [pts[0], pts[1]] : null;
	});

	// How often the side of the line agrees with the head's real 256-number decision
	const agreement = $derived.by(() => {
		if (!line || !head) return null;
		const feats = preset && testFeats ? testFeats : proj.map((e) => e.feat!);
		if (!feats.length) return null;
		let ok = 0;
		for (const f of feats) {
			const [x, y] = projectPCA(basis, f);
			const side = line.a * x + line.b * y + line.c > 0 ? 0 : 1;
			if (side === predictHead(head, f).top) ok++;
		}
		return { ok, n: feats.length, test: preset && !!testFeats };
	});

	const cname = (c: number) => (i18n.lang === 'zh' ? pca.classes[c].zh : pca.classes[c].en);
	const centroids = $derived.by(() => {
		const acc = new Map<number, [number, number, number]>();
		for (const p of basis.points) {
			const v = acc.get(p.c) ?? [0, 0, 0];
			v[0] += p.x;
			v[1] += p.y;
			v[2]++;
			acc.set(p.c, v);
		}
		return [...acc.entries()].map(([c, v]) => ({ c, x: v[0] / v[2], y: v[1] / v[2] }));
	});
	// class-name labels at the centroids, nudged apart vertically when two would overlap
	const labels = $derived.by(() => {
		const placed: { x: number; y: number; w: number }[] = [];
		const cw = i18n.lang === 'zh' ? 12 : 6.6;
		return centroids.map((c) => {
			const x = view.X(c.x);
			let y = view.Y(c.y);
			const wTxt = cname(c.c).length * cw + 8;
			for (let k = 0; k < 8; k++) {
				const hit = placed.find((p) => Math.abs(p.x - x) < (p.w + wTxt) / 2 && Math.abs(p.y - y) < 13);
				if (!hit) break;
				y = hit.y + (y >= hit.y ? 13 : -13);
			}
			placed.push({ x, y, w: wTxt });
			return { ...c, px: x, py: y };
		});
	});
	const heldCls = (c: number) => pca.classes[c]?.heldOut;
	// ladder = class A, envelope = class B in the preset pair
	const heldAB = (c: number) => (pca.classes[c].id === 'ladder' ? 0 : 1);
	const dot = (x: number, y: number, r: number) =>
		`M${(x - r).toFixed(1)} ${y.toFixed(1)}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`;
	const ctxPaths = $derived.by(() => {
		let gray = '',
			heldGray = '',
			a = '',
			b = '';
		for (const p of basis.points) {
			const x = view.X(p.x);
			const y = view.Y(p.y);
			if (!heldCls(p.c)) gray += dot(x, y, 2.3);
			else if (!preset) heldGray += dot(x, y, 2.3);
			else if (heldAB(p.c) === 0) a += dot(x, y, 2.6);
			else b += dot(x, y, 2.6);
		}
		return { gray, heldGray, a, b };
	});

	// hover: nearest mark within 16 px
	let tip = $state<{ x: number; y: number; text: string } | null>(null);
	function move(e: PointerEvent, svg: SVGSVGElement) {
		const r = svg.getBoundingClientRect();
		const px = ((e.clientX - r.left) / r.width) * w;
		const py = ((e.clientY - r.top) / r.height) * H;
		let best: { d: number; x: number; y: number; text: string; id: string | null } | null = null;
		const consider = (x: number, y: number, text: string, id: string | null, bonus = 0) => {
			const d = Math.hypot(view.X(x) - px, view.Y(y) - py) - bonus;
			if (d < 16 && (!best || d < best.d)) best = { d, x: view.X(x), y: view.Y(y), text, id };
		};
		for (const p of basis.points)
			consider(p.x, p.y, t({ en: `${cname(p.c)}: a test drawing (precomputed)`, zh: `${cname(p.c)}：一张测试图（预计算）` }), null);
		for (const e of proj)
			consider(
				e.p[0],
				e.p[1],
				`${names[e.cls]}: ${e.drawn ? t({ en: 'your drawing', zh: '你画的' }) : t({ en: 'preset example', zh: '预置样本' })}${e.flipped ? t({ en: ' (wrong label)', zh: '（标错）' }) : ''}`,
				e.id,
				6
			);
		if (padP) consider(padP[0], padP[1], t(padLabel), null, 6);
		for (const d of trailP) consider(d.p[0], d.p[1], t({ en: `your earlier drawing ${d.n}`, zh: `你之前的第 ${d.n} 张画` }), null, 4);
		const b = best as { d: number; x: number; y: number; text: string; id: string | null } | null;
		tip = b ? { x: b.x, y: b.y, text: b.text } : null;
		hover = b?.id ?? null;
	}
	function leave() {
		tip = null;
		hover = null;
	}
	function click() {
		if (hover && onselect) onselect(hover);
	}
	const evr = $derived(basis.explainedVarianceRatio.map((v) => (v * 100).toFixed(1)));
</script>

<div class="map">
	{#if showBar}
		<div class="bar">
			<div class="segmented" role="group" aria-label={t({ en: 'Which PCA map', zh: '选择哪张 PCA 地图' })}>
				<button type="button" aria-pressed={basisKey === 'all'} onclick={() => (basisKey = 'all')}>{t({ en: 'Fit on all 12 classes', zh: '在全部 12 类上拟合' })}</button>
				<button type="button" aria-pressed={basisKey === 'heldOut'} onclick={() => (basisKey = 'heldOut')}>{t({ en: 'Fit on ladder + envelope only', zh: '只在梯子 + 信封上拟合' })}</button>
			</div>
			{#if head && trained}
				<label class="chk"><input type="checkbox" bind:checked={showLine} /> {t({ en: 'Show the head’s boundary', zh: '显示输出头的分界线' })}</label>
			{/if}
		</div>
	{/if}
	<div class="plot" bind:clientWidth={w}>
		<!-- Pointer-only extras: every training example on the map is also a focusable thumbnail in the class columns (same hover link and selection), so keyboard users lose nothing. -->
		<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
		<svg
			viewBox="0 0 {w} {H}"
			width={w}
			height={H}
			role="img"
			aria-label={t({
				en: `PCA map of the frozen features: ${items.length ? `${items.length} training examples, ` : ''}${basis.points.length} precomputed test drawings of 12 classes${padP ? ', and your drawing' : ''}${trailP.length ? `, and ${trailP.length} of your earlier drawings` : ''}.`,
				zh: `冻结特征的 PCA 地图：${items.length ? `${items.length} 个训练样本，` : ''}12 个类别共 ${basis.points.length} 张预计算测试图${padP ? '，以及你的画' : ''}${trailP.length ? `，还有你之前的 ${trailP.length} 张画` : ''}。`
			})}
			onpointermove={(e) => move(e, e.currentTarget)}
			onpointerdown={(e) => move(e, e.currentTarget)}
			onpointerleave={leave}
			onclick={click}
		>
			<rect x="0" y="0" width={Math.max(0, w)} height={H} class="bg" />
			{#if line && showLine && trained}
				{#if polyA.length > 2}<polygon class="side a" points={polyA.map((p) => p.join(',')).join(' ')} />{/if}
				{#if polyB.length > 2}<polygon class="side b" points={polyB.map((p) => p.join(',')).join(' ')} />{/if}
				{#if seg}<line class="bound" x1={seg[0][0]} y1={seg[0][1]} x2={seg[1][0]} y2={seg[1][1]} />{/if}
			{/if}
			<!-- 300 context points as 4 paths (one per style): far fewer DOM nodes to lay out and paint -->
			<path class="pt" d={ctxPaths.gray} />
			<path class="pt heldgray" d={ctxPaths.heldGray} />
			<path class="pt held a" d={ctxPaths.a} />
			<path class="pt held b" d={ctxPaths.b} />
			{#each labels.filter((c) => !heldCls(c.c)) as c (c.c)}
				<text class="cl" x={c.px} y={c.py} text-anchor="middle">{cname(c.c)}</text>
			{/each}
			{#each proj as e (e.id)}
				<path
					class="ex {e.cls === 0 ? 'a' : 'b'}"
					class:hov={hover === e.id}
					class:flipped={e.flipped}
					d={markerPath(e.cls === 0 ? 'circle' : 'square', hover === e.id ? 7.5 : 5.5)}
					transform="translate({view.X(e.p[0])},{view.Y(e.p[1])})"
				/>
			{/each}
			{#each labels.filter((c) => heldCls(c.c)) as c (c.c)}
				<text class="cl strong" x={c.px} y={c.py} text-anchor="middle">{cname(c.c)}{!preset ? ' *' : ''}</text>
			{/each}
			{#each proj.filter((e) => e.id === hover) as e (e.id)}
				<!-- the hovered example again, on top of the labels -->
				<path
					class="ex hov {e.cls === 0 ? 'a' : 'b'}"
					d={markerPath(e.cls === 0 ? 'circle' : 'square', 7.5)}
					transform="translate({view.X(e.p[0])},{view.Y(e.p[1])})"
				/>
			{/each}
			{#each trailP as d (d.n)}
				<g transform="translate({view.X(d.p[0])},{view.Y(d.p[1])})" class="trailmk">
					<path d="M-4.5 -4.5L4.5 4.5M4.5 -4.5L-4.5 4.5" />
					<text x="6" y="-5">{d.n}</text>
				</g>
			{/each}
			{#if padP}
				<g transform="translate({view.X(padP[0])},{view.Y(padP[1])})" class="padmk">
					<path d="M-8 0H8M0 -8V8M-5.5 -5.5L5.5 5.5M5.5 -5.5L-5.5 5.5" />
					<text x="10" y="-8">{t(padLabel)}</text>
				</g>
			{/if}
			{#if tip}
				<g class="tip" transform="translate({Math.min(tip.x, w - 10)},{tip.y})">
					<text x="0" y="-12" text-anchor={tip.x > w * 0.6 ? 'end' : 'start'}>{tip.text}</text>
				</g>
			{/if}
		</svg>
	</div>
	<div class="foot">
		<p class="keys">
			<span><svg width="14" height="14" aria-hidden="true"><path class="ex a" d={markerPath('circle', 5)} transform="translate(7,7)" /></svg>{names[0]}</span>
			<span><svg width="14" height="14" aria-hidden="true"><path class="ex b" d={markerPath('square', 5)} transform="translate(7,7)" /></svg>{names[1]}</span>
			<span><svg width="14" height="14" aria-hidden="true"><circle class="pt" cx="7" cy="7" r="3" /></svg>{t({ en: 'test drawings of other classes', zh: '其他类别的测试图' })}</span>
			{#if !preset}<span class="muted">* {t({ en: 'held-out classes, not part of your pair', zh: '留出类别，不属于你这一对' })}</span>{/if}
			{#if badge}<ProvenanceBadge mode="live" compact note={{ en: 'Your examples and pad drawing are projected live; the gray context points and the PCA axes are precomputed (pca.json).', zh: '你的样本和画板上的画是实时投影的；灰色的参照点和 PCA 坐标轴是预计算的（pca.json）。' }} />{/if}
		</p>
		<p class="muted small">
			{t({
				en: `This map keeps only 2 of 256 directions: they explain ${evr[0]}% and ${evr[1]}% of the variation in the features. The head works in all 256.`,
				zh: `这张地图只保留了 256 个方向中的 2 个：它们分别解释了特征变化的 ${evr[0]}% 和 ${evr[1]}%。输出头用的是全部 256 个方向。`
			})}
			{#if agreement && showLine && trained}
				{t({
					en: ` The dashed line is the head’s decision on this flat slice; its side agrees with the head’s full decision for ${agreement.ok} of ${agreement.n} ${agreement.test ? 'test drawings' : 'of your examples'}.`,
					zh: ` 虚线是输出头在这个平面切片上的决策；按它划分的结果，与输出头完整决策一致的${agreement.test ? '测试图' : '样本'}有 ${agreement.ok} / ${agreement.n}。`
				})}
				{#if agreement.ok < 0.8 * agreement.n}
					<strong>
						{t({
							en: 'That is low: this head relies on directions the 2-D map does not show, so read the line with care.',
							zh: '这个比例偏低：这个输出头依赖的方向大多不在这张二维地图里，看这条线时要小心。'
						})}
					</strong>
				{/if}
			{/if}
		</p>
	</div>
</div>

<style>
	.map {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 8px;
		min-width: 0;
	}
	.bar {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 16px;
		align-items: center;
	}
	.chk {
		font-size: var(--fs-sm);
		display: inline-flex;
		gap: 6px;
		align-items: center;
	}
	.chk input {
		accent-color: var(--accent);
	}
	.plot {
		min-width: 0;
	}
	svg {
		display: block;
		max-width: 100%;
		height: auto;
		border-radius: var(--r2);
		touch-action: pan-y;
	}
	.bg {
		fill: var(--surface);
		stroke: var(--rule);
	}
	.side.a {
		fill: var(--div-pos);
		opacity: 0.08;
	}
	.side.b {
		fill: var(--div-neg);
		opacity: 0.08;
	}
	.bound {
		stroke: var(--ink);
		stroke-width: 1.5;
		stroke-dasharray: 6 4;
	}
	.pt {
		fill: var(--muted);
		opacity: 0.4;
	}
	.pt.heldgray {
		fill: var(--ink-2);
		opacity: 0.6;
	}
	.pt.held.a {
		fill: var(--div-pos);
		opacity: 0.55;
	}
	.pt.held.b {
		fill: var(--div-neg);
		opacity: 0.55;
	}
	.cl {
		font-size: var(--t-xs);
		fill: var(--muted);
		paint-order: stroke;
		stroke: var(--surface);
		stroke-width: 3px;
		stroke-linejoin: round;
		pointer-events: none;
	}
	.cl.strong {
		fill: var(--ink);
		font-weight: 700;
	}
	.ex {
		stroke: var(--surface);
		stroke-width: 2;
		paint-order: stroke;
		cursor: pointer;
	}
	.ex.a {
		fill: var(--div-pos);
	}
	.ex.b {
		fill: var(--div-neg);
	}
	.ex.hov {
		stroke: var(--ink);
	}
	.ex.flipped {
		stroke: var(--illus);
		stroke-width: 2.5;
		stroke-dasharray: 2 2;
	}
	.padmk path {
		stroke: var(--ink);
		stroke-width: 2.5;
		stroke-linecap: round;
	}
	.trailmk path {
		stroke: var(--ink-2);
		stroke-width: 1.8;
		stroke-linecap: round;
	}
	.trailmk text {
		font-size: var(--t-xs);
		font-family: var(--font-mono);
		fill: var(--ink-2);
		paint-order: stroke;
		stroke: var(--surface);
		stroke-width: 3px;
	}
	.padmk text,
	.tip text {
		font-size: 12px;
		fill: var(--ink);
		paint-order: stroke;
		stroke: var(--surface);
		stroke-width: 4px;
		stroke-linejoin: round;
		font-weight: 600;
	}
	.tip {
		pointer-events: none;
	}
	.foot {
		display: grid;
		gap: 4px;
	}
	.keys {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		align-items: center;
		font-size: var(--fs-sm);
	}
	.keys span {
		display: inline-flex;
		align-items: center;
		gap: 5px;
	}
	.small {
		margin: 0;
		font-size: var(--fs-xs);
		max-width: var(--measure);
	}
</style>

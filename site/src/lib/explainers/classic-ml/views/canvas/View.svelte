<!-- classic-ml 'canvas' (meta.ts lists what each props.show draws). The plane is [−B, B]². In the kNN steps a click (or the
     arrow keys) moves the new point; the sliders set k, the tree's depth or the number of centroids. Each edits the
     learner's input through setInput, so every source runs again on it. props.formula adds the step's arithmetic. -->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
	import { alignTex, texValues } from '../../../../atlas/ui/views/kit.parts/tex.ts';
	import { B, CENTROIDS, DMAX, R, rawOf } from '../../ml.ts';
	import { framesOf, pointsAt, question, runs, stateAt, treeAt, type Show, type TNode } from './meta.ts';

	let P: StageProps = $props();
	const show = $derived(String(P.props.show ?? 'knn') as Show);
	const tape = $derived(P.at.tape);
	const idx = $derived(P.at.index);
	const f = $derived(P.at.frame);
	const m = $derived(f?.metrics ?? {});
	const st = $derived(stateAt(tape, idx));
	const pts = $derived.by(() => {
		const xy = P.rt.pre?.tensors.xy, label = P.rt.pre?.tensors.label;
		return pointsAt(tape, idx) ?? (xy && label ? { n: xy.shape[0], xy: xy.data, label: label.data } : null);
	});
	// the input, re-read whenever the stage moves (a new input re-resolves the step, so P.at changes)
	const raw = $derived((void P.at, rawOf(P.rt.input)));
	const theme = $derived(themeState.resolved);
	const col = (c: number) => categorical(c, theme);
	const ccol = (j: number) => categorical(4 + j, theme);
	const S = 300, sx = (x: number) => ((x + B) / (2 * B)) * S, sy = (y: number) => ((B - y) / (2 * B)) * S, cell = S / R;
	const X = (i: number) => pts!.xy[2 * i], Y = (i: number) => pts!.xy[2 * i + 1];
	// the stage passes StageProps.setInput (since K1); a host without it (the view gallery) falls back to rt.setInput
	const set = (patch: Record<string, unknown>) => (P.setInput ?? ((x: unknown) => P.rt.setInput(x)))({ ...raw, ...patch });
	const classes = $derived(tape?.frame(0).metrics?.classes ?? 2);
	const per = (pre: string, n: number) => Array.from({ length: n }, (_, k) => m[`${pre}${k}`] ?? 0);

	// kNN: the new point, every distance, the k nearest
	const q = $derived(stateAt(tape, 0).q?.data ?? null);
	const order = $derived.by(() => {
		const d = st.d?.data;
		return d ? Array.from(d, (_, i) => i).sort((a, b) => d[a] - d[b] || a - b) : [];
	});
	const near = $derived(st.nn ? Array.from(st.nn.data) : order.slice(0, 5));
	function move(x: number, y: number) {
		const c = (v: number) => Math.round(Math.max(-B, Math.min(B, v)) * 100) / 100;
		set({ q: { x: c(x), y: c(y) } });
	}
	function click(e: MouseEvent) {
		const r = (e.currentTarget as SVGRectElement).getBoundingClientRect();
		move(((e.clientX - r.left) / r.width) * 2 * B - B, B - ((e.clientY - r.top) / r.height) * 2 * B);
	}
	function key(e: KeyboardEvent) {
		const d = { ArrowLeft: [-0.05, 0], ArrowRight: [0.05, 0], ArrowUp: [0, 0.05], ArrowDown: [0, -0.05] }[e.key];
		if (!d || !q) return;
		e.preventDefault();
		move(q[0] + d[0], q[1] + d[1]);
	}

	// maps: kNN's for the frame's k, or the tree's at the learner's depth
	const depth = $derived(raw.depth ?? DMAX);
	const depths = $derived(framesOf(tape, 'depth'));
	const grid = $derived(show === 'depth' ? stateAt(tape, tape?.find({ phase: 'depth', step: depth }) ?? -1).grid : show === 'map' ? st.grid : undefined);
	const best = $derived(depths.reduce((b, x) => (x.metrics!.testAcc > b.metrics!.testAcc ? x : b), depths[0]));

	// the tree so far, and its diagram: one column per leaf (left to right), a question centred over its two branches,
	// one row per depth; drawn at DX px per leaf, shrunk to the stage width down to 70 % before it scrolls
	const tree = $derived(tape && (show === 'grow' || show === 'split') ? treeAt(tape, idx) : new Map<number, TNode>());
	const nodes = $derived([...tree.values()]);
	const DX = 16, DY = 32, PAD = 26;
	const diagram = $derived.by(() => {
		const pos = new Map<number, [number, number]>();
		let leaf = 0;
		const walk = (id: number): number => {
			const n = tree.get(id);
			if (!n) return NaN;
			const x = n.kids.length ? (walk(n.kids[0]) + walk(n.kids[1])) / 2 : leaf++;
			pos.set(id, [PAD + DX / 2 + x * DX, 26 + n.depth * DY]);
			return x;
		};
		walk(0);
		const h = Math.max(0, ...nodes.map((n) => n.depth)) + 1;
		return { pos, w: leaf, W: leaf * DX + 2 * PAD, H: (h - 1) * DY + 46 };
	});
	const P0 = (id: number) => diagram.pos.get(id) ?? [0, 0];
	const cand = $derived(st.cand && f?.phase === 'node' ? Array.from({ length: st.cand.shape[0] }, (_, i) => Array.from(st.cand!.data.slice(3 * i, 3 * i + 3))) : []);
	const box = $derived(f?.phase === 'node' ? [m.x0, m.x1, m.y0, m.y1] : null);

	// k-means: clusters, centroids (and where they were), the total squared distance so far, the k sweep
	const cen = $derived(st.C ? Array.from(st.C.data) : []);
	const was = $derived(f?.phase === 'update' ? Array.from(stateAt(tape, idx - 1).C?.data ?? []) : []);
	const assign = $derived(st.assign?.data);
	const curve = $derived(tape ? Array.from({ length: idx + 1 }, (_, j) => tape.frame(j)).filter((x) => x.phase === 'assign' || x.phase === 'update').map((x) => x.metrics!.inertia) : []);
	const sweep = $derived(framesOf(tape, 'sweep').map((x) => x.metrics!));
	// the rings step (k = 2): the border between the two centroids, the straight line of spots equally far from both,
	// clipped to the plane (Liang–Barsky)
	const border = $derived.by(() => {
		if (show !== 'rings' || cen.length !== 4) return null;
		const [ax, ay, bx, by] = cen, mx = (ax + bx) / 2, my = (ay + by) / 2, dx = ay - by, dy = bx - ax;
		if (!dx && !dy) return null;
		let t0 = -Infinity, t1 = Infinity;
		for (const [p, q] of [[-dx, mx + B], [dx, B - mx], [-dy, my + B], [dy, B - my]]) {
			if (p === 0) {
				if (q < 0) return null;
				continue;
			}
			if (p < 0) t0 = Math.max(t0, q / p);
			else t1 = Math.min(t1, q / p);
		}
		return t0 < t1 ? [mx + t0 * dx, my + t0 * dy, mx + t1 * dx, my + t1 * dy] : null;
	});

	/** a small chart's scales: x from x0 to x1 and y from lo to hi on a CW × CH box, with room for tick labels */
	const CW = 260, CH = 100, CL = 34, CR = 10, CT = 8, CB = 20;
	const scale = (x0: number, x1: number, lo: number, hi: number) => ({ X: (x: number) => CL + ((x - x0) / (x1 - x0 || 1)) * (CW - CL - CR), Y: (y: number) => CH - CB - ((y - lo) / (hi - lo || 1)) * (CH - CT - CB) });
	const path = (s: number[][], sc: ReturnType<typeof scale>) => s.map(([x, y], j) => `${j ? 'L' : 'M'}${sc.X(x).toFixed(1)},${sc.Y(y).toFixed(1)}`).join('');
	const pctOf = (v: number) => `${Math.round(v * 100)}%`;
	const formula = $derived.by(() => {
		const fm = formulaOf();
		return fm ? { tex: alignTex(fm.tex), values: texValues(fm.values) } : null;
	});
	function formulaOf(): { tex: string; values: Record<string, number> } | null {
		if (!P.props.formula || !f) return null;
		const C = Array.from({ length: classes }, (_, c) => c);
		if (f.phase === 'distances') return { tex: 'd = \\sqrt{(x - x_q)^2 + (y - y_q)^2} = \\sqrt{({{dx}})^2 + ({{dy}})^2} = {{d}}', values: { dx: m.x0 - m.qx, dy: m.y0 - m.qy, d: m.d0 } };
		if (f.phase === 'gini') return { tex: `G = 1 - \\sum p^2 = 1 - ${C.map((c) => `\\left(\\frac{ {{c${c}}} }{ {{m}} }\\right)^2`).join(' - ')} = {{g}}`, values: { ...Object.fromEntries(C.map((c) => [`c${c}`, m[`c${c}`]])), m: m.m, g: m.gini } };
		if (f.phase === 'node' && m.axis >= 0) return { tex: 'G_w = \\frac{n_L}{m} G_L + \\frac{n_R}{m} G_R = \\frac{ {{nL}} }{ {{m}} } \\cdot {{gL}} + \\frac{ {{nR}} }{ {{m}} } \\cdot {{gR}} = {{w}}', values: { nL: m.nL, nR: m.nR, m: m.m, gL: m.gL, gR: m.gR, w: m.w } };
		if (f.phase === 'update' && m.cnt) return { tex: '\\bar x = \\frac{\\sum x}{n} = \\frac{ {{sx}} }{ {{cnt}} } = {{nx}} ;\\quad \\bar y = \\frac{\\sum y}{n} = \\frac{ {{sy}} }{ {{cnt}} } = {{ny}}', values: { sx: m.sx, sy: m.sy, cnt: m.cnt, nx: m.nx, ny: m.ny } };
		return null;
	}
	const dot = (i: number) => (show === 'km' || show === 'rings' ? (assign && assign[i] >= 0 ? ccol(assign[i]) : 'color-mix(in srgb, var(--ink) 35%, var(--surface))') : col(pts!.label[i]));
	const hollow = (i: number) => (show === 'gini' || show === 'split' || show === 'grow' || show === 'depth') && !!pts?.test?.[i];
</script>

<div class="cv">
	<div class="plane">
		<!-- a group while it holds the "move the new point" button (an image's children are not interactive) -->
		<svg viewBox="-8 -8 {S + 16} {S + 16}" role={show === 'knn' && q && pts ? 'group' : 'img'} aria-label={t({ en: 'Your points on the plane', zh: '平面上你的点' })}>
			<rect x="0" y="0" width={S} height={S} class="bg" />
			{#if grid}{#each runs(grid.data) as g (g.r * R + g.c)}<rect x={g.c * cell} y={g.r * cell} width={g.len * cell} height={cell} fill={col(g.v)} opacity="0.22" shape-rendering="crispEdges" />{/each}{/if}
			{#if show === 'grow'}
				{#each nodes.filter((n) => !n.kids.length) as n (n.id)}<rect x={sx(n.box[0])} y={sy(n.box[3])} width={sx(n.box[1]) - sx(n.box[0])} height={sy(n.box[2]) - sy(n.box[3])} fill={col(n.pred)} opacity="0.2" />{/each}
				{#each nodes.filter((n) => n.kids.length) as n (n.id)}<line class="cut" class:now={n.id === m.node} x1={n.axis === 0 ? sx(n.thr) : sx(n.box[0])} x2={n.axis === 0 ? sx(n.thr) : sx(n.box[1])} y1={n.axis === 0 ? sy(n.box[3]) : sy(n.thr)} y2={n.axis === 0 ? sy(n.box[2]) : sy(n.thr)} />{/each}
			{/if}
			<line x1={sx(0)} x2={sx(0)} y1="0" y2={S} class="axis" /><line y1={sy(0)} y2={sy(0)} x1="0" x2={S} class="axis" />
			{#each [-1, 1] as v (v)}
				<line x1={sx(v)} x2={sx(v)} y1={sy(0) - 3} y2={sy(0) + 3} class="axis" /><text x={sx(v)} y={sy(0) + 12} class="ax mid">{v < 0 ? '−1' : '1'}</text>
				<line y1={sy(v)} y2={sy(v)} x1={sx(0) - 3} x2={sx(0) + 3} class="axis" /><text x={sx(0) - 5} y={sy(v) + 3.5} class="ax end">{v < 0 ? '−1' : '1'}</text>
			{/each}
			<text x={S - 4} y={sy(0) - 4} class="ax end">x</text><text x={sx(0) + 5} y="10" class="ax">y</text>
			{#if box}<rect x={sx(box[0])} y={sy(box[3])} width={sx(box[1]) - sx(box[0])} height={sy(box[2]) - sy(box[3])} class="box" />{/if}
			{#if show === 'split' && m.axis >= 0}<line class="cut now" x1={m.axis === 0 ? sx(m.thr) : sx(m.x0)} x2={m.axis === 0 ? sx(m.thr) : sx(m.x1)} y1={m.axis === 0 ? sy(m.y1) : sy(m.thr)} y2={m.axis === 0 ? sy(m.y0) : sy(m.thr)} />{/if}
			{#if show === 'knn' && q && pts}
				{#if f?.phase === 'vote'}<circle cx={sx(q[0])} cy={sy(q[1])} r={(m.r / (2 * B)) * S} class="ring" />{/if}
				{#each near as i (i)}<line x1={sx(q[0])} y1={sy(q[1])} x2={sx(X(i))} y2={sy(Y(i))} class="nn" />{/each}
			{/if}
			{#if pts}{#each Array.from({ length: pts.n }, (_, i) => i) as i (i)}<circle cx={sx(X(i))} cy={sy(Y(i))} r={pts.n > 400 ? 2 : 3.2} fill={hollow(i) ? 'var(--surface)' : dot(i)} stroke={hollow(i) ? dot(i) : 'none'} stroke-width="1.4" />{/each}{/if}
			{#if show === 'knn' && q && pts}
				{#if f?.phase === 'distances'}<circle cx={sx(X(m.i0))} cy={sy(Y(m.i0))} r="7" class="hi" /><text x={sx(X(m.i0)) + 9} y={sy(Y(m.i0)) - 7} class="lbl">{m.i0}</text>{/if}
				<path d="M{sx(q[0]) - 6},{sy(q[1]) - 6}l12,12m0,-12l-12,12" class="q" />
				<rect x="0" y="0" width={S} height={S} class="hit" role="button" tabindex="0" aria-label={t({ en: 'Move the new point: click here, or use the arrow keys', zh: '移动新点：点击这里，或用方向键' })} onclick={click} onkeydown={key} />
			{/if}
			{#if (show === 'km' || show === 'rings') && cen.length}
				{#if border}<line x1={sx(border[0])} y1={sy(border[1])} x2={sx(border[2])} y2={sy(border[3])} class="border" />{/if}
				{#if f?.phase === 'assign' && show === 'km' && !P.props.chart && pts}{#each CENTROIDS.slice(0, cen.length / 2) as name, j (name)}<line x1={sx(X(m.p))} y1={sy(Y(m.p))} x2={sx(cen[2 * j])} y2={sy(cen[2 * j + 1])} class="nn" />{/each}<circle cx={sx(X(m.p))} cy={sy(Y(m.p))} r="7" class="hi" />{/if}
				{#each CENTROIDS.slice(0, cen.length / 2) as name, j (j)}
					{#if was.length}<line x1={sx(was[2 * j])} y1={sy(was[2 * j + 1])} x2={sx(cen[2 * j])} y2={sy(cen[2 * j + 1])} class="arrow" />{/if}
					<rect x={sx(cen[2 * j]) - 8} y={sy(cen[2 * j + 1]) - 8} width="16" height="16" rx="3" fill={ccol(j)} class="cen" /><text x={sx(cen[2 * j])} y={sy(cen[2 * j + 1]) + 4} class="cl">{name}</text>
				{/each}
			{/if}
		</svg>
		{#if show === 'knn'}<p class="small muted">{t({ en: 'Click the plane (or use the arrow keys) to move the new point ×.', zh: '点击平面（或用方向键）移动新点 ×。' })}</p>{/if}
	</div>

	<div class="side">
		{#if show === 'knn' && f?.phase === 'vote'}
			<div class="bars">{#each per('v', classes) as v, c (c)}<div class="bar"><span>{t({ en: `class ${c}`, zh: `第 ${c} 类` })}</span><i style:width="{(v / Math.max(1, m.k)) * 100}%" style:background={col(c)}></i><b>{v}</b></div>{/each}</div>
		{/if}
		{#if show === 'knn' && order.length}
			<table class="small"><thead><tr><th>#</th><th>{t({ en: 'point', zh: '点' })}</th><th>{t({ en: 'distance', zh: '距离' })}</th><th>{t({ en: 'class', zh: '类别' })}</th></tr></thead>
				<tbody>{#each order.slice(0, Math.min(near.length, 8)) as i, r (i)}<tr><td>{r + 1}</td><td>{i}</td><td>{fmt3(st.d!.data[i])}</td><td><i class="sw" style:background={col(pts!.label[i])}></i>{pts!.label[i]}</td></tr>{/each}</tbody></table>
			<label class="ctl">k = {raw.k ?? 5}<input type="range" min="1" max="25" value={raw.k ?? 5} onchange={(e) => set({ k: +(e.currentTarget as HTMLInputElement).value })} /></label>
		{/if}
		{#if show === 'map'}
			{@const maps = framesOf(tape, 'map')}
			<ul class="chips">{#each maps as x (x.step)}<li class:on={x.step === f?.step}>k = {x.step}: {t({ en: `${x.metrics!.patches} patch${x.metrics!.patches === 1 ? '' : 'es'}`, zh: `${x.metrics!.patches} 块区域` })}</li>{/each}</ul>
			{#if maps.length > 1 && maps.every((x) => x.metrics!.patches === maps[0].metrics!.patches)}<p class="small muted">{t({ en: 'The same number of patches for every k: your points have no stray point inside the other class, so even k = 1 leaves no island. Paint one to see it.', zh: '每个 k 的区域数都一样：你的点里没有混进另一类地盘的点，所以连 k = 1 也不留小岛。画一个这样的点试试。' })}</p>{/if}
		{/if}
		{#if show === 'gini' || show === 'split' || show === 'grow' || show === 'depth'}
			<p class="small muted"><i class="sw"></i>{t({ en: 'training point', zh: '训练点' })} &nbsp; <i class="sw o"></i>{t({ en: 'held out (the tree never sees it)', zh: '留出点（树从来看不到它）' })}</p>
		{/if}
		{#if show === 'split' && cand.length}
			{@const lo = Math.min(...cand.map((c) => c[2]))}
			{@const sc = scale(-B, B, lo, m.gini)}
			<svg viewBox="0 0 {CW} {CH}" class="chart" role="img" aria-label={t({ en: 'Weighted Gini of every cut tried', zh: '试过的每个切分的加权基尼' })}>
				<line x1={CL} x2={CW - CR} y1={sc.Y(lo)} y2={sc.Y(lo)} class="frame" /><line x1={CL} x2={CL} y1={CT} y2={sc.Y(lo)} class="frame" />
				{#each [0, 1] as a (a)}<path d={path(cand.filter((c) => c[0] === a).map((c) => [c[1], c[2]]), sc)} class="s{a}" />{/each}
				<line x1={sc.X(m.thr)} x2={sc.X(m.thr)} y1={sc.Y(m.w)} y2={sc.Y(lo) + 4} class="mark" /><circle cx={sc.X(m.thr)} cy={sc.Y(m.w)} r="4" class="best" />
				<text x={CL - 3} y={sc.Y(m.gini) + 4} class="tick end">{fmt3(m.gini)}</text><text x={CL - 3} y={sc.Y(lo) + 3} class="tick end">{fmt3(lo)}</text>
				{#if sc.X(m.thr) - CL > 40}<text x={CL} y={CH - 5} class="tick">−1.25</text>{/if}<text x={sc.X(m.thr)} y={CH - 5} class="tick mid hl">{fmt3(m.thr)}</text>{#if CW - CR - sc.X(m.thr) > 36}<text x={CW - CR} y={CH - 5} class="tick end">1.25</text>{/if}
			</svg>
			<p class="small muted"><span class="k0">— {t({ en: 'cuts on x', zh: '在 x 上切' })}</span> <span class="k1">— {t({ en: 'cuts on y', zh: '在 y 上切' })}</span> · {t({ en: 'across: where the cut is; up: its weighted Gini', zh: '横向：切在哪里；纵向：加权基尼' })} · {t({ en: `lowest: ${question(m.axis, m.thr)}, weighted Gini ${fmt3(m.w)}`, zh: `最低：${question(m.axis, m.thr)}，加权基尼 ${fmt3(m.w)}` })}</p>
		{/if}
		{#if show === 'depth' && depths.length}
			{@const accs = depths.flatMap((x) => [x.metrics!.trainAcc, x.metrics!.testAcc]).filter((v) => v >= 0)}
			{@const lo = Math.max(0, Math.floor((Math.min(...accs) - 0.02) * 20) / 20)}
			{@const sc = scale(1, DMAX, lo, 1)}
			<svg viewBox="0 0 {CW} {CH}" class="chart" role="img" aria-label={t({ en: 'Accuracy by depth', zh: '各深度的准确率' })}>
				<line x1={CL} x2={CW - CR} y1={sc.Y(lo)} y2={sc.Y(lo)} class="frame" /><line x1={CL} x2={CL} y1={CT} y2={sc.Y(lo)} class="frame" />
				<line x1={sc.X(depth)} x2={sc.X(depth)} y1={CT} y2={sc.Y(lo)} class="mark" />
				{#each ['trainAcc', 'testAcc'] as key, s (key)}
					{@const pts2 = depths.filter((x) => x.metrics![key] >= 0).map((x) => [x.metrics!.d, x.metrics![key]])}
					<path d={path(pts2, sc)} class="s{s}" />{#each pts2 as p (p[0])}<circle cx={sc.X(p[0])} cy={sc.Y(p[1])} r="2.4" class="d{s}" />{/each}
				{/each}
				<text x={CL - 3} y={sc.Y(1) + 4} class="tick end">100%</text><text x={CL - 3} y={sc.Y(lo) + 3} class="tick end">{pctOf(lo)}</text>
				{#each depths as x (x.step)}<text x={sc.X(x.metrics!.d)} y={CH - 5} class="tick mid" class:hl={x.metrics!.d === depth}>{x.metrics!.d}</text>{/each}
			</svg>
			<p class="small muted"><span class="k0">— {t({ en: 'training', zh: '训练点' })}</span> <span class="k1">— {t({ en: 'held out', zh: '留出点' })}</span> · {t({ en: 'across: depth', zh: '横向：深度' })} · {#each depths.filter((x) => x.step === depth) as x (x.step)}{t({ en: `depth ${depth}: ${fmtPct(x.metrics!.trainAcc)} / ${x.metrics!.testAcc >= 0 ? fmtPct(x.metrics!.testAcc) : '–'}`, zh: `深度 ${depth}：${fmtPct(x.metrics!.trainAcc)} / ${x.metrics!.testAcc >= 0 ? fmtPct(x.metrics!.testAcc) : '–'}` })}{/each}</p>
			<label class="ctl">{t({ en: 'depth', zh: '深度' })} = {depth}<input type="range" min="1" max={DMAX} value={depth} onchange={(e) => set({ depth: +(e.currentTarget as HTMLInputElement).value })} /></label>
			{#if best && best.step === depth && best.metrics!.testAcc >= 0}<p class="small ok">{t({ en: 'The held-out accuracy peaks here.', zh: '留出点的准确率在这里最高。' })}</p>{/if}
		{/if}
		{#if show === 'km' && f?.phase === 'assign' && !P.props.chart}
			<p class="small">{t({ en: `Point ${m.p} to each centroid:`, zh: `第 ${m.p} 号点到每个质心：` })} {#each CENTROIDS.slice(0, cen.length / 2) as name, j (j)}<span class="dist" class:on={m.j === j}>{name} {fmt3(m[`d${j}`])}</span>{/each}</p>
		{/if}
		{#if show === 'km' && P.props.chart && curve.length}
			{@const hi = Math.max(...curve)}
			{@const lo = Math.min(...curve)}
			{@const sc = scale(1, Math.max(2, curve.length), lo, hi)}
			<svg viewBox="0 0 {CW} {CH}" class="chart" role="img" aria-label={t({ en: 'Total squared distance, step by step', zh: '每一步之后的总平方距离' })}>
				<line x1={CL} x2={CW - CR} y1={sc.Y(lo)} y2={sc.Y(lo)} class="frame" /><line x1={CL} x2={CL} y1={CT} y2={sc.Y(lo)} class="frame" />
				<path d={path(curve.map((v, j) => [j + 1, v]), sc)} class="s2" />{#each curve as v, j (j)}<circle cx={sc.X(j + 1)} cy={sc.Y(v)} r="2.4" class="d1" />{/each}
				<text x={CL - 3} y={sc.Y(hi) + 4} class="tick end">{fmt3(hi)}</text>{#if hi > lo}<text x={CL - 3} y={sc.Y(lo) + 3} class="tick end">{fmt3(lo)}</text>{/if}
				<text x={CL} y={CH - 5} class="tick">{t({ en: 'step 1', zh: '第 1 步' })}</text>{#if curve.length > 1}<text x={sc.X(curve.length)} y={CH - 5} class="tick end">{curve.length}</text>{/if}
			</svg>
			<p class="small muted">{t({ en: `Total squared distance: ${fmt3(curve[0])} → ${fmt3(curve[curve.length - 1])} after ${curve.length} step${curve.length === 1 ? '' : 's'}`, zh: `总平方距离：${fmt3(curve[0])} → ${fmt3(curve[curve.length - 1])}（${curve.length} 步之后）` })}{#if f?.phase === 'done'}{t({ en: '; no point changed in the last round.', zh: '；最后一轮没有点换组。' })}{/if}</p>
		{/if}
		{#if show === 'km'}<label class="ctl">k = {raw.kc ?? 3}<input type="range" min="1" max="6" value={raw.kc ?? 3} onchange={(e) => set({ kc: +(e.currentTarget as HTMLInputElement).value })} /></label>{/if}
		{#if show === 'rings' && sweep.length}
			<div class="bars">{#each sweep as s (s.k)}<div class="bar"><span>k = {s.k}</span><i style:width="{(s.inertia / sweep[0].inertia) * 100}%"></i><b>{fmt3(s.inertia)}</b></div>{/each}</div>
			<p class="small muted">{t({ en: 'Total squared distance for each k', zh: '每个 k 的总平方距离' })}</p>
			{#if border}<p class="small"><span class="k1">- - -</span> {t({ en: 'The border: every spot on this straight line is equally far from A and B.', zh: '边界：这条直线上每一处到 A 和 B 都一样远。' })}</p>{/if}
		{/if}
		{#if formula}<FormulaCard tex={formula.tex} values={formula.values} />{/if}
	</div>
	{#if show === 'grow' && diagram.w}
		{@const nq = nodes.filter((n) => n.kids.length).length}
		<div class="wide">
			<div class="tree"><svg viewBox="0 0 {diagram.W} {diagram.H}" style:width="100%" style:min-width="{Math.round(diagram.W * 0.7)}px" style:max-width="{diagram.W}px" role="img" aria-label={t({ en: 'The tree so far', zh: '到目前为止的树' })}>
				{#each nodes as n (n.id)}{#each n.kids as kid, s (kid)}{#if diagram.pos.has(kid)}
					<line x1={P0(n.id)[0]} y1={P0(n.id)[1]} x2={P0(kid)[0]} y2={P0(kid)[1]} class="edge" />
					{#if n.depth === 0}<text x={(P0(n.id)[0] + P0(kid)[0]) / 2 + (s ? 8 : -8)} y={(P0(n.id)[1] + P0(kid)[1]) / 2 + 2} class="qn yn" class:end={!s}>{s ? t({ en: 'no', zh: '否' }) : t({ en: 'yes', zh: '是' })}</text>{/if}
				{/if}{/each}{/each}
				{#each nodes as n (n.id)}<circle cx={P0(n.id)[0]} cy={P0(n.id)[1]} r={n.kids.length ? 4 : 5.5} fill={n.kids.length ? 'var(--surface)' : col(n.pred)} class="node" class:now={n.id === m.node} />{/each}
				{#each nodes.filter((n) => n.kids.length && n.depth < 2) as n (n.id)}<text x={P0(n.id)[0]} y={P0(n.id)[1] + (n.depth ? 15 : -8)} class="qn">{question(n.axis, n.thr)}</text>{/each}
			</svg></div>
			<p class="small muted">{t({ en: `${nq} question${nq === 1 ? '' : 's'}, ${nq + 1} lea${nq ? 'ves' : 'f'} so far (left branch: yes)`, zh: `目前 ${nq} 个问题、${nq + 1} 片叶子（左分支：是）` })}</p>
		</div>
	{/if}
</div>

<style>
	.cv { display: grid; gap: var(--s3); grid-template-columns: minmax(0, 1fr); align-items: start; }
	@media (min-width: 720px) { .cv { grid-template-columns: minmax(0, 340px) minmax(0, 1fr); } }
	.plane svg { width: 100%; max-width: 360px; display: block; touch-action: manipulation; }
	.hit { fill: transparent; cursor: crosshair; }
	.hit:focus-visible { outline: 2px solid var(--accent); }
	.side { display: grid; gap: var(--s2); min-width: 0; }
	.bg { fill: var(--surface); stroke: var(--rule-strong); }
	.axis { stroke: var(--rule); stroke-width: 0.6; }
	.box { fill: none; stroke: var(--ink); stroke-width: 1.4; stroke-dasharray: 4 3; }
	.cut { stroke: var(--ink-2); stroke-width: 1; }
	.cut.now { stroke: var(--accent); stroke-width: 2.4; }
	.ring { fill: none; stroke: var(--accent); stroke-width: 1.5; stroke-dasharray: 5 3; }
	.nn { stroke: var(--ink-2); stroke-width: 1.2; stroke-dasharray: 3 2; }
	.hi { fill: none; stroke: var(--accent); stroke-width: 2; }
	.q { stroke: var(--ink); stroke-width: 3; stroke-linecap: round; }
	.lbl, .qn { font-size: 10px; fill: var(--ink); }
	.qn { text-anchor: middle; font-size: 9.5px; fill: var(--ink); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; stroke-linejoin: round; }
	.qn.yn { text-anchor: start; fill: var(--ink-2); font-size: 9px; }
	.qn.yn.end { text-anchor: end; }
	.ax { font-size: 9px; fill: var(--ink-2); }
	.ax.mid { text-anchor: middle; }
	.ax.end { text-anchor: end; }
	.border { stroke: var(--accent); stroke-width: 2; stroke-dasharray: 6 4; }
	.cen { stroke: var(--ink); stroke-width: 1.5; }
	.cl { font-size: 10px; font-weight: 700; text-anchor: middle; fill: #fff; pointer-events: none; }
	.arrow { stroke: var(--ink); stroke-width: 1.5; stroke-dasharray: 3 2; }
	.chart { width: 100%; max-width: 320px; background: var(--surface); border: 1px solid var(--rule); border-radius: var(--r1); }
	.chart path { fill: none; stroke-width: 2; }
	.s0, .k0 { stroke: var(--ink-2); color: var(--ink-2); }
	.s1, .k1 { stroke: var(--accent); color: var(--accent); stroke-dasharray: 5 3; }
	.s2 { stroke: var(--accent); }
	.d0 { fill: var(--ink-2); }
	.d1 { fill: var(--accent); }
	.frame { stroke: var(--rule-strong); stroke-width: 1; }
	.best { fill: var(--accent); }
	.tick { font-size: 10px; fill: var(--ink-2); }
	.tick.end { text-anchor: end; }
	.tick.mid { text-anchor: middle; }
	.tick.hl { fill: var(--accent); font-weight: 700; }
	.mark { stroke: var(--ink-3); stroke-dasharray: 2 2; }
	.edge { stroke: var(--rule-strong); }
	.node { stroke: var(--ink-2); }
	.node.now { stroke: var(--accent); stroke-width: 2.5; }
	.wide { grid-column: 1 / -1; display: grid; gap: var(--s2); min-width: 0; }
	.tree { overflow-x: auto; max-width: 100%; }
	.tree svg { display: block; height: auto; }
	.bars { display: grid; gap: 4px; max-width: 320px; }
	.bar { display: grid; grid-template-columns: 4.5em minmax(0, 1fr) 3.5em; gap: 6px; align-items: center; font-size: var(--fs-sm); }
	.bar i { display: block; height: 12px; border-radius: 3px; background: var(--accent); min-width: 2px; }
	table { border-collapse: collapse; } td, th { padding: 1px 8px; text-align: right; }
	.sw { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: var(--ink-2); margin-right: 4px; }
	.sw.o { background: none; border: 1.5px solid var(--ink-2); }
	.chips { display: flex; flex-wrap: wrap; gap: 4px; list-style: none; margin: 0; padding: 0; font-size: var(--fs-sm); }
	.chips li { border: 1px solid var(--rule); border-radius: 999px; padding: 1px 8px; }
	.chips li.on, .dist.on { border-color: var(--accent); background: var(--accent-soft); }
	.dist { border: 1px solid var(--rule); border-radius: 6px; padding: 0 6px; margin-left: 4px; white-space: nowrap; }
	.ctl { display: grid; grid-template-columns: 6em minmax(0, 220px); align-items: center; gap: 8px; font-size: var(--fs-sm); }
	.ok { color: var(--real); font-weight: 600; }
	p { margin: 0; }
</style>

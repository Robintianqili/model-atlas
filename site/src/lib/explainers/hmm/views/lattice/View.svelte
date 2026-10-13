<!-- hmm 'lattice' (meta.ts lists what each props.show draws). Days across (counting from 0, her activity under each),
     the weather states down. A cell's number is α (forward) or δ (Viterbi); its color is its share of the day's column
     (the sequential ramp), drawn once both cells of the day are in. Arrows run from the day before: the current cell's
     two ways in (forward), the winning way in and the losing one dashed (Viterbi), back-pointers, and the path. The task
     step adds the tables, whose sliders change the learner's input, so every source runs again. -->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical, hexToRgb, inkOn, makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { I18n, StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { alignTex, texValues } from '../../../../atlas/ui/views/kit.parts/tex.ts';
	import Tables from '../model/Tables.svelte';
	import { ACTS, I, K, STATES, type HmmRaw } from '../../hmm.ts';
	import { latOf, type Show } from './meta.ts';

	let P: StageProps = $props();
	const uid = $props.id();
	const show = $derived(String(P.props.show ?? 'forward') as Show);
	const f = $derived(P.at.frame);
	const m = $derived(f?.metrics ?? {});
	const ph = $derived(f?.phase ?? '');
	const L = $derived(latOf(P.at.tape, P.at.index));
	const vit = $derived(show === 'viterbi' || show === 'back' || show === 'task');
	const cur = $derived(ph === 'init' || ph === 'cell' ? { t: m.t, j: m.j } : ph === 'back' ? { t: m.t, j: m.s } : ph === 'end' && vit ? { t: L.n - 1, j: m.last } : null);
	const scale = $derived((void themeState.resolved, makeScale('sequential', [0, 1])));
	const cat = $derived((void themeState.resolved, [0, 1].map((k) => categorical(k))));

	// geometry: a column per day, a row per state, then the path rows
	const LW = 58, CW = 60, GAP = 24, CH = 38, HEAD = 38, RG = 16, PR = 40;
	const X = (d: number) => LW + d * (CW + GAP);
	const Y = (j: number) => HEAD + j * (CH + RG);
	type Row = { id: string; label: I18n; path: number[] };
	const extra = $derived<Row[]>(
		show === 'guess' ? [{ id: 'guess', label: I('The guess: each day’s likelier weather for its activity', '猜测：每天按活动选更可能的天气'), path: L.guess }]
		: show === 'back' ? [{ id: 'vit', label: I('Viterbi’s path (traced back)', '维特比路径（回溯得到）'), path: L.path }, { id: 'naive', label: I('From each day’s activity alone', '只看每天的活动'), path: L.naive }]
		: show === 'task' ? [{ id: 'vit', label: I('Viterbi with your tables', '用你的表格的维特比路径'), path: L.path }, { id: 'base', label: I('Viterbi with the starting tables', '用起始表格的维特比路径'), path: L.path0 }]
		: []
	);
	const W = $derived(X(Math.max(1, L.n)) - GAP + 6);
	const H = $derived(Y(K) - RG + 8 + extra.length * PR);
	const filled = (d: number, j: number) => (L.val[d]?.[j] ?? 0) > 0;
	const share = (d: number, j: number) => {
		const c = L.val[d] ?? [];
		return c.every((v) => v > 0) ? c[j] / c.reduce((a, b) => a + b, 0) : NaN;
	};
	// the arrows: [from day, from state, to state (day + 1), kind]
	type Arrow = { d: number; i: number; j: number; k: 'faint' | 'in' | 'win' | 'lose' | 'ptr' | 'path' };
	const arrows = $derived.by(() => {
		const out: Arrow[] = [];
		const route = show === 'guess' ? L.guess : show === 'back' || show === 'task' ? L.path : [];
		for (let d = 1; d < L.n; d++)
			for (let j = 0; j < K; j++) {
				const isCur = cur && ph === 'cell' && cur.t === d && cur.j === j;
				if (isCur && !vit) for (let i = 0; i < K; i++) out.push({ d: d - 1, i, j, k: 'in' });
				else if (isCur) for (let i = 0; i < K; i++) out.push({ d: d - 1, i, j, k: i === m.from ? 'win' : 'lose' });
				else if (filled(d, j) && !vit) for (let i = 0; i < K; i++) out.push({ d: d - 1, i, j, k: 'faint' });
				else if (filled(d, j) && L.psi[d]?.[j] >= 0) out.push({ d: d - 1, i: L.psi[d][j], j, k: 'ptr' });
			}
		for (let d = 1; d < route.length; d++) if (route[d] >= 0 && route[d - 1] >= 0) out.push({ d: d - 1, i: route[d - 1], j: route[d], k: 'path' });
		return out;
	});
	const onRoute = (d: number, j: number) => (show === 'guess' ? L.guess[d] === j : (show === 'back' || show === 'task') && L.path[d] === j);
	const chip = (s: number) => (s >= 0 ? { bg: cat[s], ink: inkOn(hexToRgb(cat[s])) } : null);

	// the formula of the current frame
	const SN = ['S', 'R'];
	const v = (x: number) => String(x);
	const formula = $derived.by((): { tex: string; values: Record<string, number | string> } | null => {
		const sym = vit ? '\\delta' : '\\alpha', d = m.t, j = SN[m.j] ?? 'S';
		if (ph === 'guess' && L.n > 1) return { tex: 'P(s, o) = \\pi_{s_0}\\, b_{s_0}(o_0)\\, a_{s_0 s_1}\\, b_{s_1}(o_1) \\cdots = {{pi}} \\times {{b0}} \\times {{a1}} \\times {{b1}} \\times \\cdots = {{joint}}', values: { pi: v(m.pi), b0: v(m.b0), a1: v(m.a1), b1: v(m.b1), joint: m.joint } };
		if (ph === 'guess') return { tex: 'P(s, o) = \\pi_{s_0}\\, b_{s_0}(o_0) = {{pi}} \\times {{b0}} = {{joint}}', values: { pi: v(m.pi), b0: v(m.b0), joint: m.joint } };
		if (ph === 'init') return { tex: `${sym}_{0}(${j}) = \\pi_{${j}}\\, b_{${j}}(o_{0}) = {{pi}} \\times {{b}} = {{x}}`, values: { pi: v(m.pi), b: v(m.b), x: vit ? m.delta : m.alpha } };
		if (ph === 'cell' && !vit) return { tex: `\\alpha_{${d}}(${j}) = \\big[\\alpha_{${d - 1}}(S)\\, a_{S${j}} + \\alpha_{${d - 1}}(R)\\, a_{R${j}}\\big]\\, b_{${j}}(o_{${d}}) = ({{p0}} \\times {{a0}} + {{p1}} \\times {{a1}}) \\times {{b}} = {{x}}`, values: { p0: m.p0, a0: v(m.a0), p1: m.p1, a1: v(m.a1), b: v(m.b), x: m.alpha } };
		if (ph === 'cell') return { tex: `\\delta_{${d}}(${j}) = \\max\\big[\\delta_{${d - 1}}(S)\\, a_{S${j}},\\ \\delta_{${d - 1}}(R)\\, a_{R${j}}\\big]\\, b_{${j}}(o_{${d}}) = \\max({{c0}}, {{c1}}) \\times {{b}} = {{x}}`, values: { c0: m.c0, c1: m.c1, b: v(m.b), x: m.delta } };
		if (ph === 'end' && !vit) return { tex: `P(o_0 \\ldots o_{${L.n - 1}}) = \\alpha_{${L.n - 1}}(S) + \\alpha_{${L.n - 1}}(R) = {{x0}} + {{x1}} = {{ev}}`, values: { x0: L.val[L.n - 1]?.[0] ?? NaN, x1: L.val[L.n - 1]?.[1] ?? NaN, ev: m.evidence } };
		if (ph === 'end') return { tex: `\\max\\big[\\delta_{${L.n - 1}}(S),\\ \\delta_{${L.n - 1}}(R)\\big] = \\max({{d0}}, {{d1}}) = {{best}}`, values: { d0: m.d0, d1: m.d1, best: m.pBest } };
		return null;
	});
	const WHERE = [
		{ tex: '\\pi_{j}', text: { en: 'start: P(weather j on day 0); sₜ is the weather and oₜ her activity on day t', zh: '初始概率：P(第 0 天是天气 j)；sₜ 是第 t 天的天气，oₜ 是她第 t 天的活动' } },
		{ tex: 'a_{ij}', text: { en: 'transition: P(tomorrow j | today i); S sunny, R rainy', zh: '转移概率：P(明天 j | 今天 i)；S 晴天，R 雨天' } },
		{ tex: 'b_{j}(o_{t})', text: { en: 'emission: P(day t’s activity | weather j)', zh: '发射概率：P(第 t 天的活动 | 天气 j)' } }
	];
	const set = (r: HmmRaw) => (P.setInput ?? ((x: unknown) => P.rt.setInput(x)))(r);
	const status = $derived.by(() => {
		if (show !== 'task' || ph !== 'done') return '';
		return m.changed > 0
			? t({ en: `Viterbi changed its answer on ${m.changed} of ${m.n} days (outlined).`, zh: `维特比在 ${m.n} 天中的 ${m.changed} 天改变了答案（加了框）。` })
			: t({ en: 'Same path as with the starting tables: move a slider and watch the first row.', zh: '和起始表格的路径一样：动一动滑块，看第一行。' });
	});
</script>

<div class="lat">
	<div class="scroll" use:keyScroll role="region" aria-label={t({ en: 'The lattice (scrolls sideways when wide)', zh: '网格（太宽时可以横向滚动）' })}>
		<svg width={W} height={H} viewBox="0 0 {W} {H}" role="img" aria-label={t({ en: 'Days across, weather down', zh: '横向是天数，纵向是天气' })}>
			<defs>
				<marker id="ah-{uid}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" class="head" /></marker>
				<marker id="ahx-{uid}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" class="headx" /></marker>
			</defs>
			{#each L.obs as o, d (d)}
				<text x={X(d) + CW / 2} y="13" class="day">{t({ en: `day ${d}`, zh: `第 ${d} 天` })}</text>
				<text x={X(d) + CW / 2} y="29" class="act">{t(ACTS[o])}</text>
			{/each}
			{#each STATES as s, j (j)}<text x={LW - 6} y={Y(j) + CH / 2 + 4} class="state">{t(s)}</text>{/each}
			{#each arrows as a, k (k)}
				{@const g = { x1: X(a.d) + CW, y1: Y(a.i) + CH / 2, x2: X(a.d + 1) - 1, y2: Y(a.j) + CH / 2 }}
				<line x1={g.x1} y1={g.y1} x2={g.x2} y2={g.y2} class="ar {a.k}" marker-end={a.k === 'faint' || a.k === 'ptr' || a.k === 'lose' ? `url(#ahx-${uid})` : `url(#ah-${uid})`} />
			{/each}
			{#each L.obs as _, d (d)}
				{#each STATES as _s, j (j)}
					{@const sh = share(d, j)}
					<rect x={X(d)} y={Y(j)} width={CW} height={CH} rx="5" class="cell" class:route={onRoute(d, j)} class:cur={cur && cur.t === d && cur.j === j} style:fill={Number.isFinite(sh) ? scale.css(sh) : null} />
					{#if filled(d, j)}<text x={X(d) + CW / 2} y={Y(j) + CH / 2 + 4} class="val" style:fill={Number.isFinite(sh) ? scale.ink(sh) : null}>{fmt3(L.val[d][j])}</text>{/if}
				{/each}
			{/each}
			{#each extra as row, r (row.id)}
				{@const y0 = Y(K) - RG + 8 + r * PR}
				<text x="2" y={y0 + 9} class="rl">{t(row.label)}</text>
				{#each row.path as s, d (d)}
					{@const c = chip(s)}
					{@const other = show === 'task' ? L.path0 : L.naive}
					{@const diff = row.id === 'vit' && s >= 0 && other[d] >= 0 && s !== other[d]}
					<rect x={X(d) + 4} y={y0 + 13} width={CW - 8} height="22" rx="11" class="chip" class:diff class:none={!c} style:fill={c?.bg ?? null} />
					{#if c}<text x={X(d) + CW / 2} y={y0 + 28} class="ct" style:fill={c.ink}>{t(STATES[s])}</text>{/if}
				{/each}
			{/each}
		</svg>
	</div>
	{#if formula}<div class="fx"><FormulaCard tex={alignTex(formula.tex)} values={texValues(formula.values)} where={ph === 'cell' || ph === 'guess' ? WHERE : []} /></div>{/if}
	{#if show === 'task'}
		<p class="status" aria-live="polite">{status}</p>
		<Tables raw={P.rt.input} {set} compact />
	{/if}
</div>

<style>
	.lat { display: grid; gap: 10px; min-width: 0; }
	.scroll { overflow-x: auto; max-width: 100%; }
	.scroll:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
	svg { display: block; max-width: none; }
	text { font-size: 11px; fill: var(--ink); text-anchor: middle; }
	.day { fill: var(--muted); }
	.act { font-weight: 600; }
	.state { text-anchor: end; font-weight: 600; }
	.rl { text-anchor: start; fill: var(--muted); }
	.cell { fill: var(--surface); stroke: var(--rule-strong); stroke-width: 1; }
	.cell.route { stroke: var(--accent); stroke-width: 2.5; }
	.cell.cur { stroke: var(--accent); stroke-width: 3; stroke-dasharray: 5 3; }
	.val { font-variant-numeric: tabular-nums; }
	.ar { stroke-width: 1.5; }
	.ar.faint { stroke: var(--rule-strong); }
	.ar.ptr { stroke: var(--muted); }
	.ar.in, .ar.win, .ar.path { stroke: var(--accent); stroke-width: 2.5; }
	.ar.lose { stroke: var(--muted); stroke-dasharray: 4 3; }
	.head { fill: var(--accent); }
	.headx { fill: var(--muted); }
	.chip { stroke: none; }
	.chip.none { fill: none; stroke: var(--rule-strong); stroke-dasharray: 3 3; }
	.chip.diff { stroke: var(--accent); stroke-width: 3; }
	.ct { font-size: 11px; font-weight: 600; }
	.status { margin: 0; font-size: var(--fs-sm); }
	.fx { min-width: 0; }
</style>

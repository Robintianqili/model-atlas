<!-- gaussian-process 'plot' (meta.ts lists what each props.show draws). y up, x across, on the fixed plot of the input.
     The sliders (prior and tune) edit the learner's input (ℓ and σf²), so every source runs again. -->
<script lang="ts" module>
	// a new input re-creates the stage, so the slider being moved would lose the keyboard focus: the new stage gives it back
	let refocus: { key: string; at: number } | null = null;
	let uid = 0;
</script>

<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps, Tensor } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtNum } from '../../../../atlas/core/format.ts';
	import { alignTex, texValues } from '../../../../atlas/ui/views/kit.parts/tex.ts';
	import { G, GRID, kern, LS, NS, SF2, XHI, XLO, YHI, YLO, lsOf, rawOf, sf2Of } from '../../gp.ts';
	import { phaseOf, setting, type Show } from './meta.ts';

	let P: StageProps = $props();
	const cid = `gp-clip-${++uid}`;
	const show = $derived(String(P.props.show ?? 'prior') as Show);
	const tape = $derived(P.at.tape);
	const f = $derived(P.at.frame);
	const m = $derived(f?.metrics ?? {});
	type St = Partial<Record<'x' | 'y' | 'mean' | 'sd' | 'samples' | 'K' | 'alpha' | 'lmls', Tensor>>;
	const st = $derived((tape && tape.length ? tape.state(Math.max(0, Math.min(P.at.index, tape.length - 1))) : {}) as St);
	const arr = (v?: Tensor) => Array.from(v?.data ?? []);
	const xs = $derived(arr(st.x));
	const ys = $derived(arr(st.y));
	const mean = $derived(arr(st.mean));
	const sd = $derived(arr(st.sd));
	const pr = $derived(phaseOf(tape, 'prior'));
	const kn = $derived(phaseOf(tape, 'kernel'));
	const ev = $derived(phaseOf(tape, 'evidence'));
	const dn = $derived(phaseOf(tape, 'done'));
	const has = (...s: Show[]) => s.includes(show);
	// points the posterior has used so far (the prior uses none), and whether the band is drawn
	const used = $derived(f?.phase === 'add' ? m.used : has('prior') ? 0 : xs.length);
	const bandOn = $derived(!has('kernel', 'matrix'));
	const labels = $derived(has('kernel', 'matrix', 'points', 'first') && xs.length <= 16);

	// the plot
	let W = $state(520);
	let plotEl = $state<HTMLDivElement>();
	$effect(() => {
		if (!plotEl) return;
		const ro = new ResizeObserver((e) => {
			const w = Math.round(e[0].contentRect.width);
			if (w > 0 && w !== W) W = w;
		});
		ro.observe(plotEl);
		return () => ro.disconnect();
	});
	const H = 230, ML = 30, MR = 8, MT = 8, MB = 28, Y0 = YLO - 0.6, Y1 = YHI + 0.6;
	const X = (x: number) => ML + ((x - XLO) / (XHI - XLO)) * (W - ML - MR);
	const Y = (y: number) => MT + ((Y1 - y) / (Y1 - Y0)) * (H - MT - MB);
	const line = (v: number[]) => v.map((y, g) => `${g ? 'L' : 'M'}${X(GRID[g]).toFixed(1)},${Y(y).toFixed(1)}`).join('');
	const lower = $derived(mean.length === G ? GRID.map((_, g) => G - 1 - g).map((g) => `L${X(GRID[g]).toFixed(1)},${Y(mean[g] - 2 * sd[g]).toFixed(1)}`).join('') : '');
	const band = $derived(lower ? line(mean.map((v, g) => v + 2 * sd[g])) + lower + 'Z' : '');
	const samples = $derived(has('prior', 'samples') && st.samples ? Array.from({ length: NS }, (_, s) => line(arr(st.samples).slice(s * G, (s + 1) * G))) : []);
	const lbl = (v: number) => (v < 0 ? `−${-v}` : String(v));
	/** a vertical ±2 sd bar at x */
	const bars = $derived.by(() => {
		if (has('gap') && dn.gw !== undefined) return [{ x: dn.ga, s: dn.sdA }, { x: dn.gmid, s: dn.sdMid }, { x: dn.gb, s: dn.sdB }].map((b) => ({ ...b, m: mean[Math.round(((b.x - XLO) / (XHI - XLO)) * (G - 1))] }));
		if (has('first') && f?.phase === 'add' && m.i === 0) return [{ x: m.xi, s: m.sdAt, m: m.mAt }, { x: m.xp, s: m.sdp, m: m.mp }];
		return [];
	});
	const pairs = $derived(has('kernel') && kn.iN !== undefined ? [{ a: kn.iN, b: kn.jN, k: 'near' }, { a: kn.iF, b: kn.jF, k: 'far' }] : []);

	// the kernel as a curve of the distance (kernel step)
	const KW = 240, KH = 130;
	const dmax = $derived(Math.min(XHI - XLO, Math.max(3 * (pr.ls ?? 1), kn.dF ?? 0, 1)));
	const kx = (d: number) => 30 + (d / dmax) * (KW - 38);
	const ky = (k: number) => 8 + (1 - k / Math.max(pr.sf2 ?? 1, 1e-9)) * (KH - 34);
	const kcurve = $derived(Array.from({ length: 61 }, (_, i) => (i / 60) * dmax).map((d, i) => `${i ? 'L' : 'M'}${kx(d).toFixed(1)},${ky(kern(0, d, pr.ls, pr.sf2)).toFixed(1)}`).join(''));

	// the log marginal likelihood for every length-scale (evidence, tune)
	const lmls = $derived(arr(st.lmls));
	const lr = $derived.by(() => {
		const v = lmls.filter(Number.isFinite), lo = Math.min(...v), hi = Math.max(...v), pad = 0.08 * Math.max(1, hi - lo);
		return v.length ? [lo - pad, hi + pad] : [0, 1];
	});
	const LW = 260, LH = 150;
	const lx = (i: number) => 52 + (i / (LS.length - 1)) * (LW - 64);
	const ly = (v: number) => 16 + ((lr[1] - v) / (lr[1] - lr[0])) * (LH - 52);

	const formula = $derived.by(() => {
		const fm: { tex: string; values: Record<string, number | string> } | null =
			has('kernel') && kn.iN !== undefined ? { tex: 'k(x_a, x_b) = \\sigma_f^2 \\exp\\!\\left(-\\frac{d^2}{2\\ell^2}\\right) = {{sf2}} \\times \\exp\\!\\left(-\\frac{ {{d}}^2}{2 \\times {{ls}}^2}\\right) = {{k}}', values: { sf2: setting(kn.sf2), d: kn.dN, ls: setting(kn.ls), k: kn.kN } }
			: has('first') && m.i === 0 ? { tex: `\\begin{aligned} m(x_0) &= \\tfrac{\\sigma_f^2}{\\sigma_f^2 + \\sigma_n^2}\\, y_0 \\\\ &= \\tfrac{ {{sf2}} }{ {{kd}} } \\times ${m.yi < 0 ? '({{y0}})' : '{{y0}}'} \\\\ &= {{m0}} \\\\[6pt] m(x_p) &= \\tfrac{k(x_p, x_0)}{\\sigma_f^2 + \\sigma_n^2}\\, y_0 \\\\ &= \\tfrac{ {{kp}} }{ {{kd}} } \\times ${m.yi < 0 ? '({{y0}})' : '{{y0}}'} \\\\ &= {{mp}} \\\\[6pt] \\sigma(x_p) &= \\sqrt{\\sigma_f^2 - \\tfrac{k(x_p, x_0)^2}{\\sigma_f^2 + \\sigma_n^2}} \\\\ &= {{sdp}} \\end{aligned}`, values: { sf2: setting(pr.sf2), kd: m.kd, y0: m.yi, m0: m.mAt, kp: m.kp, mp: m.mp, sdp: m.sdp } }
			: has('matrix') ? { tex: 'm(0) = \\textstyle\\sum_i \\alpha_i\\, k(0, x_i) = {{mq}}', values: { mq: kn.mq } }
			: has('evidence', 'tune') && Number.isFinite(ev.lml) ? { tex: `\\begin{aligned} \\log p(y) &= \\underbrace{ {{fit}} }_{\\text{${t({ en: 'fit', zh: '拟合' })}}} + \\underbrace{ {{cpx}} }_{\\text{${t({ en: 'complexity', zh: '复杂度' })}}} \\\\ &\\quad + \\underbrace{ ({{cst}}) }_{\\text{${t({ en: 'constant', zh: '常数' })}}} \\\\ &= {{lml}} \\end{aligned}`, values: { fit: ev.fit, cpx: ev.cpx, cst: ev.cst, lml: ev.lml } }
			: null;
		// kit issue: alignTex loops without end on a long sum (rest.map(breakSum) passes the index as breakSum's `per`), so a
		// formula with a long sum brings its own \begin{aligned}, which alignTex returns unchanged
		return fm ? { tex: alignTex(fm.tex), values: texValues(fm.values) } : null;
	});

	// the sliders (prior, tune): they change the learner's input
	const raw = $derived((void P.at, rawOf(P.rt.input)));
	const set = (patch: Record<string, unknown>, key: string) => {
		refocus = { key, at: Date.now() };
		(P.setInput ?? ((x: unknown) => P.rt.setInput(x)))({ ...raw, ...patch });
	};
	let ctlEl = $state<HTMLDivElement>();
	$effect(() => {
		if (!ctlEl || !refocus) return;
		if (Date.now() - refocus.at < 3000) ctlEl.querySelector<HTMLInputElement>(`input[data-k="${refocus.key}"]`)?.focus();
		refocus = null;
	});
	const li = $derived(LS.indexOf(lsOf(raw)));
	const si = $derived(SF2.indexOf(sf2Of(raw)));
	const n = $derived(xs.length);
	const facts = $derived.by((): string[] => {
		if (has('gap') && dn.gw !== undefined) return [t({ en: `Widest gap: x = ${fmt3(dn.ga)} to ${fmt3(dn.gb)}, ${fmt3(dn.gwl)} length-scales wide`, zh: `最宽的空白：x = ${fmt3(dn.ga)} 到 ${fmt3(dn.gb)}，宽 ${fmt3(dn.gwl)} 个长度尺度` }), t({ en: `Band at its ends ±${fmt3(2 * dn.sdA)} and ±${fmt3(2 * dn.sdB)}, in its middle ±${fmt3(2 * dn.sdMid)}`, zh: `误差带：两端 ±${fmt3(2 * dn.sdA)} 和 ±${fmt3(2 * dn.sdB)}，中间 ±${fmt3(2 * dn.sdMid)}` }), t({ en: `With no data at all: ±${fmt3(2 * dn.sd0)}`, zh: `完全没有数据时：±${fmt3(2 * dn.sd0)}` })];
		if (has('points') && f?.phase === 'add') return [t({ en: `Point ${m.i} (counting from 0) at x = ${fmt3(m.xi)}, y = ${fmt3(m.yi)}: mean there ${fmt3(m.mAt)}, band ±${fmt3(2 * m.sdAt)}`, zh: `编号 ${m.i}（从 0 数起）的点，x = ${fmt3(m.xi)}，y = ${fmt3(m.yi)}：那里的均值 ${fmt3(m.mAt)}，误差带 ±${fmt3(2 * m.sdAt)}` }), t({ en: `Average band over the plot: ±${fmt3(m.avgBand)} (before any point ±${fmt3(2 * m.sd0)})`, zh: `整张图上误差带的平均宽度：±${fmt3(m.avgBand)}（没有点时 ±${fmt3(2 * m.sd0)}）` })];
		if (has('samples') && f?.phase === 'sample') return [t({ en: `At x = ${fmt3(m.xNear)}, next to point 0: the samples lie within ${fmt3(m.spreadNear)} of each other (band ±${fmt3(2 * m.sdNear)})`, zh: `在 x = ${fmt3(m.xNear)}（编号 0 的点旁边）：样本之间相差不超过 ${fmt3(m.spreadNear)}（误差带 ±${fmt3(2 * m.sdNear)}）` }), t({ en: `At x = ${fmt3(m.xFar)}, the place farthest from every point: ${fmt3(m.spreadFar)} (band ±${fmt3(2 * m.sdFar)})`, zh: `在 x = ${fmt3(m.xFar)}（离所有点最远的地方）：${fmt3(m.spreadFar)}（误差带 ±${fmt3(2 * m.sdFar)}）` })];
		if (has('cost')) return [dn.n === 1 ? t({ en: '1 point: K holds 1 number; factoring it is one square root', zh: '1 个点：K 只有 1 个数；分解它就是开一次平方根' }) : t({ en: `${dn.n} points: K holds ${fmtNum(dn.mem)} numbers; factoring it takes about n³/3 ≈ ${fmtNum(Math.round(dn.ops))} multiplications and subtractions`, zh: `${dn.n} 个点：K 有 ${fmtNum(dn.mem)} 个数；分解它大约需要 n³/3 ≈ ${fmtNum(Math.round(dn.ops))} 次乘法和减法` }), t({ en: `${fmtNum(dn.big)} points: ${fmtNum(dn.memBig)} numbers and about ${fmt3(dn.opsBig)} arithmetic steps`, zh: `${fmtNum(dn.big)} 个点：${fmtNum(dn.memBig)} 个数，大约 ${fmt3(dn.opsBig)} 步运算` })];
		if (has('choice')) return [t({ en: `ℓ = ${setting(dn.ls)}: the mean peaks at ${fmt3(dn.peak)}, though no point is above ${fmt3(dn.ymax)}`, zh: `ℓ = ${setting(dn.ls)}：均值最高到 ${fmt3(dn.peak)}，而没有一个点高于 ${fmt3(dn.ymax)}` }), t({ en: `Largest miss at a point ${fmt3(dn.maxRes)}; average band ±${fmt3(dn.avgBand)}`, zh: `在数据点上最大的偏差 ${fmt3(dn.maxRes)}；误差带平均 ±${fmt3(dn.avgBand)}` })];
		return [];
	});
</script>

<div class="gp">
	<div class="plot" bind:this={plotEl}>
		<svg width={W} height={H} viewBox="0 0 {W} {H}" role="img" aria-label={t({ en: 'The points, the posterior mean and its band', zh: '数据点、后验均值和误差带' })}>
			<defs><clipPath id={cid}><rect x={ML} y={MT} width={W - ML - MR} height={H - MT - MB} /></clipPath></defs>
			<rect x={ML} y={MT} width={W - ML - MR} height={H - MT - MB} class="bg" />
			{#each [-3, -2, -1, 0, 1, 2, 3] as v (v)}<line x1={ML} x2={W - MR} y1={Y(v)} y2={Y(v)} class="grid" /><text x={ML - 4} y={Y(v) + 3.5} class="ax end">{lbl(v)}</text>{/each}
			{#each [-4, -2, 0, 2, 4] as v (v)}<text x={X(v)} y={H - MB + 13} class="ax mid">{lbl(v)}</text>{/each}
			<text x={W - MR} y={H - 3} class="ax end">x</text>
			<text x="9" y={MT + (H - MT - MB) / 2} class="ax mid" transform="rotate(-90 9 {MT + (H - MT - MB) / 2})">y</text>
			<g clip-path="url(#{cid})">
				{#if has('gap') && dn.gw !== undefined}<rect x={X(dn.ga)} y={MT} width={X(dn.gb) - X(dn.ga)} height={H - MT - MB} class="span" />{/if}
				{#if bandOn && band}<path d={band} class="band" /><path d={line(mean)} class="mean" class:prior={has('prior')} />{/if}
				{#each samples as s, k (k)}<path d={s} class="sample s{k}" />{/each}
				{#each bars as b, k (k)}<line x1={X(b.x)} x2={X(b.x)} y1={Y(b.m - 2 * b.s)} y2={Y(b.m + 2 * b.s)} class="bar" /><circle cx={X(b.x)} cy={Y(b.m)} r="3" class="bar-dot" />{#if has('first') && k === 1}<text x={X(b.x) + 4} y={Y(b.m + 2 * b.s) - 3} class="lbl">xₚ</text>{/if}{/each}
				{#each pairs as p (p.k)}<line x1={X(xs[p.a])} y1={Y(ys[p.a])} x2={X(xs[p.b])} y2={Y(ys[p.b])} class="pair {p.k}" />{/each}
			</g>
			{#each xs as x, i (i)}<circle cx={X(x)} cy={Y(ys[i])} r="4.2" class="pt" class:off={i >= used} />{#if labels}<text x={X(x) + 5} y={Y(ys[i]) - 5} class="lbl">{i}</text>{/if}{/each}
		</svg>
		<p class="legend">
			<span><i class="k-pt"></i>{t({ en: 'points (numbered from 0 in the order added)', zh: '数据点（按加入顺序从 0 编号）' })}</span>
			{#if bandOn}<span><i class="k-mean"></i>{t(has('prior') ? { en: 'prior mean 0, band ±2 sd', zh: '先验均值 0，误差带 ±2 个标准差' } : { en: 'posterior mean, band ±2 sd', zh: '后验均值，误差带 ±2 个标准差' })}</span>{/if}
			{#if samples.length}<span><i class="k-s"></i>{t(has('prior') ? { en: '3 functions drawn from the prior', zh: '3 条从先验中抽取的函数' } : { en: '3 functions drawn from the posterior', zh: '3 条从后验中抽取的函数' })}</span>{/if}
			{#if used < xs.length}<span>{t({ en: 'hollow: not used yet', zh: '空心：还没用上' })}</span>{/if}
			{#if pairs.length}<span><i class="k-near"></i>{t({ en: 'closest pair', zh: '最近的一对' })}</span><span><i class="k-far"></i>{t({ en: 'farthest pair', zh: '最远的一对' })}</span>{/if}
		</p>
	</div>

	<div class="side">
		{#if has('kernel')}
			<figure class="mini">
				<svg viewBox="0 0 {KW} {KH}" role="img" aria-label={t({ en: 'The kernel as a function of the distance', zh: '核函数随距离的变化' })}>
					<line x1="30" x2={KW - 8} y1={ky(0)} y2={ky(0)} class="grid" /><line x1="30" x2="30" y1="8" y2={ky(0)} class="grid" />
					<path d={kcurve} class="mean" />
					{#if kn.iN !== undefined}{#each [{ d: kn.dN, k: kn.kN, c: 'near' }, { d: kn.dF, k: kn.kF, c: 'far' }] as q (q.c)}<circle cx={kx(q.d)} cy={ky(q.k)} r="4" class="dot-{q.c}" />{/each}{/if}
					<text x="26" y={ky(pr.sf2) + 4} class="ax end">{setting(pr.sf2)}</text><text x="26" y={ky(0) + 4} class="ax end">0</text>
					<text x={KW - 8} y={KH - 4} class="ax end">{t({ en: 'distance |x − x′|', zh: '距离 |x − x′|' })}</text>
					<text x="30" y={KH - 4} class="ax">0</text><text x={kx(dmax)} y={ky(0) + 12} class="ax mid">{fmt3(dmax)}</text>
				</svg>
				<figcaption class="note">{t({ en: `k(x, x′) for ℓ = ${setting(pr.ls)} and σf² = ${setting(pr.sf2)}.`, zh: `ℓ = ${setting(pr.ls)}、σf² = ${setting(pr.sf2)} 时的 k(x, x′)。` })}{#if kn.iN !== undefined}{t({ en: ` Closest pair (points ${kn.iN} and ${kn.jN}, ${fmt3(kn.dN)} apart): ${fmt3(kn.kN)}. Farthest (points ${kn.iF} and ${kn.jF}, ${fmt3(kn.dF)} apart): ${fmt3(kn.kF)}.`, zh: ` 最近的一对（编号 ${kn.iN} 和 ${kn.jN}，相距 ${fmt3(kn.dN)}）：${fmt3(kn.kN)}。最远的一对（编号 ${kn.iF} 和 ${kn.jF}，相距 ${fmt3(kn.dF)}）：${fmt3(kn.kF)}。` })}{/if}</figcaption>
			</figure>
		{/if}
		{#if has('matrix') && st.K}
			<div class="mat">
				<p class="note">{t({ en: `K + σn² I: ${n} × ${n}, row and column i are point i (counting from 0)`, zh: `K + σn² I：${n} × ${n}，第 i 行和第 i 列对应编号 i 的点（从 0 数起）` })}</p>
				<TensorGrid data={st.K.data} rows={n} cols={n} kind="sequential" domain={[0, kn.kd]} cell={40} label={t({ en: 'The kernel matrix', zh: '核矩阵' })} readout />
			</div>
			<table class="alpha">
				<thead><tr><th>i</th><th>x</th><th>y</th><th>α</th></tr></thead>
				<tbody>{#each xs as x, i (i)}<tr class:top={i === kn.top}><td>{i}</td><td>{fmt3(x)}</td><td>{fmt3(ys[i])}</td><td>{fmt3(arr(st.alpha)[i])}</td></tr>{/each}</tbody>
			</table>
		{/if}
		{#if has('evidence', 'tune') && lmls.length}
			<figure class="mini">
				<svg viewBox="0 0 {LW} {LH}" role="img" aria-label={t({ en: 'Log marginal likelihood for each length-scale', zh: '每个长度尺度的对数边缘似然' })}>
					{#each [lr[0], (lr[0] + lr[1]) / 2, lr[1]] as v (v)}<line x1="52" x2={LW - 12} y1={ly(v)} y2={ly(v)} class="grid" /><text x="48" y={ly(v) + 3.5} class="ax end">{fmt3(v)}</text>{/each}
					<path d={lmls.map((v, i) => `${i ? 'L' : 'M'}${lx(i).toFixed(1)},${ly(v).toFixed(1)}`).join('')} class="lml" />
					{#each lmls as v, i (i)}<circle cx={lx(i)} cy={ly(v)} r={i === ev.bestI ? 5 : 3.2} class="lml-dot" class:best={i === ev.bestI} class:cur={LS[i] === ev.ls} /><text x={lx(i)} y={LH - 18} class="ax mid">{LS[i]}</text>{/each}
					<text x={LW - 12} y={LH - 4} class="ax end">{t({ en: 'length-scale ℓ', zh: '长度尺度 ℓ' })}</text><text x="4" y="10" class="ax">log p(y)</text>
				</svg>
				<figcaption class="note">{t({ en: `log p(y) for each ℓ on the slider (σf² = ${setting(ev.sf2)}). Ringed: yours, ℓ = ${setting(ev.ls)}, ${fmt3(ev.lml)}. Filled large: the best, ℓ = ${setting(ev.bestLs)}, ${fmt3(ev.bestLml)}.`, zh: `滑块上每个 ℓ 的 log p(y)（σf² = ${setting(ev.sf2)}）。带圈的是你的 ℓ = ${setting(ev.ls)}，${fmt3(ev.lml)}；实心大点是最好的 ℓ = ${setting(ev.bestLs)}，${fmt3(ev.bestLml)}。` })}</figcaption>
			</figure>
		{/if}
		{#if formula}<div class="fx"><FormulaCard tex={formula.tex} values={formula.values} /></div>{/if}
		{#if has('matrix') && Number.isFinite(kn.topTerm)}<p class="note">{t({ en: `Largest term: point ${kn.top}, α × k(0, x) = ${fmt3(kn.topA)} × ${fmt3(kn.topK)} = ${fmt3(kn.topTerm)}.`, zh: `最大的一项来自编号 ${kn.top} 的点：α × k(0, x) = ${fmt3(kn.topA)} × ${fmt3(kn.topK)} = ${fmt3(kn.topTerm)}。` })}</p>{/if}
		{#if facts.length}<ul class="facts">{#each facts as s (s)}<li>{s}</li>{/each}</ul>{/if}
		{#if has('prior', 'tune')}
			<div class="controls" bind:this={ctlEl}>
				<label class="ctl"><span>ℓ = {setting(LS[li])}</span><input type="range" min="0" max={LS.length - 1} step="1" value={li} data-k="ls" onchange={(e) => set({ ls: LS[+e.currentTarget.value] }, 'ls')} /><small class="note">{t({ en: 'length-scale: how far one point’s influence reaches', zh: '长度尺度：一个点的影响能伸多远' })}</small></label>
				<label class="ctl"><span>σf² = {setting(SF2[si])}</span><input type="range" min="0" max={SF2.length - 1} step="1" value={si} data-k="sf2" onchange={(e) => set({ sf2: SF2[+e.currentTarget.value] }, 'sf2')} /><small class="note">{t({ en: 'signal variance: how far the function strays from 0', zh: '信号方差：函数偏离 0 的幅度' })}</small></label>
				{#if has('tune') && Number.isFinite(ev.lml)}<p class="res" class:ok={ev.drop <= 0.005}>{t({ en: `Yours: log p(y) = ${fmt3(ev.lml)}. Best on the slider: ℓ = ${setting(ev.bestLs)}, ${fmt3(ev.bestLml)}.`, zh: `你的：log p(y) = ${fmt3(ev.lml)}。滑块上最好的：ℓ = ${setting(ev.bestLs)}，${fmt3(ev.bestLml)}。` })}</p>{/if}
			</div>
		{/if}
	</div>
</div>

<style>
	.gp { display: grid; gap: var(--s3); grid-template-columns: minmax(0, 1fr); align-items: start; }
	.plot { min-width: 0; }
	.plot svg { display: block; max-width: 100%; }
	.side { display: flex; flex-wrap: wrap; gap: var(--s3); align-items: flex-start; min-width: 0; }
	.side > * { flex: 1 1 230px; min-width: 0; }
	.fx { overflow-x: auto; }
	.bg { fill: var(--surface); stroke: var(--rule-strong); }
	.grid { stroke: var(--rule); stroke-width: 0.6; }
	.ax { font-size: 10px; fill: var(--ink-2); }
	.ax.mid { text-anchor: middle; }
	.ax.end { text-anchor: end; }
	.lbl { font-size: 10px; font-weight: 700; fill: var(--ink); }
	.band { fill: var(--accent); opacity: 0.16; }
	.mean { fill: none; stroke: var(--accent); stroke-width: 2.2; }
	.mean.prior { stroke-dasharray: 6 4; }
	.sample { fill: none; stroke-width: 1.4; }
	.s0 { stroke: var(--cat-2); }
	.s1 { stroke: var(--cat-3); }
	.s2 { stroke: var(--cat-4); }
	.pt { fill: var(--cat-1); stroke: var(--surface); stroke-width: 1.4; }
	.pt.off { fill: var(--surface); stroke: var(--cat-1); stroke-width: 1.6; stroke-dasharray: 2 1.5; }
	.span { fill: var(--highlight); opacity: 0.06; }
	.bar { stroke: var(--ink); stroke-width: 2.4; stroke-linecap: round; }
	.bar-dot { fill: var(--ink); }
	.pair { stroke-width: 2; }
	.pair.near, .dot-near { stroke: var(--cat-0); fill: var(--cat-0); }
	.pair.far { stroke: var(--ink-2); stroke-dasharray: 5 4; }
	.dot-far { fill: var(--ink-2); }
	.lml { fill: none; stroke: var(--accent); stroke-width: 1.6; }
	.lml-dot { fill: var(--surface); stroke: var(--accent); stroke-width: 1.6; }
	.lml-dot.best { fill: var(--accent); }
	.lml-dot.cur { stroke: var(--ink); stroke-width: 2.6; }
	.legend { display: flex; flex-wrap: wrap; gap: 2px 14px; margin: 4px 0 0; color: var(--ink-2); font-size: var(--fs-xs); }
	.legend i { display: inline-block; width: 18px; height: 0; margin-right: 5px; vertical-align: middle; border-top: 2px solid; }
	.k-pt { width: 8px !important; height: 8px !important; border: 0 !important; border-radius: 50%; background: var(--cat-1); }
	.k-mean { border-color: var(--accent) !important; border-top-width: 3px !important; }
	.k-s { border-color: var(--cat-2) !important; }
	.k-near { border-color: var(--cat-0) !important; }
	.k-far { border-top: 2px dashed var(--ink-2) !important; }
	.mini { margin: 0; display: grid; gap: 2px; }
	.mini svg { width: 100%; max-width: 300px; background: var(--surface); border: 1px solid var(--rule); border-radius: var(--r1); }
	.mat { display: grid; gap: 4px; }
	.alpha { border-collapse: collapse; font-size: var(--fs-xs); font-variant-numeric: tabular-nums; flex: 0 1 auto !important; }
	.alpha th, .alpha td { padding: 1px 8px; text-align: right; border-bottom: 1px solid var(--rule); }
	.alpha tr.top td { font-weight: 700; }
	.controls { display: grid; gap: var(--s2); }
	.ctl { display: grid; grid-template-columns: 5.5em minmax(0, 1fr); gap: 0 8px; align-items: center; font-size: var(--fs-sm); }
	.ctl span { font-family: var(--font-mono, monospace); }
	.ctl small { grid-column: 2; }
	.ok { color: var(--live); font-weight: 600; }
	.facts { margin: 0; padding-left: 1.1em; display: grid; gap: 2px; font-size: var(--fs-sm); }
	.note { font-size: var(--fs-xs); color: var(--muted); }
	.res { font-size: var(--fs-sm); }
	p { margin: 0; }
</style>

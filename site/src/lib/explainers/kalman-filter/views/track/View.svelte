<!-- kalman-filter 'track' (meta.ts lists what each props.show draws). Position (m) up, time (s) across. In the cycle
     steps the plot zooms on one second: the guess before (k − 1), the prediction (hollow, left), the reading (orange,
     right), the new estimate (teal), each with a bar of ±2 standard deviations. The tune step's sliders edit the
     learner's input (q and r), so every source runs again. -->
<script lang="ts" module>
	// a new input re-creates the stage, so the slider being moved would lose the keyboard focus: the new stage gives it back
	let refocus: { key: string; at: number } | null = null;
</script>

<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps, Tensor } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { alignTex, texValues } from '../../../../atlas/ui/views/kit.parts/tex.ts';
	import { QS, RS, qOf, rOf, rawOf } from '../../track.ts';
	import { CYCLE, doneOf, history, setting, tick, turnStats, type Show } from './meta.ts';

	let P: StageProps = $props();
	const show = $derived(String(P.props.show ?? 'noisy') as Show);
	const tape = $derived(P.at.tape);
	const idx = $derived(P.at.index);
	const f = $derived(P.at.frame);
	const m = $derived(f?.metrics ?? {});
	const ph = $derived(f?.phase ?? '');
	const cyc = $derived(CYCLE.includes(show));
	const s0 = $derived((tape && tape.length ? tape.state(0) : {}) as { truth?: Tensor; z?: Tensor; ma?: Tensor });
	const truth = $derived(Array.from(s0.truth?.data ?? []));
	const z = $derived(Array.from(s0.z?.data ?? []));
	const ma = $derived(Array.from(s0.ma?.data ?? []));
	const n = $derived(truth.length);
	const r0 = $derived(tape && tape.length ? (tape.frame(0).metrics?.r ?? 1) : 1);
	const h = $derived(history(tape, idx));
	const done = $derived(doneOf(tape));
	// while the run step replays (or waits at its first second: reduced motion, a phone that does not autoplay), the rest
	// of the run is drawn faintly, so the stage shows the whole run at every second
	const full = $derived(show === 'run' && ph !== 'done' && tape && tape.length ? history(tape, tape.length - 1) : null);
	// the last second the stage shows: the cycle's second, the playing second, or the end
	const k = $derived(ph === 'start' ? (show === 'noisy' ? n - 1 : 0) : ph === 'done' ? n - 1 : (m.k ?? 0));
	const t0 = $derived(cyc ? Math.max(0, k - 5) : Number(P.props.from ?? 0));
	const t1 = $derived(cyc ? k + 0.5 : Math.min(n - 1, Number(P.props.to ?? n - 1)));
	const secs = (a: number, b: number) => Array.from({ length: Math.max(0, b - a + 1) }, (_, i) => a + i);
	const seen = $derived(secs(t0, Math.min(t1, cyc ? k - 1 : k)));
	const ests = $derived(seen.filter((s) => h.est[s] !== undefined));
	const rest = $derived(full ? secs(k, t1).filter((s) => full.est[s] !== undefined) : []);
	// the cycle's numbers: guess before, prediction, reading, new estimate
	const c = $derived({ p0: h.est[k - 1], sd0: h.sd[k - 1], pp: m.pp, sdp: Math.sqrt(m.Pp00), z: m.z, sdz: Math.sqrt(r0), pn: ph === 'update' ? m.p : undefined, sdn: m.sd });
	const has = (...modes: Show[]) => modes.includes(show);
	const raw = $derived((void P.at, rawOf(P.rt.input)));
	// the page's setInput (ViewSlot passes it since K1); rt.setInput where a host gives none
	const set = (patch: Record<string, unknown>, key = '') => {
		if (key) refocus = { key, at: Date.now() };
		(P.setInput ?? ((x: unknown) => P.rt.setInput(x)))({ ...raw, ...patch });
	};
	let ctlEl = $state<HTMLDivElement>();
	$effect(() => {
		if (!ctlEl || !refocus) return;
		if (Date.now() - refocus.at < 3000) ctlEl.querySelector<HTMLInputElement>(`input[data-k="${refocus.key}"]`)?.focus();
		refocus = null;
	});

	// scales
	// the plot's width from a ResizeObserver (bind:clientWidth reads it on mount and forces a layout inside the step's task)
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
	const H = 230, ML = 38, MR = 12, MT = 10, MB = 30, DX = 0.28;
	const yr = $derived.by(() => {
		const vs: number[] = [];
		for (const s of secs(t0, t1)) vs.push(truth[s]);
		for (const s of show === 'noisy' ? secs(t0, t1) : seen) vs.push(z[s]);
		for (const s of ests) vs.push(h.est[s] - 2 * h.sd[s], h.est[s] + 2 * h.sd[s]);
		if (full) for (const s of rest) vs.push(z[s], full.est[s] - 2 * full.sd[s], full.est[s] + 2 * full.sd[s]);
		if (cyc) {
			if (c.pp !== undefined) vs.push(c.pp - 2 * c.sdp, c.pp + 2 * c.sdp);
			if (c.z !== undefined) vs.push(c.z - 2 * c.sdz, c.z + 2 * c.sdz);
		}
		const ok = vs.filter(Number.isFinite), lo = Math.min(...ok), hi = Math.max(...ok), pad = 0.06 * Math.max(1, hi - lo);
		return ok.length ? [lo - pad, hi + pad] : [0, 1];
	});
	const X = (s: number) => ML + ((s - t0) / Math.max(1, t1 - t0)) * (W - ML - MR);
	const Y = (v: number) => MT + ((yr[1] - v) / (yr[1] - yr[0])) * (H - MT - MB);
	const line = (pts: [number, number][]) => pts.map(([s, v], j) => `${j ? 'L' : 'M'}${X(s).toFixed(1)},${Y(v).toFixed(1)}`).join('');
	const band = $derived(show !== 'noisy' && ests.length > 1 ? line(ests.map((s) => [s, h.est[s] + 2 * h.sd[s]])) + line([...ests].reverse().map((s) => [s, h.est[s] - 2 * h.sd[s]])).replace('M', 'L') + 'Z' : '');
	const dotR = $derived(t1 - t0 <= 30 ? 3.2 : 2.2);
	/** the readings of many seconds as one path of small circles (the faint rest of the run) */
	const dots = (ss: number[]) => ss.map((s) => `M${(X(s) - dotR).toFixed(1)},${Y(z[s]).toFixed(1)}a${dotR},${dotR} 0 1 0 ${2 * dotR},0a${dotR},${dotR} 0 1 0 ${-2 * dotR},0`).join('');

	// the whole run's errors (root mean square against the truth), as the narration quotes them, also while Play replays it
	const scores = $derived(
		Number.isFinite(done.rmseKf)
			? [
					{ key: 'z', label: t({ en: 'raw readings', zh: '原始读数' }), v: done.rmseZ },
					{ key: 'ma', label: t({ en: 'moving average', zh: '移动平均' }), v: done.rmseMa },
					{ key: 'kf', label: t({ en: 'Kalman filter', zh: '卡尔曼滤波' }), v: done.rmseKf }
				]
			: []
	);
	const turn = $derived(show === 'turn' ? turnStats(tape, idx, Number(P.props.turnAt ?? 0)) : null);

	// the covariance ellipse (2 standard deviations) before and after this frame, position across and velocity up
	const ell = $derived.by(() => {
		if (!tape || idx < 1 || !has('grow', 'shrink')) return null;
		const A = (tape.state(idx - 1) as { P?: Tensor }).P?.data, B = (tape.state(idx) as { P?: Tensor }).P?.data;
		if (!A || !B) return null;
		const sp = Math.sqrt(Math.max(A[0], B[0])), sv = Math.sqrt(Math.max(A[3], B[3])), sx = 46 / (2 * sp), sy = 38 / (2 * sv);
		const one = (C: ArrayLike<number>) => {
			const a = sx * sx * C[0], b = -sx * sy * C[1], d = sy * sy * C[3], mid = (a + d) / 2, rad = Math.sqrt(((a - d) / 2) ** 2 + b * b);
			return { rx: 2 * Math.sqrt(mid + rad), ry: 2 * Math.sqrt(Math.max(0, mid - rad)), deg: (Math.atan2(2 * b, a - d) * 90) / Math.PI };
		};
		return { a: one(A), b: one(B), dp0: 2 * Math.sqrt(A[0]), dv0: 2 * Math.sqrt(A[3]), dp: 2 * Math.sqrt(B[0]), dv: 2 * Math.sqrt(B[3]) };
	});

	/** a slot that prints a negative value in parentheses: + (−0.535) */
	const par = (name: string, v: number) => (v < 0 ? `(${name})` : name);
	const formula = $derived.by(() => {
		const fm: { tex: string; values: Record<string, number | string> } | null =
			show === 'predict' ? { tex: `p^- = p + v\\,\\Delta t = {{p}} + ${par('{{v}}', m.v)} \\times 1 = {{pp}}`, values: { p: m.p, v: m.v, pp: m.pp } }
			: show === 'grow' ? { tex: '\\begin{aligned} P^- &= P_{pp} + 2P_{pv} + P_{vv} + \\tfrac{Q}{4} \\\\ &= {{a}} + 2 \\times {{b}} \\\\ &\\quad + {{c}} + \\tfrac{ {{q}} }{4} \\\\ &= {{Pp}} \\end{aligned}', values: { a: m.P00, b: m.P01, c: m.P11, q: setting(m.q), Pp: m.Pp00 } }
			: show === 'measure' ? { tex: `y = z - p^- = {{z}} - ${par('{{pp}}', m.pp)} = {{y}}`, values: { z: m.z, pp: m.pp, y: m.y } }
			: show === 'gain' ? { tex: 'K = \\frac{P^-}{P^- + R} = \\frac{ {{Pp}} }{ {{Pp}} + {{r}} } = {{K}}', values: { Pp: m.Pp00, r: setting(m.r), K: m.K } }
			: show === 'update' ? { tex: `p = p^- + K\\,y = {{pp}} + {{K}} \\times ${par('{{y}}', m.y)} = {{p}}`, values: { pp: m.pp, K: m.K, y: m.y, p: m.p } }
			: show === 'shrink' ? { tex: 'P = (1 - K)\\,P^- = (1 - {{K}}) \\times {{Pp}} = {{P}}', values: { K: m.K, Pp: m.Pp00, P: m.P00 } }
			: null;
		return fm && f && ph !== 'start' && ph !== 'done' ? { tex: alignTex(fm.tex), values: texValues(fm.values), where: show === 'grow' ? WHERE : [] } : null;
	});
	const WHERE = [
		{ tex: 'P_{pp}', text: { en: 'variance of the position', zh: '位置的方差' } },
		{ tex: 'P_{vv}', text: { en: 'variance of the velocity', zh: '速度的方差' } },
		{ tex: 'P_{pv}', text: { en: 'covariance: how the two errors move together', zh: '协方差：两个误差一起变化的程度' } },
		{ tex: '\\tfrac{Q}{4}', text: { en: 'the unexpected push: if it changes the speed by a in one second, it moves the cart a/2; a has variance Q, so a/2 has Q/4', zh: '意外的推动：如果它在一秒内让速度变化 a，就让小车多走 a/2；a 的方差是 Q，所以 a/2 的方差是 Q/4' } }
	];
	// the numbers the narration quotes that the plot and the formula do not print
	/** as the narration prints numbers: an exact integer as itself (2, not 2.00), else 3 significant digits */
	const f3 = (v: number) => (Number.isInteger(v) ? String(v).replace('-', '−') : fmt3(v));
	const sgn = (v: number) => (v < 0 ? `(${f3(v)})` : f3(v));
	const cap = $derived.by(() => {
		if (show === 'noisy' && ph === 'start') return t({ en: `${n} readings, one per second. Their typical error (root mean square): ${f3(m.rmseZ)} m.`, zh: `${n} 个读数，每秒一个。它们的典型误差（均方根）：${f3(m.rmseZ)} 米。` });
		if (show === 'measure' && ph === 'measure') return t({ en: `Prediction p⁻ = ${f3(m.pp)} m, variance P⁻ = ${f3(m.Pp00)}, band ±${f3(2 * m.sdp)} m. Reading z = ${f3(m.z)} m, variance R = ${setting(m.r)}, band ±${f3(2 * m.sdz)} m.`, zh: `预测 p⁻ = ${f3(m.pp)} 米，方差 P⁻ = ${f3(m.Pp00)}，误差带 ±${f3(2 * m.sdp)} 米。读数 z = ${f3(m.z)} 米，方差 R = ${setting(m.r)}，误差带 ±${f3(2 * m.sdz)} 米。` });
		if (show === 'update' && ph === 'update') return t({ en: `Velocity: v = ${f3(m.vp)} + ${f3(m.Kv)} × ${sgn(m.y)} = ${f3(m.v)} m/s. True position ${f3(m.truth)} m: the estimate is off by ${f3(Math.abs(m.err))} m, the reading by ${f3(Math.abs(m.errZ))} m.`, zh: `速度：v = ${f3(m.vp)} + ${f3(m.Kv)} × ${sgn(m.y)} = ${f3(m.v)} 米/秒。真实位置 ${f3(m.truth)} 米：估计偏差 ${f3(Math.abs(m.err))} 米，读数偏差 ${f3(Math.abs(m.errZ))} 米。` });
		return '';
	});
	const qi = $derived(QS.indexOf(qOf(raw)));
	const ri = $derived(RS.indexOf(rOf(raw)));
	const lbl = (v: number) => (v < 0 ? `−${-v}` : String(v));
</script>

<div class="kf">
	<div class="plot" bind:this={plotEl}>
		<svg width={W} height={H} viewBox="0 0 {W} {H}" role="img" aria-label={t({ en: 'Position over time', zh: '位置随时间的变化' })}>
			<rect x={ML} y={MT} width={W - ML - MR} height={H - MT - MB} class="bg" />
			{#each tick(yr[0], yr[1], 4) as v (v)}<line x1={ML} x2={W - MR} y1={Y(v)} y2={Y(v)} class="grid" /><text x={ML - 4} y={Y(v) + 3.5} class="ax end">{lbl(v)}</text>{/each}
			{#each tick(t0, t1, cyc ? 8 : 6).filter((s) => Number.isInteger(s)) as s (s)}<text x={X(s)} y={H - MB + 13} class="ax mid">{s}</text>{/each}
			<text x={W - MR} y={H - 3} class="ax end">{t({ en: 'time (s)', zh: '时间（秒）' })}</text>
			<text x="9" y={MT + (H - MT - MB) / 2} class="ax mid" transform="rotate(-90 9 {MT + (H - MT - MB) / 2})">{t({ en: 'position (m)', zh: '位置（米）' })}</text>
			{#if band}<path d={band} class="band" />{/if}
			{#if full && rest.length > 1}
				<g class="rest" aria-hidden="true">
					<path d={line(rest.map((s) => [s, full.est[s] + 2 * full.sd[s]])) + line([...rest].reverse().map((s) => [s, full.est[s] - 2 * full.sd[s]])).replace('M', 'L') + 'Z'} class="band" />
					<path d={line(rest.map((s) => [s, ma[s]]))} class="ma" />
					<path d={dots(rest.slice(1))} class="z" />
					<path d={line(rest.map((s) => [s, full.est[s]]))} class="est" />
				</g>
			{/if}
			<path d={line(secs(t0, cyc ? k : t1).map((s) => [s, truth[s]]))} class="truth" />
			{#if has('run', 'tune')}<path d={line(seen.map((s) => [s, ma[s]]))} class="ma" />{/if}
			{#if turn}<line x1={X(Number(P.props.turnAt))} x2={X(Number(P.props.turnAt))} y1={MT} y2={H - MB} class="mark" />{/if}
			{#each show === 'noisy' ? secs(t0, t1) : seen as s (s)}<circle cx={X(s)} cy={Y(z[s])} r={dotR} class="z" />{/each}
			{#if show !== 'noisy' && ests.length}<path d={line(ests.map((s) => [s, h.est[s]]))} class="est" />{/if}
			{#if cyc && c.pp !== undefined}
				{#if has('predict', 'grow') && c.p0 !== undefined}
					<line x1={X(k - 1)} x2={X(k - 1)} y1={Y(c.p0 - 2 * c.sd0)} y2={Y(c.p0 + 2 * c.sd0)} class="bar est-c" /><circle cx={X(k - 1)} cy={Y(c.p0)} r="5" class="dot-est" />
					<line x1={X(k - 1)} y1={Y(c.p0)} x2={X(k - DX) - 6} y2={Y(c.pp)} class="arrow" />
				{/if}
				{#if !has('predict')}<line x1={X(k - DX)} x2={X(k - DX)} y1={Y(c.pp - 2 * c.sdp)} y2={Y(c.pp + 2 * c.sdp)} class="bar pred-c" />{/if}
				<circle cx={X(k - DX)} cy={Y(c.pp)} r="5.5" class="dot-pred" />
				{#if c.z !== undefined && !has('predict', 'grow')}
					<line x1={X(k + DX)} x2={X(k + DX)} y1={Y(c.z - 2 * c.sdz)} y2={Y(c.z + 2 * c.sdz)} class="bar z-c" /><circle cx={X(k + DX)} cy={Y(c.z)} r="5" class="dot-z" />
					{#if has('measure', 'gain')}<line x1={X(k)} x2={X(k)} y1={Y(c.pp)} y2={Y(c.z)} class="gap" /><text x={X(k) + 3} y={(Y(c.pp) + Y(c.z)) / 2 + 3} class="lbl">y</text>{/if}
				{/if}
				{#if c.pn !== undefined}<line x1={X(k)} x2={X(k)} y1={Y(c.pn - 2 * c.sdn)} y2={Y(c.pn + 2 * c.sdn)} class="bar est-c thick" /><circle cx={X(k)} cy={Y(c.pn)} r="5.5" class="dot-est" />{/if}
				{#if has('update', 'shrink') && truth[k] !== undefined}<line x1={X(k) - 10} x2={X(k) + 10} y1={Y(truth[k])} y2={Y(truth[k])} class="truth-k" />{/if}
			{/if}
		</svg>
		<p class="legend">
			<span><i class="k-truth"></i>{t({ en: 'true position (hidden from the filter)', zh: '真实位置（滤波器看不到）' })}</span>
			<span><i class="k-z"></i>{t({ en: 'readings', zh: '读数' })}</span>
			{#if show !== 'noisy'}<span><i class="k-est"></i>{t({ en: 'estimate, band ±2 sd', zh: '估计，误差带 ±2 个标准差' })}</span>{/if}
			{#if cyc}<span><i class="k-pred"></i>{t({ en: 'prediction', zh: '预测' })}</span>{/if}
			{#if has('run', 'tune')}<span><i class="k-ma"></i>{t({ en: `moving average of the last ${tape?.frame(0).metrics?.w ?? 5} readings`, zh: `最近 ${tape?.frame(0).metrics?.w ?? 5} 个读数的移动平均` })}</span>{/if}
			{#if full}<span class="faint">{t({ en: 'faint: the seconds still to come', zh: '淡色：还没播放到的秒' })}</span>{/if}
		</p>
		{#if cap}<p class="cap">{cap}</p>{/if}
	</div>

	<div class="side">
		{#if ell}
			<figure class="ell">
				<svg viewBox="-80 -58 160 124" role="img" aria-label={t({ en: 'The covariance ellipse, before and after', zh: '协方差椭圆：之前和之后' })}>
					<line x1="-74" x2="74" y1="0" y2="0" class="grid" /><line y1="-50" y2="50" x1="0" x2="0" class="grid" />
					<ellipse rx={ell.a.rx} ry={ell.a.ry} transform="rotate({ell.a.deg})" class="e-a" />
					<ellipse rx={ell.b.rx} ry={ell.b.ry} transform="rotate({ell.b.deg})" class="e-b" />
					<text x="76" y="62" class="ax end">{t({ en: 'position →', zh: '位置 →' })}</text><text x="-76" y="-48" class="ax">{t({ en: '↑ velocity', zh: '↑ 速度' })}</text>
				</svg>
				<figcaption class="note">{t({ en: `The doubt about position and velocity together (2 standard deviations). Dashed, before: ±${fmt3(ell.dp0)} m and ±${fmt3(ell.dv0)} m/s. Solid, now: ±${fmt3(ell.dp)} m and ±${fmt3(ell.dv)} m/s.`, zh: `位置和速度合在一起的不确定程度（2 个标准差）。虚线是之前：±${fmt3(ell.dp0)} 米、±${fmt3(ell.dv0)} 米/秒。实线是现在：±${fmt3(ell.dp)} 米、±${fmt3(ell.dv)} 米/秒。` })}</figcaption>
			</figure>
		{/if}
		{#if has('gain', 'update') && Number.isFinite(m.K)}
			<div class="gauge" aria-hidden="true">
				<b>K = {fmt3(m.K)}</b>
				<div class="track"><i style:left="{m.K * 100}%"></i></div>
				<div class="ends"><span>{t({ en: '0: keep the prediction', zh: '0：保持预测' })}</span><span>{t({ en: '1: take the reading', zh: '1：采用读数' })}</span></div>
			</div>
		{/if}
		{#if formula}<div class="fx"><FormulaCard tex={formula.tex} values={formula.values} where={formula.where} /></div>{/if}
		{#if has('run', 'tune') && scores.length}
			<div class="bars" role="group" aria-label={t({ en: 'Typical error so far', zh: '到目前为止的典型误差' })}>
				<p class="note">{t({ en: `Typical error over all ${n} seconds (root mean square), in metres, with Q = ${setting(done.q)} and R = ${setting(done.r)}`, zh: `全部 ${n} 秒的典型误差（均方根），单位：米；Q = ${setting(done.q)}，R = ${setting(done.r)}` })}</p>
				{#each scores as s (s.key)}<div class="bar-row"><span>{s.label}</span><i class="b-{s.key}" style:width="{Math.min(100, (s.v / Math.max(...scores.map((x) => x.v), 1e-9)) * 100)}%"></i><b>{fmt3(s.v)}</b></div>{/each}
				{#if show === 'run' && Number.isFinite(done.bestRmse)}<p class="note">{t({ en: `The best pair of settings on the sliders (step 9) gets ${fmt3(done.bestRmse)} m.`, zh: `滑块上最好的一组设置（第 9 步）能做到 ${fmt3(done.bestRmse)} 米。` })}</p>{/if}
			</div>
		{/if}
		{#if turn}
			<ul class="facts">
				<li>{t({ en: `Turn at second ${P.props.turnAt}: largest error from then on ${fmt3(turn.worst)} m`, zh: `第 ${P.props.turnAt} 秒掉头：此后最大误差 ${fmt3(turn.worst)} 米` })}</li>
				<li>{t({ en: `After the turn, outside the band: ${turn.out} of ${turn.after} seconds`, zh: `掉头后落在误差带外：${turn.after} 秒中有 ${turn.out} 秒` })}</li>
				<li>{t({ en: `Typical error before the turn ${fmt3(turn.pre)} m, after it ${fmt3(turn.post)} m`, zh: `典型误差：掉头前 ${fmt3(turn.pre)} 米，掉头后 ${fmt3(turn.post)} 米` })}</li>
			</ul>
		{/if}
		{#if show === 'tune'}
			<div class="controls" bind:this={ctlEl}>
			<label class="ctl"><span>Q = {setting(QS[qi])}</span><input type="range" min="0" max={QS.length - 1} step="1" value={qi} data-k="q" onchange={(e) => set({ q: QS[+e.currentTarget.value] }, 'q')} /><small class="muted">{t({ en: 'how much the speed may change unexpectedly each second', zh: '速度每秒可能意外变化多少' })}</small></label>
			<label class="ctl"><span>R = {setting(RS[ri])}</span><input type="range" min="0" max={RS.length - 1} step="1" value={ri} data-k="r" onchange={(e) => set({ r: RS[+e.currentTarget.value] }, 'r')} /><small class="muted">{t({ en: 'how noisy you think the sensor is (its variance)', zh: '你认为传感器的噪声有多大（它的方差）' })}</small></label>
			{#if done.bestRmse}<p class="res" class:ok={done.rmseKf <= 1.02 * done.bestRmse}>{t({ en: `Your filter: ${fmt3(done.rmseKf)} m. Best pair on these sliders: ${fmt3(done.bestRmse)} m.`, zh: `你的滤波器：${fmt3(done.rmseKf)} 米。这两个滑块上最好的组合：${fmt3(done.bestRmse)} 米。` })}</p>{/if}
			</div>
		{/if}
	</div>
</div>

<style>
	.kf { display: grid; gap: var(--s3); grid-template-columns: minmax(0, 1fr); align-items: start; }
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
	.lbl { font-size: 11px; font-weight: 700; fill: var(--ink); }
	.band { fill: var(--accent); opacity: 0.16; }
	.truth { fill: none; stroke: var(--ink-2); stroke-width: 1.4; stroke-dasharray: 5 4; }
	.ma { fill: none; stroke: var(--cat-0); stroke-width: 1.8; }
	.est { fill: none; stroke: var(--accent); stroke-width: 2.2; }
	.z { fill: var(--cat-1); opacity: 0.85; }
	.bar { stroke-width: 2; stroke-linecap: round; }
	.bar.thick { stroke-width: 3.5; }
	.est-c { stroke: var(--accent); }
	.pred-c { stroke: var(--accent); stroke-dasharray: 3 2; }
	.z-c { stroke: var(--cat-1); }
	.dot-est { fill: var(--accent); stroke: var(--surface); stroke-width: 1.5; }
	.dot-pred { fill: var(--surface); stroke: var(--accent); stroke-width: 2.4; }
	.dot-z { fill: var(--cat-1); stroke: var(--surface); stroke-width: 1.5; }
	.arrow { stroke: var(--accent); stroke-width: 1.4; stroke-dasharray: 4 3; }
	.gap { stroke: var(--ink); stroke-width: 1.2; }
	.mark { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 3; }
	.truth-k { stroke: var(--ink); stroke-width: 2.4; stroke-linecap: round; }
	.rest { opacity: 0.3; }
	.faint { opacity: 0.75; font-style: italic; }
	.cap { margin-top: 4px; font-size: var(--fs-sm); color: var(--ink-2); font-variant-numeric: tabular-nums; }
	.legend { display: flex; flex-wrap: wrap; gap: 2px 14px; margin: 4px 0 0; color: var(--ink-2); font-size: var(--fs-xs); }
	.legend i { display: inline-block; width: 18px; height: 0; margin-right: 5px; vertical-align: middle; border-top: 2px solid; }
	.k-truth { border-top: 2px dashed var(--ink-2) !important; }
	.k-z { width: 8px !important; height: 8px !important; border: 0 !important; border-radius: 50%; background: var(--cat-1); }
	.k-est { border-color: var(--accent) !important; border-top-width: 3px !important; }
	.k-pred { width: 9px !important; height: 9px !important; border: 2px solid var(--accent) !important; border-radius: 50%; }
	.k-ma { border-color: var(--cat-0) !important; }
	.ell { margin: 0; display: grid; gap: 2px; }
	.ell svg { width: 100%; max-width: 260px; background: var(--surface); border: 1px solid var(--rule); border-radius: var(--r1); }
	.e-a { fill: none; stroke: var(--ink-2); stroke-width: 1.3; stroke-dasharray: 4 3; }
	.e-b { fill: var(--accent); fill-opacity: 0.18; stroke: var(--accent); stroke-width: 2; }
	.gauge { display: grid; gap: 4px; }
	.gauge .track { position: relative; height: 10px; border-radius: 5px; background: linear-gradient(90deg, var(--accent), var(--cat-1)); opacity: 0.85; }
	.gauge .track i { position: absolute; top: -4px; width: 4px; height: 18px; margin-left: -2px; background: var(--ink); border-radius: 2px; }
	.gauge .ends { display: flex; justify-content: space-between; gap: 6px; color: var(--ink-2); }
	.gauge b { color: var(--ink); }
	.bars { display: grid; gap: 4px; }
	.bar-row { display: grid; grid-template-columns: 8.5em minmax(0, 1fr) 3.2em; gap: 6px; align-items: center; font-size: var(--fs-sm); }
	.bar-row i { display: block; height: 12px; border-radius: 3px; min-width: 2px; }
	.bar-row b { text-align: right; font-variant-numeric: tabular-nums; }
	.b-z { background: var(--cat-1); }
	.b-ma { background: var(--cat-0); }
	.b-kf { background: var(--accent); }
	.controls { display: grid; gap: var(--s2); }
	.ctl { display: grid; grid-template-columns: 5.5em minmax(0, 1fr); gap: 0 8px; align-items: center; font-size: var(--fs-sm); }
	.ctl span { font-family: var(--font-mono, monospace); }
	.ctl small { grid-column: 2; }
	.ok { color: var(--live); font-weight: 600; }
	.facts { margin: 0; padding-left: 1.1em; display: grid; gap: 2px; font-size: var(--fs-sm); }
	.note { font-size: var(--fs-xs); color: var(--muted); }
	.res, .gauge .ends { font-size: var(--fs-sm); }
	p { margin: 0; }
</style>

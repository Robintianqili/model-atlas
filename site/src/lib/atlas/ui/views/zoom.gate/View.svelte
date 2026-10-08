<!--
  zoom.gate: one step of a gated recurrent unit (lstm; local gru). New, as the design asks: the gates as bars (forget f,
  input i, output o between 0 and 1; candidate g between −1 and 1) with their pre-activations, the cell-state line
  c_prev → ×f → + i·g → c → tanh → ×o → h with live numbers, and the unit's c and h over every time step (the step
  marked; click or arrow keys pick another step). Then the formula and the stored-value check.
  K1: props.gate ('f' | 'i' | 'g' | 'o' | 'c' | 'h') puts one gate (or the cell / output box) in focus and dims the rest,
  so a lesson step about the forget gate does not show the same stage as the next one; shares print as percentages the
  way the narration's |pct prints them ("<0.1%", not "4.36e−4%").
-->
<script lang="ts">
	import Num from '../kit.parts/Num.svelte';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3, fmtPct } from '../../../core/format.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../types/view.ts';
	import UnitHead from '../kit.parts/UnitHead.svelte';
	import OpFormula from '../kit.parts/OpFormula.svelte';
	import Notice from '../kit.parts/Notice.svelte';
	import { layerLabel, parentId, resolveUnit, tokenIds, tokenTexts } from '../kit.parts/stage.ts';

	let P: StageProps = $props();

	let pick = $derived.by<number[] | null>(() => {
		void P.at;
		void P.props.layer;
		return null;
	});
	const R = $derived(resolveUnit(P, pick));
	const e = $derived(R.e);
	const id = $derived(e?.layer ?? R.id ?? '');
	const base = $derived(parentId(id));
	const out = $derived(R.trace ? R.trace.t[base] : null);
	const Tn = $derived(out ? out.shape[0] : 0);
	const Hn = $derived(out ? out.shape[1] : 0);
	const tt = $derived(e ? (e.values.t ?? e.unit[0]) : 0);
	const j = $derived(e ? (e.values.j ?? e.unit[1]) : 0);
	const tokens = $derived(Tn ? tokenTexts(P, Tn, tokenIds(R.trace, R.model)) : []);
	const v = $derived(e?.values ?? {});
	const gates = $derived([
		{ k: 'f', name: { en: 'forget gate', zh: '遗忘门' }, z: v.zf, val: v.f, seq: true },
		{ k: 'i', name: { en: 'input gate', zh: '输入门' }, z: v.zi, val: v.i, seq: true },
		{ k: 'g', name: { en: 'candidate', zh: '候选值' }, z: v.zg, val: v.g, seq: false },
		{ k: 'o', name: { en: 'output gate', zh: '输出门' }, z: v.zo, val: v.o, seq: true }
	]);
	/** the gate (or box) this step is about (K1) */
	const focusGate = $derived(typeof P.props.gate === 'string' ? (P.props.gate as string) : null);
	const div = $derived(makeScale('diverging', [-1, 1], themeState.resolved));
	const seq = $derived(makeScale('sequential', [0, 1], themeState.resolved));
	// c and h of unit j over time (sub-steps id#c and the layer output h)
	const series = $derived.by(() => {
		if (!R.trace || !out) return null;
		const cT = R.trace.t[`${base}#c`];
		const h = Float64Array.from({ length: Tn }, (_, s) => out.data[s * Hn + j]);
		const c = cT ? Float64Array.from({ length: Tn }, (_, s) => cT.data[s * Hn + j]) : null;
		let m = 1;
		for (const a of [h, c]) if (a) for (const x of a) if (Number.isFinite(x)) m = Math.max(m, Math.abs(x));
		return { h, c, m };
	});
	const W = 520;
	const Hh = 150;
	const pad = { l: 34, r: 10, t: 10, b: 24 };
	const sx = (s: number) => pad.l + (Tn > 1 ? (s / (Tn - 1)) * (W - pad.l - pad.r) : (W - pad.l - pad.r) / 2);
	const sy = (y: number) => pad.t + (1 - (y + (series?.m ?? 1)) / (2 * (series?.m ?? 1))) * (Hh - pad.t - pad.b);
	const line = (a: ArrayLike<number>) => Array.from(a, (y, s) => `${sx(s).toFixed(1)},${sy(y).toFixed(1)}`).join(' ');
	function setT(s: number) {
		pick = [Math.max(0, Math.min(Tn - 1, s)), j];
	}
	function onpointer(ev: PointerEvent) {
		if (Tn < 1) return;
		const r = (ev.currentTarget as SVGSVGElement).getBoundingClientRect();
		const x = ((ev.clientX - r.left) / r.width) * W;
		setT(Math.round(((x - pad.l) / (W - pad.l - pad.r)) * (Tn - 1)));
	}
	function onkeydown(ev: KeyboardEvent) {
		if (ev.key === 'ArrowRight') setT(tt + 1);
		else if (ev.key === 'ArrowLeft') setT(tt - 1);
		else if (ev.key === 'Home') setT(0);
		else if (ev.key === 'End') setT(Tn - 1);
		else return;
		ev.preventDefault();
	}
	$effect(() => {
		if (P.autoplay && e) P.onend();
	});
</script>

{#if !e}
	<Notice error={R.error} />
{:else}
	<div class="gt">
		<UnitHead
			layer={layerLabel(R.model, id)}
			op={R.op?.def.label}
			where={{ en: `time step ${tt}${tokens[tt] && !/^\d+$/.test(tokens[tt]) ? ` “${tokens[tt]}”` : ''}, unit ${j} (counting from 0)`, zh: `时间步 ${tt}${tokens[tt] && !/^\d+$/.test(tokens[tt]) ? `“${tokens[tt]}”` : ''}、单元 ${j}（从 0 数起）` }}
			value={e.result}
		/>
		<div class="gates" role="group" aria-label={t({ en: 'The four gates of this unit', zh: '这个单元的四个门' })}>
			{#each gates as g (g.k)}
				<div class="g" class:focus={focusGate === g.k} class:dim={!!focusGate && focusGate !== g.k && focusGate !== 'c' && focusGate !== 'h'}>
					<span class="gk num">{g.k}</span>
					<span class="gn">{t(g.name)}</span>
					<span class="track" class:sym={!g.seq}>
						<span
							class="b"
							class:neg={(g.val ?? 0) < 0}
							style="--w: {g.seq ? Math.max(0, Math.min(1, g.val ?? 0)) * 100 : Math.min(1, Math.abs(g.val ?? 0)) * 50}%; background: {g.seq ? seq.css(g.val ?? 0) : div.css(g.val ?? 0)}"
						></span>
					</span>
					<span class="gv num">{g.seq ? 'σ' : 'tanh'}({fmt3(g.z)}) = <strong>{fmt3(g.val)}</strong></span>
				</div>
			{/each}
		</div>
		<div class="flow" aria-label={t({ en: 'How the cell state and the output are computed', zh: '细胞状态和输出是怎样算出来的' })}>
			<span class="box"><span class="lbl">c<sub>t−1</sub></span><Num value={v.cPrev} signed /></span>
			<span class="op">× f</span>
			<span class="box"><span class="lbl">f·c<sub>t−1</sub></span><Num value={(v.f ?? 0) * (v.cPrev ?? 0)} signed /></span>
			<span class="op">+ i·g</span>
			<span class="box hl" class:focus={focusGate === 'c'}><span class="lbl">c<sub>t</sub></span><Num value={v.c} signed /></span>
			<span class="op">tanh, × o</span>
			<span class="box hl" class:focus={focusGate === 'h'}><span class="lbl">h<sub>t</sub></span><Num value={v.h} signed /></span>
		</div>
		<p class="note">
			{t({
				en: `The forget gate keeps ${fmtPct(v.f ?? 0)} of the old cell value ${fmt3(v.cPrev)}; the input gate lets in ${fmtPct(v.i ?? 0)} of the candidate ${fmt3(v.g)}.`,
				zh: `遗忘门保留旧细胞值 ${fmt3(v.cPrev)} 的 ${fmtPct(v.f ?? 0)}；输入门放入候选值 ${fmt3(v.g)} 的 ${fmtPct(v.i ?? 0)}。`
			})}
		</p>
		{#if series && Tn > 1}
			<figure class="ts">
				<figcaption>{t({ en: `Unit ${j} over all ${Tn} time steps (0 to ${Tn - 1}): cell c (solid) and output h (dashed). Click or use the arrow keys to pick a step.`, zh: `单元 ${j} 在全部 ${Tn} 个时间步（0 到 ${Tn - 1}）中的变化：细胞值 c（实线）和输出 h（虚线）。点击或用方向键选一步。` })}</figcaption>
				<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions (arrow keys pick a time step) -->
				<svg
					viewBox="0 0 {W} {Hh}"
					role="application"
					aria-roledescription={t({ en: 'time series', zh: '时间序列' })}
					aria-label={t({ en: `cell and output of unit ${j}, time step ${tt} (counting from 0, ${Tn} steps)`, zh: `单元 ${j} 的细胞值和输出，时间步 ${tt}（从 0 数起，共 ${Tn} 步）` })}
					tabindex="0"
					onpointerdown={onpointer}
					{onkeydown}
				>
					<line x1={pad.l} x2={W - pad.r} y1={sy(0)} y2={sy(0)} class="axis" />
					<text x={pad.l - 4} y={sy(series.m) + 4} class="tick" text-anchor="end">{fmt3(series.m)}</text>
					<text x={pad.l - 4} y={sy(-series.m) + 4} class="tick" text-anchor="end">{fmt3(-series.m)}</text>
					<line x1={sx(tt)} x2={sx(tt)} y1={pad.t} y2={Hh - pad.b} class="now" />
					{#if series.c}<polyline points={line(series.c)} class="c" />{/if}
					<polyline points={line(series.h)} class="h" />
					{#each Array.from({ length: Tn }, (_, s) => s) as s (s)}
						<text x={sx(s)} y={Hh - 6} class="tick" text-anchor="middle">{Tn <= 24 || s % Math.ceil(Tn / 12) === 0 ? s : ''}</text>
					{/each}
					{#if series.c}<circle cx={sx(tt)} cy={sy(series.c[tt])} r="4.5" class="dot" />{/if}
				</svg>
			</figure>
		{/if}
		{#if Hn > 1 && Hn <= 32}
			<div class="segmented units" role="group" aria-label={t({ en: 'Unit', zh: '单元' })}>
				{#each Array.from({ length: Hn }, (_, k) => k) as k (k)}
					<button type="button" aria-pressed={k === j} onclick={() => (pick = [tt, k])}>{k}</button>
				{/each}
			</div>
		{/if}
		<OpFormula tex={R.op?.def.formula} {e} caption={{ en: `${layerLabel(R.model, id).en}, time step ${tt}, unit ${j}`, zh: `${layerLabel(R.model, id).zh ?? id}，时间步 ${tt}、单元 ${j}` }} provenance={P.provenance} />
	</div>
{/if}

<style>
	.gt {
		display: grid;
		gap: 12px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.gates {
		display: grid;
		gap: 4px;
		max-width: 620px;
	}
	.g {
		display: grid;
		grid-template-columns: 1.2em 7.5em minmax(60px, 1fr) auto;
		gap: 8px;
		align-items: center;
		font-size: var(--fs-sm);
	}
	.gk {
		font-weight: 700;
	}
	.g.focus {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
		border-radius: 3px;
	}
	.g.dim {
		opacity: 0.4;
	}
	.box.focus {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
	.gn {
		color: var(--ink-2);
		font-size: var(--fs-xs);
	}
	.track {
		position: relative;
		height: 12px;
		background: var(--surface-2);
		border-radius: 2px;
	}
	.track.sym {
		background: linear-gradient(90deg, var(--surface-2) calc(50% - 0.5px), var(--rule-strong) calc(50% - 0.5px), var(--rule-strong) calc(50% + 0.5px), var(--surface-2) calc(50% + 0.5px));
	}
	.b {
		position: absolute;
		top: 1px;
		bottom: 1px;
		left: 0;
		width: var(--w);
		border-radius: 2px;
	}
	.sym .b {
		left: 50%;
	}
	.sym .b.neg {
		left: auto;
		right: 50%;
	}
	.gv {
		font-size: var(--fs-xs);
		white-space: nowrap;
	}
	@media (max-width: 520px) {
		.g {
			grid-template-columns: 1.2em minmax(0, 1fr) auto;
		}
		.gn {
			display: none;
		}
	}
	.flow {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px 8px;
	}
	.box {
		display: inline-grid;
		gap: 0;
		justify-items: center;
		padding: 4px 8px;
		border: 1px solid var(--rule-strong);
		border-radius: var(--r1);
		background: var(--surface);
		font-size: var(--fs-sm);
	}
	.box.hl {
		border-color: var(--accent);
		box-shadow: 0 0 0 1px var(--accent);
	}
	.lbl {
		font-size: 11px;
		color: var(--muted);
		font-family: var(--font-mono);
	}
	.op {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.note {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		max-width: var(--measure);
	}
	.ts {
		margin: 0;
		max-width: 560px;
		display: grid;
		gap: 4px;
	}
	.ts figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	svg {
		width: 100%;
		height: auto;
		touch-action: pan-y;
		cursor: crosshair;
	}
	svg:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
	.axis {
		stroke: var(--rule-strong);
	}
	.now {
		stroke: var(--accent);
		stroke-dasharray: 3 2;
	}
	.c {
		fill: none;
		stroke: var(--ink);
		stroke-width: 1.8;
	}
	.h {
		fill: none;
		stroke: var(--ink-2);
		stroke-width: 1.4;
		stroke-dasharray: 4 3;
	}
	.dot {
		fill: none;
		stroke: var(--highlight);
		stroke-width: 2.2;
	}
	.tick {
		fill: var(--muted);
		font-size: 10px;
		font-family: var(--font-mono);
	}
	.units {
		flex-wrap: wrap;
	}
</style>

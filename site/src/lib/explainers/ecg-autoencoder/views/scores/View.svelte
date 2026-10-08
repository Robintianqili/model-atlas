<!-- ecg-autoencoder 'scores' (meta.ts lists the modes). Blue: normal test beats; orange: odd ones. The slider edits the
     learner's input (thr), so every step shows the same threshold. -->
<script lang="ts" module>
	// a new input re-creates the stage: the slider being moved gets the keyboard focus back
	let refocus = 0;
</script>

<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
	import { KIND_LABEL, KINDS } from '../../beats.ts';
	import { THR_MAX, THR_MIN, THR_STEP, countAt, rocOf } from '../../ecg.ts';
	import { rawOf } from '../../adapters/series/adapter.ts';
	import { SIZES, scoresData } from './meta.ts';

	let P: StageProps = $props();
	const d = $derived(scoresData(P));
	let draft = $state<number | null>(null);
	const thr = $derived(draft ?? d.thr);
	const a = $derived(d.set ? countAt(d.set, thr) : null);
	const commit = (v: number) => {
		draft = null;
		refocus = Date.now();
		(P.setInput ?? ((x: unknown) => P.rt.setInput(x)))({ ...rawOf(P.rt.input), thr: v });
	};
	let slider = $state<HTMLInputElement>();
	$effect(() => {
		if (slider && Date.now() - refocus < 3000) slider.focus();
	});

	let W = $state(520);
	let el = $state<HTMLDivElement>();
	$effect(() => {
		if (!el) return;
		const ro = new ResizeObserver((e) => {
			const w = Math.round(e[0].contentRect.width);
			if (w > 0 && w !== W) W = w;
		});
		ro.observe(el);
		return () => ro.disconnect();
	});
	// histogram: bins of 0.002 from 0.01 to 0.09
	const H = 200, ML = 34, MR = 8, MT = 8, MB = 30, LO = 0.01, HI = 0.09, BW = 0.002;
	const NB = Math.round((HI - LO) / BW);
	const X = (v: number) => ML + ((Math.min(HI, Math.max(LO, v)) - LO) / (HI - LO)) * (W - ML - MR);
	const bins = $derived.by(() => {
		const b = [Array(NB).fill(0), Array(NB).fill(0)];
		d.set?.scores.forEach((v, i) => b[d.set!.kinds[i] > 0 ? 1 : 0][Math.min(NB - 1, Math.max(0, Math.floor((v - LO) / BW)))]++);
		return b;
	});
	const ymax = $derived(Math.max(1, ...bins[0], ...bins[1]));
	const Yh = (n: number) => MT + (1 - n / ymax) * (H - MT - MB);
	// ROC: a square plot
	const S = $derived(Math.min(W, 340)), R0 = 34;
	const RX = (x: number) => R0 + x * (S - R0 - 10), RY = (y: number) => 10 + (1 - y) * (S - R0 - 10);
	const roc = $derived(d.set ? rocOf(d.set) : []);
	const shown = $derived(roc.slice(0, Math.min(roc.length, d.k + 1)));
	const rocPath = $derived(shown.map(([x, y], i) => `${i ? 'L' : 'M'}${RX(x).toFixed(1)},${RY(y).toFixed(1)}`).join(''));
	const facts = (k: string) => P.rt.num(`fact:${k}`);
	const eMax = $derived(Math.max(...SIZES.map((s) => facts(s.e))));
	const METRICS = [
		{ key: 'e' as const, label: { en: 'Rebuild error on normal training beats (smaller: rebuilds better)', zh: '正常训练心跳的重建误差（越小重建得越好）' }, w: (v: number) => (v / eMax) * 100 },
		{ key: 'auc' as const, label: { en: 'AUC, all odd beats', zh: 'AUC（全部异常心跳）' }, w: (v: number) => (v - 0.5) * 200 },
		{ key: 'p' as const, label: { en: 'AUC, P wave missing', zh: 'AUC（缺 P 波）' }, w: (v: number) => (v - 0.5) * 200 }
	];
</script>

<div class="sc" bind:this={el}>
	{#if d.mode === 'limits'}
		<div class="grp">
			{#each METRICS as g (g.key)}
				<div class="bars" role="group" aria-label={t(g.label)}>
					<p class="gl">{t(g.label)}</p>
					{#each SIZES as s (s.c)}<div class="row" class:this={s.c === 4}><span>{t({ en: `${s.c} numbers`, zh: `${s.c} 个数` })}</span><i class="bar {g.key}" style:width="{g.w(facts(s[g.key]))}%"></i><b>{fmt3(facts(s[g.key]))}</b></div>{/each}
				</div>
			{/each}
		</div>
		<p class="note">{t({ en: 'Bold: the 4-number code of this page. AUC bars start at 0.5, a coin flip; 1 separates every odd beat from every normal one.', zh: '粗体：本页用的 4 个数的编码。AUC 的条从 0.5（抛硬币）开始；1 表示把每个异常心跳和每个正常心跳都分开。' })}</p>
	{:else if !d.set || !a}
		<p class="note">{t({ en: 'Loading the test set…', zh: '正在载入测试集……' })}</p>
	{:else}
		{#if d.mode === 'hist'}
			<svg width={W} height={H} viewBox="0 0 {W} {H}" role="img" aria-label={t({ en: 'Histogram of the test beats’ scores', zh: '测试心跳分数的直方图' })}>
				<rect x={X(thr)} y={MT} width={Math.max(0, W - MR - X(thr))} height={H - MT - MB} class="flag" />
				{#each [0, 1] as c (c)}{#each bins[c] as n, i (i)}{#if n}<rect x={X(LO + i * BW) + 0.5} y={Yh(n)} width={Math.max(1, X(LO + BW) - X(LO) - 1)} height={H - MB - Yh(n)} class="h h{c}" />{/if}{/each}{/each}
				<line x1={ML} x2={W - MR} y1={H - MB} y2={H - MB} class="axis" />
				{#each [0.02, 0.04, 0.06, 0.08] as v (v)}<text x={X(v)} y={H - MB + 13} class="ax mid">{v}</text>{/each}
				<text x={W - MR} y={H - 3} class="ax end">{t({ en: 'anomaly score', zh: '异常分数' })}</text>
				<text x={ML - 4} y={MT + 9} class="ax end">{ymax}</text><text x={ML - 4} y={H - MB} class="ax end">0</text>
				<line x1={X(thr)} x2={X(thr)} y1={MT - 2} y2={H - MB + 4} class="thr" />
				<text x={X(thr) + 4} y={MT + 10} class="lbl">{t({ en: 'flagged →', zh: '标为异常 →' })}</text>
				{#if Number.isFinite(d.mine)}<path d="M{X(d.mine)},{H - MB + 2}l-6,10h12z" class="you" /><text x={X(d.mine)} y={H - 2} class="ax mid you-t">{t({ en: 'you', zh: '你' })}</text>{/if}
			</svg>
			<p class="legend"><span><i class="k0"></i>{t({ en: `normal test beats (${a.neg})`, zh: `正常测试心跳（${a.neg}）` })}</span><span><i class="k1"></i>{t({ en: `odd test beats (${a.pos})`, zh: `异常测试心跳（${a.pos}）` })}</span>{#if Number.isFinite(d.mine)}<span><i class="kyou"></i>{t({ en: `your beat: ${fmt3(d.mine)}`, zh: `你的心跳：${fmt3(d.mine)}` })}</span>{/if}</p>
		{:else}
			<div class="rocw">
			<svg width={S} height={S} viewBox="0 0 {S} {S}" role="img" aria-label={t({ en: 'ROC curve', zh: 'ROC 曲线' })}>
				<rect x={R0} y="10" width={S - R0 - 10} height={S - R0 - 10} class="bg" />
				<line x1={RX(0)} y1={RY(0)} x2={RX(1)} y2={RY(1)} class="diag" />
				{#if Number.isFinite(d.auc)}<path d="{rocPath}L{RX(1)},{RY(0)}Z" class="area" />{/if}
				<path d={rocPath} class="roc" />
				{#each [0, 0.5, 1] as v (v)}<text x={RX(v)} y={S - R0 + 13} class="ax mid">{v}</text><text x={R0 - 4} y={RY(v) + 3.5} class="ax end">{v}</text>{/each}
				<text x={S - 10} y={S - 4} class="ax end">{t({ en: 'false-alarm rate →', zh: '误报率 →' })}</text>
				<text x={R0 + 4} y="22" class="ax">{t({ en: '↑ catch rate (recall)', zh: '↑ 检出率（召回率）' })}</text>
				<circle cx={RX(a.fpr)} cy={RY(a.recall)} r="6" class="dot" />
			</svg>
			<p class="cap">
				{#if d.m.thr !== undefined}{t({ en: `Sweep: threshold ${fmt3(d.m.thr)}, ${d.m.tp} caught, ${d.m.fp} false alarms.`, zh: `扫描：阈值 ${fmt3(d.m.thr)}，抓到 ${d.m.tp} 个，误报 ${d.m.fp} 个。` })}{/if}
				{#if Number.isFinite(d.auc)}<b>{t({ en: `Area under the curve (AUC): ${fmt3(d.auc)}`, zh: `曲线下面积（AUC）：${fmt3(d.auc)}` })}</b>
					<span>{t({ en: 'Each kind against the normal beats:', zh: '每种异常各自对正常心跳：' })} {KINDS.slice(1).map((k) => `${t(KIND_LABEL[k])} ${fmt3(facts(`auc_${k}`))}`).join(' · ')}</span>{/if}
			</p>
			</div>
		{/if}
		<div class="ctl">
			<label><span>{t({ en: 'threshold', zh: '阈值' })} {fmt3(thr)}</span><input bind:this={slider} type="range" min={THR_MIN} max={THR_MAX} step={THR_STEP} value={thr} oninput={(e) => (draft = +e.currentTarget.value)} onchange={(e) => commit(+e.currentTarget.value)} /></label>
			<dl class="nums">
				<div><dt>{t({ en: 'odd beats caught (recall)', zh: '抓到的异常（召回率）' })}</dt><dd>{a.tp} / {a.pos} · {fmtPct(a.recall)}</dd></div>
				<div><dt>{t({ en: 'false alarms', zh: '误报' })}</dt><dd>{a.fp} / {a.neg} · {fmtPct(a.fpr)}</dd></div>
				<div><dt>{t({ en: 'flagged beats that are odd (precision)', zh: '被标记的心跳里真异常的比例（精确率）' })}</dt><dd>{Number.isFinite(a.precision) ? fmtPct(a.precision) : '–'}</dd></div>
				<div class="wide"><dt>{t({ en: 'caught, by kind', zh: '按类型抓到的' })}</dt><dd class="kinds">{#each KINDS.slice(1) as k, j (k)}<span>{t(KIND_LABEL[k])} {a.per[j + 1]}/{a.of[j + 1]}</span>{/each}</dd></div>
			</dl>
		</div>
	{/if}
</div>

<style>
	.sc { display: grid; gap: var(--s2); grid-template-columns: minmax(0, 1fr); min-width: 0; }
	svg { display: block; max-width: 100%; }
	.bg { fill: var(--surface); stroke: var(--rule-strong); }
	.flag { fill: var(--generated); opacity: 0.07; }
	.h { opacity: 0.6; }
	.h0 { fill: var(--cat-0); }
	.h1 { fill: var(--cat-1); }
	.axis { stroke: var(--rule-strong); }
	.ax { font-size: 10px; fill: var(--ink-2); }
	.ax.mid { text-anchor: middle; }
	.ax.end { text-anchor: end; }
	.lbl { font-size: 11px; font-weight: 600; fill: var(--ink); }
	.thr { stroke: var(--ink); stroke-width: 2; }
	.you { fill: var(--real); }
	.you-t { fill: var(--real-text); font-weight: 600; }
	.diag { stroke: var(--rule-strong); stroke-dasharray: 4 4; }
	.roc { fill: none; stroke: var(--accent); stroke-width: 2.4; }
	.area { fill: var(--accent); opacity: 0.12; }
	.dot { fill: var(--ink); stroke: var(--surface); stroke-width: 2; }
	.legend { display: flex; flex-wrap: wrap; gap: 2px 14px; margin: 0; color: var(--ink-2); font-size: var(--fs-xs); }
	.legend i { display: inline-block; width: 12px; height: 10px; margin-right: 5px; vertical-align: middle; opacity: 0.7; }
	.k0 { background: var(--cat-0); } .k1 { background: var(--cat-1); } .kyou { background: var(--real); }
	.cap { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); display: flex; flex-wrap: wrap; gap: 4px 12px; font-variant-numeric: tabular-nums; }
	.cap b { color: var(--ink); }
	.rocw { display: flex; flex-wrap: wrap; gap: var(--s3); align-items: center; }
	.rocw .cap { flex: 1 1 200px; flex-direction: column; }
	.ctl { display: grid; gap: var(--s2); }
	.ctl label { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 8px; align-items: center; font-size: var(--fs-sm); }
	.ctl label span { font-family: var(--font-mono, monospace); white-space: nowrap; }
	.nums .wide { grid-column: 1 / -1; }
	.kinds { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 0 12px; }
	.kinds span { white-space: nowrap; }
	.nums { margin: 0; display: grid; gap: 2px 16px; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); font-size: var(--fs-sm); }
	.nums div { display: flex; justify-content: space-between; gap: 8px; border-bottom: 1px solid var(--rule); padding: 2px 0; }
	.nums dt { color: var(--ink-2); }
	.nums dd { margin: 0; text-align: right; font-variant-numeric: tabular-nums; }
	.grp { display: grid; gap: var(--s3); grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
	.bars { display: grid; gap: 4px; align-content: start; }
	.gl { margin: 0; font-size: var(--fs-xs); color: var(--ink-2); }
	.row { display: grid; grid-template-columns: 5.5em minmax(0, 1fr) 3.6em; gap: 6px; align-items: center; font-size: var(--fs-sm); }
	.row.this { font-weight: 700; }
	.row b { text-align: right; font-variant-numeric: tabular-nums; }
	.bar { display: block; height: 10px; border-radius: 3px; min-width: 2px; max-width: 100%; }
	.bar.e { background: var(--generated); opacity: 0.7; }
	.bar.auc, .bar.p { background: var(--accent); }
	.note { margin: 0; font-size: var(--fs-xs); color: var(--muted); }
</style>

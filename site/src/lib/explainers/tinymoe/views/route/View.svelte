<!--
  tinymoe/route: routing in a mixture-of-experts layer (props.mode, see data.ts).
  token: one character's 4 router scores, chances (bars to scale, 0 to 1) and the 2 experts kept with their gates.
  grid:  every character of the text with a 2 × 4 grid under it (rows: blocks, columns: experts), shaded by the gate;
         props.mark { block, expert } outlines the characters that expert took.
  load:  the dispatch process: slots taken per expert against the capacity line, and the characters so far with each
         chosen expert kept (number) or dropped (×); at the balance frame also f, P and f × P per expert and their sum.
         props.capacity false (the balance step, before capacity is taught): the slots sent only, no line and no ×.
  dense: the measured comparison with the dense twin (facts).
  Nothing animates by itself (the lesson's scrubber moves 'load'); the grid wraps, so nothing scrolls sideways.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtInt, fmtPct } from '../../../../atlas/core/format.ts';
	import { denseOf, gridOf, loadOf, modeOf, tokenOf } from './data.ts';

	let P: StageProps = $props();
	const mode = $derived(modeOf(P));
	const seq = $derived(makeScale('sequential', [0, 1], themeState.resolved));
	const idx = (n: number) => Array.from({ length: n || 0 }, (_, i) => i);
	const tok = $derived(mode === 'token' ? tokenOf(P) : null);
	const grid = $derived(mode === 'grid' ? gridOf(P) : null);
	const ld = $derived(mode === 'load' ? loadOf(P) : null);
	const dense = $derived(mode === 'dense' ? denseOf(P) : []);
	const mark = $derived((P.props?.mark ?? null) as { block: number; expert: number } | null);
	const zmax = $derived(tok ? Math.max(1e-9, ...tok.z.map(Math.abs)) : 1);
	const div = $derived(makeScale('diverging', [-zmax, zmax], themeState.resolved));
	const lmax = $derived(ld ? Math.max(ld.cap, ...ld.load, ...ld.sent, 1) : 1);
	const pmax = $derived(Math.max(...dense.map((r) => r.total).filter(Number.isFinite), 1));
	const marked = (g: number[][]) => !!mark && (g[mark.block - 1]?.[mark.expert] ?? 0) > 0;
</script>

{#if mode === 'token' && tok}
	{@const sm = P.props?.softmax === true}
	<div class="tok">
		<p class="cap">{t({ en: `Block ${tok.block} · the character “${tok.c}” (position ${tok.t}, counting from 0)`, zh: `第 ${tok.block} 块 · 字符“${tok.c}”（位置 ${tok.t}，从 0 数起）` })}</p>
		<table>
			<thead><tr><th scope="col">{t({ en: 'Expert', zh: '专家' })}</th><th scope="col">{t({ en: 'Router score', zh: '路由分数' })}</th>{#if sm}<th scope="col">{t({ en: 'e^(score − largest)', zh: 'e^(分数 − 最大分数)' })}</th>{/if}<th scope="col">{t({ en: 'Chance (0 to 1)', zh: '概率（0 到 1）' })}</th>{#if !sm}<th scope="col">{t({ en: 'Gate', zh: '门控' })}</th>{/if}</tr></thead>
			<tbody>
				{#each tok.p as p, e (e)}
					{@const on = !sm && (e === tok.a || e === tok.b)}
					<tr class:on>
						<th scope="row">{e}</th>
						<td class="num"><span class="sbar"><span style="width: {(50 * Math.abs(tok.z[e])) / zmax}%; {tok.z[e] < 0 ? 'right: 50%' : 'left: 50%'}; background: {div.css(tok.z[e])}"></span></span>{fmt3(tok.z[e])}</td>
						{#if sm}<td class="num">{fmt3(tok.ex[e])}</td>{/if}
						<td class="num"><span class="bar"><span style="width: {100 * p}%; background: {seq.css(p)}"></span></span>{fmtPct(p)}</td>
						{#if !sm}<td class="num">{on ? fmt3(tok.g[e]) : t({ en: 'not run', zh: '不运行' })}</td>{/if}
					</tr>
				{/each}
			</tbody>
		</table>
		<p class="note">
			{#if sm}{t({ en: `Each chance = e^(score − ${fmt3(tok.zTop)}) ÷ ${fmt3(tok.sum)}, the sum of the third column; the four chances add up to 1.`, zh: `每个概率 = e^(分数 − ${fmt3(tok.zTop)}) ÷ ${fmt3(tok.sum)}，也就是第三列的和；四个概率加起来是 1。` })}
			{:else}{t({ en: `Kept: experts ${tok.a} and ${tok.b}. Gates = their chances ÷ (${fmt3(tok.pa)} + ${fmt3(tok.pb)}), so the two add up to 1.`, zh: `保留：专家 ${tok.a} 和 ${tok.b}。门控 = 它们的概率 ÷（${fmt3(tok.pa)} + ${fmt3(tok.pb)}），所以两个加起来是 1。` })}{/if}
		</p>
	</div>
{:else if mode === 'grid' && grid}
	<div class="gridwrap">
		<div class="key">
			<span class="mini legend" aria-hidden="true">{#each idx(grid.L) as b (b)}<span class="mrow">{#each idx(grid.E) as e (e)}<span class="cell lab">{e}</span>{/each}</span>{/each}</span>
			<span>{t({ en: 'Under each character: block 1 (top row) and block 2 (bottom row), experts 0 to 3 from left to right. A colored cell is a chosen expert; the stronger the color, the larger its gate.', zh: '每个字符下面：第 1 块（上一行）和第 2 块（下一行），从左到右是专家 0 到 3。有颜色的格子是被选中的专家；颜色越浓，门控越大。' })}</span>
			<ColorLegend kind="sequential" domain={[0, 1]} label={{ en: 'gate', zh: '门控' }} width={140} />
		</div>
		<ul class="chips" aria-label={t({ en: 'Routing of every character', zh: '每个字符的分派' })}>
			{#each grid.chars as ch (ch.t)}
				<li class="chip" class:marked={marked(ch.g)}>
					<span class="ch">{ch.c}</span>
					<span class="mini" aria-hidden="true">
						{#each ch.g as g, b (b)}<span class="mrow">{#each g as v, e (e)}<span class="cell" style={v > 0 ? `background: ${seq.css(v)}` : ''}></span>{/each}</span>{/each}
					</span>
				</li>
			{/each}
		</ul>
	</div>
{:else if mode === 'load' && ld}
	{@const capOn = P.props?.capacity !== false}
	<div class="load">
		<p class="cap">{capOn ? t({ en: `Block ${ld.block} · each expert may take ${fmtInt(ld.cap)} slots (dashed line)`, zh: `第 ${ld.block} 块 · 每个专家最多接 ${fmtInt(ld.cap)} 个位置（虚线）` }) : t({ en: `Block ${ld.block} · the slots sent to each expert`, zh: `第 ${ld.block} 块 · 每个专家被送来的位置` })}</p>
		<div class="experts">
			{#each ld.load as l, e (e)}
				<div class="erow">
					<span class="ename">{t({ en: `Expert ${e}`, zh: `专家 ${e}` })}</span>
					<span class="track">
						<span class="sent" style="width: {(100 * ld.sent[e]) / lmax}%"></span>
						<span class="taken" style="width: {(100 * (capOn ? l : ld.sent[e])) / lmax}%"></span>
						{#if capOn}<span class="capline" style="left: {(100 * ld.cap) / lmax}%"></span>{/if}
					</span>
					<span class="num">{capOn ? t({ en: `${fmtInt(l)} taken, ${fmtInt(ld.sent[e] - l)} dropped`, zh: `接下 ${fmtInt(l)}，丢掉 ${fmtInt(ld.sent[e] - l)}` }) : t({ en: `${fmtInt(ld.sent[e])} slots`, zh: `${fmtInt(ld.sent[e])} 个位置` })}</span>
				</div>
			{/each}
		</div>
		{#if ld.balance && P.props?.table !== false}
			{@const m = ld.balance}
			<table class="bal">
				<thead><tr><th scope="col">{t({ en: 'Expert', zh: '专家' })}</th><th scope="col">{t({ en: 'f: share of slots sent', zh: 'f：分到的位置份额' })}</th><th scope="col">{t({ en: 'P: mean chance', zh: 'P：平均概率' })}</th><th scope="col">f × P</th></tr></thead>
				<tbody>{#each ld.load as _, e (e)}<tr><th scope="row">{e}</th><td class="num">{fmt3(m[`f${e}`])}</td><td class="num">{fmt3(m[`P${e}`])}</td><td class="num">{fmt3(m[`fp${e}`])}</td></tr>{/each}</tbody>
				<tfoot><tr><th scope="row">Σ</th><td class="num">{fmt3(ld.load.reduce((a, _, e) => a + m[`f${e}`], 0))}</td><td class="num">{fmt3(ld.load.reduce((a, _, e) => a + m[`P${e}`], 0))}</td><td class="num">{ld.load.length} × {ld.load.reduce((a, _, e) => a + m[`fp${e}`], 0).toFixed(4)} = {Number(m.aux).toFixed(4)}</td></tr></tfoot>
			</table>
		{/if}
		<ul class="chips small" aria-label={t({ en: 'Characters routed so far', zh: '已经分派的字符' })}>
			{#each ld.chars as ch (ch.t)}
				<li class="chip"><span class="ch">{ch.c}</span>{#each ch.picks as pk (pk.e)}<span class="pk" class:drop={capOn && !pk.ok}>{pk.ok || !capOn ? pk.e : '×'}</span>{/each}</li>
			{/each}
		</ul>
		<p class="note">{capOn ? t({ en: 'Under each character: the experts that took it; × is a slot dropped because that expert was full.', zh: '每个字符下面：接下它的专家；× 表示那个专家已经满了，这个位置被丢掉。' }) : t({ en: 'Under each character: the experts it was sent to.', zh: '每个字符下面：它被送去的专家。' })}</p>
	</div>
{:else if mode === 'dense'}
	<table class="dense">
		<thead><tr><th scope="col">{t({ en: 'Model', zh: '模型' })}</th><th scope="col">{t({ en: 'Parameters: all (whole bar) and used per character (colored part)', zh: '参数：全部（整条）和每个字符用到的（彩色部分）' })}</th><th scope="col">{t({ en: 'Loss on held-out stories (nats per character, lower is better)', zh: '在没见过的故事上的损失（每个字符的奈特，越低越好）' })}</th></tr></thead>
		<tbody>
			{#each dense as r (r.id)}
				<tr>
					<th scope="row">{t(r.name)}</th>
					<td><span class="pbar"><span class="all" style="width: {(100 * r.total) / pmax}%"></span><span class="act" style="width: {(100 * r.active) / pmax}%"></span></span><span class="num">{fmtInt(r.total)} / {fmtInt(r.active)}</span></td>
					<td class="num">{r.loss.toFixed(4)}</td>
				</tr>
			{/each}
		</tbody>
	</table>
{/if}

<style>
	.cap { margin: 0 0 8px; font-weight: 600; }
	.note { margin: 8px 0 0; color: var(--ink-2); font-size: var(--fs-sm); }
	table { border-collapse: collapse; width: 100%; font-size: var(--fs-sm); }
	th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--rule); }
	thead th { color: var(--ink-2); font-weight: 600; }
	.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
	tr.on th, tr.on td { font-weight: 700; }
	tr:not(.on) td:last-child { color: var(--muted); }
	.bar, .sbar { position: relative; display: inline-block; width: 64px; height: 10px; margin-right: 6px; vertical-align: middle; background: var(--surface-2); }
	.bar > span { position: absolute; left: 0; top: 0; bottom: 0; }
	.sbar > span { position: absolute; top: 0; bottom: 0; }
	.sbar::after { content: ''; position: absolute; left: 50%; top: -2px; bottom: -2px; border-left: 1px solid var(--ink-2); }
	.key { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; font-size: var(--fs-sm); color: var(--ink-2); margin-bottom: 10px; }
	.key > span:nth-child(2) { flex: 1 1 220px; }
	.chips { list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 6px 4px; }
	.chip { display: inline-flex; flex-direction: column; align-items: center; min-width: 38px; padding: 2px; border: 1px solid transparent; border-radius: 4px; }
	.chip.marked { border-color: var(--accent); background: var(--accent-soft); }
	.ch { font-family: var(--font-mono, monospace); font-size: var(--fs-sm); line-height: 1.3; }
	.mini { display: inline-flex; flex-direction: column; gap: 1px; }
	.mrow { display: inline-flex; gap: 1px; }
	.cell { width: 8px; height: 8px; outline: 1px solid var(--rule); }
	.legend .cell { width: 12px; height: 12px; font-size: 9px; line-height: 12px; text-align: center; outline: none; }
	.experts { display: grid; gap: 6px; margin-bottom: 10px; }
	.erow { display: grid; grid-template-columns: 5.5em minmax(80px, 1fr) auto; gap: 8px; align-items: center; font-size: var(--fs-sm); }
	.track { position: relative; height: 14px; background: var(--surface-2); }
	.track .sent { position: absolute; left: 0; top: 0; bottom: 0; background: var(--rule-strong); }
	.track .taken { position: absolute; left: 0; top: 0; bottom: 0; background: var(--accent); }
	.capline { position: absolute; top: -3px; bottom: -3px; border-left: 2px dashed var(--ink); }
	.bal { margin-bottom: 10px; }
	.bal tfoot th, .bal tfoot td { font-weight: 700; border-bottom: none; }
	.small .chip { min-width: 22px; }
	.pk { font-size: var(--fs-xs); line-height: 1.2; color: var(--ink-2); }
	.pk.drop { color: var(--div-neg-text); font-weight: 700; }
	.pbar { position: relative; display: block; max-width: 120px; height: 10px; margin-bottom: 2px; background: var(--surface-2); }
	@media (max-width: 480px) { .bar, .sbar { width: 32px; } th, td { padding: 4px; } }
	.pbar > span { position: absolute; left: 0; top: 0; bottom: 0; }
	.pbar .all { background: var(--rule-strong); }
	.pbar .act { background: var(--accent); }
</style>

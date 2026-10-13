<!-- transe 'step': one training step (see meta.ts). Distances are drawn to scale on one axis (0 at the left); the dashed
     line is d(fact) + γ, and the outlined stretch is how far the corrupted copy falls short of it: the loss. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical } from '$lib/kit/color';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { ENT, REL } from '../../kg.ts';
	import { worked } from '../../space.ts';
	import { GAMMA } from './meta.ts';

	let P: StageProps = $props();
	const w = $derived(worked(P.at, GAMMA));
	const upd = $derived(P.props.mode === 'update');
	const theme = $derived(themeState.resolved);
	const n = (i: number) => t(ENT[i].name);
	const par = (v: number) => (v < 0 ? `(${fmt3(v)})` : fmt3(v));
	const max = $derived(w ? Math.max(w.dPos + w.gamma, w.dNeg, w.dPos1 + w.gamma, w.dNeg1) * 1.06 : 1);
	const X = (v: number) => 92 + (v / max) * 296;
	const ticks = $derived(Array.from({ length: Math.floor(max) + 1 }, (_, i) => i));
	const rows = $derived(w ? [{ y: 18, dp: w.dPos, dn: w.dNeg, loss: w.loss, label: { en: 'before', zh: '之前' } }, ...(upd ? [{ y: 92, dp: w.dPos1, dn: w.dNeg1, loss: w.loss1, label: { en: 'after', zh: '之后' } }] : [])] : []);
</script>

<div class="st" style:min-height="340px">
	{#if !w}
		<p class="muted">{t({ en: 'Waiting for the first training step…', zh: '等待第一步训练…' })}</p>
	{:else}
		<p class="small"><span class="tag" style:border-color={categorical(0, theme)}>{t({ en: 'fact', zh: '事实' })}: {n(w.h)} · {t(REL[w.r])} · {n(w.t)}</span> <span class="tag" style:border-color={categorical(1, theme)}>{t({ en: 'corrupted', zh: '损坏副本' })}: {n(w.h2)} · {t(REL[w.r])} · {n(w.t2)}</span></p>
		<svg viewBox="0 0 400 {upd ? 176 : 102}" role="img" aria-label={t({ en: 'The two distances and the margin, to scale', zh: '两个距离和间隔，按比例画出' })}>
			{#each rows as row (row.y)}
				<text x="2" y={row.y + 8} class="lab">{t(row.label)}</text>
				<rect x={X(0)} y={row.y} width={X(row.dp) - X(0)} height="14" fill={categorical(0, theme)} />
				<rect x={X(0)} y={row.y + 20} width={X(row.dn) - X(0)} height="14" fill={categorical(1, theme)} />
				<text x={X(row.dp) + 3} y={row.y + 11}>{fmt3(row.dp)}</text>
				<text x={X(row.dn) + 3} y={row.y + 31}>{fmt3(row.dn)}</text>
				<line x1={X(row.dp + w.gamma)} x2={X(row.dp + w.gamma)} y1={row.y - 4} y2={row.y + 38} class="m" />
				{#if row.loss > 0}<rect x={X(row.dn)} y={row.y + 18} width={X(row.dp + w.gamma) - X(row.dn)} height="18" class="gap" />{/if}
				<text x={X(row.dp + w.gamma) - 3} y={row.y + 48} text-anchor="end" class="lab">d + γ = {fmt3(row.dp + w.gamma)} · {t({ en: 'loss', zh: '损失' })} {fmt3(row.loss)}</text>
			{/each}
			{#each ticks as v (v)}<text x={X(v)} y={upd ? 172 : 98} text-anchor="middle" class="tick">{v}</text>{/each}
		</svg>
		<p class="small muted">{t({ en: 'distance ‖h + r − t‖ along the axis; first bar the fact, second its corrupted copy; dashed line: d(fact) + γ', zh: '横轴是距离 ‖h + r − t‖；第一条是事实，第二条是它的损坏副本；虚线：d(事实) + γ' })}</p>
		{#if upd}
			<div class="scroll" use:keyScroll>
				<table class="num-table small">
					<thead><tr><th>k</th><th>h<sub>k</sub> ({n(w.h)})</th><th>r<sub>k</sub></th><th>t<sub>k</sub> ({n(w.t)})</th><th>x<sub>k</sub> ({n(w.x)})</th><th>{t({ en: 'gradient g', zh: '梯度 g' })}</th><th>{t({ en: 'new h', zh: '新的 h' })}<sub>k</sub></th></tr></thead>
					<tbody>
						{#each w.before.E[w.h] as hk, k (k)}
							<tr class:hi={k === w.k}><td class="muted">{k}</td><td class="num">{fmt3(hk)}</td><td class="num">{fmt3(w.before.R[w.r][k])}</td><td class="num">{fmt3(w.before.E[w.t][k])}</td><td class="num">{fmt3(w.before.E[w.x][k])}</td><td class="num">{fmt3(w.grad.E[w.h][k])}</td><td class="num">{fmt3(w.after.E[w.h][k])}</td></tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p class="small num">{t({ en: `row k = ${w.k}:`, zh: `k = ${w.k} 那一行：` })} {fmt3(w.old)} − {w.lr} × {par(w.g)} = {fmt3(w.new)}</p>
		{/if}
	{/if}
</div>

<style>
	.st { display: grid; gap: 6px; min-width: 0; align-content: start; } p { margin: 0; } .small { font-size: var(--fs-sm); } .scroll { overflow-x: auto; }
	svg { width: 100%; height: auto; display: block; } svg text { font: 500 10px var(--font-mono); fill: var(--ink); } .lab { font-family: var(--font-sans); } .tick { fill: var(--ink-2); }
	.m { stroke: var(--ink); stroke-width: 1.3; stroke-dasharray: 4 3; } .gap { fill: none; stroke: var(--highlight); stroke-width: 2; }
	.tag { padding: 0 6px; border: 2px solid; border-radius: var(--r1); }
	table { border-collapse: collapse; } th, td { padding: 1px 6px; text-align: right; white-space: nowrap; } th { font-weight: 600; } tr.hi td { outline: 2px solid var(--highlight); outline-offset: -2px; background: var(--accent-soft); }
	@media (max-width: 560px) { .st .scroll table { font-size: 10.5px; } th, td { padding: 1px 2px; } th { white-space: normal; } }
</style>

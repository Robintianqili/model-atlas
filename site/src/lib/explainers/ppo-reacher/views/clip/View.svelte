<!-- ppo-reacher 'clip': every sample of the minibatch at (ratio r, advantage A). With props.clip (default): the dashed
     lines r = 1 − ε and r = 1 + ε, the two shaded regions where the clipped objective stops pulling, clipped samples
     hollow, the worked sample ringed and its numbers under the chart. Ratios beyond the chart sit at its edge. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps, Tensor } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { mbOf, workedOf } from './meta.ts';

	let P: StageProps = $props();
	const m = $derived(P.at.tape && P.at.index >= 0 && P.at.index < P.at.tape.length ? mbOf((P.at.tape.state(P.at.index) as { batch?: Record<string, Tensor> }).batch) : null);
	const show = $derived(P.props.clip !== false);
	const w = $derived(m ? workedOf(m) : null);
	const R = $derived(m ? [Math.max(0, Math.min(0.5, ...m.r) - 0.05), Math.min(3, Math.max(1.5, ...m.r) + 0.05)] : [0, 2]);
	const am = $derived(m ? Math.max(0.5, ...m.A.map(Math.abs)) * 1.08 : 1);
	const X = (r: number) => 60 + ((Math.max(R[0], Math.min(R[1], r)) - R[0]) / (R[1] - R[0])) * 520;
	const Y = (a: number) => 140 - (a / am) * 125;
	const k = $derived(m ? m.clipped.reduce((s, x) => s + x, 0) : 0);
	const off = $derived(!m || m.eps >= 10);
</script>

{#if m && w}
	<div class="cp">
		<svg viewBox="0 0 600 300" role="img" aria-label={t({ en: 'Each sample at its ratio and advantage', zh: '每个样本按它的比率和优势放置' })}>
			{#if show && !off}
				<rect x={X(1 + m.eps)} y="15" width={Math.max(0, 580 - X(1 + m.eps))} height="125" class="zone" />
				<rect x="60" y="140" width={Math.max(0, X(1 - m.eps) - 60)} height="125" class="zone" />
				<line x1={X(1 - m.eps)} y1="15" x2={X(1 - m.eps)} y2="265" class="eps" /><line x1={X(1 + m.eps)} y1="15" x2={X(1 + m.eps)} y2="265" class="eps" />
				<text x={X(1 - m.eps) - 4} y="28" class="tick end">1 − ε</text><text x={X(1 + m.eps) + 4} y="28" class="tick">1 + ε</text>
			{/if}
			<line x1="60" y1="140" x2="580" y2="140" class="axis" /><line x1={X(1)} y1="15" x2={X(1)} y2="265" class="one" />
			{#each m.r as r, i (i)}
				<circle cx={X(r)} cy={Y(m.A[i])} r="4" class={show && m.clipped[i] ? 'pt hollow' : 'pt'} />
			{/each}
			{#if show}<circle cx={X(w.r)} cy={Y(w.A)} r="10" class="ring" />{/if}
			<text x="60" y="285" class="tick">{fmt3(R[0])}</text><text x={X(1)} y="285" class="tick mid">1</text><text x="580" y="285" class="tick end">{fmt3(R[1])}</text>
			<text x="320" y="299" class="tick mid">{t({ en: 'ratio r = π(a | s) / π_old(a | s)', zh: '比率 r = π(a | s) / π_old(a | s)' })}</text>
			<text x="54" y="24" class="tick end">A</text><text x="54" y="144" class="tick end">0</text>
		</svg>
		<p class="small">{t({ en: `Update ${m.j} (counting from 0) of this batch: ${m.n} samples, one dot each; up = advantage above 0.`, zh: `这一批的第 ${m.j} 次更新（从 0 数起）：${m.n} 个样本，每个一点；越往上优势越大。` })}
			{#if show}{off ? t({ en: ' ε is off: nothing is clipped.', zh: ' ε 已关闭：什么都不截断。' }) : t({ en: ` Shaded: where the objective stops pulling. Hollow: the ${k} clipped samples.`, zh: ` 阴影：目标不再拉动的区域。空心点：被截断的 ${k} 个样本。` })}{/if}</p>
		{#if show}
			<p class="small num">{t({ en: 'Ringed', zh: '圈出的样本' })}: r = {fmt3(w.r)} · A = {fmt3(w.A)} · r·A = {fmt3(w.s1)} · clip(r)·A = {fmt3(w.c)} × {fmt3(w.A)} = {fmt3(w.s2)} · min = {fmt3(w.obj)} → {t(w.clipped ? { en: 'clipped: no pull', zh: '被截断：不再拉动' } : { en: 'not clipped: it pulls', zh: '没有截断：仍在拉动' })}</p>
		{/if}
	</div>
{:else}
	<p class="muted">{t({ en: 'Computing…', zh: '计算中…' })}</p>
{/if}

<style>
	.cp { display: grid; gap: 6px; grid-template-columns: minmax(0, 1fr); min-width: 0; }
	svg { width: 100%; height: auto; max-width: 640px; }
	.zone { fill: var(--surface-2); } .eps { stroke: var(--ink-2); stroke-dasharray: 5 4; stroke-width: 1.5; }
	.axis { stroke: var(--rule-strong); stroke-width: 1.5; } .one { stroke: var(--rule-strong); stroke-width: 1; }
	.pt { fill: var(--accent); } .pt.hollow { fill: var(--surface); stroke: var(--div-neg); stroke-width: 2; } .ring { fill: none; stroke: var(--ink); stroke-width: 2; }
	.tick { font: 15px var(--font-mono); fill: var(--ink-2); } .mid { text-anchor: middle; } .end { text-anchor: end; }
	.small { margin: 0; font-size: var(--fs-xs); }
</style>

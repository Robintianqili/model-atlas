<!-- ppo-reacher 'cells': one row per quantity (an episode: r, V, δ, A) or per episode (a batch: A), one cell per step,
     time to the right. Orange above 0, blue below, stronger color = farther from 0 (each row of an episode on its own scale);
     grey: not computed yet. The ringed column is the step the narration works out; the GAE formula is under it. -->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { Frame, StageProps } from '../../../../atlas/types/index.ts';
	import { alignTex, texValues } from '../../../../atlas/ui/views/kit.parts/tex.ts';
	import { cellsOf } from './meta.ts';

	let P: StageProps = $props();
	const c = $derived(cellsOf(P));
	const DELTA = '\\delta_{ {{t}} } = r + \\gamma\\,V(s_{ {{t1}} }) - V(s_{ {{t}} }) = {{r}} + {{gamma}} \\times {{v1}} - {{v}} = {{delta}}';
	const H = $derived(c?.mode === 'batch' ? 13 : 30);
	const cw = $derived(c ? 500 / c.T : 1);
	const color = (v: number) => (!Number.isFinite(v) ? 'var(--surface-2)' : v >= 0 ? 'var(--div-pos)' : 'var(--div-neg)');
	const alpha = (v: number, m: number) => (!Number.isFinite(v) ? 1 : 0.12 + 0.88 * Math.min(1, Math.abs(v) / m));
	const formula = $derived.by(() => {
		const f = P.at.frame, ph = f?.phase === 'gae' ? (P.at.tape?.phases?.gae as { formula?: { tex: string; values: (x: Frame) => Record<string, number | string> } } | undefined) : undefined;
		// props.delta (the value step): the TD error of the frame's step instead of its advantage
		if (f && ph && P.props.delta) return { tex: alignTex(DELTA), values: texValues({ ...f.metrics!, gamma: String(f.metrics!.gamma) }) };
		return f && ph?.formula ? { tex: alignTex(ph.formula.tex), values: texValues(ph.formula.values(f)) } : null;
	});
	const LEG = {
		episode: { en: 'Rows: reward r, value V(s), TD error δ, advantage A. Orange above 0, blue below; stronger color = farther from 0 (each row on its own scale); grey: not worked out yet.', zh: '各行：奖励 r、价值 V(s)、TD 误差 δ、优势 A。橙色大于 0，蓝色小于 0；颜色越浓，离 0 越远（每行各用自己的尺度）；灰色：还没算到。' },
		batch: { en: 'One row per episode of the batch (episode 0 is yours), one cell per step; the color is the normalized advantage: orange above 0, blue below.', zh: '这一批里每个回合一行（编号 0 的是你的），每一步一格；颜色是归一化后的优势：橙色大于 0，蓝色小于 0。' }
	};
</script>

{#if c}
	<div class="cl">
		<svg viewBox="0 0 600 {c.rows.length * (H + 3) + 22}" role="img" aria-label={t({ en: 'The steps as cells', zh: '用格子表示的每一步' })}>
			<text x="92" y="13" class="tick">{t({ en: 'step 0 →', zh: '第 0 步 →' })}</text>
			{#each c.rows as row, k (k)}
				<text x="86" y={20 + k * (H + 3) + H / 2 + 5} class="tick end">{t(row.label)}</text>
				{#each row.v as v, i (i)}
					<rect x={92 + i * cw} y={20 + k * (H + 3)} width={cw - 1} height={H} fill={color(v)} fill-opacity={alpha(v, c.maxAbs[k])} />
				{/each}
			{/each}
			{#if c.focus !== null}<rect x={92 + c.focus * cw - 1.5} y="18" width={cw + 2} height={c.rows.length * (H + 3) + 1} class="ring" />{/if}
		</svg>
		<p class="small">{t(LEG[c.mode])}</p>
		{#if formula}<FormulaCard tex={formula.tex} values={formula.values} provenance={P.provenance} />{/if}
	</div>
{:else}
	<p class="muted">{t({ en: 'Computing…', zh: '计算中…' })}</p>
{/if}

<style>
	.cl { display: grid; gap: 6px; grid-template-columns: minmax(0, 1fr); min-width: 0; }
	svg { width: 100%; height: auto; max-width: 760px; }
	.tick { font: 15px var(--font-mono); fill: var(--ink-2); } .end { text-anchor: end; }
	.ring { fill: none; stroke: var(--ink); stroke-width: 2; }
	.small { margin: 0; font-size: var(--fs-xs); }
</style>

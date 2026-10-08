<!-- cart-pole 'round': one row per episode, one cell per step (time runs left to right). G: darker = larger return (grey: not
     computed yet). A and pull: orange positive, blue negative, stronger = larger. The ringed cell is the one the
     narration works out; its numbers are under the picture. -->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { Frame, StageProps } from '../../../../atlas/types/index.ts';
	import { alignTex, texValues } from '../../../../atlas/ui/views/kit.parts/tex.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { roundOf } from './meta.ts';

	let P: StageProps = $props();
	const d = $derived(roundOf(P));
	const H = $derived((d?.rows.length ?? 0) > 1 ? 16 : 34);
	const cw = $derived(d ? 500 / Math.max(10, d.maxLen) : 1);
	// the cell's color: G shades one color; A and pull take the sign of the advantage (a pull's size sets the strength)
	const color = (c: { G: number; A: number }, v: number) => (!Number.isFinite(v) ? 'var(--surface-2)' : d!.mode === 'G' ? 'var(--accent)' : c.A >= 0 ? 'var(--div-pos)' : 'var(--div-neg)');
	const alpha = (v: number) => (!Number.isFinite(v) ? 1 : 0.15 + 0.85 * Math.min(1, Math.abs(v) / d!.maxAbs));
	const c = $derived(d?.focus ? d.rows[d.focus.ep]?.[d.focus.t] : undefined);
	// on a 'return' frame of the episode tape: G = r + γ·G(next) with the frame's numbers
	const formula = $derived.by(() => {
		const f = P.at.frame;
		const ph = f?.phase === 'return' ? (P.at.tape?.phases?.return as { formula?: { tex: string; values: (x: Frame) => Record<string, number> } } | undefined) : undefined;
		// γ as written (0.95), as the narration says it, not to 3 digits (0.950)
		return f && ph?.formula ? { tex: alignTex(ph.formula.tex), values: texValues({ ...ph.formula.values(f), gamma: String(f.metrics?.gamma) }) } : null;
	});
	const LEG = {
		G: { en: 'Each cell is one step of the episode; darker = larger return G. Grey: not computed yet.', zh: '每个格子是回合中的一步；颜色越深，回报 G 越大。灰色：还没算到。' },
		A: { en: 'Orange: advantage above 0 (better than usual), blue: below 0. Stronger color = larger size.', zh: '橙色：优势大于 0（比平常好），蓝色：小于 0。颜色越浓，数值越大。' },
		pull: { en: 'Each step’s pull on the output score, A × (p(left) − y) / N; its color is the sign of the advantage, its strength the size of the pull.', zh: '每一步对输出分数的拉力 A × (p(左) − y) / N；颜色是优势的正负，深浅是拉力的大小。' }
	};
</script>

{#if d}
	<div class="rd">
		<svg viewBox="0 0 600 {d.rows.length * (H + 3) + 22}" role="img" aria-label={t({ en: 'The steps of the round', zh: '这一轮的每一步' })}>
			<text x="92" y="13" class="tick">{t({ en: 'step 0 →', zh: '第 0 步 →' })}</text>
			{#each d.rows as row, e (e)}
				<text x="86" y={20 + e * (H + 3) + H / 2 + 5} class="tick end">{d.rows.length > 1 ? `${e} · ${row.length}` : `T = ${row.length}`}</text>
				{#each row as cell, i (i)}
					<rect x={92 + i * cw} y={20 + e * (H + 3)} width={Math.max(1, cw - (cw > 3 ? 1 : 0))} height={H} fill={color(cell, cell[d.mode])} fill-opacity={alpha(cell[d.mode])} />
				{/each}
			{/each}
			{#if d.focus}<rect x={92 + d.focus.t * cw - 1.5} y={20 + d.focus.ep * (H + 3) - 1.5} width={Math.max(4, cw) + 3} height={H + 3} class="ring" />{/if}
		</svg>
		<p class="small muted">{t(LEG[d.mode])}{d.rows.length > 1 ? t({ en: ' Left of each row: the episode number · its length in steps.', zh: '每行左边：回合编号 · 它的长度（步）。' }) : ''}</p>
		{#if c && d.focus}
			<p class="num small">
				{t({ en: `Ringed: episode ${d.focus.ep}, step ${d.focus.t}`, zh: `圈出的格子：回合 ${d.focus.ep}，第 ${d.focus.t} 步` })} · G = {fmt3(c.G)}{#if d.mode !== 'G'}{' · '}A = {fmt3(c.A)} · {t(c.left ? { en: 'pushed left (y = 1)', zh: '向左推（y = 1）' } : { en: 'pushed right (y = 0)', zh: '向右推（y = 0）' })} · {t({ en: 'p(left)', zh: 'p(左)' })} = {fmt3(c.p)}{/if}{#if d.mode === 'pull'}{' · '}{t({ en: 'pull', zh: '拉力' })} = {fmt3(c.pull)}{/if}
			</p>
			{#if d.mode === 'A'}<p class="num small">{t({ en: `All ${d.N} steps: mean G = ${fmt3(d.mean)} · standard deviation ${fmt3(d.sd)} · A = (G − mean) / standard deviation`, zh: `全部 ${d.N} 步：G 的平均值 = ${fmt3(d.mean)} · 标准差 ${fmt3(d.sd)} · A = (G − 平均值) / 标准差` })}</p>{/if}
		{/if}
		{#if formula}<FormulaCard tex={formula.tex} values={formula.values} provenance={P.provenance} />{/if}
		{#if d.mode === 'pull'}<p class="num small">{t({ en: `All ${d.N} pulls added up: ${fmt3(d.sum)} = the gradient of the output bias`, zh: `全部 ${d.N} 个拉力加起来：${fmt3(d.sum)} = 输出偏置的梯度` })}</p>{/if}
	</div>
{:else}
	<p class="muted">{t({ en: 'Computing…', zh: '计算中…' })}</p>
{/if}

<style>
	.rd { display: grid; gap: 6px; grid-template-columns: minmax(0, 1fr); min-width: 0; }
	svg { width: 100%; height: auto; max-width: 760px; }
	.tick { font: 15px var(--font-mono); fill: var(--ink-2); } .end { text-anchor: end; }
	.ring { fill: none; stroke: var(--ink); stroke-width: 2; }
	.small { margin: 0; font-size: var(--fs-xs); }
</style>

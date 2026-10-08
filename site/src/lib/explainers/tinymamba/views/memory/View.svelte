<!-- tinymamba 'memory' (meta.ts): the scan of one channel unrolled into a look-back matrix (outputs × inputs; the
     upper triangle is empty: no token reads a later one), and the numbers each model keeps. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { makeScale } from '$lib/kit/color';
	import { fmt3, fmtInt } from '$lib/kit/format';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { memoryOf } from './meta.ts';

	let P: StageProps = $props();
	const M = $derived(memoryOf(P));
	const n = $derived(M.toks.length);
	let cw = $state(400);
	const c = $derived(Math.max(8, Math.min(24, Math.floor((Math.min(cw, 480) - 20) / Math.max(1, n)))));
	const sc = $derived(makeScale('diverging', [-1, Math.max(1e-9, ...(M.M ?? []).flat().map(Math.abs))], themeState.resolved));
	let hover = $state('');
	const bar = (v: number) => `${Math.max(1, (100 * v) / Math.max(M.state, M.kv))}%`;
</script>

<div class="mem" bind:clientWidth={cw}>
	<div class="box">
		<span class="lab">{t({ en: `Channel ${M.ch} of block ${M.b}, unrolled: row = the output at a token, column = the input u of a token read so far`, zh: `第 ${M.b} 块通道 ${M.ch} 展开：行 = 某个词元处的输出，列 = 已经读过的某个词元的输入 u` })}</span>
		<svg width={16 + n * c} height={16 + n * c} role="img" aria-label={t({ en: 'look-back matrix', zh: '回看矩阵' })} onpointerleave={() => (hover = '')}>
			{#each M.toks as k, i (i)}
				<text x={16 + i * c + c / 2} y={11} text-anchor="middle" class="tk">{k}</text>
				<text x={8} y={16 + i * c + c - 2} text-anchor="middle" class="tk">{k}</text>
			{/each}
			{#each M.M ?? [] as row, r (r)}
				{#each row as v, s (s)}
					{#if s <= r}<rect x={16 + s * c} y={16 + r * c} width={c - 1} height={c - 1} fill={sc.css(v)} role="presentation" onpointerenter={() => (hover = t({ en: `output at token ${r} “${M.toks[r]}”, input of token ${s} “${M.toks[s]}”: ${fmt3(v)}`, zh: `第 ${r} 个词元“${M.toks[r]}”的输出、第 ${s} 个词元“${M.toks[s]}”的输入：${fmt3(v)}` }))}><title>{fmt3(v)}</title></rect>{/if}
				{/each}
			{/each}
		</svg>
		<p class="small">{hover || t({ en: 'Hover a cell to read its weight.', zh: '把指针移到格子上可以读出权重。' })}</p>
	</div>
	<div class="bars">
		<div><span>{t({ en: `This model’s state: ${fmtInt(M.state)} numbers (${M.L} blocks × ${M.D} channels × ${M.N} slots), for any length`, zh: `这个模型的状态：${fmtInt(M.state)} 个数（${M.L} 块 × ${M.D} 个通道 × ${M.N} 个槽），多长都一样` })}</span><i style:width={bar(M.state)} class="a"></i></div>
		<div><span>{t({ en: `Attention, same width: ${fmtInt(M.kv)} numbers for these ${M.n} tokens (${M.perToken} more per token)`, zh: `同样宽度的注意力：这 ${M.n} 个词元要 ${fmtInt(M.kv)} 个数（每多一个词元多 ${M.perToken} 个）` })}</span><i style:width={bar(M.kv)} class="b"></i></div>
		<p class="small">{t({ en: `At 1,000 tokens: still ${fmtInt(M.state)} against ${fmtInt(M.kv1000)}.`, zh: `到 1,000 个词元时：仍然是 ${fmtInt(M.state)}，对比 ${fmtInt(M.kv1000)}。` })}</p>
	</div>
</div>

<style>
	.mem { display: grid; grid-template-columns: minmax(0, 1fr); gap: 12px; min-width: 0; }
	.box { overflow-x: auto; min-width: 0; display: grid; gap: 4px; }
	svg { display: block; }
	.tk { font: 600 10px var(--font-mono, monospace); fill: var(--muted); }
	.lab, .small, .bars span { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); }
	.bars { display: grid; gap: 8px; }
	.bars div { display: grid; gap: 2px; }
	.bars i { display: block; height: 12px; border-radius: 2px; }
	.bars i.a { background: var(--accent); }
	.bars i.b { background: var(--muted); }
</style>

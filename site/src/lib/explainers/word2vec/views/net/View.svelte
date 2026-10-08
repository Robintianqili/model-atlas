<!-- word2vec 'net': one training step on one pair, one table row per word (see meta.ts). Center word: accent; true context
     word: outlined. Vector cells are colored by sign and size (blue negative, orange positive); hover for the value. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
	import { worked, workedWeight } from '../../vectors.ts';

	let P: StageProps = $props();
	const w = $derived(worked(P.rt.input as never, P.at));
	/** the update step's worked number (the narration's), outlined in the ΔW_in column */
	const ww = $derived(w ? workedWeight(w) : null);
	const show = $derived(String(P.props.show ?? 'row'));
	const cols = $derived(({ onehot: ['x'], row: ['x', 'win'], scores: ['wout', 'u'], softmax: ['u', 'p'], loss: ['p'], update: ['din', 'dout', 'u1'] } as Record<string, string[]>)[show] ?? ['x', 'win']);
	const has = (k: string) => cols.includes(k);
	const sub = (a: number[][] | undefined, b: number[][]) => (a ? a.map((r, i) => r.map((x, k) => x - b[i][k])) : b);
	const din = $derived(w ? sub(w.win1, w.win) : []);
	const dout = $derived(w ? sub(w.wout1, w.wout) : []);
	const u1 = $derived(w?.win1 && w.wout1 ? w.wout1.map((r) => r.reduce((s, x, k) => s + x * w.win1![w.c][k], 0)) : []);
	const scale = (m: number[][]) => Math.max(1e-12, ...m.flat().map(Math.abs));
	const fill = (v: number, s: number) => `color-mix(in srgb, var(${v >= 0 ? '--div-pos' : '--div-neg'}) ${Math.round(Math.min(1, Math.sqrt(Math.abs(v) / s)) * 85)}%, var(--surface))`;
	const par = (v: number) => (v < 0 ? `(${fmt3(v)})` : fmt3(v));
	const L = { x: 'x', win: 'W_in', wout: { en: 'output vector', zh: '输出向量' }, u: { en: 'score u', zh: '分数 u' }, p: { en: 'share p', zh: '份额 p' }, din: 'ΔW_in', dout: { en: 'Δ output vector', zh: 'Δ 输出向量' }, u1: { en: 'new score', zh: '新分数' } };
</script>

{#snippet cells(v: number[], s: number, hi: number)}<span class="vec">{#each v as x, k (k)}<span class="cell" class:hi={k === hi} style:background={fill(x, s)} title={fmt3(x)}></span>{/each}</span>{/snippet}

<div class="net" style:min-height="360px">
	{#if !w}
		<p class="muted">{t({ en: 'Waiting for the first training step…', zh: '等待第一步训练…' })}</p>
	{:else}
		{@const C = w.words[w.c]}
		{@const O = w.words[w.o]}
		<p class="head small">
			<span class="tag c">{t({ en: 'center', zh: '中心词' })} {C}</span> <span class="tag o">{t({ en: 'context', zh: '上下文词' })} {O}</span>
			{#if show === 'row' || show === 'scores'}<span class="num">h = ({w.h.map((x) => fmt3(x)).join(', ')})</span>{/if}
			{#if show === 'scores'}<span class="num">u({O}) = {w.h.map((x, k) => `${par(x)}×${par(w.wout[w.o][k])}`).join(' + ')} = {fmt3(w.u[w.o])}</span>{/if}
			{#if show === 'softmax'}<span class="num">p = e<sup>u</sup> / Σ e<sup>u</sup>, Σ e<sup>u</sup> = {fmt3(w.u.reduce((s, x) => s + Math.exp(x), 0))}</span>{/if}
			{#if show === 'loss'}<span class="num">{t({ en: 'loss', zh: '损失' })} = −ln p({O}) = −ln {fmt3(w.p[w.o], 4)} = {fmt3(w.loss, 4)}</span>{/if}
			{#if show === 'update'}<span class="num">{t({ en: 'learning rate', zh: '学习率' })} {w.lr} · {t({ en: 'loss on this pair', zh: '这一对的损失' })} {fmt3(w.loss, 4)} → {fmt3(w.loss1 ?? NaN, 4)}</span>{/if}
			{#if show === 'update' && ww}<span class="num">{t({ en: `${C}’s row, number ${ww.k}:`, zh: `${C} 那一行第 ${ww.k} 个数：` })} {fmt3(ww.old)} − {w.lr} × {par(ww.g)} = {fmt3(ww.new)}</span><span class="num">{t({ en: 'score', zh: '分数' })} u({O}): {fmt3(w.u[w.o])} → {fmt3(w.u1 ?? NaN)}</span>{/if}
		</p>
		<div class="scroll" use:keyScroll>
			<table class="num-table">
				<thead><tr><th class="id">{t({ en: 'id', zh: '编号' })}</th><th>{t({ en: 'word', zh: '词' })}</th>{#each cols as k (k)}<th>{t(L[k as keyof typeof L])}</th>{/each}</tr></thead>
				<tbody>
					{#each w.words.slice(0, w.V) as word, j (j)}
						<tr class:c={j === w.c} class:o={j === w.o}>
							<td class="muted id">{j}</td><td class="word">{word}</td>
							{#if has('x')}<td class="num" class:one={j === w.c}>{j === w.c ? 1 : 0}</td>{/if}
							{#if has('win')}<td>{@render cells(w.win[j], scale(w.win), -1)}</td>{/if}
							{#if has('wout')}<td>{@render cells(w.wout[j], scale(w.wout), -1)}</td>{/if}
							{#if has('u')}<td class="num">{fmt3(w.u[j])}</td>{/if}
							{#if has('p')}<td class="num"><span class="bar" style:width="{Math.round(Math.min(120, w.p[j] * 400))}px"></span> {fmtPct(w.p[j])}</td>{/if}
							{#if has('din')}<td>{@render cells(din[j], scale(din), j === w.c ? (ww?.k ?? -1) : -1)}</td>{/if}
							{#if has('dout')}<td>{@render cells(dout[j], scale(dout), -1)}</td>{/if}
							{#if has('u1')}<td class="num">{fmt3(u1[j])}</td>{/if}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>

<style>
	.net { display: grid; gap: 6px; min-width: 0; align-content: start; } .head { display: flex; flex-wrap: wrap; gap: 4px 10px; margin: 0; align-items: center; }
	.tag { padding: 0 6px; border-radius: var(--r1); font-weight: 600; } .tag.c { background: var(--accent-soft); } .tag.o { outline: 2px solid var(--highlight); }
	.scroll { overflow-x: auto; max-height: 640px; overflow-y: auto; } table { border-collapse: collapse; font-size: var(--fs-sm); } .small { font-size: var(--fs-sm); } .one { font-weight: 700; color: var(--accent-strong); }
	th, td { padding: 1px 6px; text-align: left; white-space: nowrap; } th { position: sticky; top: 0; background: var(--surface); font-weight: 600; }
	tr.c td { background: var(--accent-soft); } tr.o td { box-shadow: inset 0 2px 0 var(--highlight), inset 0 -2px 0 var(--highlight); } .word { font-weight: 600; }
	.vec { display: inline-flex; gap: 1px; } .cell { width: 15px; height: 15px; border: 1px solid var(--rule); border-radius: 2px; } .cell.hi { outline: 2px solid var(--highlight); outline-offset: 1px; }
	.bar { display: inline-block; height: 10px; background: var(--seq-mid); border-radius: 2px; vertical-align: middle; }
	/* phones keep the id column: the narration reads ids ("a 1 at id 4", "row 4") */
	@media (max-width: 560px) { th, td { padding: 1px 3px; } th { white-space: normal; word-break: keep-all; vertical-align: bottom; } th.id { white-space: nowrap; } .cell { width: 12px; height: 12px; } }
</style>

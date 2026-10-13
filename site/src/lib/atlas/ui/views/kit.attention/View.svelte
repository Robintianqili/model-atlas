<!--
  kit.attention: the kit's AttentionMatrix on an [H, Tq, Tk] layer of attention weights, one head at a time, with the
  tokens as row and column labels, the causal mask hatched, and the query row of the stage's unit (or props.query)
  outlined. Under it, the query's row as words: where this token looks, largest weight first. props: layer, head, query.
  K4: props.tokens names the queries and props.keys the keys (cross-attention: an encoder-decoder's decoder positions
  look at the encoder's), each a list of strings or { en, zh } names.
-->
<script lang="ts">
	import AttentionMatrix from '$lib/kit/AttentionMatrix.svelte';
	import { fmtPct } from '../../../core/format.ts';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../types/view.ts';
	import Notice from '../kit.parts/Notice.svelte';
	import { keyTexts, stageModel, stageTrace, tokenIds, tokenTexts } from '../kit.parts/stage.ts';
	import { attnOf } from './attn.ts';

	let P: StageProps = $props();
	let headPick = $derived.by<number | null>(() => {
		void P.at;
		return null;
	});
	let row = $state<number | null>(null);
	let col = $state<number | null>(null);
	const A = $derived(attnOf(P, headPick));
	const ids = $derived(tokenIds(stageTrace(P), stageModel(P)));
	const tokens = $derived(A ? tokenTexts(P, A.tq, ids, i18n.lang) : []);
	const keys = $derived(A ? keyTexts(P, A.tk, ids, i18n.lang) : []);
	const q = $derived(A ? Math.max(0, Math.min(A.tq - 1, row ?? A.query)) : 0);
	const shownRow = $derived(A ? q : null);
	const ranked = $derived.by(() => {
		if (!A) return [];
		const r = A.w.subarray(q * A.tk, (q + 1) * A.tk);
		return Array.from(r, (p, j) => ({ j, p })).filter((x) => x.p > 0).sort((a, b) => b.p - a.p).slice(0, 5);
	});
	$effect(() => {
		if (P.autoplay && A) P.onend();
	});
</script>

{#if !A}
	<Notice text={{ en: 'No attention weights to show: the stage needs an [H, Tq, Tk] layer.', zh: '没有可显示的注意力权重：需要一个 [H, Tq, Tk] 的层。' }} />
{:else}
	<div class="at">
		{#if A.heads > 1}
			<div class="segmented" role="group" aria-label={t({ en: 'Head', zh: '头' })}>
				{#each Array.from({ length: A.heads }, (_, h) => h) as h (h)}
					<button type="button" aria-pressed={h === A.head} onclick={() => (headPick = h)}>{t({ en: `head ${h}`, zh: `头 ${h}` })}</button>
				{/each}
			</div>
		{/if}
		<AttentionMatrix weights={A.w} queries={tokens} keys={keys} causal={A.causal} bind:row={() => shownRow, (v) => (row = v)} bind:col maxCell={30} label={{ en: `${A.label.en}, head ${A.head}`, zh: `${A.label.zh ?? A.label.en}，头 ${A.head}` }} />
		<p class="row">
			<span class="muted">{t({ en: `Query “${tokens[q]}” looks at:`, zh: `查询“${tokens[q]}”关注：` })}</span>
			{#each ranked as r (r.j)}<span class="k"><span class="tok">{keys[r.j]}</span> <span class="num">{fmtPct(r.p)}</span></span>{/each}
		</p>
	</div>
{/if}

<style>
	.at {
		display: grid;
		gap: 10px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.segmented {
		justify-self: start;
		flex-wrap: wrap;
	}
	.row {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 12px;
		font-size: var(--fs-sm);
		align-items: baseline;
	}
	.k {
		white-space: nowrap;
	}
	.tok {
		font-family: var(--font-mono);
		padding: 0 4px;
		border-radius: 3px;
		background: var(--surface-2);
		box-shadow: inset 0 0 0 1.5px var(--k);
		white-space: pre;
	}
</style>

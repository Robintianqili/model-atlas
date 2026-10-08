<!-- word2vec 'window': the sentences as word chips (id under each word); at a 'center' frame the center word is filled,
     the words of its window outlined, and its pairs listed (see meta.ts). -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { corpusOf } from '../../corpus.ts';
	import { pairsAt } from './meta.ts';

	let P: StageProps = $props();
	const c = $derived(corpusOf(P.rt.input as never));
	const m = $derived(P.at.frame?.phase === 'center' && P.props.mode !== 'vocab' ? P.at.frame.metrics : undefined);
	const inWin = (s: number, i: number) => !!m && s === m.s && i !== m.i && Math.abs(i - m.i) <= c.window;
</script>

<div class="win" style:min-height="260px">
	<div class="text" use:keyScroll aria-label={t({ en: 'Your sentences, word by word', zh: '你的句子，逐词显示' })}>
		{#each c.sentences as ids, s (s)}
			<span class="sent" class:cur={m?.s === s}>
				{#each ids as id, i (i)}<span class="w" class:center={m?.s === s && m?.i === i} class:ctx={inWin(s, i)}>{c.words[id]}<sub>{id}</sub></span>{/each}
			</span>
		{/each}
	</div>
	{#if m}
		<p class="pairs small"><strong>{t({ en: 'Pairs from this center word:', zh: '这个中心词给出的训练对：' })}</strong> {#each pairsAt(c, m.s, m.i) as [a, b], k (k)}<span class="pair">({a}, {b})</span>{/each}</p>
		<p class="small muted">{t({ en: `window ${c.window} · ${m.n} of ${c.pairs.length} pairs so far`, zh: `窗口 ${c.window} · 到目前为止 ${m.n} / ${c.pairs.length} 对` })}</p>
	{:else}
		<p class="vocab small" aria-label={t({ en: 'The vocabulary: id, word, count', zh: '词表：编号、词、次数' })}>
			{#each c.words as w, i (i)}<span><b class="num">{i}</b> {w} <span class="muted">×{c.counts[i]}</span></span>{/each}
		</p>
		{#if c.dropped.length}<p class="small muted">{t({ en: `${c.dropped.length} rarer words left out`, zh: `去掉了 ${c.dropped.length} 个更少见的词` })}</p>{/if}
	{/if}
</div>

<style>
	.win { display: grid; gap: 8px; min-width: 0; align-content: start; } p { margin: 0; } .small { font-size: var(--fs-sm); }
	.text { display: flex; flex-wrap: wrap; gap: 6px 12px; line-height: 1.2; } .sent { display: inline-flex; flex-wrap: wrap; gap: 2px; padding: 2px; border-radius: var(--r1); }
	.sent.cur { background: var(--surface-2); } .w { display: inline-flex; flex-direction: column; align-items: center; padding: 1px 4px; border: 1px solid transparent; border-radius: var(--r1); font-size: var(--fs-sm); }
	.w sub { font-size: 10px; color: var(--muted); } .w.center { background: var(--accent-soft); border-color: var(--accent); font-weight: 700; } .w.ctx { border-color: var(--highlight); border-style: dashed; }
	.pairs { display: flex; flex-wrap: wrap; gap: 4px 8px; } .pair { font-family: var(--font-mono); } .vocab { display: flex; flex-wrap: wrap; gap: 2px 12px; }
</style>

<!-- tiny-encdec 'cross' (meta.ts): what the model wrote (and the right answer), the head buttons, then the cross-attention
     matrix (kit AttentionMatrix: rows = output symbols, columns = input characters; props.row outlined). props.hide: the
     matrix stays hidden while the stage is inert (an open predict question). props.tries: buttons with dates in formats
     the model never saw (setInput). props.gru: the measured comparison with tinyseq2seq's GRU, from facts.json.
     props.play (on the 'decode' process, step play 'view'): a player walks the process frames; it starts on the first
     frame when it autoplays and on the last one otherwise (reduced motion). Pattern: tinyseq2seq's 'align' view. -->
<script lang="ts">
	import AttentionMatrix from '$lib/kit/AttentionMatrix.svelte';
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import { base } from '$app/paths';
	import { atFor } from '../../../../atlas/runtime/sources.ts';
	import { frameLabel } from '../../../../atlas/ui/scrub.ts';
	import { fmtPct } from '../../../../atlas/core/format.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { crossOf, TRIES, type Pick } from './meta.ts';
	import { verdict } from '../../look.ts';

	let P: StageProps = $props();
	const tape = $derived(P.props.play ? P.at.tape : undefined);
	let k = $state(0);
	let pick = $state<Pick | null>(null);
	$effect(() => void (k = P.autoplay ? 0 : Math.max(0, (tape?.length ?? 1) - 1)));
	const want = $derived<Pick>(pick ?? (typeof P.props.head === 'number' ? P.props.head : 'avg'));
	const A = $derived(crossOf(tape ? { ...P, at: atFor(P.rt, P.source, Math.min(k, tape.length - 1)) } : P, want));
	const set = (text: string) => (P.setInput ?? P.rt.setInput)(text);
	const f = (key: string) => fmtPct(Number(P.rt.fact(`fact:${key}`)));
	const GRU = [
		[{ en: 'year first', zh: '年份在前' }, 'YearFirst'],
		[{ en: 'dashes', zh: '短横线' }, 'Dashes'],
		[{ en: 'no comma', zh: '没有逗号' }, 'NoComma'],
		[{ en: 'day as a word', zh: '日写成单词' }, 'Spelled'],
		[{ en: 'no such day: digits copied', zh: '不存在的日子：照抄数字' }, 'Impossible']
	] as const;
</script>

{#snippet player(tp: NonNullable<typeof tape>)}
	<StepPlayer count={tp.length} bind:step={k} interval={450} autoplay={P.autoplay} onend={P.onend} labels={Array.from({ length: tp.length }, (_, i) => t(frameLabel(tp, i, { unit: { en: 'decoder step', zh: '解码步' } })))} counter={{ en: 'Frame {i} of {n}', zh: '第 {i} / {n} 帧' }} />
{/snippet}

<div class="cx">
	<div class:ans={P.props.hide}>
		<p class="res">
			<span class="lab">{t(A.finished ? { en: 'The model wrote', zh: '模型写下' } : { en: 'Written so far', zh: '已经写下' })}</span>
			<b class="num out">{A.wrote || '—'}</b>
			{#if A.finished}<span class={A.right ? 'ok' : 'bad'}>{A.right ? '✓' : A.want ? '✗' : ''}{t(verdict(P.rt)).replace(/^[,，]\s*/, ' ')}</span>{/if}
		</p>
	</div>
	{#if P.props.tries}
		<div class="row" role="group" aria-label={t({ en: 'Formats it never saw', zh: '它没见过的格式' })}>
			{#each TRIES as x (x)}<button type="button" class="btn" aria-pressed={A.text === x} onclick={() => set(x)}>{x}</button>{/each}
		</div>
	{/if}
	{#if A.H > 1}
		<div class="segmented" role="group" aria-label={t({ en: 'Which head', zh: '哪个头' })}>
			{#each ['avg', ...Array.from({ length: A.H }, (_, h) => h)] as h (h)}
				<button type="button" aria-pressed={A.h === h} onclick={() => (pick = h as Pick)}>{t(h === 'avg' ? { en: 'both heads, averaged', zh: '两个头取平均' } : { en: `head ${h}`, zh: `头 ${h}` })}</button>
			{/each}
		</div>
	{/if}
	{#if tape && tape.length > 1 && !P.ctx?.reduced}{@render player(tape)}{/if}
	<div class:ans={P.props.hide}>
		{#if A.R > 0}
			<AttentionMatrix weights={A.weights} queries={A.rows} keys={A.keys} row={typeof P.props.row === 'number' && P.props.row < A.R ? P.props.row : null} maxCell={30} label={{ en: 'Rows: symbols written · columns: your characters', zh: '行：写下的符号 · 列：你的字符' }} />
		{:else}
			<p class="lab">{t({ en: 'No row yet: the decoder has not written anything.', zh: '还没有行：解码器还没写任何东西。' })}</p>
		{/if}
	</div>
	{#if tape && tape.length > 1 && P.ctx?.reduced}{@render player(tape)}{/if}
	{#if P.props.gru}
		<table class="gru">
			<caption>{t({ en: 'Right answers on 1,000 dates per format it never saw', zh: '每种没见过的格式各 1,000 个日期上的答对率' })}</caption>
			<thead><tr><th scope="col">{t({ en: 'Format', zh: '格式' })}</th><th scope="col">{t({ en: 'This Transformer', zh: '本 Transformer' })}</th><th scope="col"><a href="{base}/m/tinyseq2seq/">{t({ en: 'GRU + attention', zh: 'GRU + 注意力' })}</a></th></tr></thead>
			<tbody>
				{#each GRU as [name, key] (key)}<tr><th scope="row">{t(name)}</th><td class="num">{f(`off${key}`)}</td><td class="num">{f(`gruOff${key}`)}</td></tr>{/each}
			</tbody>
		</table>
	{/if}
</div>

<style>
	.cx { display: grid; gap: 10px; min-width: 0; }
	.res { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: baseline; margin: 0; }
	.lab { font-size: var(--fs-sm); color: var(--ink-2); margin: 0; }
	.out { font-size: var(--fs-lg); letter-spacing: 0.06em; }
	.ok { color: var(--live); font-weight: 600; }
	.bad { color: var(--ink-2); font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); font-family: var(--font-mono); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	.segmented { justify-self: start; flex-wrap: wrap; }
	.gru { border-collapse: collapse; font-size: var(--fs-sm); justify-self: start; }
	.gru caption { text-align: left; color: var(--ink-2); padding-bottom: 4px; }
	.gru th, .gru td { padding: 2px 10px 2px 0; text-align: left; border-bottom: 1px solid var(--line); }
	.gru td { text-align: right; }
	.gru thead a { white-space: nowrap; } /* 390 px zh: never break 注意力 in the middle */
	:global([inert]) .ans { visibility: hidden; }
	/* one character per column: upright labels (the kit tilts them for word tokens) */
	.cx :global(.am .cl > span) { transform: translateX(-50%); font-size: min(15px, calc(var(--cell) * 0.75)); font-weight: 600; }
</style>

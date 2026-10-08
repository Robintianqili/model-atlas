<!-- tinyseq2seq 'align' (meta.ts): what the model wrote (and the right answer), then the alignment matrix (kit
     AttentionMatrix: rows = output symbols, columns = input characters). props.tries: buttons with dates in formats the
     model never saw, each typed into the input with setInput. props.play (on the 'decode' process, step play 'view'):
     a player walks the process frames; it starts on the first frame when it autoplays and on the last one otherwise
     (reduced motion), so the narration's whole answer is on the stage without animation. Its controls sit above the
     growing matrix while it plays, and below it (with the long frame list) under reduced motion. -->
<script lang="ts">
	import AttentionMatrix from '$lib/kit/AttentionMatrix.svelte';
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import { atFor } from '../../../../atlas/runtime/sources.ts';
	import { frameLabel } from '../../../../atlas/ui/scrub.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { alignOf, TRIES } from './meta.ts';
	import { verdict } from '../../look.ts';

	let P: StageProps = $props();
	const tape = $derived(P.props.play ? P.at.tape : undefined);
	let k = $state(0);
	$effect(() => void (k = P.autoplay ? 0 : Math.max(0, (tape?.length ?? 1) - 1)));
	const A = $derived(alignOf(tape ? { ...P, at: atFor(P.rt, P.source, Math.min(k, tape.length - 1)) } : P));
	const set = (text: string) => (P.setInput ?? P.rt.setInput)(text);
</script>

{#snippet player(tp: NonNullable<typeof tape>)}
	<StepPlayer count={tp.length} bind:step={k} interval={450} autoplay={P.autoplay} onend={P.onend} labels={Array.from({ length: tp.length }, (_, i) => t(frameLabel(tp, i, { unit: { en: 'decoder step', zh: '解码步' } })))} counter={{ en: 'Frame {i} of {n}', zh: '第 {i} / {n} 帧' }} />
{/snippet}

<div class="al">
	<p class="res">
		<span class="lab">{t(A.finished ? { en: 'The model wrote', zh: '模型写下' } : { en: 'Written so far', zh: '已经写下' })}</span>
		<b class="num out">{A.wrote || '—'}</b>
		{#if A.finished}
			<span class={A.right ? 'ok' : 'bad'}>{A.right ? '✓' : A.want ? '✗' : ''}{t(verdict(P.rt)).replace(/^[,，]\s*/, ' ')}</span>
		{/if}
	</p>
	{#if P.props.tries}
		<div class="row" role="group" aria-label={t({ en: 'Formats it never saw', zh: '它没见过的格式' })}>
			{#each TRIES as x (x)}<button type="button" class="btn" aria-pressed={A.text === x} onclick={() => set(x)}>{x}</button>{/each}
		</div>
	{/if}
	{#if tape && tape.length > 1 && !P.ctx?.reduced}{@render player(tape)}{/if}
	{#if A.R > 0}
		<AttentionMatrix weights={A.weights} queries={A.rows} keys={A.keys} maxCell={30} label={{ en: 'Rows: output symbols · columns: your characters', zh: '行：输出符号 · 列：你的字符' }} />
	{:else}
		<p class="small">{t({ en: 'No row yet: the decoder has not written anything.', zh: '还没有行：解码器还没写任何东西。' })}</p>
	{/if}
	{#if tape && tape.length > 1 && P.ctx?.reduced}{@render player(tape)}{/if}
</div>

<style>
	.al { display: grid; gap: 10px; min-width: 0; }
	.res { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: baseline; margin: 0; }
	.lab, .small { font-size: var(--fs-sm); color: var(--ink-2); margin: 0; }
	.out { font-size: var(--fs-lg); letter-spacing: 0.06em; }
	.ok { color: var(--live); font-weight: 600; }
	.bad { color: var(--ink-2); font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); font-family: var(--font-mono); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	/* one character per column: upright labels (the kit tilts them for word tokens) */
	.al :global(.am .cl > span) { transform: translateX(-50%); font-size: min(15px, calc(var(--cell) * 0.75)); font-weight: 600; }
</style>

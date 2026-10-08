<!--
  word2vec Capture (local adapter 'corpus'): the learner's mini-corpus (presets from TinyStories, or their own sentences)
  and the window size. Reports { text, window }; typing is a live change (the page coalesces it per frame).
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/index.ts';
	import { corpusOf, textOf, windowOf, WINDOWS, type CorpusRaw } from '../../corpus.ts';

	let { raw, presets = [], onchange, compact = false }: CaptureProps<CorpusRaw> = $props();
	const cur = $derived(textOf(raw));
	const win = $derived(windowOf(raw));
	const c = $derived(corpusOf(raw));
	let text = $state('');
	$effect(() => {
		const v = cur;
		untrack(() => v !== text && (text = v));
	});
	const set = (p: { text?: string; window?: number }, live = false) => onchange?.({ text: p.text ?? cur, window: p.window ?? win }, { live });
</script>

<div class="cap">
	<div class="row" role="group" aria-label={t({ en: 'Example corpora', zh: '示例语料' })}>
		{#each presets as p (p.id)}<button type="button" class="btn" aria-pressed={textOf(p.raw) === cur} onclick={() => set({ text: textOf(p.raw) })}>{t(p.label)}</button>{/each}
	</div>
	<TextInputBar bind:value={text} maxLength={800} multiline rows={compact ? 3 : 5} label={t({ en: 'Your sentences', zh: '你的句子' })} oninput={(v) => set({ text: v }, true)} />
	<div class="row" role="group" aria-label={t({ en: 'Window size', zh: '窗口大小' })}>
		<span>{t({ en: 'Window: words on each side', zh: '窗口：中心词每侧的词数' })}</span>
		{#each WINDOWS as w (w)}<button type="button" class="btn" aria-pressed={w === win} onclick={() => set({ window: w })}>{w}</button>{/each}
	</div>
	<p class="small muted">{t({ en: `${c.words.length} different words · ${c.pairs.length} (center, context) pairs`, zh: `${c.words.length} 个不同的词 · ${c.pairs.length} 个（中心词，上下文词）对` })}</p>
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); } .row { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); } .btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	p { margin: 0; }
</style>

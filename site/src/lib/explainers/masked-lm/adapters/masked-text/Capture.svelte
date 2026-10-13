<!--
  masked-lm Capture (local adapter 'masked-text', adapted from tinybert's, credit: the tinybert package): the learner's
  text, the example sentences, and the text's tokens as chips; a click on a chip hides that token ([MASK]) or, on a chosen
  one, gives it back. Typed text is encoded with TinyGPT's tokenizer, fetched read-only from TinyGPT's own folder
  (static/models/tinygpt/tokenizer.json; this concept package has no copy); the word nearest the middle is hidden until
  the learner picks tokens.
-->
<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { base } from '$app/paths';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/index.ts';
	import { tokenizerOf } from '../../../../atlas/adapters/text-bpe/adapter.ts';
	import { contentOf, fatesOf, toggled, vocab, type MaskedRaw } from './adapter.ts';

	let { raw, presets = [], onchange, compact = false }: CaptureProps<MaskedRaw> = $props();
	let tokJson = $state<unknown>(null);
	onMount(() => {
		fetch(`${base}/models/tinygpt/tokenizer.json`)
			.then((r) => (r.ok ? r.json() : null))
			.then((j) => (tokJson = j))
			.catch(() => {});
	});
	const tok = $derived(tokenizerOf(tokJson));
	let text = $state('');
	$effect(() => {
		const v = raw?.text;
		untrack(() => {
			if (typeof v === 'string' && v !== text) text = v;
		});
	});
	$effect(() => {
		const v = text;
		const k = tok;
		untrack(() => {
			if (k && v !== (raw?.text ?? '')) onchange?.({ text: v, ids: k.encode(v) }, { live: true });
		});
	});
	const ids = $derived(contentOf(raw));
	const F = $derived(fatesOf(raw));
	const chips = $derived(ids.map((_, i) => ({ text: vocab(F.shown[i]), id: F.shown[i], special: F.fate[i] === 1, tone: F.fate[i] === 2 ? ('generated' as const) : undefined })));
	const chosen = $derived(F.fate.flatMap((f, i) => (f > 0 ? [i] : [])));
</script>

<div class="cap" class:compact>
	<TextInputBar bind:value={text} maxLength={compact ? 160 : 300} multiline={!compact} rows={2} label={t({ en: 'Your text', zh: '你的文字' })} />
	{#if !compact && presets.length}
		<div class="row" role="group" aria-label={t({ en: 'Example sentences', zh: '例句' })}>
			{#each presets as p (p.id)}<button type="button" class="btn" aria-pressed={p.raw?.text === raw?.text} onclick={() => onchange?.(p.raw)}>{t(p.label)}</button>{/each}
		</div>
	{/if}
	{#if chips.length}
		<TokenChips tokens={chips} selected={null} marked={chosen} showIds={!compact} label={{ en: 'Click a token to hide it, or to give a hidden one back', zh: '点一个词元把它藏起来，再点一次还原' }} onselect={(i) => onchange?.(toggled(raw, i))} />
	{/if}
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
</style>

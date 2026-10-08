<!--
  tinybert Capture (local adapter 'masked-text'): the learner's text, the preset sentences, and the text's tokens as
  chips; a click on a chip hides that token. Typed text is encoded with TinyGPT's tokenizer (tokenizer.json, the same
  file TinyGPT uses), and the word nearest the middle is hidden until the learner picks one.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/index.ts';
	import { tokenizerOf } from '../../../../atlas/adapters/text-bpe/adapter.ts';
	import { contentOf, maskOf, vocabOf, type MaskedRaw, type Spec } from './adapter.ts';

	let { rt, raw, presets = [], onchange, compact = false }: CaptureProps<MaskedRaw> = $props();
	const spec = $derived((rt?.spec ?? null) as Spec);
	const tok = $derived(tokenizerOf((rt as { assets?: Record<string, unknown> } | undefined)?.assets?.['tokenizer.json']));
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
	const m = $derived(maskOf(raw, spec));
	const chips = $derived(ids.map((id, i) => (i === m ? { text: '[MASK]', id: 1024, special: true } : { text: vocabOf(spec)(id), id })));
</script>

<div class="cap" class:compact>
	<TextInputBar bind:value={text} maxLength={compact ? 160 : 300} multiline={!compact} rows={2} label={t({ en: 'Your text', zh: '你的文字' })} />
	{#if !compact && presets.length}
		<div class="row" role="group" aria-label={t({ en: 'Example sentences', zh: '例句' })}>
			{#each presets as p (p.id)}<button type="button" class="btn" aria-pressed={p.raw?.text === raw?.text} onclick={() => onchange?.(p.raw)}>{t(p.label)}</button>{/each}
		</div>
	{/if}
	{#if chips.length}
		<TokenChips tokens={chips} selected={m} showIds={!compact} label={{ en: 'Click a token to hide it', zh: '点一个词元把它藏起来' }} onselect={(i) => onchange?.({ ...(raw ?? {}), mask: i, picked: true })} />
	{/if}
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
</style>

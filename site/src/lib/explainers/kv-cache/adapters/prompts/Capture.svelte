<!--
  kv-cache Capture (local adapter 'prompts'): prompt A (TinyGPT continues it) and prompt B (a second request), the preset
  pairs, and both prompts as token chips with their shared prefix marked. Typed text is encoded with TinyGPT's tokenizer,
  fetched read-only from TinyGPT's own folder (static/models/tinygpt/tokenizer.json), as self-supervised-objectives does.
-->
<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { base } from '$app/paths';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/index.ts';
	import { tokenizerOf } from '../../../../atlas/adapters/text-bpe/adapter.ts';
	import { promptA, promptB, sharedPrefix, type PromptsRaw } from '../../data.ts';
	import { VOCAB } from '../../vocab.ts';

	let { raw, presets = [], onchange, compact = false }: CaptureProps<PromptsRaw> = $props();
	let tokJson = $state<unknown>(null);
	onMount(() => {
		fetch(`${base}/models/tinygpt/tokenizer.json`)
			.then((r) => (r.ok ? r.json() : null))
			.then((j) => (tokJson = j))
			.catch(() => {});
	});
	const tok = $derived(tokenizerOf(tokJson));
	let textA = $state('');
	let textB = $state('');
	$effect(() => {
		const a = raw?.text;
		const b = raw?.text2;
		untrack(() => {
			if (typeof a === 'string' && a !== textA) textA = a;
			if (typeof b === 'string' && b !== textB) textB = b;
		});
	});
	$effect(() => {
		const a = textA;
		const b = textB;
		const k = tok;
		untrack(() => {
			if (k && (a !== (raw?.text ?? '') || b !== (raw?.text2 ?? ''))) onchange?.({ text: a, ids: k.encode(a), text2: b, ids2: k.encode(b) }, { live: true });
		});
	});
	const A = $derived(promptA(raw));
	const B = $derived(promptB(raw));
	const s = $derived(sharedPrefix(A, B));
	const chips = (ids: number[]) => ids.map((id) => ({ text: VOCAB[id] ?? `#${id}`, id, special: id === 1023 }));
	const shared = $derived(Array.from({ length: s }, (_, i) => i));
</script>

<div class="cap" class:compact>
	<TextInputBar bind:value={textA} maxLength={compact ? 160 : 240} label={t({ en: 'Prompt A (TinyGPT continues it)', zh: '提示词 A（TinyGPT 接着往下写）' })} />
	<TextInputBar bind:value={textB} maxLength={compact ? 160 : 240} label={t({ en: 'Prompt B (a second request)', zh: '提示词 B（第二个请求）' })} />
	{#if !compact && presets.length}
		<div class="row" role="group" aria-label={t({ en: 'Example prompt pairs', zh: '示例提示词对' })}>
			{#each presets as p (p.id)}<button type="button" class="btn" aria-pressed={p.raw?.text === raw?.text && p.raw?.text2 === raw?.text2} onclick={() => onchange?.(p.raw)}>{t(p.label)}</button>{/each}
		</div>
	{/if}
	{#if !compact}
		<p class="note">{t({ en: `Shared prefix: ${s} tokens (marked)`, zh: `共享前缀：${s} 个词元（已标出）` })}</p>
		<TokenChips tokens={chips(A)} selectable={false} marked={shared} label={{ en: 'Prompt A as tokens', zh: '提示词 A 的词元' }} />
		<TokenChips tokens={chips(B)} selectable={false} marked={shared} label={{ en: 'Prompt B as tokens', zh: '提示词 B 的词元' }} />
	{/if}
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	.note { margin: 0; font-size: var(--fs-xs); color: var(--muted); }
</style>

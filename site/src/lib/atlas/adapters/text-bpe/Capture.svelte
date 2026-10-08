<!--
  text-bpe Capture: a text field (kit TextInputBar) and the tokens it becomes (kit TokenChips). It tokenizes with the
  model's own tokenizer.json (from the runtime's auto assets or adapter options, else fetched once from the asset folder)
  and reports { text, ids }, so the ids that run are exactly the ids shown.
-->
<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import type { TokenItem } from '$lib/kit/types';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../ui/capture.ts';
	import { say } from '../../ui/strings.ts';
	import { tokenizerOf } from './adapter.ts';
	import type { BPETokenizer } from '../../../models/bpe.ts';

	let { rt, raw, presets, onchange, compact = false }: CaptureProps = $props();

	let tok = $state.raw<BPETokenizer | null>(null);
	let text = $state('');
	const current = $derived(raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as { text?: string; ids?: number[] }) : typeof raw === 'string' ? { text: raw } : null);
	$effect(() => {
		const c = current?.text;
		untrack(() => {
			if (typeof c === 'string' && c !== text) text = c;
		});
	});
	onMount(() => {
		const opt = rt.pkg.manifest.input?.options?.tokenizer;
		const local = tokenizerOf(rt.assets['tokenizer.json'] ?? opt);
		if (local) {
			tok = local;
			return;
		}
		fetch(`${rt.dir}/tokenizer.json`)
			.then((r) => (r.ok ? r.json() : null))
			.then((j) => (tok = tokenizerOf(j)))
			.catch(() => (tok = null));
	});
	// typing and the TextInputBar presets both change `text`: report it when it differs from the shared input
	$effect(() => {
		const v = text;
		const k = tok;
		untrack(() => {
			if (k && v !== (current?.text ?? '')) onchange({ text: v, ids: k.encode(v) }, { live: true });
		});
	});
	const ids = $derived(current?.ids ?? (tok && current?.text ? tok.encode(current.text) : []));
	const chips = $derived<TokenItem[]>(tok ? ids.slice(0, compact ? 24 : 96).map((id) => ({ text: tok!.tokenToDisplay(id, { visibleWhitespace: true }), id })) : []);
	const presetTexts = $derived(presets.map((p) => ({ text: (p.raw as { text?: string })?.text ?? '', label: p.label })).filter((p) => p.text));
</script>

<div class="cap" class:compact>
	<TextInputBar bind:value={text} maxLength={compact ? 160 : 400} multiline={!compact} rows={2} label={t({ en: 'Your text', zh: '你的文字' })} presets={compact ? [] : presetTexts} />
	{#if chips.length}
		<TokenChips tokens={chips} showIds={!compact} selectable={false} label={say('whatModelSees')} />
	{/if}
</div>

<style>
	.cap {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
</style>

<!--
  text-char Capture: a text field (kit TextInputBar) and its characters as chips (unknown characters are marked by the
  adapter's note under the input). Reports the text itself; the adapter maps characters to ids with the model's alphabet.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import type { TokenItem } from '$lib/kit/types';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../ui/capture.ts';
	import { say } from '../../ui/strings.ts';

	let { raw, presets, onchange, compact = false }: CaptureProps = $props();

	const current = $derived(typeof raw === 'string' ? raw : raw && typeof raw === 'object' ? ((raw as { text?: string }).text ?? '') : '');
	let text = $state('');
	$effect(() => {
		const c = current;
		untrack(() => {
			if (c !== text) text = c;
		});
	});
	$effect(() => {
		const v = text;
		untrack(() => {
			if (v !== current) onchange(v, { live: true });
		});
	});
	const show = (c: string) => (c === ' ' ? '␣' : c === '\n' ? '↵' : c === '\t' ? '⇥' : c);
	const chips = $derived<TokenItem[]>([...current].slice(0, compact ? 40 : 160).map((c) => ({ text: show(c) })));
	const presetTexts = $derived(presets.map((p) => ({ text: typeof p.raw === 'string' ? p.raw : ((p.raw as { text?: string })?.text ?? ''), label: p.label })).filter((p) => p.text));
</script>

<div class="cap">
	<TextInputBar bind:value={text} maxLength={compact ? 160 : 400} multiline={!compact} rows={2} label={t({ en: 'Your text', zh: '你的文字' })} presets={compact ? [] : presetTexts} />
	{#if chips.length}
		<TokenChips tokens={chips} selectable={false} label={say('whatModelSees')} />
	{/if}
</div>

<style>
	.cap {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
</style>

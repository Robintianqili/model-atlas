<!--
  positional-encodings Capture (local adapter 'pe-text'; the text part copied from self-supervised-objectives'
  masked-text): the learner's text and example sentences, its tokens as chips, a Shuffle button (a new seed for the
  shuffled order) and two sliders, the key's position p and the distance k to the query. Typed text is encoded with
  TinyGPT's tokenizer, fetched read-only from TinyGPT's own folder (static/models/tinygpt/tokenizer.json).
-->
<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { base } from '$app/paths';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/index.ts';
	import { tokenizerOf } from '../../../../atlas/adapters/text-bpe/adapter.ts';
	import { tokText } from '../../vocab.ts';
	import { K_MAX, P_MAX, peOf, type PeRaw } from './adapter.ts';

	let { raw, presets = [], onchange, compact = false }: CaptureProps<PeRaw> = $props();
	let tokJson = $state<unknown>(null);
	onMount(() => {
		fetch(`${base}/models/tinygpt/tokenizer.json`)
			.then((r) => (r.ok ? r.json() : null))
			.then((j) => (tokJson = j))
			.catch(() => {});
	});
	const tok = $derived(tokenizerOf(tokJson));
	const s = $derived(peOf(raw));
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
			if (k && v !== (raw?.text ?? '')) onchange?.({ ...(raw ?? {}), text: v, ids: k.encode(v) }, { live: true });
		});
	});
	const set = (patch: Partial<NonNullable<PeRaw>>, live = false) => onchange?.({ ...(raw ?? {}), ...patch }, { live });
	const chips = $derived(s.ids.map((id) => ({ text: tokText(id), id })));
</script>

<div class="cap" class:compact>
	<TextInputBar bind:value={text} maxLength={compact ? 160 : 300} multiline={!compact} rows={2} label={t({ en: 'Your text', zh: '你的文字' })} />
	{#if !compact && presets.length}
		<div class="row" role="group" aria-label={t({ en: 'Example sentences', zh: '例句' })}>
			{#each presets as p (p.id)}<button type="button" class="btn" aria-pressed={p.raw?.text === raw?.text} onclick={() => set({ text: p.raw?.text, ids: p.raw?.ids })}>{t(p.label)}</button>{/each}
		</div>
	{/if}
	{#if chips.length && !compact}<TokenChips tokens={chips} showIds={false} label={{ en: 'Your tokens', zh: '你的词元' }} />{/if}
	<div class="row">
		<button type="button" class="btn" onclick={() => set({ shuffle: s.shuffle + 1 })}>{t({ en: 'Shuffle again', zh: '再打乱一次' })}</button>
		<span class="muted num">{t({ en: `shuffled order no. ${s.shuffle}`, zh: `第 ${s.shuffle} 种打乱顺序` })}</span>
	</div>
	<label class="sl">
		<span class="name">{t({ en: 'Key position p', zh: '键的位置 p' })}</span>
		<input type="range" min="0" max={P_MAX} step="1" value={s.p} oninput={(e) => set({ p: Number(e.currentTarget.value) }, true)} onchange={(e) => set({ p: Number(e.currentTarget.value) })} />
		<output class="num">{s.p}</output>
	</label>
	<label class="sl">
		<span class="name">{t({ en: 'Distance k', zh: '距离 k' })}</span>
		<input type="range" min="1" max={K_MAX} step="1" value={s.k} oninput={(e) => set({ k: Number(e.currentTarget.value) }, true)} onchange={(e) => set({ k: Number(e.currentTarget.value) })} />
		<output class="num">{s.k}</output>
	</label>
	<p class="muted small">{t({ en: `The query sits at position p + k = ${s.p + s.k}.`, zh: `查询位于位置 p + k = ${s.p + s.k}。` })}</p>
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; }
	.row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	.sl { display: grid; grid-template-columns: minmax(0, 8rem) minmax(0, 1fr) 2.5rem; align-items: center; gap: 8px; font-size: var(--fs-sm); }
	.sl input { width: 100%; min-width: 0; accent-color: var(--accent); }
	.sl output { text-align: right; }
	.name, .muted { color: var(--muted); }
	.small { margin: 0; font-size: var(--fs-xs); }
</style>

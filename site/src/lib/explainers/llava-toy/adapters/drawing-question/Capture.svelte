<!--
  llava-toy Capture (local adapter 'drawing-question'): the kit's drawing28 Capture (pad and presets) for the drawing,
  plus the question: a text box and three example questions. The drawing and the question travel together in one raw
  input, so a new drawing keeps the question and a new question keeps the drawing.
-->
<script lang="ts">
	import Draw from '$lib/atlas/adapters/drawing28/Capture.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '$lib/atlas/ui/capture';
	import { START_Q, VOCAB, type QRaw } from './adapter.ts';

	let props: CaptureProps = $props();
	const r = $derived((props.raw ?? {}) as QRaw);
	const q = $derived(r.question ?? START_Q);
	const drawing = $derived(typeof r.b64 === 'string' ? { b64: r.b64 } : { strokes: r.strokes ?? [] });
	const keepQ = (d: unknown, o?: { live?: boolean }) => {
		const x = (d ?? {}) as QRaw;
		props.onchange({ ...(typeof x.b64 === 'string' ? { b64: x.b64, preset: x.preset } : { strokes: x.strokes ?? [] }), question: q }, o);
	};
	const setQ = (v: string, live = false) => props.onchange({ ...r, question: v }, { live });
	const EXAMPLES = ['what is this ?', 'is this a house ?', 'is it an apple ?'];
</script>

<div class="cap">
	<Draw {...props} raw={drawing} onchange={keepQ} />
	<label class="in">
		<span class="muted">{t({ en: 'Question', zh: '问题' })}</span>
		<input type="text" value={q} maxlength="80" spellcheck="false" oninput={(e) => setQ(e.currentTarget.value, true)} onchange={(e) => setQ(e.currentTarget.value)} />
	</label>
	<div class="ex" role="group" aria-label={t({ en: 'Example questions', zh: '示例问题' })}>
		{#each EXAMPLES as e (e)}<button type="button" class="chip" aria-pressed={q === e} onclick={() => setQ(e)}>{e}</button>{/each}
	</div>
	{#if !props.compact}<p class="muted words">{t({ en: 'Words it knows:', zh: '它认识的词：' })} {VOCAB.slice(1).join(' · ')}</p>{/if}
</div>

<style>
	.cap { display: grid; gap: 8px; justify-items: start; min-width: 0; width: 100%; }
	.in { display: grid; grid-template-columns: 5.5em minmax(0, 1fr); gap: 6px; align-items: center; width: 100%; font-size: var(--fs-sm); }
	.in input { min-width: 0; width: 100%; min-height: 32px; }
	.ex { display: flex; flex-wrap: wrap; gap: 4px 6px; font-size: var(--fs-xs); }
	.chip { cursor: pointer; font-size: var(--fs-xs); padding: 2px 8px; }
	.chip[aria-pressed='true'] { background: var(--accent-soft); border-color: var(--accent); }
	.words { margin: 0; font-size: var(--fs-xs); }
</style>

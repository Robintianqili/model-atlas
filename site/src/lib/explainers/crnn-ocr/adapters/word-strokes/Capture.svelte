<!--
  crnn-ocr word-strokes Capture: the kit DrawPad (mouse, touch, pen) with a thin pen for writing a number across the
  pad, a box for the digits the learner meant (the lesson compares them with what the model reads), and the presets
  (shown on the pad, so they can be drawn over). Two Captures on one page (the input section and the lesson's side
  panel) stay in sync through `raw`.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import DrawPad from '$lib/kit/DrawPad.svelte';
	import type { Stroke } from '$lib/kit/draw';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/ui/capture.ts';
	import { say } from '../../../../atlas/ui/strings.ts';
	import { textOf } from './adapter.ts';

	let { raw, presets, onchange, compact = false }: CaptureProps = $props();

	let strokes = $state.raw<Stroke[]>([]);
	const rawStrokes = $derived(raw && typeof raw === 'object' && Array.isArray((raw as { strokes?: unknown }).strokes) ? (raw as { strokes: Stroke[] }).strokes : null);
	$effect(() => {
		const r = rawStrokes;
		untrack(() => {
			if (r && r !== strokes) strokes = r;
			else if (!r && strokes.length) strokes = [];
		});
	});
	const text = $derived(textOf(raw));
	const presetOn = $derived((raw as { preset?: string } | null)?.preset ?? null);
	let open = $state(false);
</script>

<div class="cap" class:compact>
	{#if !compact || open}
		<DrawPad
			bind:strokes
			brush={0.03}
			size={compact ? 220 : 300}
			showBrush={false}
			label={t({ en: 'Writing pad: write a number across it', zh: '书写板：在上面横着写一个数' })}
			onchange={(s) => onchange({ strokes: s, text })}
			ondraw={(s) => onchange({ strokes: s, text }, { live: true })}
		/>
	{/if}
	<label class="wrote">
		<span>{t({ en: 'The digits you meant:', zh: '你想写的数字：' })}</span>
		<input
			type="text"
			inputmode="numeric"
			maxlength="12"
			autocomplete="off"
			value={text}
			oninput={(e) => onchange({ strokes, text: e.currentTarget.value.replace(/[^0-9]/g, '') })}
		/>
	</label>
	{#if presets.length && (!compact || open)}
		<div class="presets" role="group" aria-label={say('presets')}>
			<span class="muted">{say('presets')}:</span>
			{#each presets as p (p.id)}
				<button type="button" class="chip" aria-pressed={presetOn === p.id} onclick={() => onchange(p.raw)}>{t(p.label)}</button>
			{/each}
		</div>
	{/if}
	{#if compact}
		<button type="button" class="btn small" aria-expanded={open} onclick={() => (open = !open)}>
			<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M11 2.5 13.5 5 6 12.5H3.5V10z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" /></svg>
			{open ? say('hideInput') : say('redrawHere')}
		</button>
	{/if}
</div>

<style>
	.cap {
		display: grid;
		gap: 8px;
		justify-items: start;
		min-width: 0;
	}
	.wrote {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
		font-size: var(--fs-sm);
	}
	.wrote input {
		width: 9em;
		font: inherit;
		font-family: var(--font-mono);
		padding: 4px 8px;
		min-height: 32px;
		border: 1px solid var(--rule-strong);
		border-radius: var(--r1);
		background: var(--surface);
		color: var(--ink);
	}
	.presets {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 6px;
		align-items: center;
		font-size: var(--fs-xs);
	}
	.presets .chip {
		cursor: pointer;
		font-size: var(--fs-xs);
		padding: 2px 8px;
	}
	.presets .chip[aria-pressed='true'] {
		background: var(--accent-soft);
		border-color: var(--accent);
	}
	.small {
		gap: 6px;
		min-height: 32px;
		padding: 0 12px;
		font-size: var(--fs-sm);
	}
</style>

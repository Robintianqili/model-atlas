<!--
  drawing28 Capture: the kit DrawPad (mouse, touch, pen) plus the adapter's presets. Strokes go to the runtime as
  { strokes } (preprocessStrokes renders them like the dataset); a preset goes as { b64 }. Two Captures on one page (the
  input section and the lesson's side panel) stay in sync through `raw`.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import DrawPad from '$lib/kit/DrawPad.svelte';
	import type { Stroke } from '$lib/kit/draw';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../ui/capture.ts';
	import { say } from '../../ui/strings.ts';

	let { raw, presets, onchange, compact = false }: CaptureProps = $props();

	let strokes = $state.raw<Stroke[]>([]);
	const rawStrokes = $derived(raw && typeof raw === 'object' && Array.isArray((raw as { strokes?: unknown }).strokes) ? ((raw as { strokes: Stroke[] }).strokes) : null);
	// follow the shared input: another Capture drew, or a preset replaced the strokes
	$effect(() => {
		const r = rawStrokes;
		untrack(() => {
			if (r && r !== strokes) strokes = r;
			else if (!r && strokes.length) strokes = [];
		});
	});
	const presetOn = $derived(raw && typeof raw === 'object' && typeof (raw as { b64?: unknown }).b64 === 'string' ? (raw as { b64: string }).b64 : null);
	let open = $state(false);
</script>

<div class="cap" class:compact>
	{#if !compact || open}
		<DrawPad
			bind:strokes
			brush={0.055}
			size={compact ? 190 : 280}
			showBrush={!compact}
			label={t({ en: 'Drawing pad', zh: '画板' })}
			onchange={(s) => onchange({ strokes: s })}
			ondraw={(s) => onchange({ strokes: s }, { live: true })}
		/>
	{/if}
	{#if presets.length && (!compact || open)}
		<div class="presets" role="group" aria-label={say('presets')}>
			<span class="muted">{say('presets')}:</span>
			{#each presets as p (p.id)}
				<button
					type="button"
					class="chip"
					aria-pressed={!!presetOn && (p.raw as { b64?: string }).b64 === presetOn}
					onclick={() => {
						strokes = [];
						onchange(p.raw);
					}}>{t(p.label)}</button
				>
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

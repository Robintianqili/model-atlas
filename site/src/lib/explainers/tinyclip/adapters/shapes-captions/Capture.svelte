<!--
  tinyclip Capture (local adapter 'shapes-captions'): the framed drawing pad (the frame is part of every drawing the model
  sees), the presets and three caption boxes. Any edit drops the preset mark, so a task can tell the learner's own
  captions from a preset's. The compact copy beside the lesson shows the captions and opens the pad on request.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import DrawPad from '$lib/kit/DrawPad.svelte';
	import type { Stroke } from '$lib/kit/draw';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '$lib/atlas/ui/capture';
	import { say } from '$lib/atlas/ui/strings';
	import { PRESETS, VOCAB } from '../../scenes.ts';
	import { rawOf, type ClipRaw } from './adapter.ts';

	let { raw, onchange, compact = false }: CaptureProps = $props();
	const r = $derived((raw ?? {}) as ClipRaw);
	const caps = $derived([...(r.captions ?? []), '', '', ''].slice(0, 3));
	let strokes = $state.raw<Stroke[]>([]);
	$effect(() => {
		const s = r.strokes ?? [];
		untrack(() => (strokes = s.map((x) => (Array.isArray(x) ? { points: x, size: 0.045 } : { points: x.points, size: x.size ?? 0.045 }))));
	});
	const send = (p: Partial<ClipRaw>, live = false) => onchange({ strokes, captions: caps, ...p, preset: undefined }, { live });
	const setCap = (i: number, v: string, live: boolean) => send({ captions: caps.map((c, j) => (j === i ? v : c)) }, live);
	let open = $state(false);
</script>

<div class="cap" class:compact>
	{#if !compact || open}
		<div class="frame">
			<DrawPad bind:strokes brush={0.045} size={compact ? 180 : 240} showBrush={false} label={t({ en: 'Drawing pad with a frame', zh: '带画框的画板' })} onchange={(s) => send({ strokes: s })} ondraw={(s) => send({ strokes: s }, true)} />
		</div>
		<div class="presets" role="group" aria-label={say('presets')}>
			{#each PRESETS as p (p.id)}<button type="button" class="chip" aria-pressed={r.preset === p.id} onclick={() => onchange(rawOf(p))}>{t(p.label)}</button>{/each}
		</div>
	{/if}
	{#each caps as c, i (i)}
		<label class="in"><span class="muted">{t({ en: `Caption ${i + 1}`, zh: `描述 ${i + 1}` })}</span><input type="text" value={c} maxlength="80" spellcheck="false" oninput={(e) => setCap(i, e.currentTarget.value, true)} onchange={(e) => setCap(i, e.currentTarget.value, false)} /></label>
	{/each}
	{#if !compact}<p class="muted words">{t({ en: 'Words it knows:', zh: '它认识的词：' })} {VOCAB.slice(1).join(' · ')}</p>{/if}
	{#if compact}<button type="button" class="btn small" aria-expanded={open} onclick={() => (open = !open)}>{open ? say('hideInput') : say('redrawHere')}</button>{/if}
</div>

<style>
	.cap { display: grid; gap: 8px; justify-items: start; min-width: 0; width: 100%; }
	.frame :global(canvas) { outline: 3px solid var(--ink); outline-offset: -3px; }
	.presets { display: flex; flex-wrap: wrap; gap: 4px 6px; font-size: var(--fs-xs); }
	.chip { cursor: pointer; font-size: var(--fs-xs); padding: 2px 8px; }
	.chip[aria-pressed='true'] { background: var(--accent-soft); border-color: var(--accent); }
	.in { display: grid; grid-template-columns: 5.5em minmax(0, 1fr); gap: 6px; align-items: center; width: 100%; font-size: var(--fs-sm); }
	.in input { min-width: 0; width: 100%; min-height: 32px; }
	.words { margin: 0; font-size: var(--fs-xs); }
	.small { min-height: 32px; padding: 0 12px; font-size: var(--fs-sm); }
</style>

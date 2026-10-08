<!--
  ViewSlot: one view of the registry, loaded lazily, with the StageProps contract (design §4.8). An unregistered key
  renders FallbackView (with a note in the Lesson Lab). Below the view: the view's describe() text, in a "Show as text"
  disclosure, so every stage has a text form for screen readers and for learners who prefer it.
-->
<script lang="ts">
	import type { Component } from 'svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../types/view.ts';
	import type { I18n } from '../types/core.ts';
	import FallbackView from './FallbackView.svelte';
	import { describeFallback } from './fallback.ts';
	import { hasView, loadView, viewMeta } from './registry.ts';
	import { say } from './strings.ts';

	interface Props {
		view: string;
		stage: StageProps;
		/** show "not registered" notes (Lesson Lab, dev) */
		lab?: boolean;
		/** show the describe() disclosure (default true) */
		text?: boolean;
	}
	let { view, stage, lab = false, text = true }: Props = $props();

	const registered = $derived(hasView(view));
	const meta = $derived(viewMeta(view));
	let Comp = $state<Component<any> | null>(null);
	let loadedKey = $state('');
	$effect(() => {
		const key = view;
		if (!registered) {
			Comp = null;
			loadedKey = key;
			return;
		}
		let alive = true;
		loadView(key).then((c) => {
			if (!alive) return;
			Comp = c;
			loadedKey = key;
		});
		return () => {
			alive = false;
		};
	});
	const ready = $derived(loadedKey === view);
	const described = $derived.by((): I18n => {
		try {
			return meta && Comp ? meta.describe(stage) : describeFallback(stage);
		} catch (e) {
			return { en: `(describe() failed: ${e instanceof Error ? e.message : String(e)})`, zh: '（describe() 出错）' };
		}
	});
</script>

<div class="slot" style:min-height={meta?.minHeight ? `${meta.minHeight}px` : undefined} data-view={view}>
	{#if lab && !registered}
		<p class="miss" role="note">{say('viewMissing', { key: view })}</p>
	{/if}
	{#if ready}
		{#if Comp}
			<!-- explicit props, not {...stage}: spread props would pull Svelte's spread helpers into the chunk every page shares.
			     K1: setInput and shared reach the view too (setInput was dropped here, so views never received it). -->
			<Comp rt={stage.rt} source={stage.source} at={stage.at} focus={stage.focus} sel={stage.sel} ctx={stage.ctx} provenance={stage.provenance} autoplay={stage.autoplay} onend={stage.onend} props={stage.props} setInput={stage.setInput} shared={stage.shared} />
		{:else}
			<FallbackView rt={stage.rt} source={stage.source} at={stage.at} focus={stage.focus} sel={stage.sel} ctx={stage.ctx} provenance={stage.provenance} autoplay={stage.autoplay} onend={stage.onend} props={stage.props} setInput={stage.setInput} shared={stage.shared} />
		{/if}
	{/if}
	{#if text && ready}
		<details class="as-text">
			<summary>{say('showAsText')}</summary>
			<p>{t(described)}</p>
		</details>
	{/if}
</div>

<style>
	.slot {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.miss {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--illus);
		border-left: 3px solid var(--illus);
		padding-left: 8px;
	}
	.as-text {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.as-text summary {
		cursor: pointer;
		color: var(--muted);
		width: fit-content;
	}
	.as-text p {
		margin: 4px 0 0;
	}
</style>

<!--
  ConceptPanel: one free-standing stage of a concept page (hook, toy, real, compare): a compiled StepDef shown outside the
  Walkthrough. It uses the lesson's StageHost (provenance badge with the source's parity record, "Computing… k / n", the
  empty note, predict and task gating, a compare's two panels, a play range's scrubber, glossary terms) on the page's one
  runtime, so its numbers are the lesson's. Differences from a lesson step:
    - its moment is resolved with panelPosition(), which never touches the lesson's selection;
    - it is re-resolved after every new input; the old stage stays until the new numbers arrive (no flicker while the
      learner drags a dial);
    - its animation (a self-playing view, a play range) starts only once the panel has been on screen (never on load, and
      never with reduced motion);
    - the narration is shown under the title as the panel's text, with its live numbers, then the "Look at" line.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import { motion } from '$lib/kit/theme.svelte';
	import type { WalkContext } from '$lib/kit/types';
	import type { AtlasCompiledStep } from '../../lesson/compile.ts';
	import type { StepPosition } from '../../runtime/sources.ts';
	import type { AtlasStore } from '../runtime.svelte.ts';
	import StageHost from '../StageHost.svelte';
	import { panelPosition } from './panels.ts';

	interface Props {
		store: AtlasStore;
		step: AtlasCompiledStep;
		lab?: boolean;
		/** show the step title as a heading (the section has its own title) */
		heading?: boolean;
	}
	let { store, step, lab = false, heading = true }: Props = $props();

	let pos = $state.raw<StepPosition | undefined>(undefined);
	let gen = 0;
	async function resolveIt() {
		const g = ++gen;
		const p = await panelPosition(store.rt, step);
		if (g !== gen) return;
		pos = p;
		store.touch();
	}
	$effect(() => {
		void store.inputGen;
		void step;
		untrack(() => void resolveIt());
	});

	// latched once the panel has been on screen (a self-playing view or a play range starts then, never on load)
	let shown = $state(false);
	let box = $state<HTMLElement | null>(null);
	$effect(() => {
		const el = box;
		if (!el || shown) return;
		if (typeof IntersectionObserver === 'undefined') {
			shown = true;
			return;
		}
		const io = new IntersectionObserver((es) => {
			if (es.some((e) => e.isIntersecting)) {
				shown = true;
				io.disconnect();
			}
		});
		io.observe(el);
		return () => io.disconnect();
	});
	const ctx = $derived<WalkContext>({
		index: 0,
		step: { id: step.id, title: step.title, narration: '' },
		playing: false,
		visible: shown,
		shown,
		reduced: motion.reduced,
		done: () => {}
	});

	const text = $derived.by(() => {
		void store.version;
		if (!pos || pos.error || pos.skipped || pos.at.index < 0) return null;
		try {
			const n = step.narration(store.rt, pos.at);
			const l = step.look?.(store.rt, pos.at);
			return { n, l };
		} catch {
			return null;
		}
	});
</script>

<div class="cp" bind:this={box} data-panel={step.id}>
	{#if heading}<h3>{t(step.title)}</h3>{/if}
	{#if text}
		<p class="ptext">{t(text.n)}</p>
		{#if text.l}<p class="look"><span class="tag">{ui('lessonLook')}</span> {t(text.l)}</p>{/if}
	{/if}
	<div class="stage">
		<StageHost {store} {step} {pos} {ctx} {lab} />
	</div>
</div>

<style>
	.cp {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	h3 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.ptext,
	.look {
		margin: 0;
		max-width: var(--measure);
	}
	.look {
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.tag {
		font-size: var(--fs-xs);
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--accent-strong);
		margin-right: 4px;
	}
	.stage {
		padding: 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
		min-width: 0;
	}
</style>

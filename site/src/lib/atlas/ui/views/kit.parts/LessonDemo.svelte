<!--
  Views inside the kit Walkthrough, as a lesson stage shows them (the stage column is ~60 % wide; narrow screens put the
  narration first). Gallery only: it checks the stage contract end to end: `autoplay` = ctx.shown && !ctx.reduced,
  `onend` → ctx.done() for a waitForDone step, and narration from the views' own describe() text. The real lesson
  player is W1-D1's LessonPlayer; this demo uses illustrative demo models.
-->
<script lang="ts">
	import type { Component } from 'svelte';
	import Walkthrough from '$lib/kit/Walkthrough.svelte';
	import type { WalkStep } from '$lib/kit/types';
	import type { I18n, StageProps, ViewMeta } from '../../../types/index.ts';

	interface Item {
		id: string;
		title: I18n;
		meta: ViewMeta;
		view: Component<StageProps>;
		stage: Omit<StageProps, 'ctx' | 'autoplay' | 'onend'>;
		waitForDone?: boolean;
	}
	let { items }: { items: Item[] } = $props();
	let index = $state(0);
	const steps = $derived<WalkStep[]>(
		items.map((it) => ({
			id: it.id,
			title: it.title,
			narration: it.meta.describe({ ...it.stage, autoplay: false, onend: () => {} }),
			waitForDone: it.waitForDone
		}))
	);
</script>

<Walkthrough {steps} bind:index id="gallery-lesson" syncHash={false} stageMinHeight={420} label={{ en: 'Views in a lesson', zh: '课程里的视图' }}>
	{#snippet stage(step, ctx)}
		{@const it = items.find((x) => x.id === step.id)}
		{#if it}
			{@const View = it.view}
			<View {...it.stage} {ctx} autoplay={ctx.shown && !ctx.reduced} onend={ctx.done} />
		{/if}
	{/snippet}
</Walkthrough>

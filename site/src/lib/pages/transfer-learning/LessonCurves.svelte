<!--
  Lesson step 7 (precomputed, finetune.json): head only vs full fine-tune vs from scratch on the
  same task, by drawings per class. A player walks from 1 to 2,048 per class; the chart keeps the
  lesson's own selection, so the Part 2 chart below is not moved.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import { fmtInt } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import CurveChart from './CurveChart.svelte';
	import type { LessonState } from './lessonState.svelte';

	interface Props {
		L: LessonState;
		autoplay?: boolean;
		ondone?: () => void;
	}
	let { L, autoplay = false, ondone }: Props = $props();

	const ns = $derived(L.finetune?.ns ?? []);
	// every visit starts at 1 drawing per class, where the difference is largest
	let k = $state(0);
	// player → chart, and chart (pointer / arrow keys) → player
	$effect(() => {
		const n = ns[k];
		if (n !== undefined && untrack(() => L.curve.part2N) !== n) L.curve.part2N = n;
	});
	$effect(() => {
		const i = ns.indexOf(L.curve.part2N);
		if (i >= 0 && i !== untrack(() => k)) k = i;
	});
	const labels = $derived(
		L.finetune
			? ns.map((n, i) => {
					const r = L.finetune!.results;
					const p = (m: 'head' | 'full' | 'scratch') => (r[m][i].mean * 100).toFixed(1);
					return {
						en: `${fmtInt(n)} per class: head ${p('head')}, full ${p('full')}, scratch ${p('scratch')}`,
						zh: `每类 ${fmtInt(n)} 张：输出头 ${p('head')}，全量 ${p('full')}，从零 ${p('scratch')}`
					};
				})
			: []
	);
</script>

{#if L.finetune}
	<div class="lc">
		<StepPlayer count={ns.length} bind:step={k} interval={1500} {labels} {autoplay} onend={ondone} label={{ en: 'Walk through the training-set sizes', zh: '逐个查看训练集大小' }}>
			{#snippet children()}
				<CurveChart finetune={L.finetune!} sel={L.curve} showLive={false} compact />
			{/snippet}
		</StepPlayer>
	</div>
{/if}

<style>
	.lc {
		min-width: 0;
	}
</style>

<!--
  "Predict first": a multiple-choice question asked before the learner plays. The choice is
  saved and shown again by <PredictReveal> at the end of the page.
  <PredictFirst q={question} />   (q: PredictQuestion from $lib/kit/types)
-->
<script lang="ts">
	import { predictions, savePrediction, clearPrediction, restorePrediction } from './predict.svelte';
	import { ui, t, colon } from '$lib/i18n/i18n.svelte';
	import type { PredictQuestion } from './types';

	let { q }: { q: PredictQuestion } = $props();
	const gid = $props.id();
	let choice = $state<number | null>(null);
	const saved = $derived(predictions[q.id]);

	$effect(() => {
		restorePrediction(q.id);
	});
</script>

<div class="pf">
	<p class="eyebrow">{ui('secPredict')}</p>
	<p class="q" id="{gid}-q">{t(q.question)}</p>
	{#if saved === undefined}
		<div class="opts" role="radiogroup" aria-labelledby="{gid}-q">
			{#each q.options as o, i (i)}
				<label class="opt" class:on={choice === i}>
					<input type="radio" name={gid} value={i} bind:group={choice} />
					<span>{t(o)}</span>
				</label>
			{/each}
		</div>
		<button type="button" class="btn primary" disabled={choice === null} onclick={() => choice !== null && savePrediction(q.id, choice)}>
			{ui('savePrediction')}
		</button>
	{:else}
		<p class="saved">
			<span class="muted">{ui('yourPrediction')}{colon()}</span> <strong>{t(q.options[saved])}</strong>
			<button type="button" class="link" onclick={() => clearPrediction(q.id)}>{ui('changePrediction')}</button>
		</p>
		<p class="muted small">{ui('predictionSaved')}</p>
	{/if}
</div>

<style>
	.pf {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 10px;
		padding: 16px 18px;
		border-radius: var(--r3);
		background: var(--here-bg);
		border: 1px solid color-mix(in srgb, var(--ink) 35%, transparent);
	}
	.eyebrow {
		margin: 0;
		color: var(--ink);
	}
	.q {
		margin: 0;
		font-family: var(--font-sans);
		font-size: var(--t-lg);
		font-weight: 600;
	}
	.opts {
		display: grid;
		gap: 6px;
	}
	.opt {
		display: flex;
		gap: 10px;
		align-items: flex-start;
		padding: 8px 12px;
		border-radius: var(--r2);
		background: var(--panel);
		border: 1px solid var(--rule);
		cursor: pointer;
	}
	.opt.on {
		border-color: var(--ink);
	}
	.opt input {
		margin-top: 0.3em;
		accent-color: var(--ink);
	}
	.btn {
		justify-self: start;
	}
	.saved {
		margin: 0;
	}
	.link {
		border: 0;
		background: none;
		color: var(--ink);
		text-decoration: underline;
		padding: 0 4px;
	}
	.small {
		margin: 0;
		font-size: var(--t-xs);
	}
</style>

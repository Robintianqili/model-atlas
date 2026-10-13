<!--
  The checkpoint half of "predict first": shows the learner's saved prediction, then the
  answer and the explanation. <PredictReveal q={question} />
-->
<script lang="ts">
	import { predictions, restorePrediction } from './predict.svelte';
	import { ui, t, colon } from '$lib/i18n/i18n.svelte';
	import type { PredictQuestion } from './types';

	let { q }: { q: PredictQuestion } = $props();
	let revealed = $state(false);
	const saved = $derived(predictions[q.id]);

	$effect(() => {
		restorePrediction(q.id);
	});
</script>

<div class="pr">
	<p class="q">{t(q.question)}</p>
	<p>
		<span class="muted">{ui('yourPrediction')}{colon()}</span>
		{#if saved !== undefined}<strong>{t(q.options[saved])}</strong>{:else}<em class="muted">{ui('noPredictionYet')}</em>{/if}
	</p>
	{#if !revealed}
		<button type="button" class="btn" onclick={() => (revealed = true)}>{ui('revealAnswer')}</button>
	{:else}
		<div class="ans" role="status">
			{#if saved !== undefined}
				<span class="verdict" class:ok={saved === q.answer}>{saved === q.answer ? ui('correct') : ui('notQuite')}</span>
			{/if}
			<p><strong>{ui('answer')}{colon()}</strong> {t(q.options[q.answer])}</p>
			<p>{t(q.explain)}</p>
		</div>
	{/if}
</div>

<style>
	.pr {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 8px;
		padding: 16px 18px;
		border-radius: var(--r3);
		background: var(--panel);
		border: 1px solid var(--rule);
	}
	.pr p {
		margin: 0;
	}
	.q {
		font-size: var(--t-md);
		line-height: var(--lh-md);
		font-weight: var(--w-ui);
	}
	.btn {
		justify-self: start;
	}
	.ans {
		display: grid;
		gap: 6px;
		padding-top: 6px;
		border-top: 1px solid var(--rule);
	}
	/* feedback colours (spec §2.3): ok / no, with the word */
	.verdict {
		justify-self: start;
		font-weight: 700;
		font-size: var(--t-sm);
		padding: 1px 10px;
		border-radius: var(--r-pill);
		background: var(--no-soft);
		color: var(--no);
	}
	.verdict.ok {
		background: var(--ok-soft);
		color: var(--ok);
	}
</style>

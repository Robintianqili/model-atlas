<!--
  Misconception check: a statement, "true or false?", then the answer with live evidence.
  <Misconception statement={{ en, zh }} truth={false}>{#snippet evidence()}…{/snippet}</Misconception>
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import { t, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		statement: Text;
		truth: boolean;
		verdict: Text;
		evidence: Snippet;
	}
	let { statement, truth, verdict, evidence }: Props = $props();
	let answer = $state<boolean | null>(null);
	const gid = $props.id();
</script>

<div class="mc">
	<p class="eyebrow">{t({ en: 'Misconception check', zh: '误区检查' })}</p>
	<p class="st" id="{gid}-q">“{t(statement)}”</p>
	<div class="btns" role="group" aria-labelledby="{gid}-q">
		<button type="button" class="btn" aria-pressed={answer === true} onclick={() => (answer = true)}>{t({ en: 'True', zh: '对' })}</button>
		<button type="button" class="btn" aria-pressed={answer === false} onclick={() => (answer = false)}>{t({ en: 'False', zh: '错' })}</button>
	</div>
	{#if answer !== null}
		<div class="ans" role="status">
			<p><span class="v" class:ok={answer === truth}>{answer === truth ? t({ en: 'Right.', zh: '答对了。' }) : t({ en: 'Not quite.', zh: '不太对。' })}</span> {t(verdict)}</p>
			{@render evidence()}
		</div>
	{/if}
</div>

<style>
	.mc {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 10px;
		padding: 14px 16px;
		border: 1px dashed var(--rule-strong);
		border-radius: var(--r3);
		background: var(--surface);
	}
	.eyebrow,
	.st {
		margin: 0;
	}
	.st {
		font-family: var(--font-display);
		font-size: var(--fs-lg);
		font-weight: 600;
	}
	.btns {
		display: flex;
		gap: 8px;
	}
	.ans {
		display: grid;
		gap: 8px;
		padding-top: 8px;
		border-top: 1px solid var(--rule);
	}
	.ans p {
		margin: 0;
	}
	.v {
		font-weight: 700;
		color: var(--illus);
	}
	.v.ok {
		color: var(--live);
	}
</style>

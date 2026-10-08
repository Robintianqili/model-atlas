<!--
  A misconception check: a statement, "True" / "False", then the verdict with evidence and an
  optional button that sets a widget's state to show the evidence.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import { t, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		statement: Text;
		/** is the statement true? */
		truth: boolean;
		evidence: Snippet;
		show?: { label: Text; action: () => void };
	}
	let { statement, truth, evidence, show }: Props = $props();
	let choice = $state<boolean | null>(null);
	const gid = $props.id();
</script>

<div class="mc">
	<p class="eyebrow">{t({ en: 'Misconception check', zh: '误区检查' })}</p>
	<p class="st" id="{gid}-s">“{t(statement)}”</p>
	<div class="row" role="group" aria-labelledby="{gid}-s">
		<button type="button" class="btn" aria-pressed={choice === true} onclick={() => (choice = true)}>{t({ en: 'True', zh: '对' })}</button>
		<button type="button" class="btn" aria-pressed={choice === false} onclick={() => (choice = false)}>{t({ en: 'False', zh: '错' })}</button>
	</div>
	{#if choice !== null}
		<div class="ans" role="status">
			<p class="verdict" class:ok={choice === truth}>
				{choice === truth ? t({ en: 'Right.', zh: '答对了。' }) : t({ en: 'Not quite.', zh: '不太对。' })}
				{truth ? t({ en: 'The statement is true.', zh: '这句话是对的。' }) : t({ en: 'The statement is false.', zh: '这句话是错的。' })}
			</p>
			<div class="ev">{@render evidence()}</div>
			{#if show}<button type="button" class="btn ghost" onclick={show.action}>{t(show.label)}</button>{/if}
		</div>
	{/if}
</div>

<style>
	.mc {
		display: grid;
		gap: 8px;
		padding: 14px 16px;
		border-radius: var(--r3);
		border: 1px solid var(--rule);
		background: var(--surface);
	}
	.eyebrow {
		margin: 0;
		color: var(--illus);
	}
	.st {
		margin: 0;
		font-family: var(--font-display);
		font-size: var(--fs-lg);
		font-weight: 600;
	}
	.row {
		display: flex;
		gap: 8px;
	}
	.btn[aria-pressed='true'] {
		border-color: var(--accent);
		box-shadow: 0 0 0 1px var(--accent);
	}
	.ans {
		display: grid;
		gap: 8px;
		padding-top: 8px;
		border-top: 1px solid var(--rule);
		justify-items: start;
	}
	.verdict {
		margin: 0;
		font-weight: 700;
		color: var(--illus);
	}
	.verdict.ok {
		color: var(--live);
	}
	.ev {
		max-width: var(--measure);
	}
	.ev :global(p) {
		margin: 0 0 0.5em;
	}
</style>

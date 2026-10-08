<!--
  A checkpoint question: pick an option, check, read the explanation.
-->
<script lang="ts">
	import { t, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		question: Text;
		options: Text[];
		answer: number;
		explain: Text;
	}
	let { question, options, answer, explain }: Props = $props();
	let choice = $state<number | null>(null);
	let checked = $state(false);
	const gid = $props.id();
</script>

<div class="qz">
	<p class="q" id="{gid}-q">{t(question)}</p>
	<div class="opts" role="radiogroup" aria-labelledby="{gid}-q">
		{#each options as o, i (i)}
			<label class="opt" class:on={choice === i} class:right={checked && i === answer} class:wrong={checked && choice === i && i !== answer}>
				<input type="radio" name={gid} value={i} bind:group={choice} onchange={() => (checked = false)} />
				<span>{t(o)}</span>
			</label>
		{/each}
	</div>
	<button type="button" class="btn primary" disabled={choice === null} onclick={() => (checked = true)}>{t({ en: 'Check', zh: '检查答案' })}</button>
	{#if checked && choice !== null}
		<p class="ex" role="status">
			<b class:ok={choice === answer}>{choice === answer ? t({ en: 'Correct.', zh: '正确。' }) : t({ en: 'Not quite.', zh: '不太对。' })}</b>
			{t(explain)}
		</p>
	{/if}
</div>

<style>
	.qz {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 10px;
		padding: 16px 18px;
		border-radius: var(--r3);
		background: var(--surface);
		border: 1px solid var(--rule);
	}
	.q {
		margin: 0;
		font-family: var(--font-display);
		font-size: var(--fs-lg);
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
		border: 1px solid var(--rule);
		cursor: pointer;
	}
	.opt.on {
		border-color: var(--accent);
	}
	.opt.right {
		border-color: var(--live);
		background: var(--live-soft);
	}
	.opt.wrong {
		border-color: var(--illus);
		background: var(--illus-soft);
	}
	.opt input {
		margin-top: 0.3em;
		accent-color: var(--accent);
	}
	.btn {
		justify-self: start;
	}
	.ex {
		margin: 0;
	}
	.ex b {
		color: var(--illus);
	}
	.ex b.ok {
		color: var(--live);
	}
</style>

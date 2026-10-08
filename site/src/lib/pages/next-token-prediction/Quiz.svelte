<!--
  A multiple-choice check with immediate feedback (misconception checks and checkpoint questions).
-->
<script lang="ts">
	import { t, type Text } from '$lib/i18n/i18n.svelte';

	type Q = { id: string; question: Text; options: Text[]; answer: number; explain: Text; hint?: Text };
	let { q }: { q: Q } = $props();
	const gid = $props.id();
	let choice = $state<number | null>(null);
	let checked = $state(false);
	const right = $derived(checked && choice === q.answer);
</script>

<div class="quiz">
	<p class="q" id="{gid}-q">{t(q.question)}</p>
	{#if q.hint}<p class="hint">{t(q.hint)}</p>{/if}
	<div class="opts" role="radiogroup" aria-labelledby="{gid}-q">
		{#each q.options as o, i (i)}
			<label class="opt" class:on={choice === i} class:good={checked && i === q.answer} class:bad={checked && choice === i && i !== q.answer}>
				<input
					type="radio"
					name={gid}
					value={i}
					bind:group={choice}
					onchange={() => (checked = false)}
				/>
				<span>{t(o)}</span>
			</label>
		{/each}
	</div>
	<div class="act">
		<button type="button" class="btn" disabled={choice === null} onclick={() => (checked = true)}>
			{t({ en: 'Check', zh: '检查' })}
		</button>
		{#if checked}
			<p class="fb" class:ok={right} role="status">
				<strong>{right ? t({ en: 'Right.', zh: '对。' }) : t({ en: 'Not quite.', zh: '不太对。' })}</strong>
				{t(q.explain)}
			</p>
		{/if}
	</div>
</div>

<style>
	.quiz {
		display: grid;
		gap: 8px;
		padding: 14px 16px;
		border-radius: var(--r3);
		border: 1px solid var(--rule);
		background: var(--surface);
	}
	.q {
		margin: 0;
		font-weight: 650;
	}
	.hint {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.opts {
		display: grid;
		gap: 6px;
	}
	.opt {
		display: flex;
		gap: 8px;
		align-items: flex-start;
		padding: 6px 10px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		cursor: pointer;
		font-size: var(--fs-sm);
	}
	.opt input {
		margin-top: 3px;
	}
	.opt.on {
		border-color: var(--accent);
		background: var(--accent-soft);
	}
	.opt.good {
		border-color: var(--live);
		background: var(--live-soft);
	}
	.opt.bad {
		border-color: var(--k);
	}
	.act {
		display: grid;
		gap: 8px;
	}
	.act .btn {
		justify-self: start;
	}
	.fb {
		margin: 0;
		font-size: var(--fs-sm);
		padding: 8px 10px;
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	.fb.ok strong {
		color: var(--live);
	}
</style>

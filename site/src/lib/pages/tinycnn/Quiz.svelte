<!--
  Checkpoint questions answered by finding a value in the widgets above. Choose, then check.
-->
<script lang="ts">
	import { t, type Text } from '$lib/i18n/i18n.svelte';

	interface Q {
		id: string;
		q: Text;
		hint: Text;
		options: Text[];
		answer: number;
		explain: Text;
	}
	let { questions }: { questions: Q[] } = $props();
	let picked = $state<Record<string, number>>({});
	let checked = $state<Record<string, boolean>>({});
</script>

<div class="quiz">
	{#each questions as q, n (q.id)}
		<fieldset class="qq">
			<legend><span class="qn num">{n + 1}</span>{t(q.q)}</legend>
			<p class="hint muted">{t(q.hint)}</p>
			<div class="opts">
				{#each q.options as o, i (i)}
					<label class="opt" class:on={picked[q.id] === i}>
						<input type="radio" name={q.id} value={i} checked={picked[q.id] === i} onchange={() => { picked[q.id] = i; checked[q.id] = false; }} />
						<span>{t(o)}</span>
					</label>
				{/each}
			</div>
			<button type="button" class="btn" disabled={picked[q.id] === undefined} onclick={() => (checked[q.id] = true)}>{t({ en: 'Check', zh: '检查' })}</button>
			{#if checked[q.id]}
				<p class="res" class:ok={picked[q.id] === q.answer} role="status">
					<strong>{picked[q.id] === q.answer ? t({ en: 'Correct.', zh: '正确。' }) : t({ en: 'Not quite.', zh: '不完全对。' })}</strong>
					{t(q.explain)}
				</p>
			{/if}
		</fieldset>
	{/each}
</div>

<style>
	.quiz {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr));
		gap: 14px;
	}
	.qq {
		margin: 0;
		display: grid;
		gap: 8px;
		align-content: start;
		padding: 14px 16px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
		min-width: 0;
	}
	legend {
		display: flex;
		gap: 8px;
		align-items: baseline;
		font-weight: 650;
		padding: 0;
		float: left;
		width: 100%;
	}
	.qn {
		color: var(--accent);
		font-size: var(--fs-sm);
	}
	.hint {
		margin: 0;
		font-size: var(--fs-xs);
	}
	.opts {
		display: grid;
		gap: 4px;
	}
	.opt {
		display: flex;
		gap: 8px;
		align-items: flex-start;
		padding: 6px 10px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		font-size: var(--fs-sm);
		cursor: pointer;
	}
	.opt.on {
		border-color: var(--accent);
	}
	.opt input {
		accent-color: var(--accent);
		margin-top: 0.3em;
	}
	.btn {
		justify-self: start;
	}
	.res {
		margin: 0;
		font-size: var(--fs-sm);
		padding: 6px 10px;
		border-radius: var(--r1);
		background: var(--illus-soft);
	}
	.res.ok {
		background: var(--live-soft);
	}
</style>

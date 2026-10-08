<!--
  "Try this": guided experiments. Each card asks the learner to predict first, then a button sets
  the page state for the experiment (brief §4.2 item 8).
-->
<script lang="ts">
	import { t, type Text } from '$lib/i18n/i18n.svelte';

	type Item = { title: Text; predict: Text; action: Text; run: () => void };
	let { items, heading }: { items: Item[]; heading?: Text } = $props();
	let done = $state<Record<number, boolean>>({});
</script>

<aside class="try" aria-label={t(heading ?? { en: 'Try this', zh: '试一试' })}>
	<p class="eyebrow">{t(heading ?? { en: 'Try this', zh: '试一试' })}</p>
	<ol>
		{#each items as it, i (i)}
			<li>
				<strong>{t(it.title)}</strong>
				<span class="q">{t(it.predict)}</span>
				<button
					type="button"
					class="btn"
					onclick={() => {
						it.run();
						done[i] = true;
					}}
				>
					{t(it.action)}{#if done[i]}<span class="ok" aria-hidden="true"> ✓</span>{/if}
				</button>
			</li>
		{/each}
	</ol>
</aside>

<style>
	.try {
		padding: 14px 16px;
		border-radius: var(--r3);
		border: 1px dashed color-mix(in srgb, var(--accent) 50%, var(--rule));
		background: color-mix(in srgb, var(--accent-soft) 45%, var(--surface));
	}
	.eyebrow {
		margin: 0 0 8px;
		color: var(--accent-strong);
	}
	ol {
		margin: 0;
		padding: 0;
		list-style: none;
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr));
		gap: 12px 18px;
	}
	li {
		display: grid;
		gap: 6px;
		align-content: start;
		font-size: var(--fs-sm);
	}
	.q {
		color: var(--ink-2);
	}
	.btn {
		justify-self: start;
		font-size: var(--fs-sm);
	}
	.ok {
		color: var(--live);
	}
</style>

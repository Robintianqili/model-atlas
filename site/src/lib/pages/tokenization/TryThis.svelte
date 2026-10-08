<!--
  "Try this": guided experiments. Each card asks you to predict first, then a button sets the
  page state (text, number of merges, …) and scrolls to the control.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { TryItem } from './types';

	let { items }: { items: TryItem[] } = $props();
</script>

<div class="try" role="group" aria-label={t({ en: 'Try this', zh: '试一试' })}>
	<p class="eyebrow">{t({ en: 'Try this: predict first, then check', zh: '试一试：先预测，再验证' })}</p>
	<ol>
		{#each items as it, i (i)}
			<li>
				<span class="q">{t(it.q)}</span>
				<button type="button" class="btn" onclick={it.run}>{t(it.go)} →</button>
			</li>
		{/each}
	</ol>
</div>

<style>
	.try {
		display: grid;
		gap: 8px;
		padding: 12px 14px;
		border-radius: var(--r3);
		background: var(--accent-soft);
		border: 1px solid color-mix(in srgb, var(--accent) 35%, transparent);
	}
	.eyebrow {
		margin: 0;
		color: var(--accent-strong);
	}
	ol {
		margin: 0;
		padding-left: 1.3em;
		display: grid;
		gap: 8px;
	}
	li {
		padding-left: 2px;
	}
	.q {
		display: block;
		margin-bottom: 4px;
	}
	.btn {
		min-height: 34px;
	}
</style>

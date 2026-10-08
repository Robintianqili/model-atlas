<!--
  Glossary term with a definition tooltip (hover, focus or tap) and a link to /glossary/.
  <Term id="softmax" />              renders the term name in the current language
  <Term id="logit">logits</Term>     custom visible text
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import { base } from '$app/paths';
	import Tooltip from './Tooltip.svelte';
	import { GLOSSARY_BY_ID } from '$lib/catalog/glossary';
	import { ui, t } from '$lib/i18n/i18n.svelte';

	let { id, children }: { id: string; children?: Snippet } = $props();
	const g = $derived(GLOSSARY_BY_ID[id]);
</script>

{#if g}
	<Tooltip width={300}>
		{#snippet content()}
			<strong>{t(g.term)}</strong> — {t(g.def)}
			<br /><a href="{base}/glossary/#{id}">{ui('glossaryLink')} →</a>
		{/snippet}
		<span class="term">{#if children}{@render children()}{:else}{t(g.term)}{/if}</span>
	</Tooltip>
{:else}
	<span class="term missing" title="glossary id not found: {id}">{#if children}{@render children()}{:else}{id}{/if}</span>
{/if}

<style>
	.term {
		text-decoration: underline dotted;
		text-decoration-color: var(--accent);
		text-underline-offset: 3px;
		text-decoration-thickness: 1.5px;
	}
	.missing {
		text-decoration-color: red;
	}
</style>

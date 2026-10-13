<!--
  Glossary term with a definition tooltip (hover, focus or tap) and a link to /glossary/.
  <Term id="softmax" />              renders the term name in the current language
  <Term id="logit">logits</Term>     custom visible text
  The names come from glossary-names.ts (generated, a few KB); the definitions (glossary.ts) load on the first hover,
  focus or tap of any term on the page, as one cached chunk (redesign wave F: no page shell carries them).
-->
<script lang="ts" module>
	import type { GlossaryTerm } from '$lib/catalog/types';

	let defs: Promise<Record<string, GlossaryTerm>> | null = null;
	/** The definitions, loaded once per page view and shared by every Term. */
	function loadDefs(): Promise<Record<string, GlossaryTerm>> {
		defs ??= import('$lib/catalog/glossary').then((m) => m.GLOSSARY_BY_ID);
		return defs;
	}
</script>

<script lang="ts">
	import type { Snippet } from 'svelte';
	import { base } from '$app/paths';
	import Tooltip from './Tooltip.svelte';
	import { GLOSSARY_NAMES } from '$lib/catalog/glossary-names';
	import { i18n, ui, t } from '$lib/i18n/i18n.svelte';

	let { id, children }: { id: string; children?: Snippet } = $props();
	const names = $derived(GLOSSARY_NAMES[id]);
	const name = $derived(names ? (i18n.lang === 'zh' ? names[1] : names[0]) : id);
	let def = $state<GlossaryTerm | null>(null);

	function open() {
		if (def?.id === id) return;
		loadDefs()
			.then((all) => (def = all[id] ?? null))
			.catch(() => {
				defs = null; // a failed chunk load (offline): try again on the next open
			});
	}
</script>

{#if names}
	<Tooltip width={300} onopen={open}>
		{#snippet content()}
			<strong>{name}</strong>{#if def && def.id === id}
				— {t(def.def)}{/if}
			<br /><a href="{base}/glossary/#{id}">{ui('glossaryLink')} →</a>
		{/snippet}
		<span class="term">{#if children}{@render children()}{:else}{name}{/if}</span>
	</Tooltip>
{:else}
	<span class="term missing" title="glossary id not found: {id}">{#if children}{@render children()}{:else}{id}{/if}</span>
{/if}

<style>
	.term {
		text-decoration: underline dotted;
		text-decoration-color: color-mix(in srgb, currentColor 60%, transparent);
		text-underline-offset: 3px;
		text-decoration-thickness: 1.5px;
	}
	.missing {
		text-decoration-color: red;
	}
</style>

<script lang="ts">
	import '$lib/styles/global';
	import { onMount, type Snippet } from 'svelte';
	import { page } from '$app/state';
	import SiteHeader from '$lib/kit/SiteHeader.svelte';
	import SiteFooter from '$lib/kit/SiteFooter.svelte';
	import { initThemeAndMotion } from '$lib/kit/theme.svelte';
	import { initLang, ui } from '$lib/i18n/i18n.svelte';

	let { children }: { children: Snippet } = $props();

	// Model and concept pages (/m/…, /train/<slug>/) set their own description through
	// ModelPageShell / ConceptPageShell; everywhere else the site tagline is the description.
	const ownDescription = $derived(/^\/(m|train)\/./.test(page.route.id ?? ''));

	// The chrome switch (redesign spec §6.1): a lesson route's load returns { chrome: 'lesson' } and renders without the
	// site header and footer. Such a page brings its own skip link, lesson top bar, <main id="main"> and controls.
	// Catalog pages, planned model pages, the glossary, About and DevKit keep the site chrome.
	const lesson = $derived((page.data as { chrome?: string }).chrome === 'lesson');

	onMount(() => {
		initThemeAndMotion();
		initLang();
	});
</script>

<svelte:head>
	{#if !ownDescription}<meta name="description" content={ui('siteTagline')} />{/if}
</svelte:head>

{#if lesson}
	{@render children()}
{:else}
	<SiteHeader />
	<main id="main" tabindex="-1">
		{@render children()}
	</main>
	<SiteFooter />
{/if}

<style>
	main {
		display: block;
		min-height: 60vh;
		outline: none;
	}
</style>

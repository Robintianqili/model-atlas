<!--
  Error page. Also what the GitHub Pages 404.html (adapter-static fallback) shows for any unknown URL:
  the fallback boots the app, the router finds no page and renders this with status 404.
-->
<script lang="ts">
	import { page } from '$app/state';
	import { base } from '$app/paths';
	import { t, ui } from '$lib/i18n/i18n.svelte';

	const notFound = $derived(page.status === 404);
</script>

<svelte:head><title>{notFound ? ui('notFoundTitle') : page.status} · Model Atlas</title></svelte:head>

<div class="container err">
	<p class="eyebrow num">{page.status}</p>
	{#if notFound}
		<h1>{ui('notFoundTitle')}</h1>
		<p class="lede">{ui('notFoundText')}</p>
		<p class="path num">{page.url.pathname}</p>
	{:else}
		<h1>{t({ en: 'Something went wrong', zh: '出了点问题' })}</h1>
		<p class="lede">{page.error?.message}</p>
	{/if}
	<p class="ctas">
		<a class="btn primary" href="{base}/">{ui('notFoundHome')}</a>
		<a class="btn" href="{base}/atlas/">{ui('browseAtlas')}</a>
		<a class="btn" href="{base}/m/tinycnn/#lesson">{ui('lessonStart')}</a>
	</p>
</div>

<style>
	.err {
		padding-top: var(--s8);
		padding-bottom: var(--s8);
		display: grid;
		gap: var(--s3);
	}
	.err h1,
	.err p {
		margin: 0;
	}
	.path {
		font-size: var(--fs-sm);
		color: var(--muted);
		overflow-wrap: anywhere;
	}
	.ctas {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		margin-top: var(--s3) !important;
	}
</style>

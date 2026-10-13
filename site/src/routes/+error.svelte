<!--
  Error page (redesign spec §6.9): the mark, one sentence, a search and links to the lessons and the paths. Also what the
  GitHub Pages 404.html (adapter-static fallback) shows for any unknown URL: the fallback boots the app, the router finds
  no page and renders this with status 404 (no server data there, so nothing here reads page.data).
-->
<script lang="ts">
	import { page } from '$app/state';
	import { base } from '$app/paths';
	import Icon from '$lib/kit/Icon.svelte';
	import { search } from '$lib/kit/icons';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import { ct } from '$lib/kit/catalog/text';

	const notFound = $derived(page.status === 404);
</script>

<svelte:head><title>{notFound ? ui('notFoundTitle') : page.status} · Model Atlas</title></svelte:head>

<div class="container err">
	<!-- the mark as the favicon file: importing Mark.svelte here split it out of the root layout chunk (+0.5 KB on every page) -->
	<img class="mark" src="{base}/favicon.svg" alt="" width="48" height="48" />
	<h1>{notFound ? ui('notFoundTitle') : t({ en: 'Something went wrong', zh: '出了点问题' })}</h1>
	<p class="lede">{notFound ? ct('notFoundBody') : page.error?.message}</p>
	{#if notFound}<p class="path num">{page.url.pathname}</p>{/if}
	<form class="find" role="search" action="{base}/atlas/" method="get">
		<label for="err-q" class="visually-hidden">{ui('searchLabel')}</label>
		<span class="glass" aria-hidden="true"><Icon d={search} size={18} /></span>
		<input id="err-q" type="search" name="q" placeholder={ui('searchLabel')} autocomplete="off" enterkeyhint="search" />
		<button type="submit" class="btn">{ui('searchOpen')}</button>
	</form>
	<p class="ctas">
		<a class="btn btn-act" href="{base}/atlas/">{ct('allLessons')}</a>
		<a class="btn" href="{base}/#paths">{ct('paths')}</a>
	</p>
</div>

<style>
	.err {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 16px;
		max-width: 720px;
		padding-top: var(--s8);
		padding-bottom: var(--s7);
	}
	.err h1,
	.err p {
		margin: 0;
	}
	.path {
		font-size: var(--t-sm);
		color: var(--muted);
		overflow-wrap: anywhere;
	}
	.find {
		position: relative;
		display: flex;
		gap: 8px;
		width: min(100%, 480px);
	}
	.glass {
		position: absolute;
		left: 12px;
		top: 50%;
		transform: translateY(-50%);
		color: var(--muted);
		pointer-events: none;
	}
	input {
		flex: 1;
		min-width: 0;
		height: 44px;
		padding: 0 12px 0 40px;
		border: 1px solid var(--line-control);
		border-radius: var(--r3);
		background: var(--panel);
		color: var(--ink);
		font-size: var(--t-md);
	}
	input::placeholder {
		color: var(--muted);
		opacity: 1;
	}
	.ctas {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
</style>

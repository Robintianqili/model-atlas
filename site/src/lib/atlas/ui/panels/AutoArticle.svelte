<!--
  AutoArticle (redesign spec §5.9): a kit model page's How it works view, loaded only when the view opens: the package's
  article (explainers/<slug>/content/En.svelte and Zh.svelte) in the essay column, or its sections/Article.svelte.
  KaTeX's stylesheet comes with this chunk (a formula in the article), not with the lesson shell.
-->
<script lang="ts">
	import { onMount, type Component } from 'svelte';
	import 'katex/dist/katex.min.css';
	import { i18n, ui } from '$lib/i18n/i18n.svelte';
	import type { AtlasSpec } from '../../types/spec.ts';
	import type { PackageModules } from '../../types/runtime.ts';
	import type { AtlasStore } from '../runtime.svelte.ts';
	import { loadArticle } from '../packages.ts';
	import { loadOverrides, type Overrides } from './overrides.ts';

	interface Props {
		vp: { store: AtlasStore | null; pkg: PackageModules | null; spec: AtlasSpec | null; slug: string };
		section?: string;
	}
	let { vp }: Props = $props();
	let ov = $state.raw<Overrides>({});
	let article = $state.raw<{ en: Component<any>; zh?: Component<any> } | null>(null);
	onMount(() => {
		loadOverrides(vp.slug, ['Article']).then((o) => {
			ov = o;
			if (!o.Article) loadArticle(vp.slug).then((a) => (article = a));
		});
	});
	const Ov = $derived(ov.Article ?? null);
</script>

{#if Ov}
	<Ov store={vp.store} pkg={vp.pkg} spec={vp.spec} />
{:else if article}
	{@const En = article.en}
	{@const Zh = article.zh}
	<div class="article prose" lang={i18n.lang === 'zh' && Zh ? 'zh-CN' : 'en'}>
		{#if i18n.lang === 'zh' && Zh}
			<Zh />
		{:else}
			{#if i18n.lang === 'zh'}<p class="nt" lang="zh-CN">{ui('notTranslatedYet')}</p>{/if}
			<En />
		{/if}
	</div>
{/if}

<style>
	.article {
		font-size: var(--t-lg);
		line-height: 30px;
	}
	:global(html[lang='zh-CN']) .article {
		line-height: 32px;
	}
	.nt {
		padding-left: 10px;
		border-left: 3px solid var(--rule-strong);
		color: var(--muted);
		font-size: var(--t-sm);
	}
</style>

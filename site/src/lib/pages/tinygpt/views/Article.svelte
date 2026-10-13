<!--
  TinyGPT's How it works view (redesign spec §5.9), loaded only when the view opens: the article in the essay column.
  KaTeX's stylesheet comes with this chunk (the article's formulas), not with the lesson shell.
-->
<script lang="ts">
	import 'katex/dist/katex.min.css';
	import Article from '$lib/kit/Article.svelte';
	import ArticleEn from '../ArticleEn.svelte';
	import ArticleZh from '../ArticleZh.svelte';

	let { vp: _vp }: { vp: unknown; section?: string } = $props();
</script>

<Article en={ArticleEn} zh={ArticleZh} />

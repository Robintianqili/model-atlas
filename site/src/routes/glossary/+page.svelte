<!-- /glossary/ (redesign spec §6.6): the only page that imports the full glossary (definitions); the rest show names from
     glossary-names.ts and load the definitions on first use (<Term>). "Used in" comes from +page.server.ts. -->
<script lang="ts">
	import GlossaryBrowser from '$lib/kit/catalog/GlossaryBrowser.svelte';
	import { GLOSSARY } from '$lib/catalog/glossary';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
</script>

<svelte:head><title>{ui('navGlossary')} · Model Atlas</title></svelte:head>

<div class="container glossary">
	<header class="head">
		<h1>{ui('navGlossary')}</h1>
		<p class="lede">
			{t({
				en: `${GLOSSARY.length} terms in English and Chinese, each with a short definition and the lessons that use it.`,
				zh: `${GLOSSARY.length} 个术语，中英对照，每个都有简短的定义和用到它的课程。`
			})}
		</p>
	</header>
	<GlossaryBrowser terms={GLOSSARY} used={data.used} lessons={data.lessons} areas={data.areas} />
</div>

<style>
	.head {
		padding: 24px 0 16px;
	}
	h1 {
		margin: 0 0 4px;
	}
	.lede {
		margin: 0;
		font-size: var(--t-md);
		line-height: var(--lh-md);
	}
	@media (max-width: 959px) {
		.head {
			padding: 20px 0 12px;
		}
	}
</style>

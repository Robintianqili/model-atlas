<!--
  /devkit/guide/<section>/: one top-level section of site/DEVKIT.md, rendered from the file at build time: the title, a
  short Chinese summary, a note when notes about the maintainers' own computer were left out, the text, and previous /
  next. The docs tree and "On this page" (the section's own headings) come from the layout (DevkitPage.svelte).
-->
<script lang="ts">
	import { base } from '$app/paths';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import Glyph from './Glyph.svelte';
	import { info } from './icons.ts';
	import Html from './Html.svelte';
	import Pager from './Pager.svelte';
	import type { SectionData, SectionRef } from './types.ts';

	let { page }: { page: SectionData } = $props();
	const s = $derived(page.section);
	const name = (x: SectionRef) => (i18n.lang === 'zh' ? x.zh : x.title);
	const prev = $derived(page.prev ? { href: `/devkit/guide/${page.prev.slug}/`, title: `${page.prev.n}. ${name(page.prev)}` } : { href: '/devkit/guide/', title: t({ en: 'The guide', zh: '指南' }) });
	const next = $derived(page.next ? { href: `/devkit/guide/${page.next.slug}/`, title: `${page.next.n}. ${name(page.next)}` } : { href: '/devkit/reference/', title: t({ en: 'Reference', zh: '参考' }) });
</script>

<article class="gs">
	<header class="head">
		<p class="eyebrow"><a href="{base}/devkit/guide/">{t({ en: 'Guide', zh: '指南' })}</a> · {t({ en: `Section ${s.n} of ${page.sections.length}`, zh: `第 ${s.n} 节，共 ${page.sections.length} 节` })}</p>
		<h1>{s.n}. {name(s)}</h1>
		{#if i18n.lang === 'zh'}<p class="en-title" lang="en">{s.title}</p>{/if}
	</header>

	{#if s.summary}
		{#if i18n.lang === 'zh'}
			<section class="zh" lang="zh-CN" aria-label="本节摘要">
				<p class="zh-h">本节摘要</p>
				<p>{s.summary}</p>
				<p class="zh-note">下面的正文是英文，直接来自 site/DEVKIT.md（唯一来源）。</p>
			</section>
		{:else}
			<details class="zh-en">
				<summary>Chinese summary · <span lang="zh-CN">中文摘要</span></summary>
				<p lang="zh-CN">{s.summary}</p>
			</details>
		{/if}
	{/if}

	{#if s.omitted}
		<p class="machine" role="note">
			<Glyph d={info} size={18} />
			<span>
				{t({
					en: 'This page leaves out a few notes that only apply to the maintainers’ own computer. To set up yours, follow',
					zh: '本页略去了几条只适用于维护者自己电脑的说明。准备你自己的环境，请看'
				})}
				<a href="{base}/devkit/contribute/#how">{t({ en: 'How to contribute', zh: '怎样贡献' })}</a>{t({ en: '.', zh: '。' })}
			</span>
		</p>
	{/if}

	<Html html={s.html} class="body" />

	<Pager label={t({ en: 'Previous and next section', zh: '上一节和下一节' })} {prev} {next} />
</article>

<style>
	.head {
		margin-bottom: var(--s5);
	}
	.head .eyebrow {
		margin: 0 0 var(--s2);
	}
	.eyebrow a {
		color: inherit;
	}
	.head h1 {
		margin: 0;
	}
	.en-title {
		margin: var(--s1) 0 0;
		color: var(--muted);
		font-size: var(--t-sm);
	}
	.zh {
		margin-bottom: var(--s5);
		padding: var(--s4);
		border: 1px solid var(--rule);
		border-left: 3px solid var(--line-control);
		border-radius: 0 var(--r3) var(--r3) 0;
		background: var(--panel);
	}
	.zh p {
		margin: 0 0 var(--s2);
		line-height: 1.8;
	}
	.zh-h {
		font-weight: var(--w-ui);
		font-size: var(--t-sm);
		color: var(--ink);
	}
	.zh .zh-note {
		margin: 0;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	.zh-en {
		margin-bottom: var(--s5);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.zh-en summary {
		padding-block: 6px;
		cursor: pointer;
		color: var(--muted);
	}
	.zh-en p {
		margin: var(--s1) 0 0;
		max-width: 75ch;
		line-height: 1.8;
	}
	.machine {
		display: flex;
		gap: var(--s3);
		align-items: flex-start;
		margin: 0 0 var(--s5);
		padding: var(--s3) var(--s4);
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--panel);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.machine :global(.icon) {
		flex: none;
		margin-top: 1px;
		color: var(--pre);
	}
</style>

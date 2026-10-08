<!--
  /devkit/guide/<section>/: one top-level section of site/DEVKIT.md, rendered from the file at build time, with a section
  sidebar (a collapsible list on phones), the section's own headings, a short Chinese summary and previous / next links.
-->
<script lang="ts">
	import { base } from '$app/paths';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import Html from './Html.svelte';
	import type { SectionData, SectionRef } from './types.ts';

	let { page }: { page: SectionData } = $props();
	const s = $derived(page.section);
	const name = (x: SectionRef) => (i18n.lang === 'zh' ? x.zh : x.title);
</script>

{#snippet nav()}
	<ol class="toc">
		{#each page.sections as x (x.slug)}
			<li>
				<a href="{base}/devkit/guide/{x.slug}/" aria-current={x.slug === s.slug ? 'page' : undefined}><span class="num">{x.n}.</span> {name(x)}</a>
				{#if x.slug === s.slug && s.subs.length}
					<ul class="subs">
						{#each s.subs as h (h.id)}<li><a href="#{h.id}" lang="en">{h.text}</a></li>{/each}
					</ul>
				{/if}
			</li>
		{/each}
	</ol>
{/snippet}

<div class="container gs">
	<aside class="side" aria-label={t({ en: 'Guide sections', zh: '指南章节' })}>
		<p class="side-h"><a href="{base}/devkit/guide/">{t({ en: 'The guide', zh: '指南' })}</a></p>
		{@render nav()}
	</aside>

	<article class="main">
		<details class="phone-nav">
			<summary>{t({ en: `Sections (${page.sections.length})`, zh: `章节（${page.sections.length}）` })} · {s.n}. {name(s)}</summary>
			{@render nav()}
		</details>

		<header class="head">
			<p class="eyebrow">{t({ en: 'Guide', zh: '指南' })} · {t({ en: `Section ${s.n}`, zh: `第 ${s.n} 节` })}</p>
			<h1>{s.n}. {name(s)}</h1>
			{#if i18n.lang === 'zh'}<p class="en-title" lang="en">{s.title}</p>{/if}
		</header>

		{#if s.summary}
			{#if i18n.lang === 'zh'}
				<section class="zh panel" lang="zh-CN" aria-label="本节摘要">
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
				{t({
					en: 'This page leaves out a few notes that only apply to the maintainers’ own computer. To set up yours, follow',
					zh: '本页略去了几条只适用于维护者自己电脑的说明。准备你自己的环境，请看'
				})}
				<a href="{base}/devkit/contribute/#how">{t({ en: 'How to contribute', zh: '怎样贡献' })}</a>{t({ en: '.', zh: '。' })}
			</p>
		{/if}
		<Html html={s.html} class="body" />

		<nav class="pn" aria-label={t({ en: 'Previous and next section', zh: '上一节和下一节' })}>
			{#if page.prev}
				<a class="btn" href="{base}/devkit/guide/{page.prev.slug}/" rel="prev">← {page.prev.n}. {name(page.prev)}</a>
			{:else}
				<a class="btn" href="{base}/devkit/guide/">← {t({ en: 'The guide', zh: '指南' })}</a>
			{/if}
			{#if page.next}
				<a class="btn" href="{base}/devkit/guide/{page.next.slug}/" rel="next">{page.next.n}. {name(page.next)} →</a>
			{/if}
		</nav>
	</article>
</div>

<style>
	.gs {
		display: grid;
		grid-template-columns: 250px minmax(0, 1fr);
		gap: var(--s6);
		padding-top: var(--s5);
		align-items: start;
	}
	.side {
		position: sticky;
		top: calc(var(--header-h) + 12px);
		max-height: calc(100vh - var(--header-h) - 24px);
		overflow-y: auto;
		padding-right: 6px;
		font-size: var(--fs-sm);
	}
	.side-h {
		margin: 0 0 8px;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		letter-spacing: 0.06em;
		text-transform: uppercase;
	}
	.side-h a {
		color: var(--muted);
		text-decoration: none;
	}
	.toc {
		margin: 0;
		padding: 0;
		list-style: none;
		display: grid;
		gap: 1px;
	}
	.toc a {
		display: block;
		padding: 4px 8px;
		border-radius: var(--r1);
		color: var(--ink-2);
		text-decoration: none;
		line-height: 1.35;
	}
	.toc a:hover {
		background: var(--surface-2);
		color: var(--ink);
	}
	.toc a[aria-current='page'] {
		background: var(--accent-soft);
		color: var(--ink);
		font-weight: 600;
	}
	.toc .num {
		color: var(--muted);
		font-size: 0.85em;
	}
	.subs {
		margin: 2px 0 6px 18px;
		padding: 0 0 0 8px;
		border-left: 1px solid var(--rule);
		list-style: none;
		font-size: var(--fs-xs);
	}
	.subs a {
		padding: 3px 6px;
	}
	.main {
		min-width: 0;
	}
	.phone-nav {
		display: none;
	}
	.head h1 {
		font-size: var(--fs-2xl);
		margin-bottom: 4px;
	}
	.en-title {
		margin: 0 0 var(--s3);
		color: var(--muted);
		font-size: var(--fs-sm);
	}
	.head {
		margin-bottom: var(--s4);
	}
	.zh {
		margin-bottom: var(--s5);
		background: var(--surface-2);
		border-left: 3px solid var(--accent);
		border-radius: 0 var(--r3) var(--r3) 0;
		max-width: calc(var(--measure) + 40px);
	}
	.zh p {
		margin: 0 0 8px;
		line-height: 1.75;
	}
	.zh-h {
		font-weight: 700;
		font-size: var(--fs-sm);
		color: var(--accent-strong);
	}
	.zh .zh-note {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.machine {
		max-width: calc(var(--measure) + 40px);
		margin: 0 0 var(--s4);
		padding: 8px 12px;
		border-left: 3px solid var(--pre);
		border-radius: 0 var(--r2) var(--r2) 0;
		background: var(--surface-2);
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.zh-en {
		margin-bottom: var(--s4);
		font-size: var(--fs-sm);
		color: var(--ink-2);
		max-width: var(--measure);
	}
	.zh-en summary {
		cursor: pointer;
		color: var(--muted);
	}
	.zh-en p {
		margin: 6px 0 0;
		line-height: 1.75;
	}
	.pn {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		gap: 10px;
		margin-top: var(--s6);
		padding-top: var(--s4);
		border-top: 1px solid var(--rule);
	}
	.pn .btn {
		max-width: 100%;
		height: auto;
		min-height: 40px;
		padding-block: 8px;
		line-height: 1.3;
		text-align: left;
	}
	@media (max-width: 900px) {
		.gs {
			grid-template-columns: minmax(0, 1fr);
			gap: 0;
		}
		.side {
			display: none;
		}
		.phone-nav {
			display: block;
			margin-bottom: var(--s4);
			padding: 8px 12px;
			border: 1px solid var(--rule);
			border-radius: var(--r2);
			background: var(--surface);
			font-size: var(--fs-sm);
		}
		.phone-nav summary {
			cursor: pointer;
			font-weight: 600;
		}
		.phone-nav .toc {
			margin-top: 8px;
		}
	}
</style>

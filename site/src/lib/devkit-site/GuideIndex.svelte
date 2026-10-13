<!--
  /devkit/guide/: the guide's introduction (from site/DEVKIT.md) and its sections as rows (in Chinese: the Chinese title
  over the English one; each section page opens with its Chinese summary). Sections 1 to 4 are the ones to read first
  (the guide's own advice); the rest is looked up when needed.
-->
<script lang="ts">
	import { base } from '$app/paths';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import Html from './Html.svelte';
	import Pager from './Pager.svelte';
	import type { GuideIndexData } from './types.ts';

	let { page }: { page: GuideIndexData } = $props();
	const first = $derived(page.sections.filter((s) => s.n <= 4));
	const rest = $derived(page.sections.filter((s) => s.n > 4));
	const sub = (n: number) => t({ en: n === 1 ? '1 subsection' : `${n} subsections`, zh: `${n} 小节` });
</script>

{#snippet rows(list: GuideIndexData['sections'])}
	<ol class="secs">
		{#each list as s (s.slug)}
			<li>
				<a class="sec" href="{base}/devkit/guide/{s.slug}/">
					<span class="n">{s.n}</span>
					<span class="body">
						<span class="title">{i18n.lang === 'zh' ? s.zh : s.title}</span>
						{#if i18n.lang === 'zh'}
							<span class="en" lang="en">{s.title}</span>
						{/if}
					</span>
					{#if s.subs}<span class="subs">{sub(s.subs)}</span>{/if}
				</a>
			</li>
		{/each}
	</ol>
{/snippet}

<article class="gi">
	<header class="head">
		<p class="eyebrow">DevKit · {t({ en: 'Guide', zh: '指南' })}</p>
		<h1>{t({ en: 'The DevKit guide', zh: 'DevKit 指南' })}</h1>
		<p class="lede">
			{t({
				en: 'How to build an illustration, from setting up to the last check. This is site/DEVKIT.md, rendered from the file itself when the site is built, one page per section.',
				zh: '怎样做一个图解，从准备环境到最后的检查。这就是 site/DEVKIT.md，在网站构建时直接从文件生成，每节一页。'
			})}
		</p>
		{#if i18n.lang === 'zh'}
			<p class="note" lang="zh-CN">指南正文是英文（这个文件是唯一来源）；每一节开头都有一段中文摘要。</p>
		{/if}
	</header>

	<div class="intro">
		<Html html={page.introHtml} />
		<p class="small">
			{t({
				en: 'The guide is also the maintainers’ working manual, so a few notes describe their own computer; the sections that have such notes say so at the top.',
				zh: '这份指南同时也是维护者的工作手册，所以有几处说明的是他们自己的电脑；有这类说明的章节会在开头注明。'
			})}
		</p>
	</div>

	<section aria-labelledby="gi-first">
		<h2 id="gi-first">{t({ en: 'Read these first', zh: '先读这几节' })}</h2>
		{@render rows(first)}
	</section>
	<section aria-labelledby="gi-rest">
		<h2 id="gi-rest">{t({ en: 'Look up when you need it', zh: '需要时再查' })}</h2>
		{@render rows(rest)}
	</section>

	<Pager label={t({ en: 'Previous and next page', zh: '上一页和下一页' })} prev={{ href: '/devkit/tutorial/', title: t({ en: 'Tutorial: build hello-xor', zh: '教程：做出 hello-xor' }) }} next={first[0] ? { href: `/devkit/guide/${first[0].slug}/`, title: `${first[0].n}. ${i18n.lang === 'zh' ? first[0].zh : first[0].title}` } : null} />
</article>

<style>
	.head .eyebrow {
		margin: 0 0 var(--s2);
	}
	.head h1 {
		margin-bottom: var(--s3);
	}
	.note {
		font-size: var(--t-sm);
		color: var(--ink-2);
	}
	.intro {
		margin: var(--s5) 0 var(--s6);
		padding: var(--s4) var(--s5);
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
	}
	.small {
		margin: var(--s3) 0 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
	}
	section + section {
		margin-top: var(--s6);
	}
	h2 {
		margin-bottom: var(--s3);
	}
	.gi {
		container-type: inline-size;
	}
	.head .lede,
	.intro {
		max-width: 760px;
	}
	.secs {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		column-gap: var(--s5);
		margin: 0;
		padding: 0;
		list-style: none;
		border-top: 1px solid var(--rule);
	}
	@container (min-width: 760px) {
		.secs {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}
	.sec {
		display: grid;
		grid-template-columns: 2.2em minmax(0, 1fr) auto;
		align-items: baseline;
		gap: var(--s3);
		height: 100%;
		padding: var(--s3) var(--s3) var(--s3) 0;
		border-bottom: 1px solid var(--rule);
		color: var(--ink);
		text-decoration: none;
		transition: background var(--dur-1) var(--ease);
	}
	.sec:hover {
		background: var(--here-bg);
	}
	.n {
		color: var(--muted);
		font-variant-numeric: tabular-nums;
		text-align: right;
	}
	.body {
		display: grid;
		gap: 2px;
		min-width: 0;
	}
	.title {
		font-weight: var(--w-ui);
	}
	.en,
	.subs {
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	.subs {
		white-space: nowrap;
	}
	@media (max-width: 559px) {
		.sec {
			grid-template-columns: 1.6em minmax(0, 1fr);
		}
		.subs {
			grid-column: 2;
		}
	}
</style>

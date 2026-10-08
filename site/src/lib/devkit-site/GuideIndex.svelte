<!-- /devkit/guide/: the guide's introduction (from site/DEVKIT.md) and its sections, each with its Chinese summary. -->
<script lang="ts">
	import { base } from '$app/paths';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import Html from './Html.svelte';
	import type { GuideIndexData } from './types.ts';

	let { page }: { page: GuideIndexData } = $props();
</script>

<div class="container gi">
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
		<p class="muted small">
			{t({
				en: 'The guide is also the maintainers’ working manual, so a few notes describe their own computer; the sections that have such notes say so at the top.',
				zh: '这份指南同时也是维护者的工作手册，所以有几处说明的是他们自己的电脑；有这类说明的章节会在开头注明。'
			})}
		</p>
	</header>

	<div class="intro panel">
		<Html html={page.introHtml} />
	</div>

	<ol class="secs">
		{#each page.sections as s (s.slug)}
			<li>
				<a class="sec" href="{base}/devkit/guide/{s.slug}/">
					<span class="n num">{s.n}</span>
					<span class="body">
						<span class="title">{i18n.lang === 'zh' ? s.zh : s.title}</span>
						{#if i18n.lang === 'zh'}
							<span class="en" lang="en">{s.title}</span>
							<span class="sum" lang="zh-CN">{s.summary}</span>
						{:else if s.subs}
							<span class="en">{s.subs} {s.subs === 1 ? 'subsection' : 'subsections'}</span>
						{/if}
					</span>
				</a>
			</li>
		{/each}
	</ol>
</div>

<style>
	.gi {
		padding-top: var(--s6);
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s5);
	}
	.head h1 {
		margin-bottom: var(--s3);
	}
	.head p {
		margin-bottom: 8px;
	}
	.small {
		font-size: var(--fs-sm);
		max-width: var(--measure);
	}
	.note {
		font-size: var(--fs-sm);
		color: var(--ink-2);
		border-left: 3px solid var(--accent);
		padding-left: 10px;
		max-width: var(--measure);
	}
	.intro {
		max-width: calc(var(--measure) + 40px);
	}
	.secs {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr));
		gap: 10px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.sec {
		display: flex;
		gap: 12px;
		height: 100%;
		padding: 12px 14px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		color: var(--ink);
		text-decoration: none;
	}
	.sec:hover {
		border-color: var(--accent);
	}
	.n {
		flex: none;
		width: 2ch;
		color: var(--accent-strong);
		font-weight: 700;
		text-align: right;
	}
	.body {
		display: grid;
		gap: 2px;
		min-width: 0;
	}
	.title {
		font-weight: 600;
	}
	.en {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.sum {
		margin-top: 4px;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		line-height: 1.6;
	}
</style>

<!--
  /devkit/contribute/: CONTRIBUTING.md, the design's checklist (§6.2), the gates of `atlas check` (DEVKIT.md section 10)
  and every error code (scripts/atlas/errors.ts, explained by DEVKIT.md section 13), all rendered from those files at
  build time. The Chinese meanings of the codes come from codes.ts (check.ts fails the build when the list changes).
-->
<script lang="ts">
	import { base } from '$app/paths';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import Html from './Html.svelte';
	import { withBase } from './base.ts';
	import type { ContributeData } from './types.ts';

	let { page }: { page: ContributeData } = $props();
	const groups = $derived(page.groups.map((g) => ({ ...g, codes: page.codes.filter((c) => c.group === g.id) })).filter((g) => g.codes.length));
</script>

<div class="container co">
	<header class="head">
		<p class="eyebrow">DevKit · {t({ en: 'Contribute', zh: '参与贡献' })}</p>
		<h1>{t({ en: 'Contribute a model', zh: '贡献一个模型' })}</h1>
		<p class="lede">
			{t({
				en: 'The path for contributors, the rules a package must keep, the checklist and the gates that enforce it, and what every error code means. Everything below is rendered from the repository’s own files when the site is built.',
				zh: '给贡献者的路径、包必须遵守的规则、检查清单和执行它们的关卡，以及每个错误代码的含义。下面的内容都在网站构建时由仓库里的文件生成。'
			})}
		</p>
		<nav class="jump" aria-label={t({ en: 'On this page', zh: '本页内容' })}>
			<a href="#how">{t({ en: 'How to contribute', zh: '怎样贡献' })}</a>
			<a href="#checklist">{t({ en: 'The checklist', zh: '检查清单' })}</a>
			<a href="#gates">{t({ en: 'The gates', zh: '关卡' })}</a>
			<a href="#error-codes">{t({ en: 'Error codes', zh: '错误代码' })}</a>
		</nav>
	</header>

	<section id="how" aria-labelledby="how-h">
		<h2 id="how-h">{t({ en: 'How to contribute', zh: '怎样贡献' })}</h2>
		<p class="src">{t({ en: 'From CONTRIBUTING.md', zh: '来自 CONTRIBUTING.md（英文原文）' })}</p>
		{#if i18n.lang === 'zh'}
			<p class="zh panel" lang="zh-CN">{page.contributingZh}</p>
		{/if}
		<Html html={page.contributingHtml} />
	</section>

	<section id="checklist" aria-labelledby="checklist-h">
		<h2 id="checklist-h">{t({ en: 'The checklist', zh: '检查清单' })}</h2>
		<p class="src">{t({ en: 'From the DevKit design, §6.2: each item names the gate that enforces it', zh: '来自 DevKit 设计文档 §6.2：每一项都写明了执行它的关卡' })}</p>
		<Html html={page.checklistHtml} />
	</section>

	<section id="gates" aria-labelledby="gates-h">
		<h2 id="gates-h">{t({ en: 'The gates', zh: '关卡' })}</h2>
		<p class="src">{t({ en: 'What npm run atlas -- check <slug> runs, from the guide:', zh: 'npm run atlas -- check <slug> 运行的内容，来自指南：' })} <a href={withBase(page.gatesGuide, base)}>{t({ en: 'Checks and gates', zh: '检查与关卡' })}</a></p>
		<Html html={page.gatesHtml} />
		<Html html={page.qualityHtml} />
	</section>

	<section id="error-codes" aria-labelledby="codes-h">
		<h2 id="codes-h">{t({ en: 'Error codes', zh: '错误代码' })}</h2>
		<p class="src">
			{t({ en: `${page.codes.length} codes, from the kit's error list; each explanation is the guide's own.`, zh: `${page.codes.length} 个代码，来自工具包的错误列表；每段说明都取自指南（英文）。` })}
			<a href={withBase(page.errorsGuide, base)}>{t({ en: 'Error codes in the guide', zh: '指南中的错误代码' })}</a>
		</p>
		{#each groups as g (g.id)}
			<h3 class="gh">{g.id}xx · {t(g.label)}</h3>
			<dl class="codes">
				{#each g.codes as c (c.code)}
					<div class="code-row" id={c.code.toLowerCase()}>
						<dt>
							<a class="code num" href="#{c.code.toLowerCase()}">{c.code}</a>
							<span class="mean">{i18n.lang === 'zh' ? c.zh : c.en}</span>
							{#if i18n.lang === 'zh'}<span class="en" lang="en">{c.en}</span>{/if}
						</dt>
						<dd><Html html={c.html} /></dd>
					</div>
				{/each}
			</dl>
		{/each}
	</section>
</div>

<style>
	.co {
		padding-top: var(--s6);
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s6);
	}
	.head h1 {
		margin-bottom: var(--s3);
	}
	.jump {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 16px;
		font-size: var(--fs-sm);
	}
	section {
		min-width: 0;
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	section h2 {
		font-size: var(--fs-xl);
		margin-bottom: 4px;
	}
	.src {
		margin: 0 0 var(--s4);
		font-size: var(--fs-sm);
		color: var(--muted);
	}
	.zh {
		max-width: calc(var(--measure) + 40px);
		background: var(--surface-2);
		border-left: 3px solid var(--accent);
		line-height: 1.75;
	}
	.gh {
		margin: var(--s5) 0 var(--s2);
		font-size: var(--fs-lg);
	}
	.codes {
		margin: 0;
		display: grid;
		gap: 0;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
	}
	.code-row {
		display: grid;
		grid-template-columns: minmax(12em, 18em) minmax(0, 1fr);
		gap: 6px 18px;
		padding: 10px 14px;
		border-bottom: 1px solid var(--rule);
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	.code-row:last-child {
		border-bottom: 0;
	}
	.code-row:target {
		background: var(--accent-soft);
	}
	dt {
		display: grid;
		align-content: start;
		gap: 2px;
	}
	.code {
		font-weight: 700;
		text-decoration: none;
	}
	.mean {
		font-weight: 600;
		font-size: var(--fs-sm);
	}
	.en {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	dd {
		margin: 0;
		min-width: 0;
		font-size: var(--fs-sm);
	}
	dd :global(.dk-md p:last-child) {
		margin-bottom: 0;
	}
	@media (max-width: 700px) {
		.code-row {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>

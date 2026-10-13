<!--
  /devkit/contribute/: CONTRIBUTING.md, the design's checklist (§6.2), the gates of `atlas check` (DEVKIT.md section 10)
  and every error code (scripts/atlas/errors.ts, explained by DEVKIT.md section 13), all rendered from those files at
  build time. The Chinese meanings of the codes come from codes.ts (check.ts fails the build when the list changes).
  The codes are rows that open in place (code and meaning first, the guide's explanation inside); a link to a code
  (#ak604, as the CLI prints it) opens its row, and "Open all" opens every row.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import Html from './Html.svelte';
	import Pager from './Pager.svelte';
	import { withBase } from './base.ts';
	import type { ContributeData } from './types.ts';

	let { page }: { page: ContributeData } = $props();
	const groups = $derived(page.groups.map((g) => ({ ...g, codes: page.codes.filter((c) => c.group === g.id) })).filter((g) => g.codes.length));

	let open = $state<Record<string, boolean>>({});
	const allOpen = $derived(page.codes.every((c) => open[c.code]));
	function setAll(v: boolean) {
		const next: Record<string, boolean> = {};
		for (const c of page.codes) next[c.code] = v;
		open = next;
	}
	// a link to a code (#ak604) opens its row
	onMount(() => {
		const fromHash = () => {
			const id = location.hash.slice(1).toLowerCase();
			const c = page.codes.find((x) => x.code.toLowerCase() === id);
			if (!c) return;
			open[c.code] = true;
			requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView({ block: 'start' }));
		};
		fromHash();
		addEventListener('hashchange', fromHash);
		return () => removeEventListener('hashchange', fromHash);
	});
</script>

<article class="co">
	<header class="head">
		<p class="eyebrow">DevKit · {t({ en: 'Contribute', zh: '参与贡献' })}</p>
		<h1>{t({ en: 'Contribute a model', zh: '贡献一个模型' })}</h1>
		<p class="lede">
			{t({
				en: 'The path for contributors, the rules a package must keep, the checklist and the gates that enforce it, and what every error code means. Everything below is rendered from the repository’s own files when the site is built.',
				zh: '给贡献者的路径、包必须遵守的规则、检查清单和执行它们的关卡，以及每个错误代码的含义。下面的内容都在网站构建时由仓库里的文件生成。'
			})}
		</p>
	</header>

	<section id="how" aria-labelledby="how-h">
		<h2 id="how-h">{t({ en: 'How to contribute', zh: '怎样贡献' })}</h2>
		<p class="src">{t({ en: 'From CONTRIBUTING.md', zh: '来自 CONTRIBUTING.md（英文原文）' })}</p>
		{#if i18n.lang === 'zh'}
			<p class="zh" lang="zh-CN">{page.contributingZh}</p>
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
		<div class="ch">
			<h2 id="codes-h">{t({ en: 'Error codes', zh: '错误代码' })}</h2>
			<button type="button" class="btn btn-quiet" aria-pressed={allOpen} onclick={() => setAll(!allOpen)}>
				{allOpen ? t({ en: 'Close all', zh: '全部收起' }) : t({ en: 'Open all', zh: '全部展开' })}
			</button>
		</div>
		<p class="src">
			{t({ en: `${page.codes.length} codes, from the kit's error list; each explanation is the guide's own. Open a row to read it.`, zh: `${page.codes.length} 个代码，来自工具包的错误列表；每段说明都取自指南（英文）。展开一行即可阅读。` })}
			<a href={withBase(page.errorsGuide, base)}>{t({ en: 'Error codes in the guide', zh: '指南中的错误代码' })}</a>
		</p>
		{#each groups as g (g.id)}
			<h3 class="gh" id="codes-{g.id}">{g.id}xx · {t(g.label)}</h3>
			<ul class="codes">
				{#each g.codes as c (c.code)}
					<li>
						<details
							class="code-row"
							id={c.code.toLowerCase()}
							open={open[c.code] ?? false}
							ontoggle={(e) => (open[c.code] = (e.currentTarget as HTMLDetailsElement).open)}
						>
							<summary>
								<span class="code num">{c.code}</span>
								<span class="mean">{i18n.lang === 'zh' ? c.zh : c.en}{#if i18n.lang === 'zh'}<span class="en" lang="en">{c.en}</span>{/if}</span>
							</summary>
							<div class="ex">
								<Html html={c.html} class="compact" />
								<a class="self" href="#{c.code.toLowerCase()}">{t({ en: `Link to ${c.code}`, zh: `链接到 ${c.code}` })}</a>
							</div>
						</details>
					</li>
				{/each}
			</ul>
		{/each}
	</section>

	<Pager label={t({ en: 'Previous page', zh: '上一页' })} prev={{ href: '/devkit/reference/', title: t({ en: 'Reference', zh: '参考' }) }} />
</article>

<style>
	.co {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s7);
	}
	.head .eyebrow {
		margin: 0 0 var(--s2);
	}
	.head h1 {
		margin-bottom: var(--s3);
	}
	.head .lede {
		margin: 0;
	}
	section {
		min-width: 0;
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	section h2 {
		margin-bottom: var(--s1);
	}
	.src {
		margin: 0 0 var(--s4);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
	}
	.zh {
		padding: var(--s3) var(--s4);
		border: 1px solid var(--rule);
		border-left: 3px solid var(--line-control);
		border-radius: 0 var(--r3) var(--r3) 0;
		background: var(--panel);
		line-height: 1.8;
	}
	.ch {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--s2);
	}
	.ch h2 {
		margin: 0;
	}
	.gh {
		margin: var(--s5) 0 var(--s2);
		font-size: var(--t-md);
		line-height: var(--lh-md);
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	.codes {
		margin: 0;
		padding: 0;
		list-style: none;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--panel);
		overflow: hidden;
	}
	.codes > li + li {
		border-top: 1px solid var(--rule);
	}
	.code-row {
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	summary {
		display: grid;
		grid-template-columns: 4.5em minmax(0, 1fr);
		align-items: baseline;
		gap: var(--s3);
		min-height: 44px;
		padding: 10px var(--s4);
		cursor: pointer;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	summary:hover {
		background: var(--here-bg);
	}
	.code-row[open] > summary,
	.code-row:target > summary {
		background: var(--here-bg);
		box-shadow: inset 3px 0 0 var(--here);
	}
	.code {
		font-weight: var(--w-ui);
	}
	.mean {
		font-weight: var(--w-ui);
	}
	.en {
		display: block;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		font-weight: var(--w-body);
		color: var(--muted);
	}
	.ex {
		display: grid;
		gap: var(--s2);
		padding: var(--s3) var(--s4) var(--s4) calc(4.5em + var(--s4) + var(--s3));
		min-width: 0;
	}
	.self {
		justify-self: start;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	@media (max-width: 559px) {
		.ex {
			padding-left: var(--s4);
		}
	}
	.co > :global(.pg) {
		margin-top: 0;
	}
</style>

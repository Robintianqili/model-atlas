<!--
  /devkit/reference/views/: every view a lesson step can show, generated at build time from the view metas (key, title,
  kinds, minimum height, the text a screen reader hears for its demo, the ops that use it as their zoom view), each
  with its live demos (the /dev/atlas/ gallery's, badged Illustrative). The kit views come first, each with its demo
  beside its facts. The views of packages follow as a compact list grouped by package: a row opens in place to show
  its demo (loaded only then), and a link to #view-<key> opens its row.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import { t } from '$lib/i18n/i18n.svelte';
	import Demo from './Demo.svelte';
	import { withBase } from './base.ts';
	import type { ViewData, ViewsData } from './types.ts';

	let { page }: { page: ViewsData } = $props();
	const kit = $derived(page.views.filter((v) => !v.pkg));
	const pkg = $derived(page.views.filter((v) => v.pkg));
	const groups = $derived.by(() => {
		const out: { pkg: string; lesson: ViewData['lesson']; views: ViewData[] }[] = [];
		for (const v of pkg) {
			const g = out.find((x) => x.pkg === v.pkg);
			if (g) g.views.push(v);
			else out.push({ pkg: v.pkg!, lesson: v.lesson, views: [v] });
		}
		return out;
	});
	const anchor = (key: string) => `view-${key.replace(/[^\w-]/g, '-')}`;
	const KIND: Record<string, { en: string; zh: string }> = {
		trace: { en: 'A · trace', zh: 'A · 推理轨迹' },
		train: { en: 'B · train', zh: 'B · 训练' },
		process: { en: 'C · process', zh: 'C · 过程' }
	};
	/** package rows opened at least once (their demo stays mounted) and open now */
	let seen = $state<Record<string, boolean>>({});
	let open = $state<Record<string, boolean>>({});

	// a link to a package view's anchor opens its row
	onMount(() => {
		const fromHash = () => {
			const id = decodeURIComponent(location.hash.slice(1));
			const v = pkg.find((x) => anchor(x.key) === id);
			if (!v) return;
			open[v.key] = true;
			seen[v.key] = true;
			requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView({ block: 'start' }));
		};
		fromHash();
		addEventListener('hashchange', fromHash);
		return () => removeEventListener('hashchange', fromHash);
	});
</script>

{#snippet kinds(v: ViewData)}
	{#each v.kinds as k (k)}<span class="chip">{t(KIND[k] ?? { en: k, zh: k })}</span>{/each}
{/snippet}

<div class="ref">
	<header class="head">
		<p class="eyebrow"><a href="{base}/devkit/reference/">{t({ en: 'Reference', zh: '参考' })}</a></p>
		<h1>{t({ en: 'View reference', zh: '视图参考' })}</h1>
		<p class="lede">
			{t({
				en: `Every view a lesson step can show: ${kit.length} kit views and ${pkg.length} views that packages made for themselves, generated from their meta.ts files when the site is built. A view draws one step's stage from the runtime and the step's frame; each demo runs the real view on demo data.`,
				zh: `课程每一步可以显示的视图：${kit.length} 个工具包视图，以及 ${pkg.length} 个由各个包为自己做的视图，在网站构建时由它们的 meta.ts 生成。视图根据运行时和这一步的帧画出舞台；每个演示都在演示数据上运行真实的视图。`
			})}
		</p>
		<p class="small">
			{t({ en: 'The same list in the guide:', zh: '指南里的同一张表：' })} <a href={withBase(page.guide, base)}>{t({ en: 'View reference (guide)', zh: '视图参考（指南）' })}</a>
		</p>
	</header>

	<nav class="index" aria-label={t({ en: 'Views', zh: '视图' })}>
		<p class="il">{t({ en: `Kit views · ${kit.length}`, zh: `工具包视图 · ${kit.length}` })}</p>
		<ul>
			{#each kit as v (v.key)}<li><a href="#{anchor(v.key)}"><code>{v.key}</code></a></li>{/each}
		</ul>
		{#if groups.length}
			<p class="il">{t({ en: `Views made by packages · ${pkg.length} in ${groups.length} packages`, zh: `各个包自己做的视图 · ${groups.length} 个包共 ${pkg.length} 个` })}</p>
			<ul>
				{#each groups as g (g.pkg)}<li><a href="#pkg-{g.pkg}"><code>{g.pkg}</code></a></li>{/each}
			</ul>
		{/if}
	</nav>

	{#each kit as v (v.key)}
		<section class="view" id={anchor(v.key)} aria-labelledby="{anchor(v.key)}-h">
			<header class="vh">
				<h2 id="{anchor(v.key)}-h"><code>{v.key}</code> <span class="lab">{t(v.title)}</span></h2>
				<p class="chips">
					{@render kinds(v)}
					{#if v.minHeight}<span class="chip">{t({ en: `min height ${v.minHeight} px`, zh: `最小高度 ${v.minHeight} 像素` })}</span>{/if}
				</p>
			</header>
			<div class="grid">
				<div class="facts">
					<p>{t({ en: `A lesson step uses it as view: '${v.key}'.`, zh: `课程的一步这样使用它：view: '${v.key}'。` })}</p>
					{#if v.ops.length}
						<p>
							{t({ en: 'The zoom view of', zh: '这些算子的放大视图：' })}{' '}
							{#each v.ops as o, i (o)}<a href="{base}/devkit/reference/ops/#op-{o}"><code>{o}</code></a>{i < v.ops.length - 1 ? ', ' : ''}{/each}
						</p>
					{/if}
					{#if v.describe}
						<p class="say"><span class="k">{t({ en: 'Its demo, as a screen reader hears it', zh: '它的演示，读屏软件读出的内容' })}</span>{t(v.describe)}</p>
					{/if}
				</div>
				<Demo view={v.key} key={v.key} minHeight={v.minHeight} say={false} />
			</div>
		</section>
	{/each}

	{#if groups.length}
		<section class="pkgs" aria-labelledby="pkg-h">
			<h2 id="pkg-h">{t({ en: 'Views made by packages', zh: '各个包自己做的视图' })}</h2>
			<p class="small">{t({ en: "A package can add its own view (explainers/<slug>/views/<key>/, used as '<slug>/<key>'); these are the ones that exist today. Open a row to run its demo.", zh: '包可以添加自己的视图（explainers/<slug>/views/<key>/，用作 \'<slug>/<key>\'）；下面是现有的这些。展开一行即可运行它的演示。' })}</p>
			{#each groups as g (g.pkg)}
				<section class="pg" id="pkg-{g.pkg}" aria-labelledby="pkg-{g.pkg}-h">
					<h3 id="pkg-{g.pkg}-h">
						<code>{g.pkg}</code>
						{#if g.lesson}<a class="ls" href={withBase(g.lesson.href, base)}>{t(g.lesson.name)}</a>{/if}
						<span class="n">{t({ en: g.views.length === 1 ? '1 view' : `${g.views.length} views`, zh: `${g.views.length} 个视图` })}</span>
					</h3>
					<ul class="rows">
						{#each g.views as v (v.key)}
							<li id={anchor(v.key)}>
								<details
									open={open[v.key] ?? false}
									ontoggle={(e) => {
										const o = (e.currentTarget as HTMLDetailsElement).open;
										open[v.key] = o;
										if (o) seen[v.key] = true;
									}}
								>
									<summary>
										<code class="key">{v.key.slice(g.pkg.length + 1) || v.key}</code>
										<span class="ti">{t(v.title)}</span>
										<span class="ks">{@render kinds(v)}</span>
										<span class="go" aria-hidden="true">{open[v.key] ? t({ en: 'Hide', zh: '收起' }) : t({ en: 'Demo', zh: '演示' })}</span>
									</summary>
									<div class="rb">
										<p class="use">{t({ en: `A step of ${g.pkg} uses it as view: '${v.key}'.`, zh: `${g.pkg} 的课程步骤这样使用它：view: '${v.key}'。` })}{#if v.minHeight}{' '}{t({ en: `Min height ${v.minHeight} px.`, zh: `最小高度 ${v.minHeight} 像素。` })}{/if}</p>
										{#if v.describe}<p class="say"><span class="k">{t({ en: 'Its demo, as a screen reader hears it', zh: '它的演示，读屏软件读出的内容' })}</span>{t(v.describe)}</p>{/if}
										{#if seen[v.key]}<Demo view={v.key} key={v.key} minHeight={v.minHeight} say={false} />{/if}
									</div>
								</details>
							</li>
						{/each}
					</ul>
				</section>
			{/each}
		</section>
	{/if}
</div>

<style>
	.ref {
		container-type: inline-size;
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s6);
	}
	.head .eyebrow {
		margin: 0 0 var(--s2);
	}
	.eyebrow a {
		color: inherit;
	}
	.head h1 {
		margin-bottom: var(--s3);
	}
	.small {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
		max-width: 75ch;
	}

	/* the index */
	.index {
		display: grid;
		gap: var(--s2);
		padding: var(--s4) var(--s5);
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
	}
	.il {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
	}
	.il:not(:first-child) {
		margin-top: var(--s2);
	}
	.index ul {
		margin: 0;
		padding: 0;
		list-style: none;
		display: flex;
		flex-wrap: wrap;
		gap: 0 var(--s3);
		font-size: var(--t-sm);
		line-height: 26px;
	}

	/* a kit view */
	.view {
		display: grid;
		gap: var(--s3);
		min-width: 0;
		padding-top: var(--s6);
		border-top: 1px solid var(--rule);
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	.vh {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		justify-content: space-between;
		gap: var(--s2) var(--s4);
	}
	.vh h2 {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: var(--s1) var(--s3);
		min-width: 0;
		overflow-wrap: anywhere;
	}
	.lab {
		font-weight: var(--w-body);
		font-size: var(--t-md);
		color: var(--ink-2);
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: var(--s2);
		margin: 0;
	}
	.grid {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4) var(--s5);
		align-items: start;
	}
	@container (min-width: 820px) {
		.grid {
			grid-template-columns: 260px minmax(0, 1fr);
		}
	}
	.facts {
		display: grid;
		gap: var(--s3);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.facts p {
		margin: 0;
	}
	.k {
		display: block;
		margin-bottom: 2px;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}

	/* views made by packages: a compact list, one group per package */
	.pkgs {
		display: grid;
		gap: var(--s4);
		padding-top: var(--s6);
		border-top: 2px solid var(--rule-strong);
	}
	.pkgs > h2 {
		margin: 0;
	}
	.pg {
		display: grid;
		gap: var(--s2);
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	.pg h3 {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: var(--s1) var(--s3);
		margin: var(--s3) 0 0;
		font-size: var(--t-md);
		line-height: var(--lh-md);
	}
	.ls {
		font-size: var(--t-sm);
		font-weight: var(--w-body);
	}
	.pg .n {
		font-size: var(--t-xs);
		font-weight: var(--w-body);
		color: var(--muted);
	}
	.rows {
		margin: 0;
		padding: 0;
		list-style: none;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--panel);
		overflow: hidden;
	}
	.rows > li {
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	.rows > li + li {
		border-top: 1px solid var(--rule);
	}
	summary {
		display: grid;
		grid-template-columns: minmax(8em, 13em) minmax(0, 1fr) auto auto;
		align-items: center;
		gap: var(--s1) var(--s3);
		min-height: 44px;
		padding: var(--s1) var(--s4);
		cursor: pointer;
		list-style: none;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	summary::-webkit-details-marker {
		display: none;
	}
	summary:hover {
		background: var(--here-bg);
	}
	.key {
		overflow-wrap: anywhere;
	}
	.ti {
		color: var(--ink-2);
	}
	.ks {
		display: flex;
		gap: var(--s1);
	}
	.go {
		min-width: 4em;
		text-align: right;
		font-weight: var(--w-ui);
		text-decoration: underline;
		text-decoration-thickness: 1px;
		text-underline-offset: 3px;
	}
	details[open] > summary {
		background: var(--here-bg);
		box-shadow: inset 3px 0 0 var(--here);
	}
	.rb {
		display: grid;
		gap: var(--s3);
		padding: var(--s3) var(--s4) var(--s4);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.rb p {
		margin: 0;
	}
	@container (max-width: 640px) {
		summary {
			grid-template-columns: minmax(0, 1fr) auto;
		}
		.ti {
			grid-column: 1;
		}
		.ks {
			grid-column: 1;
			flex-wrap: wrap;
		}
		.go {
			grid-column: 2;
			grid-row: 1 / span 3;
		}
	}
</style>

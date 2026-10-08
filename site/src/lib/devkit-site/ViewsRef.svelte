<!--
  /devkit/reference/views/: every view a lesson step can show, generated at build time from the view metas (key, title,
  kinds, minimum height, the text a screen reader hears for its demo, the ops that use it as their zoom view), each
  with its live demos (the /dev/atlas/ gallery's, badged Illustrative). Kit views first, then the views of packages.
-->
<script lang="ts">
	import { base } from '$app/paths';
	import { t } from '$lib/i18n/i18n.svelte';
	import Demo from './Demo.svelte';
	import { withBase } from './base.ts';
	import type { ViewsData } from './types.ts';

	let { page }: { page: ViewsData } = $props();
	const kit = $derived(page.views.filter((v) => !v.pkg));
	const pkg = $derived(page.views.filter((v) => v.pkg));
	const anchor = (key: string) => `view-${key.replace(/[^\w-]/g, '-')}`;
	const KIND: Record<string, { en: string; zh: string }> = {
		trace: { en: 'A · trace', zh: 'A · 推理轨迹' },
		train: { en: 'B · train', zh: 'B · 训练' },
		process: { en: 'C · process', zh: 'C · 过程' }
	};
</script>

{#snippet card(v: (typeof page.views)[number])}
	<article class="view" id={anchor(v.key)} aria-labelledby="{anchor(v.key)}-h">
		<header class="vh">
			<h2 id="{anchor(v.key)}-h"><code>{v.key}</code> <span class="lab">{t(v.title)}</span></h2>
			<div class="chips">
				{#each v.kinds as k (k)}<span class="chip">{t(KIND[k] ?? { en: k, zh: k })}</span>{/each}
				{#if v.minHeight}<span class="chip">{t({ en: `min height ${v.minHeight} px`, zh: `最小高度 ${v.minHeight} 像素` })}</span>{/if}
				{#if v.pkg}<span class="chip">{t({ en: `local to ${v.pkg}`, zh: `${v.pkg} 包自己的视图` })}</span>{/if}
			</div>
		</header>
		{#if v.ops.length}
			<p class="ops small">
				{t({ en: 'The zoom view of', zh: '这些算子的放大视图：' })}{' '}
				{#each v.ops as o, i (o)}<a href="{base}/devkit/reference/ops/#op-{o}"><code>{o}</code></a>{i < v.ops.length - 1 ? ', ' : ''}{/each}
			</p>
		{/if}
		<p class="use small">
			{#if v.pkg}
				{t({ en: `Used by its package as view: '${v.key}'.`, zh: `由它所在的包使用：view: '${v.key}'。` })}
			{:else}
				{t({ en: `A lesson step uses it as view: '${v.key}'.`, zh: `课程的一步这样使用它：view: '${v.key}'。` })}
			{/if}
		</p>
		{#if v.describe}
			<p class="say small"><span class="muted">{t({ en: 'Its demo, as a screen reader hears it:', zh: '它的演示，读屏软件读出的内容：' })}</span> {t(v.describe)}</p>
		{/if}
		<Demo view={v.key} key={v.key} minHeight={v.minHeight} say={false} />
	</article>
{/snippet}

<div class="container ref">
	<header class="head">
		<p class="eyebrow">DevKit · {t({ en: 'Reference', zh: '参考' })}</p>
		<h1>{t({ en: 'View reference', zh: '视图参考' })}</h1>
		<p class="lede">
			{t({
				en: `Every view a lesson step can show: ${kit.length} kit views and ${pkg.length} views that packages made for themselves, generated from their meta.ts files when the site is built. A view draws one step's stage from the runtime and the step's frame; each demo below runs the real view on demo data.`,
				zh: `课程每一步可以显示的视图：${kit.length} 个工具包视图，以及 ${pkg.length} 个由各个包为自己做的视图，在网站构建时由它们的 meta.ts 生成。视图根据运行时和这一步的帧画出舞台；下面每个演示都在演示数据上运行真实的视图。`
			})}
		</p>
		<p class="muted small">
			{t({ en: 'The same list in the guide:', zh: '指南里的同一张表：' })} <a href={withBase(page.guide, base)}>{t({ en: 'View reference (guide)', zh: '视图参考（指南）' })}</a>
		</p>
		<nav class="jump" aria-label={t({ en: 'Views', zh: '视图' })}>
			{#each page.views as v (v.key)}<a href="#{anchor(v.key)}"><code>{v.key}</code></a>{/each}
		</nav>
	</header>

	{#each kit as v (v.key)}{@render card(v)}{/each}

	{#if pkg.length}
		<section class="pkgs" aria-labelledby="pkg-h">
			<h2 id="pkg-h" class="sub">{t({ en: 'Views made by packages', zh: '各个包自己做的视图' })}</h2>
			<p class="muted small">{t({ en: "A package can add its own view (explainers/<slug>/views/<key>/, used as '<slug>/<key>'); these are the ones that exist today.", zh: '包可以添加自己的视图（explainers/<slug>/views/<key>/，用作 \'<slug>/<key>\'）；下面是现有的这些。' })}</p>
		</section>
		{#each pkg as v (v.key)}{@render card(v)}{/each}
	{/if}
</div>

<style>
	.ref {
		padding-top: var(--s6);
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s5);
	}
	.head h1 {
		margin-bottom: var(--s3);
	}
	.small {
		font-size: var(--fs-sm);
	}
	.jump {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 10px;
		margin-top: var(--s3);
		font-size: var(--fs-sm);
	}
	.jump code {
		font-size: 0.85em;
	}
	.view {
		padding-top: var(--s5);
		border-top: 1px solid var(--rule);
		scroll-margin-top: calc(var(--header-h) + 12px);
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.vh {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 8px 16px;
	}
	.vh h2 {
		margin: 0;
		font-size: var(--fs-xl);
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 12px;
		min-width: 0;
		overflow-wrap: anywhere;
	}
	.vh h2 code {
		font-size: 0.85em;
	}
	.lab {
		font-family: var(--font-body);
		font-weight: 400;
		font-size: var(--fs-md);
		color: var(--ink-2);
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.ops,
	.use,
	.say {
		margin: 0;
		color: var(--ink-2);
	}
	.pkgs {
		padding-top: var(--s5);
		border-top: 2px solid var(--rule-strong);
	}
	.sub {
		font-size: var(--fs-xl);
		margin-bottom: 6px;
	}
</style>

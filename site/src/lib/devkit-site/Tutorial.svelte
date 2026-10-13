<!--
  /devkit/tutorial/: "Build hello-xor step by step", played by the kit Walkthrough (the lesson format, dogfooded). Each
  step's stage is the command (from DEVKIT.md section 4), the real file (an excerpt cut out of the hello-xor package at
  build time) and what it produces (read or run at build time: spec.json, the training tape, the frames of the
  slowed-down step, ...). The finished hello-xor lesson can be loaded at the end of the page.
-->
<script lang="ts">
	import { base } from '$app/paths';
	import Walkthrough from '$lib/kit/Walkthrough.svelte';
	import type { WalkStep } from '$lib/kit/types';
	import { t } from '$lib/i18n/i18n.svelte';
	import Html from './Html.svelte';
	import Produces from './Produces.svelte';
	import HelloEmbed from './HelloEmbed.svelte';
	import Pager from './Pager.svelte';
	import { withBase } from './base.ts';
	import type { TutorialData } from './types.ts';

	let { page }: { page: TutorialData } = $props();
	let index = $state(0);
	const steps = $derived<WalkStep[]>(page.steps.map((s) => ({ id: s.id, title: s.title, narration: s.narration, look: s.look ?? undefined })));
	const cur = $derived(page.steps[Math.max(0, Math.min(page.steps.length - 1, index))]);
</script>

{#snippet stage()}
	<div class="st">
		{#if cur.commandHtml}
			<div>
				<p class="k">{t({ en: 'Run', zh: '运行' })}</p>
				<Html html={cur.commandHtml} />
			</div>
		{/if}
		{#if cur.fileHtml}
			<div>
				<p class="k">{t({ en: 'The real file (hello-xor)', zh: '真实的文件（hello-xor）' })}</p>
				<Html html={cur.fileHtml} />
			</div>
		{/if}
		<Produces p={cur.produces} />
	</div>
{/snippet}

{#snippet end()}
	<p class="endnote">
		{t({ en: 'That is the whole path. Below, load the finished hello-xor lesson and step through it.', zh: '整个流程就是这样。在下面加载完成的 hello-xor 课程，逐步看一遍。' })}
		<a href="#hello-xor">{t({ en: 'The finished lesson', zh: '完成的课程' })} ↓</a>
	</p>
{/snippet}

<div class="tut">
	<header class="head">
		<p class="eyebrow">DevKit · {t({ en: 'Tutorial', zh: '教程' })} · {t({ en: `${page.steps.length} steps`, zh: `${page.steps.length} 步` })}</p>
		<h1>{t({ en: 'Build hello-xor step by step', zh: '一步一步做出 hello-xor' })}</h1>
		<p class="lede">
			{t({
				en: 'hello-xor is the kit’s own example: a 2-2-1 network on XOR that uses all three kinds. This tutorial follows section 4 of the guide. The commands build your package, my-xor, as the guide does; the files and the results are hello-xor’s, cut out of the real package when this site was built.',
				zh: 'hello-xor 是工具包自带的例子：一个学习 XOR 的 2-2-1 网络，三种类型都用到了。本教程按指南第 4 节的顺序进行。命令和指南一样，做的是你自己的包 my-xor；文件和结果则来自 hello-xor，是在网站构建时从真实的包里截取的。'
			})}
		</p>
		<p class="small">
			{t({ en: 'Use ← and → or the buttons to move between steps. The full text of each step is in the guide:', zh: '用 ← 和 → 键或按钮切换步骤。每一步的完整说明见指南：' })}
			<a href={withBase(page.guide, base)}>{t({ en: 'Walk-through: your first package', zh: '动手做第一个包' })}</a>
		</p>
	</header>

	<Walkthrough {steps} bind:index id="tutorial" label={{ en: 'Build hello-xor step by step', zh: '一步一步做出 hello-xor' }} {stage} {end} stageMinHeight={380} />

	<section id="hello-xor" class="embed" aria-labelledby="hello-xor-h">
		<h2 id="hello-xor-h">{t({ en: 'The finished lesson', zh: '完成的课程' })}</h2>
		<p class="intro">
			{t({
				en: 'This is what the files above make: the real hello-xor package, loaded and run in your browser by the same kit as every model page. hello-xor is a lab-only package, so it has no page of its own on the site.',
				zh: '这就是上面那些文件做出来的东西：真实的 hello-xor 包，由与每个模型页相同的工具包在你的浏览器里加载并运行。hello-xor 只用于实验室，所以它在网站上没有自己的页面。'
			})}
		</p>
		<HelloEmbed slug={page.hello.slug} spec={page.hello.spec} />
	</section>

	<Pager label={t({ en: 'Previous and next page', zh: '上一页和下一页' })} prev={{ href: '/devkit/', title: t({ en: 'Overview', zh: '概览' }) }} next={{ href: '/devkit/guide/', title: t({ en: 'The DevKit guide', zh: 'DevKit 指南' }) }} />
</div>

<style>
	.tut {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s6);
	}
	.head .eyebrow {
		margin: 0 0 var(--s2);
	}
	.head h1 {
		margin-bottom: var(--s3);
	}
	.head .lede {
		margin-bottom: var(--s3);
	}
	.small {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
	}
	.st {
		display: grid;
		gap: var(--s3);
		min-width: 0;
	}
	.st > div {
		min-width: 0;
	}
	.k {
		margin: 0 0 var(--s1);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
		color: var(--ink-2);
	}
	.st :global(.dk-md .code) {
		margin-bottom: 0;
	}
	.endnote {
		margin: 0;
		font-size: var(--t-sm);
	}
	.embed {
		padding-top: var(--s5);
		border-top: 1px solid var(--rule);
		scroll-margin-top: calc(var(--header-h) + 12px);
		min-width: 0;
	}
	.intro {
		color: var(--ink-2);
		max-width: 75ch;
	}
	.tut > :global(.pg) {
		margin-top: 0;
	}
</style>

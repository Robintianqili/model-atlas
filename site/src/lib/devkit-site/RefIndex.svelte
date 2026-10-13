<!-- /devkit/reference/: the two reference pages, as two link cards with their counts. -->
<script lang="ts">
	import { base } from '$app/paths';
	import { t } from '$lib/i18n/i18n.svelte';
	import Glyph from './Glyph.svelte';
	import { arrow, code, lesson } from './icons.ts';
	import type { ReferenceData } from './types.ts';

	let { page }: { page: ReferenceData } = $props();
</script>

<div class="ri">
	<header>
		<p class="eyebrow">DevKit · {t({ en: 'Reference', zh: '参考' })}</p>
		<h1>{t({ en: 'Reference', zh: '参考' })}</h1>
		<p class="lede">{t({ en: 'The parts a lesson is made of, generated from the kit when the site is built, each with a live demo.', zh: '组成一节课的各个部件，在网站构建时由工具包生成，每个都有实时演示。' })}</p>
	</header>
	<div class="cards">
		<a class="card" href="{base}/devkit/reference/ops/">
			<span class="top"><span class="tile" aria-hidden="true"><Glyph d={code} size={18} /></span><strong>{t({ en: 'Ops', zh: '算子' })}</strong><span class="n num">{page.ops}</span></span>
			<span class="d">{t({ en: 'The arithmetic of a trained model, layer by layer: formula, narration templates, attributes, parity bar and PyTorch equivalent of every core op.', zh: '训练好的模型逐层的计算：每个核心算子的公式、旁白模板、属性、一致性阈值和对应的 PyTorch 写法。' })}</span>
			<span class="go">{t({ en: 'Op reference', zh: '算子参考' })}<Glyph d={arrow} size={16} /></span>
		</a>
		<a class="card" href="{base}/devkit/reference/views/">
			<span class="top"><span class="tile" aria-hidden="true"><Glyph d={lesson} size={18} /></span><strong>{t({ en: 'Views', zh: '视图' })}</strong><span class="n num">{page.views + page.packageViews}</span></span>
			<span class="d">{t({ en: `What a lesson step can show: ${page.views} kit views and ${page.packageViews} views made by packages.`, zh: `课程的一步可以显示什么：${page.views} 个工具包视图和 ${page.packageViews} 个包自己做的视图。` })}</span>
			<span class="go">{t({ en: 'View reference', zh: '视图参考' })}<Glyph d={arrow} size={16} /></span>
		</a>
	</div>
</div>

<style>
	.ri {
		display: grid;
		gap: var(--s5);
	}
	header .eyebrow {
		margin: 0 0 var(--s2);
	}
	h1 {
		margin-bottom: var(--s3);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr));
		gap: var(--s4);
	}
	.card {
		display: grid;
		gap: var(--s3);
		align-content: start;
		padding: var(--s5);
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
		color: var(--ink);
		text-decoration: none;
		transition: border-color var(--dur-1) var(--ease);
	}
	.card:hover {
		border-color: var(--ink);
	}
	.top {
		display: flex;
		align-items: center;
		gap: var(--s3);
	}
	.top strong {
		font-size: var(--t-xl);
		line-height: var(--lh-xl);
	}
	.tile {
		display: inline-grid;
		place-items: center;
		width: 36px;
		height: 36px;
		border-radius: var(--r2);
		background: var(--here-bg);
		color: var(--ink);
	}
	.n {
		margin-left: auto;
		font-size: var(--t-2xl);
		line-height: var(--lh-2xl);
		font-weight: var(--w-ui);
	}
	.d {
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.go {
		display: inline-flex;
		align-items: center;
		gap: var(--s1);
		font-weight: var(--w-ui);
		text-decoration: underline;
		text-decoration-thickness: 1px;
		text-underline-offset: 3px;
	}
</style>

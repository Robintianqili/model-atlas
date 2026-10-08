<!-- /devkit/reference/: the two reference pages. -->
<script lang="ts">
	import { base } from '$app/paths';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { ReferenceData } from './types.ts';

	let { page }: { page: ReferenceData } = $props();
</script>

<div class="container ri">
	<header>
		<p class="eyebrow">DevKit · {t({ en: 'Reference', zh: '参考' })}</p>
		<h1>{t({ en: 'Reference', zh: '参考' })}</h1>
		<p class="lede">{t({ en: 'The parts a lesson is made of, generated from the kit when the site is built, each with a live demo.', zh: '组成一节课的各个部件，在网站构建时由工具包生成，每个都有实时演示。' })}</p>
	</header>
	<div class="cards">
		<a class="panel card" href="{base}/devkit/reference/ops/">
			<span class="n num">{page.ops}</span>
			<strong>{t({ en: 'Ops', zh: '算子' })}</strong>
			<span>{t({ en: 'The arithmetic of a trained model, layer by layer: formula, narration templates, attributes, parity bar and PyTorch equivalent of every core op.', zh: '训练好的模型逐层的计算：每个核心算子的公式、旁白模板、属性、一致性阈值和对应的 PyTorch 写法。' })}</span>
		</a>
		<a class="panel card" href="{base}/devkit/reference/views/">
			<span class="n num">{page.views + page.packageViews}</span>
			<strong>{t({ en: 'Views', zh: '视图' })}</strong>
			<span>{t({ en: `What a lesson step can show: ${page.views} kit views and ${page.packageViews} views made by packages.`, zh: `课程的一步可以显示什么：${page.views} 个工具包视图和 ${page.packageViews} 个包自己做的视图。` })}</span>
		</a>
	</div>
</div>

<style>
	.ri {
		padding-top: var(--s6);
		display: grid;
		gap: var(--s5);
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
		gap: 6px;
		color: var(--ink);
		text-decoration: none;
	}
	.card:hover {
		border-color: var(--accent);
	}
	.n {
		font-size: var(--fs-2xl);
		font-weight: 700;
		color: var(--accent-strong);
	}
	.card strong {
		font-size: var(--fs-lg);
	}
	.card span:last-child {
		color: var(--ink-2);
		font-size: var(--fs-sm);
	}
</style>

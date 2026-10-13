<!-- TinyGPT's credits and licences: the About sheet and the Training view show them. -->
<script lang="ts">
	import { base } from '$app/paths';
	import { t } from '$lib/i18n/i18n.svelte';
</script>

<p class="credits">
	{t({
		en: 'Model, tokenizer and page by Model Atlas. Training data: TinyStories (Eldan & Li 2023, CDLA-Sharing-1.0). Ideas from nanoGPT/minGPT (MIT), Transformer Explainer (MIT), BertViz (Apache-2.0) and the logit lens post; no code copied. Full list on the About page.',
		zh: '模型、分词器和页面由 Model Atlas 制作。训练数据：TinyStories（Eldan 与 Li 2023，CDLA-Sharing-1.0）。借鉴了 nanoGPT/minGPT（MIT）、Transformer Explainer（MIT）、BertViz（Apache-2.0）和 logit lens 一文的思路，未复制代码。完整列表见“关于”页面。'
	})}
	<a href="{base}/about/">{t({ en: 'About and licenses', zh: '关于与许可' })}</a>
</p>

<style>
	.credits {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
</style>

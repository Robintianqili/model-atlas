<!-- TinyCNN's credits and licences: the About sheet and the Training view show them. -->
<script lang="ts">
	import { base } from '$app/paths';
	import { t } from '$lib/i18n/i18n.svelte';
</script>

<ul class="credits">
	<li>
		{t({ en: 'Training data and preset drawings', zh: '训练数据与预设画' })}:
		<a href="https://github.com/googlecreativelab/quickdraw-dataset" rel="noopener">Quick, Draw! dataset</a>, Google Creative Lab, CC BY 4.0.
	</li>
	<li>
		{t({ en: 'Layout idea (TinyVGG, per-cell convolution arithmetic)', zh: '布局思路（TinyVGG、逐格卷积计算）' })}:
		<a href="https://github.com/poloclub/cnn-explainer" rel="noopener">CNN Explainer</a>, Wang et al., Georgia Tech Polo Club (MIT). {t({ en: 'No code copied.', zh: '未复制代码。' })}
	</li>
	<li>{t({ en: 'Model, weights and this page: Model Atlas. Full list on the About page.', zh: '模型、权重和本页：Model Atlas。完整列表见“关于”页。' })} <a href="{base}/about/">{t({ en: 'Licenses', zh: '许可' })}</a></li>
</ul>

<style>
	.credits {
		margin: 0;
		padding-left: 1.2em;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.credits li + li {
		margin-top: 6px;
	}
</style>

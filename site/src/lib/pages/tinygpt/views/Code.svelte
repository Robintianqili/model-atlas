<!--
  TinyGPT's Code view (redesign spec §5.9), loaded only when the view opens: one block and the forward pass in PyTorch
  (used for training), how this page runs the model, and the parity check against PyTorch.
-->
<script lang="ts">
	import CodeBlock from '$lib/kit/CodeBlock.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { BLOCK_CODE, USAGE_CODE } from '../code';

	let { vp: _vp }: { vp: unknown; section?: string } = $props();
</script>

<div class="codes">
	<p class="parity">
		{t({
			en: 'Every one of the 66 traced tensors matches PyTorch within 1e-4 (the largest difference measured on fresh prompts was 4.2e-5, at a full 64-token context).',
			zh: '66 个被追踪的张量全部与 PyTorch 相差在 1e-4 以内（在新提示词上实测的最大差异为 4.2e-5，出现在满 64 个词元的上下文时）。'
		})}
	</p>
	<CodeBlock code={BLOCK_CODE} lang="python" caption={{ en: 'pipeline/tinygpt/model.py · one block and the forward pass (PyTorch, used for training)', zh: 'pipeline/tinygpt/model.py · 一个块与前向计算（PyTorch，用于训练）' }} />
	<CodeBlock code={USAGE_CODE} lang="ts" caption={{ en: 'How this page runs the model (TypeScript)', zh: '本页如何运行模型（TypeScript）' }} />
	<p class="muted small">
		{t({ en: 'The Inspector in Explore shows the exact line of the PyTorch reference for each of the 66 tensors. Further reading:', zh: '“探索”里的检查器会为 66 个张量逐一显示 PyTorch 参考实现中对应的那一行。延伸阅读：' })}
		<a href="https://github.com/karpathy/nanoGPT" rel="noopener">nanoGPT</a>,
		<a href="https://zh.d2l.ai/chapter_attention-mechanisms/index.html" rel="noopener">{t({ en: 'd2l.ai attention (中文)', zh: '《动手学深度学习》注意力机制' })}</a>,
		<a href="https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html" rel="noopener">d2l.ai (English)</a>.
	</p>
</div>

<style>
	.codes {
		display: grid;
		gap: 12px;
	}
	.parity {
		max-width: var(--measure);
		margin: 0;
		color: var(--ink-2);
	}
	.small {
		font-size: var(--t-sm);
	}
</style>

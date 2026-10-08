<!--
  The page's hook: the learner's sentence (shared by the lesson and every part below) with a live
  one-line summary: tokens, training examples and TinyGPT's loss on it.
-->
<script lang="ts">
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Num from '$lib/kit/Num.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import Term from '$lib/kit/Term.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { NtpState } from './state.svelte';
	import { MAX_CHARS, PRESETS } from './facts';

	let { S }: { S: NtpState } = $props();
	const n = $derived(S.ids.length);
	const loss = $derived(S.fresh && S.score && S.score.scored > 0 ? S.score.meanNats : null);
</script>

<div class="ys">
	<div class="prose">
		<Lang>
			{#snippet en()}
				<p>
					A GPT-style language model is <Term id="pre-training">pre-trained</Term> on one task only: guess the next
					<Term id="token" />. This page shows that task on a sentence of your own, with TinyGPT (364,040 parameters, trained
					on children’s stories) running in your browser. Type a sentence or keep this one, then start the lesson.
				</p>
			{/snippet}
			{#snippet zh()}
				<p>
					GPT 一类的语言模型在<Term id="pre-training">预训练</Term>时只做一件事：猜下一个<Term id="token">词元</Term>。本页就在你自己的一句话上演示这件事，用的是在你的浏览器里运行的
					TinyGPT（364,040 个参数，用儿童故事训练）。输入一句话，或者就用这一句，然后开始课程。
				</p>
			{/snippet}
		</Lang>
	</div>
	<TextInputBar bind:value={S.text} maxLength={MAX_CHARS} presets={PRESETS} label={{ en: 'Your sentence (the lesson and every part below use it)', zh: '你的句子（课程和下面各部分共用）' }} />
	<p class="sum small">
		{#if S.error}
			<span class="err">{S.error}</span>
		{:else if !S.tok}
			<span class="muted">{t({ en: 'Loading TinyGPT…', zh: '正在加载 TinyGPT…' })}</span>
		{:else if n === 0}
			{t({ en: 'Type something to begin.', zh: '先输入一点文字。' })}
		{:else}
			<strong class="num">{n}</strong>
			{t({ en: n === 1 ? 'token' : 'tokens', zh: '个词元' })} →
			<strong class="num">{Math.max(0, n - 1)}</strong>
			{t({ en: n - 1 === 1 ? 'training example' : 'training examples', zh: '个训练样本' })}
			{#if loss !== null}
				· {t({ en: 'TinyGPT’s loss on it:', zh: 'TinyGPT 在它上面的损失：' })}
				<strong><Num value={loss} /></strong>
				{t({ en: 'nats/token', zh: '奈特/词元' })}
				<ProvenanceBadge mode="live" compact />
			{/if}
		{/if}
	</p>
</div>

<style>
	.ys {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s3);
	}
	.prose {
		max-width: var(--measure);
	}
	.prose p {
		margin: 0;
	}
	.sum {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 6px;
	}
	.small {
		font-size: var(--fs-sm);
	}
	.err {
		color: var(--k-text);
	}
</style>

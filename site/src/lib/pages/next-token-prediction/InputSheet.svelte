<!--
  The input sheet (redesign spec §4.4 InputSheet; the stage header's chip and #input open it): the learner's sentence,
  shared by the lesson and every part of Explore, with the live summary (tokens, training examples, TinyGPT's loss).
-->
<script lang="ts">
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import type { NtpState } from './state.svelte';
	import { MAX_CHARS, PRESETS } from './facts';

	let { S }: { S: NtpState } = $props();
	const n = $derived(S.ids.length);
	const loss = $derived(S.fresh && S.score && S.score.scored > 0 ? S.score.meanNats : null);
</script>

<div class="sheet">
	<TextInputBar bind:value={S.text} maxLength={MAX_CHARS} presets={PRESETS} label={{ en: 'Your sentence (the lesson and Explore use it)', zh: '你的句子（课程和“探索”共用）' }} />
	<p class="sum" aria-live="polite">
		{#if S.error}
			<span class="err">{S.error}</span>
		{:else if !S.tok}
			{t({ en: 'Loading TinyGPT…', zh: '正在加载 TinyGPT…' })}
		{:else if n === 0}
			{t({ en: 'Type something to begin.', zh: '先输入一点文字。' })}
		{:else}
			<b class="num">{n}</b>
			{t({ en: n === 1 ? 'token' : 'tokens', zh: '个词元' })} →
			<b class="num">{Math.max(0, n - 1)}</b>
			{t({ en: n - 1 === 1 ? 'training example' : 'training examples', zh: '个训练样本' })}{#if loss !== null}{' · '}{t({ en: 'TinyGPT’s loss on it:', zh: 'TinyGPT 在它上面的损失：' })}
				<b class="num">{fmt3(loss)}</b> {t({ en: 'nats/token', zh: '奈特/词元' })}{/if}
		{/if}
	</p>
	<p class="hint">{t({ en: 'The step you are on updates as you type. Your text never leaves your device.', zh: '你一边输入，当前这一步就一边更新。你的文本不会离开你的设备。' })}</p>
</div>

<style>
	.sheet {
		display: grid;
		gap: var(--s3);
	}
	.sum,
	.hint {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.hint {
		color: var(--muted);
	}
	.err {
		color: var(--no);
	}
</style>

<!--
  The lesson's side panel: the learner's sentence, its token count and TinyGPT's loss on it, and
  an "Edit here" toggle so the sentence can change mid-lesson (every step updates at once).
-->
<script lang="ts">
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { NtpLesson } from './lessonState.svelte';
	import { MAX_CHARS, PRESETS } from './facts';

	let { L }: { L: NtpLesson } = $props();
	const S = $derived(L.S);
	let open = $state(false);
	const loss = $derived(S.fresh && S.score && S.score.scored > 0 ? S.score.meanNats : null);
</script>

<div class="you">
	<div class="you-h">
		<h4>{t({ en: 'Your sentence', zh: '你的句子' })}</h4>
		{#if S.tok}
			<span class="meta small">
				<span class="num">{L.n}</span>
				{t({ en: L.n === 1 ? 'token' : 'tokens', zh: '个词元' })}{#if loss !== null}{' '}·
					{t({ en: 'loss', zh: '损失' })} <Num value={loss} />{/if}
			</span>
		{/if}
	</div>
	{#if !open}
		<p class="sent">{S.text || t({ en: '(empty)', zh: '（空）' })}</p>
	{/if}
	{#if open}
		<TextInputBar
			bind:value={S.text}
			maxLength={MAX_CHARS}
			presets={PRESETS}
			label={{ en: 'Your sentence (the whole page uses it)', zh: '你的句子（全页共用）' }}
		/>
	{/if}
	<p class="hint">
		{t({ en: 'Change it at any time: the step you are on updates at once.', zh: '随时可以改：你所在的这一步会立刻更新。' })}
	</p>
	<button type="button" class="btn small" aria-expanded={open} onclick={() => (open = !open)}>
		<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M11 2.5 13.5 5 6 12.5H3.5V10z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" /></svg>
		{open ? t({ en: 'Done', zh: '完成' }) : t({ en: 'Edit here', zh: '在这里修改' })}
	</button>
</div>

<style>
	.you {
		display: grid;
		gap: 8px;
		padding: 10px 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface-2);
		min-width: 0;
	}
	.you-h {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		justify-content: space-between;
		gap: 4px 12px;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.meta {
		color: var(--ink-2);
	}
	.small {
		font-size: var(--fs-xs);
	}
	.sent {
		margin: 0;
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		line-height: 1.5;
		overflow-wrap: anywhere;
		display: -webkit-box;
		-webkit-line-clamp: 3;
		line-clamp: 3;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}
	.hint {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.btn.small {
		justify-self: start;
		gap: 6px;
		min-height: 32px;
		padding: 0 12px;
		font-size: var(--fs-sm);
	}
</style>

<!--
  Lesson step 1: the sentence becomes tokens. Text -> pieces from TinyGPT's 1,024-token list ->
  the numbers the model actually receives. Live: TinyGPT's own byte-level BPE tokenizer.
-->
<script lang="ts">
	import { base } from '$app/paths';
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { TokenItem } from '$lib/kit/types';
	import type { NtpLesson } from './lessonState.svelte';
	import { MAX_TOKENS } from './facts';

	let { L }: { L: NtpLesson } = $props();
	const S = $derived(L.S);

	const chars = $derived([...S.text].length);
	const items = $derived<TokenItem[]>(
		S.tok ? S.ids.map((id) => ({ text: S.disp(id), id, bytes: Array.from(S.tok!.tokenBytes(id)) })) : []
	);
</script>

<div class="tk">
	<div class="row">
		<p class="eyebrow">{t({ en: `Your text · ${chars} characters`, zh: `你的文本 · ${chars} 个字符` })}</p>
		<p class="raw">{S.text || '∅'}</p>
	</div>
	<p class="arrow" aria-hidden="true"><span>↓</span> {t({ en: 'cut into pieces from a fixed list of 1,024 tokens', zh: '按一张固定的 1,024 词元表切成小块' })}</p>
	<div class="row">
		<p class="eyebrow">{t({ en: `${L.n} ${L.n === 1 ? 'token' : 'tokens'}`, zh: `${L.n} 个词元` })}</p>
		{#if items.length}
			<TokenChips tokens={items} showIds selectable={false} label={{ en: 'Your sentence as tokens', zh: '你的句子切成的词元' }} />
		{:else}
			<p class="muted small">{t({ en: 'Type something to begin.', zh: '先输入一点文字。' })}</p>
		{/if}
	</div>
	<p class="arrow" aria-hidden="true"><span>↓</span> {t({ en: 'each token is replaced by its number (its id)', zh: '每个词元换成它的编号（id）' })}</p>
	<div class="row">
		<p class="eyebrow">{t({ en: 'What TinyGPT receives', zh: 'TinyGPT 收到的输入' })}</p>
		<p class="ids num">[{S.ids.join(', ')}]</p>
	</div>
	{#if S.cut > 0}
		<p class="note small">
			{t({ en: `Only the first ${MAX_TOKENS} tokens are used; ${S.cut} more were cut.`, zh: `只用前 ${MAX_TOKENS} 个词元，另外 ${S.cut} 个被截掉了。` })}
		</p>
	{/if}
	{#if S.hasCjk}
		<p class="warn small">
			{t({
				en: 'TinyGPT’s tokens were learned from English stories only, so each Chinese character falls back to 3 byte tokens (shown as <0x..>). The model has almost never seen them.',
				zh: 'TinyGPT 的词元只从英文故事里学来，所以每个汉字都退回成 3 个字节词元（显示为 <0x..>）。模型几乎从没见过它们。'
			})}
		</p>
	{/if}
	<p class="small muted">
		{t({ en: 'How the 1,024 tokens were chosen:', zh: '这 1,024 个词元是怎么选出来的：' })}
		<a href="{base}/train/tokenization/">{t({ en: 'Tokenization', zh: '分词' })} →</a>
	</p>
</div>

<style>
	.tk {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.row {
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	.eyebrow {
		margin: 0;
	}
	.raw {
		margin: 0;
		padding: 8px 10px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		font-size: var(--fs-md);
		overflow-wrap: anywhere;
		white-space: pre-wrap;
	}
	.arrow {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.arrow span {
		font-size: var(--fs-lg);
		color: var(--muted);
		width: 1.2em;
		text-align: center;
	}
	.ids {
		margin: 0;
		padding: 8px 10px;
		border-radius: var(--r2);
		background: var(--surface-2);
		font-size: var(--fs-sm);
		overflow-wrap: anywhere;
		line-height: 1.6;
	}
	.small {
		font-size: var(--fs-xs);
		margin: 0;
	}
	.note {
		color: var(--ink-2);
	}
	.warn {
		padding: 8px 10px;
		border-left: 3px solid var(--illus);
		background: var(--illus-soft);
		border-radius: 0 var(--r1) var(--r1) 0;
	}
</style>

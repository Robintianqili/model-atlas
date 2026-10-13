<!--
  The lesson's cover stage (redesign spec §5.6, answer first): the learner's sentence through TinyGPT, live: its tokens,
  the (context → next token) exercises it holds, and the final model's average loss and perplexity on it.
-->
<script lang="ts">
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import type { TokenItem } from '$lib/kit/types';
	import { sci } from './lessonMath';
	import type { NtpState } from './state.svelte';

	let { S }: { S: NtpState } = $props();
	const n = $derived(S.ids.length);
	const items = $derived<TokenItem[]>(S.tok ? S.ids.map((id) => ({ text: S.disp(id), id, bytes: Array.from(S.tok!.tokenBytes(id)) })) : []);
	const sc = $derived(S.fresh && S.score && S.score.scored > 0 ? S.score : null);
</script>

<figure class="cover">
	<blockquote class="sent">{S.text || t({ en: '(empty)', zh: '（空）' })}</blockquote>
	{#if S.error}
		<p class="err">{S.error}</p>
	{:else if !S.tok}
		<p class="muted">{t({ en: 'Loading TinyGPT…', zh: '正在加载 TinyGPT…' })}</p>
	{:else}
		<TokenChips tokens={items} showIds selectable={false} label={{ en: 'Your sentence as tokens', zh: '你的句子切成的词元' }} />
		<p class="flow" aria-live="polite">
			<span><b class="num">{n}</b> {t({ en: n === 1 ? 'token' : 'tokens', zh: '个词元' })}</span>
			<span class="arrow" aria-hidden="true">→</span>
			<span><b class="num">{Math.max(0, n - 1)}</b> {t({ en: 'exercises (context → next token)', zh: '道练习题（上下文 → 下一个词元）' })}</span>
			<span class="arrow" aria-hidden="true">→</span>
			<span>{t({ en: 'TinyGPT’s loss', zh: 'TinyGPT 的损失' })} <b class="num big">{sc ? fmt3(sc.meanNats) : '…'}</b> {t({ en: 'nats per token', zh: '奈特/词元' })}</span>
			<span class="sep" aria-hidden="true">·</span>
			<span>{t({ en: 'perplexity', zh: '困惑度' })} <b class="num">{sc ? sci(sc.perplexity) : '…'}</b></span>
		</p>
	{/if}
	<figcaption>
		{t({
			en: 'Your sentence through TinyGPT (364,040 parameters, trained on children’s stories), computed in your browser. The lesson takes these numbers apart, one step at a time.',
			zh: '这是 TinyGPT（364,040 个参数，用儿童故事训练）读你的句子得到的结果，在你的浏览器里现场算出。课程会一步一步拆开这些数。'
		})}
	</figcaption>
</figure>

<style>
	.cover {
		display: grid;
		gap: var(--s4);
		margin: 0;
		min-width: 0;
	}
	.sent {
		margin: 0;
		padding: var(--s3) var(--s4);
		border-left: 3px solid var(--ink);
		border-radius: 0 var(--r3) var(--r3) 0;
		background: var(--panel);
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
		overflow-wrap: anywhere;
	}
	.flow {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 10px;
		margin: 0;
		color: var(--ink-2);
	}
	.flow b {
		color: var(--ink);
		font-weight: var(--w-ui);
	}
	.flow .big {
		font-size: var(--t-xl);
	}
	.arrow,
	.sep {
		color: var(--muted);
	}
	figcaption {
		max-width: var(--measure);
		color: var(--muted);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.err {
		color: var(--no);
	}
</style>

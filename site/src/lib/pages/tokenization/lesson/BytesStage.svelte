<!--
  Lesson step 1: the learner's text as UTF-8 bytes. Every character sits on top of its bytes (hex);
  characters that take several bytes are drawn on a card. Below: how many characters take 1, 2, 3
  or 4 bytes (count × length = bytes), and a fixed key of four characters encoded live.
-->
<script lang="ts">
	import ByteView from '../ByteView.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtInt } from '$lib/kit/format';
	import { charSummary, charRanges, hexOf } from '../lesson';

	let { text }: { text: string } = $props();

	const cs = $derived(charSummary(text));
	/** characters drawn: fewer when most of them are 3-byte characters */
	const max = $derived(cs.bytes > cs.chars * 1.5 ? 40 : 64);
	const ranges = $derived(charRanges(text, max));
	let hover = $state<number | null>(null);

	const KINDS = [
		null,
		{ en: 'English letters, digits, spaces, basic punctuation', zh: '英文字母、数字、空格、半角标点' },
		{ en: 'accented Latin, Greek, Cyrillic, Arabic, …', zh: '带重音的拉丁字母、希腊字母、西里尔字母、阿拉伯字母等' },
		{ en: 'Chinese, Japanese, Korean, full-width punctuation, …', zh: '汉字、日文、韩文、全角标点等' },
		{ en: 'most emoji and rare characters', zh: '大多数表情符号和生僻字' }
	];
	const KEY = [
		{ ch: 'a', what: { en: 'English letter', zh: '英文字母' } },
		{ ch: 'é', what: { en: 'accented letter', zh: '带重音的字母' } },
		{ ch: '我', what: { en: 'Chinese character', zh: '汉字' } },
		{ ch: '🙂', what: { en: 'emoji', zh: '表情符号' } }
	];
	const enc = new TextEncoder();
	const maxRow = $derived(Math.max(1, ...cs.byLen.map((n, i) => n * i)));
</script>

<div class="ls">
	<h4>{t({ en: 'Each character becomes 1 to 4 bytes', zh: '每个字符变成 1 到 4 个字节' })}</h4>
	{#if cs.chars}
		<p class="flow" aria-live="polite">
			<span><b class="num">{fmtInt(cs.chars)}</b> {t({ en: 'characters', zh: '个字符' })}</span>
			<span class="arr" aria-hidden="true">→</span>
			<span><b class="num">{fmtInt(cs.bytes)}</b> {t({ en: 'UTF-8 bytes', zh: '个 UTF-8 字节' })}</span>
			<span class="arr" aria-hidden="true">→</span>
			<span><b class="num big">{fmtInt(cs.bytes)}</b> {t({ en: 'tokens to start with (one per byte)', zh: '个起始词元（每个字节一个）' })}</span>
		</p>
		<ByteView {text} {ranges} bind:hover {max} label={{ en: 'Your text as UTF-8 bytes', zh: '你的文本的 UTF-8 字节' }} />
		<p class="note">{t({ en: 'Under each character: its byte or bytes, written as base-16 numbers (00 to FF means 0 to 255).', zh: '每个字符下面是它的字节，用十六进制数写出（00 到 FF，即 0 到 255）。' })}</p>

		<div class="split">
			<p class="sh">{t({ en: 'Bytes per character in your text', zh: '你的文本里每个字符的字节数' })}</p>
			{#each cs.byLen as n, len (len)}
				{#if len > 0 && n > 0}
					<div class="row">
						<span class="lab"><b class="num">{len}</b> {len === 1 ? t({ en: 'byte', zh: '个字节' }) : t({ en: 'bytes', zh: '个字节' })}<span class="kind muted small">{t(KINDS[len]!)}</span></span>
						<span class="track"><span class="fill" style:width="{(100 * n * len) / maxRow}%"></span></span>
						<span class="eq num">{fmtInt(n)} × {len} = <b>{fmtInt(n * len)}</b></span>
					</div>
				{/if}
			{/each}
			{#if cs.byLen.filter((n) => n > 0).length > 1}
				<p class="sum num">= <b>{fmtInt(cs.bytes)}</b> {t({ en: 'bytes', zh: '个字节' })}</p>
			{/if}
		</div>

		<div class="key" role="list" aria-label={t({ en: 'Examples of UTF-8 lengths', zh: 'UTF-8 长度示例' })}>
			{#each KEY as k (k.ch)}
				<span class="kc" role="listitem">
					<span class="kch" lang={k.ch === '我' ? 'zh-CN' : undefined}>{k.ch}</span>
					<span class="num kb">{hexOf(k.ch)}</span>
					<span class="muted small">{t(k.what)} · {enc.encode(k.ch).length} {enc.encode(k.ch).length === 1 ? t({ en: 'byte', zh: '个字节' }) : t({ en: 'bytes', zh: '个字节' })}</span>
				</span>
			{/each}
		</div>
	{:else}
		<p class="muted">{t({ en: 'Your text is empty. Open “Your text” at the top of the stage: pick a preset or type something.', zh: '你的文本是空的。打开演示区顶部的“你的文本”，选一个预设，或者输入一些内容。' })}</p>
	{/if}
</div>

<style>
	.ls {
		container: ls / inline-size;
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 12px;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.flow {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 10px;
		color: var(--ink-2);
	}
	.arr {
		color: var(--muted);
	}
	.big {
		font-size: var(--fs-xl);
		color: var(--ink);
	}
	.split {
		display: grid;
		gap: 4px;
		padding: 10px 12px;
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	.sh {
		margin: 0 0 2px;
		font-size: var(--fs-sm);
		font-weight: 650;
	}
	.row {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(40px, 0.6fr) auto;
		align-items: center;
		gap: 4px 10px;
		font-size: var(--fs-sm);
	}
	.lab {
		min-width: 0;
	}
	.kind {
		display: block;
	}
	.track {
		height: 10px;
		border-radius: 3px;
		background: var(--surface);
		overflow: hidden;
	}
	.fill {
		display: block;
		height: 100%;
		background: var(--accent);
	}
	.eq {
		text-align: right;
		white-space: nowrap;
	}
	.sum {
		margin: 0;
		text-align: right;
		font-size: var(--fs-sm);
	}
	.small {
		font-size: var(--fs-xs);
	}
	.note {
		margin: -6px 0 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.key {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 10px;
	}
	.kc {
		display: inline-grid;
		grid-template-columns: auto auto;
		align-items: baseline;
		gap: 0 8px;
		padding: 4px 10px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
	}
	.kc .small {
		grid-column: 1 / -1;
	}
	.kch {
		font-family: var(--font-mono);
		font-size: 1.1rem;
	}
	.kb {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	@container ls (max-width: 520px) {
		.row {
			grid-template-columns: minmax(0, 1fr) auto;
		}
		.track {
			display: none;
		}
	}
</style>

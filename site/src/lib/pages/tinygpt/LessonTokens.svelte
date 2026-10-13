<!--
  Lesson step 1: the learner's text → BPE tokens (chips with ids) → the list of ids the model reads,
  and the 64-token context window. Clicking a chip picks the token the lesson follows (lesson state).
-->
<script lang="ts">
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { TokenItem } from '$lib/kit/types';
	import type { TGState } from './state.svelte';
	import { CTX } from './state.svelte';

	interface Props {
		/** the lesson's view state (shares the run with the page) */
		LS: TGState;
		text: string;
		promptIds: number[];
		genIds: number[];
	}
	let { LS, text, promptIds, genIds }: Props = $props();

	const all = $derived([...promptIds, ...genIds]);
	const dropped = $derived(Math.max(0, all.length - CTX));
	const chips = $derived<TokenItem[]>(
		LS.tok
			? all.map((id, i) => ({
					text: LS.tok!.tokenToDisplay(id),
					id,
					bytes: Array.from(LS.tok!.tokenBytes(id)),
					special: LS.tok!.isSpecial(id),
					tone: i < dropped ? 'muted' : i >= promptIds.length ? 'generated' : undefined
				}))
			: []
	);
	let selected = $state<number | null>(null);
	$effect(() => {
		selected = LS.T ? LS.pos + dropped : null;
	});
	function onselect(i: number) {
		if (i < dropped) {
			selected = LS.T ? LS.pos + dropped : null;
			return;
		}
		LS.selectPos(i - dropped);
	}
	const chars = $derived([...text].length);
	const genText = $derived(LS.tok && genIds.length ? LS.tok.decode(genIds) : '');
	const inWindow = $derived(all.slice(dropped));
	const perTok = $derived(promptIds.length ? chars / promptIds.length : 0);
	const fill = $derived(Math.min(1, all.length / CTX));
	const byteTokens = $derived(promptIds.filter((id) => id >= 128 && id < 256).length);
</script>

<div class="lt">
	<div class="box">
		<p class="cap">
			{t({ en: 'Your text', zh: '你的文本' })} · <span class="num">{chars}</span>
			{t({ en: chars === 1 ? 'character' : 'characters', zh: '个字符' })}
			{#if genIds.length}<span class="muted">+ {t({ en: 'generated', zh: '生成的部分' })}</span>{/if}
		</p>
		<p class="txt">{text}<span class="gen">{genText}</span></p>
	</div>

	<p class="arrow"><span aria-hidden="true">↓</span> {t({ en: 'the tokenizer splits it into pieces it knows', zh: '分词器把它切成认识的片段' })}</p>

	<div class="box">
		<p class="cap">
			{t({ en: 'Tokens', zh: '词元' })} · <span class="num">{all.length}</span>
			{#if perTok > 0}<span class="muted">· {t({ en: `about ${perTok.toFixed(1)} characters per token`, zh: `平均每个词元约 ${perTok.toFixed(1)} 个字符` })}</span>{/if}
		</p>
		<TokenChips tokens={chips} showIds bind:selected {onselect} label={{ en: 'Tokens of your text (click one to follow it)', zh: '你的文本的词元（点击一个来跟踪它）' }} />
		<p class="hint muted">
			{t({ en: '␣ marks a space; the small number under each chip is its id in the vocabulary of 1,024.', zh: '␣ 表示空格；每个色块下方的小数字是它在 1024 个词元的词表中的 id。' })}
			{#if byteTokens > 0}
				<strong>{t({ en: `${byteTokens} raw byte tokens: TinyGPT only learned English pieces, so other scripts fall back to bytes.`, zh: `${byteTokens} 个原始字节词元：TinyGPT 只学过英文片段，其他文字会退回到字节。` })}</strong>
			{/if}
		</p>
	</div>

	<p class="arrow"><span aria-hidden="true">↓</span> {t({ en: 'what the model actually receives: a list of ids', zh: '模型真正收到的：一串 id' })}</p>

	<div class="box">
		<p class="ids num">[{inWindow.join(', ')}]</p>
		<div class="win">
			<span class="bar" aria-hidden="true"><span style="width: {fill * 100}%" class:full={all.length >= CTX}></span></span>
			<span class="num">{inWindow.length} / {CTX}</span>
			<span class="muted">
				{dropped > 0
					? t({ en: `the context window is full: the first ${dropped} tokens are dropped (crossed out above)`, zh: `上下文窗口已满：最前面的 ${dropped} 个词元被丢掉（上面已划掉）` })
					: t({ en: 'tokens in the 64-token context window', zh: '个词元在 64 个词元的上下文窗口内' })}
			</span>
		</div>
	</div>
</div>

<style>
	/* the kit chips print their ids at 10 px: keep them at the 12 px floor here */
	.lt :global(.id) {
		font-size: var(--t-xs);
	}
	.lt {
		display: grid;
		gap: 6px;
		grid-template-columns: minmax(0, 1fr);
	}
	.box {
		display: grid;
		gap: 6px;
		padding: 10px 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		min-width: 0;
	}
	.cap {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		font-weight: 650;
	}
	.cap .muted {
		font-weight: 400;
	}
	.txt {
		margin: 0;
		font-family: var(--font-mono);
		font-size: 0.95rem;
		white-space: pre-wrap;
		word-break: break-word;
		max-height: 9em;
		overflow-y: auto;
		font-variant-ligatures: none;
	}
	.gen {
		color: var(--generated-text);
		font-weight: 600;
	}
	.arrow {
		margin: 0;
		padding-left: 14px;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.arrow span {
		font-size: 1rem;
		color: var(--accent);
		margin-right: 4px;
	}
	.hint {
		margin: 0;
		font-size: var(--fs-xs);
	}
	.ids {
		margin: 0;
		font-size: 0.86rem;
		word-break: break-word;
		color: var(--ink);
	}
	.win {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 8px;
		font-size: var(--fs-xs);
	}
	.win .bar {
		width: 90px;
		height: 6px;
		border-radius: 3px;
		background: var(--surface-3);
		overflow: hidden;
	}
	.win .bar span {
		display: block;
		height: 100%;
		background: var(--accent);
	}
	.win .bar span.full {
		background: var(--illus);
	}
</style>

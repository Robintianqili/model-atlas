<!--
  Lesson step 5: repeat count → merge until nothing is left. A StepPlayer runs over the merges (its
  scrubber is the vocabulary-size slider); the readout, the newest merge and the start of the text
  follow it, and the chart shows tokens vs vocabulary size for the learner's tokenizer and for
  TinyGPT's first k merges on the same text (tap the chart to jump).
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import { t, i18n } from '$lib/i18n/i18n.svelte';
	import { fmtInt } from '$lib/kit/format';
	import { bytesToDisplay } from '$lib/models/bpe';
	import Tok from '../Tok.svelte';
	import TokenCurve from '../TokenCurve.svelte';
	import { chipHtml } from '../Chips.svelte';
	import type { TrainTrace } from '../train';
	import { SegCursor, firstSingleton, previewSegs, tokensAt } from '../lesson';

	interface Props {
		trace: TrainTrace;
		text: string;
		pattern: string;
		/** merges applied (bindable) */
		k?: number;
		autoplay?: boolean;
		onend?: () => void;
	}
	let { trace, text, pattern, k = $bindable(0), autoplay = false, onend }: Props = $props();

	const K = $derived(trace.steps.length);
	const kk = $derived(Math.max(0, Math.min(k, K)));
	/** about 9 s from where the step starts to the last merge */
	const iv = untrack(() => Math.max(25, Math.min(500, Math.round(9000 / Math.max(1, trace.steps.length - k)))));
	const disp = (id: number) => bytesToDisplay(trace.bytes[id]);
	const now = $derived(tokensAt(trace, kk));
	const last = $derived(kk > 0 ? trace.steps[kk - 1] : null);
	const single = $derived(firstSingleton(trace));
	const cursor = $derived(new SegCursor(text, pattern, trace.merges));
	const preview = $derived(previewSegs(cursor.at(kk), 96));
	const nSegs = $derived(trace.pretokens);
	const curve = $derived([
		{ id: 'yours', name: { en: 'your BPE (trained on this text)', zh: '你的 BPE（用这段文本训练）' }, color: 'var(--cat-0)', points: [trace.initialTokenCount, ...trace.steps.map((s) => s.tokenCount)] },
		{ id: 'tiny', name: { en: "TinyGPT's merges (trained on stories)", zh: 'TinyGPT 的合并规则（用故事训练）' }, color: 'var(--cat-1)', points: trace.tiny, dashed: true }
	]);
	const ratio = $derived(now ? (trace.initialTokenCount / now).toFixed(2) : '–');
	let playing = $state(false);
	/** a jump chosen by the learner stops the merge player there */
	function jump(v: number) {
		playing = false;
		k = v;
	}
</script>

<div class="ls">
	{#if K > 0}
		<StepPlayer count={K + 1} bind:step={k} bind:playing interval={iv} {autoplay} {onend} speeds={[0.5, 1, 2]} label={{ en: 'Count, merge, repeat: press Play or drag the slider', zh: '统计、合并、重复：点“播放”或拖动滑块' }}>
			{#snippet children()}
				<p class="read" aria-live="off">
					<span>{t({ en: 'Vocabulary', zh: '词表' })} = <span class="num">256</span> <span class="muted small">{t({ en: 'bytes', zh: '个字节' })}</span> + <span class="num acc">{kk}</span> <span class="muted small">{t({ en: 'merges', zh: '次合并' })}</span> = <b class="num big">{256 + kk}</b></span>
					<span class="sep" aria-hidden="true">·</span>
					<span>{t({ en: 'your text', zh: '你的文本' })} = <b class="num big">{fmtInt(now)}</b> {t({ en: 'tokens', zh: '个词元' })}</span>
					<span class="muted small num">{t({ en: `(${ratio} bytes per token)`, zh: `（平均每个词元 ${ratio} 个字节）` })}</span>
				</p>
				<div class="row">
					{#if last}
						<span class="lm">
							<span class="muted small">{t({ en: 'newest merge', zh: '最新的合并' })}</span>
							<span class="muted num">#{kk}</span>
							<Tok text={disp(last.pair[0])} small /> <span class="op">+</span> <Tok text={disp(last.pair[1])} small />
							<span class="op">→</span> <Tok text={disp(last.newId)} small mark="new" />
							<span class="cnt" class:one={last.count === 1}><span class="muted small">{t({ en: 'pair seen', zh: '这一对出现' })}</span> <b class="num">{last.count}</b>{t({ en: '×', zh: ' 次' })}</span>
						</span>
					{:else}
						<span class="muted">{t({ en: 'No merges yet: every byte is a token.', zh: '还没有合并：每个字节都是一个词元。' })}</span>
					{/if}
				</div>
				<p class="muted small pvh">{t({ en: 'The start of your text now: one box per token, gaps between pre-tokens', zh: '你的文本开头此刻的样子：一个方块一个词元，空隙分开预分词片段' })}</p>
				<div class="pv" aria-label={t({ en: 'The start of your text as tokens', zh: '你的文本开头部分的词元' })}>
					{#each preview as s, i (i)}
						<span class="grp">{#each s.ids as id, j (j)}<span class="ch c{j % 2}" class:new={last && id === last.newId}>{@html chipHtml(disp(id))}</span>{/each}</span>
					{/each}
					{#if preview.length < nSegs || preview.cut}<span class="muted small">…</span>{/if}
				</div>
			{/snippet}
		</StepPlayer>
		<TokenCurve series={curve} k={kk} pickMax={K} endLabels={false} label={{ en: 'Tokens in your text vs vocabulary size', zh: '文本词元数与词表大小' }} onpick={jump} />
		<p class="note">
			{#if single > 0}
				{#if i18n.lang === 'zh'}
					从第 <b class="num">{single}</b> 次合并开始，胜出的对只出现 1 次：分词器在背整个词，而不是在发现规律。
				{:else}
					From merge <b class="num">#{single}</b> on, the winning pair occurs only once: the tokenizer is memorizing whole words, not finding patterns.
				{/if}
				<button type="button" class="btn ghost small" onclick={() => jump(single)}>{t({ en: `Jump to merge #${single}`, zh: `跳到第 ${single} 次合并` })}</button>
			{/if}
			{#if trace.finalPairs > 0}
				{#if i18n.lang === 'zh'}
					训练在第 <b class="num">{K}</b> 次合并后停止（和 TinyGPT 一样多），这时还剩 <b class="num">{fmtInt(trace.finalPairs)}</b> 种相邻对没有合并；文本是 <b class="num">{fmtInt(tokensAt(trace, K))}</b> 个词元。
				{:else}
					Training stops after <b class="num">{K}</b> merges (as many as TinyGPT has), with <b class="num">{fmtInt(trace.finalPairs)}</b> different pairs still unmerged; the text is <b class="num">{fmtInt(tokensAt(trace, K))}</b> tokens.
				{/if}
			{:else if i18n.lang === 'zh'}
				最多 <b class="num">{K}</b> 次合并后，每个预分词片段都成了一个词元（<b class="num">{fmtInt(tokensAt(trace, K))}</b> 个）。
			{:else}
				After all <b class="num">{K}</b> merges every pre-token is one token (<b class="num">{fmtInt(tokensAt(trace, K))}</b>).
			{/if}
		</p>
	{:else}
		<h4>{t({ en: 'Nothing to repeat', zh: '没有可重复的' })}</h4>
		<p class="muted">{t({ en: 'Your text has no pair to merge. Type a longer text.', zh: '你的文本里没有可合并的对。请输入更长的文本。' })}</p>
	{/if}
</div>

<style>
	.ls {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 10px;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.read {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 10px;
	}
	.big {
		font-size: var(--fs-lg);
	}
	.acc {
		color: var(--accent);
		font-weight: 700;
	}
	.sep {
		color: var(--muted);
	}
	.small {
		font-size: var(--fs-xs);
	}
	.row {
		min-height: 30px;
		display: flex;
		align-items: center;
	}
	.lm {
		display: inline-flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px;
	}
	.op {
		font-family: var(--font-mono);
		color: var(--muted);
	}
	.cnt {
		color: var(--seq-hi);
		font-weight: 700;
		margin-left: 4px;
	}
	.cnt.one {
		color: var(--illus);
	}
	.pvh {
		margin: 0 0 -6px;
	}
	.pv {
		min-width: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 5px 7px;
		align-items: center;
		min-height: 64px;
		padding: 8px;
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	.grp {
		display: inline-flex;
		flex-wrap: wrap;
		gap: 2px;
		min-width: 0;
		max-width: 100%;
	}
	.ch {
		display: inline-block;
		min-width: 1.2em;
		padding: 1px 4px;
		border-radius: 4px;
		font-family: var(--font-mono);
		font-size: 0.86rem;
		line-height: 1.45;
		white-space: pre;
		text-align: center;
		color: var(--ink);
	}
	.ch :global(.hex) {
		font-size: 0.72em;
		color: var(--ink-2);
	}
	.c0 {
		background: color-mix(in srgb, var(--cat-0) 16%, var(--surface));
	}
	.c1 {
		background: color-mix(in srgb, var(--cat-3) 22%, var(--surface));
	}
	.ch.new {
		box-shadow: 0 0 0 2px var(--accent);
		background: var(--accent-soft);
	}
	.note {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.btn.small {
		min-height: 30px;
		padding: 0 10px;
		font-size: var(--fs-xs);
		margin: 2px 0;
	}
</style>

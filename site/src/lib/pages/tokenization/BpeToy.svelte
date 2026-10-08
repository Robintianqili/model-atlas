<!--
  PART 1: build a byte-level BPE tokenizer from the learner's own text.
  Training runs in the Web Worker (trainTrace = trainBPE + pair counts); everything shown is the
  state after the first k merges: pair counts, the merge just made, the text cut into tokens,
  how one token was built, the learned vocabulary, and tokens vs vocabulary size.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { t, i18n } from '$lib/i18n/i18n.svelte';
	import { fmtInt } from '$lib/kit/format';
	import { bytesToDisplay, type BPETokenizer } from '$lib/models/bpe';
	import Chips from './Chips.svelte';
	import Tok from './Tok.svelte';
	import type { ChipItem } from './types';
	import TokenCurve from './TokenCurve.svelte';
	import { segment, mergeTree, type TrainTrace, type MergeTree, type PairCount } from './train';
	import { PRESETS } from './presets';
	import { conn } from './state.svelte';

	interface Props {
		text: string;
		k: number;
		trace: TrainTrace | null;
		/** the text the current trace was trained on (bindable; `text` may be newer while training) */
		trained?: string;
		tiny: BPETokenizer | null;
	}
	let { text = $bindable(), k = $bindable(), trace = $bindable(), trained = $bindable(''), tiny }: Props = $props();

	const MAX_MERGES = 767; // the same number of merges TinyGPT's tokenizer has
	let training = $state(false);
	let ms = $state(0);
	let error = $state('');
	let seq = 0;
	// raw: the pairs come from the (raw) trace; a $state proxy of one would never be === to it
	let hoverPair = $state.raw<PairCount | null>(null);
	/** the selected chip, remembered by the byte it starts at, so it stays on the same text across merges */
	let selByte = $state<number | null>(null);
	let showAllVocab = $state(false);
	// a new training text starts without a selection
	$effect(() => {
		void text;
		selByte = null;
	});

	async function run(txt: string) {
		const c = conn.client;
		if (!c || !tiny) return;
		const my = ++seq;
		training = true;
		try {
			const r = await c.train(txt, MAX_MERGES, tiny.pattern, tiny.merges);
			if (my !== seq) return;
			trace = r.trace;
			trained = txt;
			ms = r.ms;
			error = '';
			if (k > r.trace.merges.length) k = r.trace.merges.length;
		} catch (e) {
			if (my === seq) error = e instanceof Error ? e.message : String(e);
		} finally {
			if (my === seq) training = false;
		}
	}

	// re-train (debounced) whenever the text changes or the worker becomes available.
	// `trace` is read untracked: tracking it made every new trace schedule another training run
	// (a loop of about 4 trainings per second while the page sat idle).
	$effect(() => {
		const txt = text;
		if (!conn.client || !tiny) return;
		const id = setTimeout(() => run(txt), untrack(() => trace) ? 250 : 0);
		return () => clearTimeout(id);
	});

	const K = $derived(trace?.merges.length ?? 0);
	// keep k inside 0 … K (a "Try this" may ask for the last merge before training has finished)
	$effect(() => {
		if (trace && k > K) k = K;
	});
	const kk = $derived(Math.min(k, K));
	const segs = $derived(trace && tiny ? segment(text, tiny.pattern, trace.merges, kk) : []);
	const tokensNow = $derived(trace ? (kk === 0 ? trace.initialTokenCount : trace.steps[kk - 1].tokenCount) : 0);
	const bytesN = $derived(trace?.initialTokenCount ?? 0);
	const disp = (id: number) => (trace ? bytesToDisplay(trace.bytes[id]) : '');
	const hex = (id: number) => (trace ? trace.bytes[id].map((b) => b.toString(16).padStart(2, '0')).join(' ') : '');

	// display strings per token id, computed once per trained tokenizer
	const info = $derived.by(() => {
		const m = new Map<number, ChipItem>();
		return (id: number): ChipItem => {
			let it = m.get(id);
			if (!it) {
				const d = disp(id);
				it = { text: d, id, hex: hex(id), kind: d.includes('<0x') ? 'partial' : undefined };
				m.set(id, it);
			}
			return it;
		};
	});

	// chips for the whole text, with marks for the newest token and the hovered pair
	const chipData = $derived.by(() => {
		const items: ChipItem[] = [];
		const marks: (string | null)[] = [];
		/** byte offset where each chip starts */
		const starts: number[] = [];
		const newId = kk > 0 ? 255 + kk : -1;
		const get = info;
		let off = 0;
		for (const s of segs) {
			s.ids.forEach((id, j) => {
				const base = get(id);
				items.push(j === 0 ? { ...base, gap: true } : base);
				marks.push(id === newId ? 'new' : null);
				starts.push(off);
				off += trace ? trace.bytes[id].length : 1;
			});
		}
		if (hoverPair) {
			let i = 0;
			for (const s of segs) {
				for (let j = 0; j < s.ids.length - 1; j++) {
					if (s.ids[j] === hoverPair.a && s.ids[j + 1] === hoverPair.b) {
						marks[i + j] = 'pair';
						marks[i + j + 1] = 'pair';
					}
				}
				i += s.ids.length;
			}
		}
		return { items, marks, starts };
	});
	// index of the chip that contains the selected byte (the last chip starting at or before it)
	const selected = $derived.by<number | null>(() => {
		if (selByte === null) return null;
		const st = chipData.starts;
		let lo = 0;
		let hi = st.length - 1;
		let ans = -1;
		while (lo <= hi) {
			const mid = (lo + hi) >> 1;
			if (st[mid] <= selByte) {
				ans = mid;
				lo = mid + 1;
			} else hi = mid - 1;
		}
		return ans >= 0 ? ans : null;
	});

	const top = $derived(trace ? (kk < K ? trace.steps[kk].top : trace.finalTop) : []);
	const pairsNow = $derived(trace ? (kk < K ? trace.steps[kk].pairs : trace.finalPairs) : 0);
	const last = $derived(trace && kk > 0 ? trace.steps[kk - 1] : null);
	const before = $derived(trace && kk > 0 ? (kk === 1 ? trace.initialTokenCount : trace.steps[kk - 2].tokenCount) : 0);

	// how the selected chip's token was built
	const selId = $derived(selected !== null && chipData.items[selected] ? chipData.items[selected].id ?? null : null);
	function internal(tree: MergeTree, out: MergeTree[] = []): MergeTree[] {
		if (tree.kids) {
			internal(tree.kids[0], out);
			internal(tree.kids[1], out);
			out.push(tree);
		}
		return out;
	}
	const built = $derived(trace && selId !== null ? internal(mergeTree(selId, trace.merges)).sort((a, b) => a.id - b.id) : []);

	const vocab = $derived(trace ? trace.steps.slice(0, kk).map((s) => s).reverse() : []);
	const curve = $derived(
		trace
			? [
					{ id: 'yours', name: { en: 'your BPE (trained on this text)', zh: '你的 BPE（用这段文本训练）' }, color: 'var(--cat-0)', points: [trace.initialTokenCount, ...trace.steps.map((s) => s.tokenCount)] },
					{ id: 'tiny', name: { en: "TinyGPT's merges (trained on stories)", zh: 'TinyGPT 的合并规则（用故事训练）' }, color: 'var(--cat-1)', points: trace.tiny, dashed: true }
				]
			: []
	);
	const tinyAtK = $derived(trace ? trace.tiny[Math.min(kk, trace.tiny.length - 1)] : 0);
	const maxBar = $derived(top.length ? top[0].n : 1);
</script>

<div class="toy">
	<TextInputBar
		bind:value={text}
		multiline
		rows={5}
		maxLength={2000}
		label={{ en: 'Training text (type or paste your own)', zh: '训练文本（可输入或粘贴你自己的）' }}
		presets={PRESETS.train.map((p) => ({ text: p.text, label: p.label }))}
	/>

	<p class="stats" aria-live="polite">
		{#if trace}
			<span><b class="num">{fmtInt(bytesN)}</b> {t({ en: 'UTF-8 bytes', zh: '个 UTF-8 字节' })}</span>
			<span><b class="num">{fmtInt(trace.pretokens)}</b> {t({ en: 'pre-tokens', zh: '个预分词片段' })} ({fmtInt(trace.distinct)} {t({ en: 'different', zh: '种' })})</span>
			<span>{t({ en: 'learned', zh: '学到' })} <b class="num">{K}</b> {t({ en: 'merges in', zh: '次合并，用时' })} <span class="num">{ms < 1 ? '<1' : Math.round(ms)} ms</span> {t({ en: '(in a Web Worker)', zh: '（在 Web Worker 中）' })}</span>
			{#if training}<span class="muted">{t({ en: 'training…', zh: '训练中…' })}</span>{/if}
		{:else if error}
			<span class="err">{error}</span>
		{:else}
			<span class="muted">{t({ en: 'Loading TinyGPT’s pre-tokenizer…', zh: '正在加载 TinyGPT 的预分词规则…' })}</span>
		{/if}
		<ProvenanceBadge mode="live" compact note={{ en: 'Trained in your browser right now. Same algorithm and tie-break as the Python trainer that built TinyGPT’s tokenizer.', zh: '此刻在你的浏览器里训练。算法与平局规则和训练 TinyGPT 分词器的 Python 程序完全相同。' }} />
	</p>

	{#if trace}
		<StepPlayer count={K + 1} bind:step={k} interval={700} label={{ en: 'Merge by merge', zh: '逐次合并' }}>
			{#snippet children()}
				<p class="vocab-read" id="vocab-size">
					<span>{t({ en: 'Vocabulary', zh: '词表' })} = <span class="num">256</span> {t({ en: 'bytes', zh: '个字节' })} + <span class="num acc">{kk}</span> {t({ en: 'merges', zh: '次合并' })} = <b class="num">{256 + kk}</b></span>
					<span class="sep">·</span>
					<span>{t({ en: 'your text', zh: '你的文本' })} = <b class="num">{fmtInt(tokensNow)}</b> {t({ en: 'tokens', zh: '个词元' })}</span>
				</p>
				<div class="grid2">
					<div class="panel card">
						<h4>{kk < K ? t({ en: 'Pair counts now', zh: '当前的相邻对计数' }) : t({ en: 'Pairs left', zh: '剩下的相邻对' })}</h4>
						{#if top.length}
							<p class="hint">{t({ en: 'Every adjacent pair of tokens inside a pre-token, counted over the whole text', zh: '统计整段文本里，每个预分词片段内部相邻两个词元组成的对' })}{t({ en: ' (', zh: '（' })}{fmtInt(pairsNow)} {t({ en: 'different pairs', zh: '种不同的对' })}{t({ en: ').', zh: '）。' })}</p>
							<ol class="pairs">
								{#each top as p, i (i)}
									<li>
										<button
											type="button"
											class="pair-row"
											class:chosen={i === 0 && kk < K}
											onpointerenter={() => (hoverPair = p)}
											onpointerleave={() => (hoverPair = null)}
											onfocus={() => (hoverPair = p)}
											onblur={() => (hoverPair = null)}
											onclick={() => (hoverPair = p)}
											aria-label="{disp(p.a)} + {disp(p.b)}: {p.n}"
										>
											<span class="pp"><Tok text={disp(p.a)} small /> <Tok text={disp(p.b)} small /></span>
											<span class="bar"><span style:width="{(100 * p.n) / maxBar}%"></span></span>
											<span class="n num">{p.n}</span>
										</button>
										{#if i === 0 && kk < K}<span class="next">{t({ en: '← merged next', zh: '← 下一次合并它' })}</span>{/if}
									</li>
								{/each}
							</ol>
							<p class="hint">{t({ en: 'Hover or tap a pair to see where it occurs in your text.', zh: '把鼠标移到某个对上（或点一下），看它在文本中出现在哪里。' })}</p>
						{:else}
							<p>{t({ en: 'No pairs left: every pre-token is now a single token. Training stops here.', zh: '没有相邻对了：每个预分词片段都已是一个词元，训练到此为止。' })}</p>
						{/if}
					</div>
					<div class="panel card">
						{#if last}
							<h4>{t({ en: 'Merge', zh: '合并' })} #{kk}</h4>
							<div class="merge">
								<Tok text={disp(last.pair[0])} />
								<span class="op">+</span>
								<Tok text={disp(last.pair[1])} />
								<span class="op">→</span>
								<Tok text={disp(last.newId)} mark="new" />
								<span class="muted small">ID <b class="num">{last.newId}</b></span>
							</div>
							<p class="eq">
								<span class="num">{fmtInt(before)}</span> − <span class="num pc">{fmtInt(before - last.tokenCount)}</span> = <b class="num">{fmtInt(last.tokenCount)}</b>
								<span class="muted">{t({ en: 'tokens', zh: '个词元' })}</span>
							</p>
							<p class="hint">
								{t({ en: 'The pair occurred', zh: '这个对出现了' })} <b class="num pc">{last.count}</b> {t({ en: 'times; each occurrence becomes one token, so the text gets that many tokens shorter', zh: '次；每次出现都变成一个词元，所以文本少了这么多个词元' })}{#if before - last.tokenCount < last.count}{t({ en: ' (overlapping pairs, as in “aaa”, merge only once)', zh: '（重叠的对，如 “aaa”，只合并一次）' })}{/if}{t({ en: '.', zh: '。' })}
								{#if last.count === 1}<br /><b>{t({ en: 'Count 1:', zh: '计数为 1：' })}</b> {t({ en: 'from here on the tokenizer is memorizing words that appear only once in this text.', zh: '从这里开始，分词器是在“背下”只出现一次的词。' })}{/if}
							</p>
						{:else}
							<h4>{t({ en: 'Start: bytes only', zh: '起点：只有字节' })}</h4>
							<p>{t({ en: 'No merges yet. The vocabulary is the 256 possible byte values, so every byte of your text is one token:', zh: '还没有任何合并。词表就是 256 种可能的字节，所以文本中的每个字节都是一个词元：' })} <b class="num">{fmtInt(bytesN)}</b> {t({ en: 'tokens.', zh: '个词元。' })}</p>
							<p class="hint">{t({ en: 'Press ▶ or → to merge the most frequent pair.', zh: '按 ▶ 或 → 键，合并出现最多的那一对。' })}</p>
						{/if}
					</div>
				</div>
			{/snippet}
		</StepPlayer>

		<div class="seg">
			<h4>{t({ en: 'Your text, cut into tokens', zh: '你的文本被切成的词元' })} <span class="muted small">{t({ en: '(', zh: '（' })}{fmtInt(tokensNow)} {t({ en: 'tokens; small gaps separate pre-tokens; tap a token to see how it was built', zh: '个词元；小空隙分隔预分词片段；点一个词元看它是怎么拼出来的' })}{t({ en: ')', zh: '）' })}</span></h4>
			<Chips
				items={chipData.items}
				marks={chipData.marks}
				{selected}
				onselect={(i) => (selByte = chipData.starts[i] ?? null)}
				maxHeight={260}
				limit={1000}
				label={{ en: 'Training text as tokens', zh: '训练文本的词元' }}
			/>
			<p class="legend small">
				<span><i class="lg new"></i>{t({ en: 'made by the last merge', zh: '上一次合并产生的' })}</span>
				<span><i class="lg pair"></i>{t({ en: 'the pair under your pointer', zh: '你指向的那一对' })}</span>
				<span><i class="lg partial"></i>{t({ en: 'raw bytes (part of a character)', zh: '原始字节（一个字符的一部分）' })}</span>
			</p>
			{#if selId !== null}
				<div class="panel built" aria-live="polite">
					<h4>
						{t({ en: 'How', zh: '词元' })} <Tok text={disp(selId)} /> {t({ en: 'was built', zh: '是怎么拼出来的' })}
						<span class="muted small">(ID {selId}{selId < 256 ? t({ en: ', a single byte', zh: '，单个字节' }) : ''})</span>
					</h4>
					{#if built.length}
						<ol class="steps">
							{#each built as node (node.id)}
								<li>
									<span class="muted num">#{node.id - 255}</span>
									{#if node.kids}
										<Tok text={disp(node.kids[0].id)} small /> <span class="op">+</span> <Tok text={disp(node.kids[1].id)} small />
										<span class="op">→</span>
										<Tok text={disp(node.id)} small />
										<span class="muted small num">×{trace.steps[node.id - 256].count}</span>
									{/if}
								</li>
							{/each}
						</ol>
						<p class="hint">{t({ en: 'Merges in the order they were learned (#1 first); ×n is how often the pair occurred when it was merged.', zh: '按学到的先后顺序列出（#1 最早）；×n 是合并时这个对出现的次数。' })}</p>
					{:else}
						<p class="hint">{t({ en: 'This token is one of the 256 starting bytes:', zh: '这个词元是最初的 256 个字节之一：' })} <span class="num">{hex(selId)}</span></p>
					{/if}
				</div>
			{/if}
		</div>

		<div class="grid2 lower">
			<div class="chart">
				<h4>{t({ en: 'Tokens in your text vs vocabulary size', zh: '文本词元数与词表大小' })}</h4>
				<TokenCurve series={curve} k={kk} pickMax={K} label={{ en: 'Tokens in your text vs vocabulary size', zh: '文本词元数与词表大小' }} onpick={(v) => (k = v)} />
				<p class="hint">
					{#if i18n.lang === 'zh'}
						用上 TinyGPT 全部 767 次合并，你的文本是 <b class="num">{fmtInt(trace.tiny[trace.tiny.length - 1])}</b> 个词元；只用它的前 <span class="num">{kk}</span> 次合并（和你现在一样多），是 <b class="num">{fmtInt(tinyAtK)}</b> 个。
					{:else}
						With all 767 of TinyGPT’s merges, your text is <b class="num">{fmtInt(trace.tiny[trace.tiny.length - 1])}</b> tokens; with the same number of merges as yours now ({kk}), <b class="num">{fmtInt(tinyAtK)}</b>.
					{/if}
					{t({ en: 'Your tokenizer was trained on this very text, so it compresses it better, but it would be poor at other text. Tap the chart to jump.', zh: '你的分词器就是用这段文本训练的，所以压缩得更好，但换一段文本就未必了。点图表可以跳到对应位置。' })}
				</p>
			</div>
			<div class="vocab">
				<h4>{t({ en: 'Learned vocabulary', zh: '学到的词表' })} <span class="muted small">(256 {t({ en: 'bytes', zh: '个字节' })} + {kk})</span></h4>
				{#if vocab.length}
					<ol class="vlist" reversed>
						{#each showAllVocab ? vocab : vocab.slice(0, 40) as s (s.rank)}
							<li><span class="muted num">#{s.rank + 1}</span> <Tok text={disp(s.newId)} small /> <span class="muted num small">×{s.count}</span></li>
						{/each}
					</ol>
					{#if vocab.length > 40}
						<button type="button" class="btn ghost small" onclick={() => (showAllVocab = !showAllVocab)}>
							{showAllVocab ? t({ en: 'Show fewer', zh: '收起' }) : t({ en: `Show all ${vocab.length}`, zh: `显示全部 ${vocab.length} 个` })}
						</button>
					{/if}
				{:else}
					<p class="hint">{t({ en: 'Nothing learned yet: only the 256 bytes.', zh: '还没学到新词元：只有 256 个字节。' })}</p>
				{/if}
			</div>
		</div>
	{/if}
</div>

<style>
	.toy {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
	}
	.stats {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 16px;
		align-items: center;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.err {
		color: var(--k-text);
	}
	.vocab-read {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 10px;
		font-size: var(--fs-md);
	}
	.vocab-read .acc {
		color: var(--accent);
		font-weight: 700;
	}
	.sep {
		color: var(--muted);
	}
	.grid2 {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s3);
	}
	@media (min-width: 820px) {
		.grid2 {
			grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		}
		.grid2.lower {
			grid-template-columns: minmax(0, 1.6fr) minmax(0, 1fr);
		}
	}
	.card {
		min-width: 0;
		display: grid;
		gap: 8px;
		align-content: start;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px;
	}
	.hint {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.small {
		font-size: var(--fs-xs);
		font-weight: 400;
	}
	.pairs {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 3px;
	}
	.pairs li {
		display: flex;
		align-items: center;
		gap: 6px;
		min-width: 0;
	}
	.pair-row {
		flex: 1;
		min-width: 0;
		display: grid;
		grid-template-columns: minmax(0, auto) minmax(40px, 1fr) 3.2em;
		align-items: center;
		gap: 8px;
		padding: 2px 6px;
		border: 1px solid transparent;
		border-radius: var(--r1);
		background: none;
		text-align: left;
		color: inherit;
	}
	.pair-row:hover,
	.pair-row:focus-visible {
		background: var(--surface-2);
	}
	.pair-row.chosen {
		border-color: var(--accent);
	}
	.pp {
		min-width: 0;
		overflow: hidden;
	}
	.bar {
		height: 10px;
		background: var(--surface-2);
		border-radius: 3px;
		overflow: hidden;
	}
	.bar span {
		display: block;
		height: 100%;
		background: var(--seq-mid);
	}
	.n {
		text-align: right;
		font-size: var(--fs-sm);
	}
	.next {
		font-size: var(--fs-xs);
		color: var(--accent);
		white-space: nowrap;
	}
	.merge {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px;
	}
	.op {
		font-family: var(--font-mono);
		color: var(--muted);
	}
	.eq {
		margin: 0;
		font-size: var(--fs-lg);
	}
	.pc {
		color: var(--seq-hi);
		font-weight: 700;
	}
	.seg {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.legend {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		color: var(--ink-2);
	}
	.lg {
		display: inline-block;
		width: 14px;
		height: 12px;
		border-radius: 3px;
		margin-right: 5px;
		vertical-align: -1px;
	}
	.lg.new {
		box-shadow: inset 0 0 0 2px var(--accent);
	}
	.lg.pair {
		background: color-mix(in srgb, var(--seq-mid) 30%, var(--surface));
		box-shadow: inset 0 0 0 2px var(--seq-mid);
	}
	.lg.partial {
		border: 1px dashed var(--ink-2);
		background: var(--surface-2);
	}
	.built {
		display: grid;
		gap: 8px;
	}
	.steps {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 4px;
	}
	.steps li {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px;
	}
	.chart,
	.vocab {
		min-width: 0;
		display: grid;
		gap: 8px;
		align-content: start;
	}
	.vlist {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 10px;
		max-height: 300px;
		overflow-y: auto;
	}
	.vlist li {
		display: inline-flex;
		align-items: center;
		gap: 3px;
	}
	.btn.small {
		min-height: 32px;
		justify-self: start;
	}
</style>

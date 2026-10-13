<!--
  PART 2: TinyGPT's real tokenizer on the learner's sentence (live, bpe.ts encodeWithSteps):
  pre-tokens -> the merges inside one pre-token (StepPlayer, with the rank rule shown) ->
  final ids -> UTF-8 byte view, all hover-linked. The training counts of each merge come from
  tokenizer.json (precomputed when the tokenizer was trained).
-->
<script lang="ts">
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtInt } from '$lib/kit/format';
	import { visibleWhitespace, type BPETokenizer } from '$lib/models/bpe';
	import Chips from './Chips.svelte';
	import Tok from './Tok.svelte';
	import ByteView from './ByteView.svelte';
	import { isCJK } from './realtok';
	import type { ChipItem } from './types';

	interface Props {
		tok: BPETokenizer;
		sentence: string;
		/** pair counts at training time, per merge rank (tokenizer.json mergeCounts) */
		counts?: number[];
	}
	let { tok, sentence, counts = [] }: Props = $props();

	const es = $derived(tok.encodeWithSteps(sentence));
	const disp = (id: number) => tok.tokenToDisplay(id);
	const hex = (id: number) => Array.from(tok.tokenBytes(id), (b) => b.toString(16).padStart(2, '0')).join(' ');

	// flat token list with byte ranges and the pre-token each came from
	const flat = $derived.by(() => {
		const out: { id: number; start: number; end: number; pt: number }[] = [];
		let off = 0;
		es.pretokens.forEach((p, pi) => {
			for (const id of p.ids) {
				const n = tok.tokenBytes(id).length;
				out.push({ id, start: off, end: off + n, pt: pi });
				off += n;
			}
		});
		return out;
	});
	let hover = $state<number | null>(null); // flat token index
	let sel = $state<number | null>(null); // selected pre-token
	let userPicked = $state(false);
	let step = $state(0);

	// default pre-token: the one with the most merges (re-chosen when the text changes, unless picked)
	$effect(() => {
		const n = es.pretokens.length;
		if (!n) {
			sel = null;
			return;
		}
		if (!userPicked || sel === null || sel >= n) {
			let best = 0;
			es.pretokens.forEach((p, i) => {
				if (p.steps.length > es.pretokens[best].steps.length) best = i;
			});
			sel = best;
		}
	});
	$effect(() => {
		void sel;
		void sentence;
		step = 0;
	});

	const pt = $derived(sel !== null ? es.pretokens[sel] : null);
	/** clamp: right after the text changes the player may still hold a step of the old pre-token */
	const clampStep = (s: number) => (pt ? Math.max(0, Math.min(s, pt.steps.length)) : 0);
	const stateAt = (s0: number) => {
		const s = clampStep(s0);
		return pt ? (s === 0 ? pt.bytes : pt.steps[s - 1].tokens) : [];
	};
	const labels = $derived(
		pt
			? [
					{ en: `Start: ${pt.bytes.length} bytes, one token each`, zh: `起点：${pt.bytes.length} 个字节，各是一个词元` },
					...pt.steps.map((s) => ({
						en: `Merge rank ${s.rank}: "${disp(s.pair[0])}" + "${disp(s.pair[1])}" → "${disp(s.newId)}" (ID ${s.newId})`,
						zh: `合并（排名 ${s.rank}）：“${disp(s.pair[0])}” + “${disp(s.pair[1])}” → “${disp(s.newId)}”（ID ${s.newId}）`
					}))
				]
			: []
	);
	/** adjacent pairs at state s and their merge rank (undefined = never learned) */
	const candidates = $derived.by(() => {
		const ids = stateAt(step);
		const out: { a: number; b: number; r: number | undefined }[] = [];
		for (let i = 0; i < ids.length - 1; i++) out.push({ a: ids[i], b: ids[i + 1], r: tok.rank(ids[i], ids[i + 1]) });
		return out;
	});
	const minRank = $derived(Math.min(...candidates.map((c) => c.r ?? Infinity)));

	const chipItems = $derived<ChipItem[]>(
		flat.map((f, i) => ({
			text: disp(f.id),
			id: f.id,
			gap: i > 0 && flat[i - 1].pt !== f.pt,
			hex: hex(f.id),
			kind: tok.isSpecial(f.id) ? 'special' : disp(f.id).includes('<0x') ? 'partial' : undefined
		}))
	);
	const chipMarks = $derived(flat.map((_, i) => (i === hover ? 'link' : null)));

	// Chinese characters, and the single-byte tokens that lie inside them (punctuation, accents and
	// emoji are not counted here, so the callout's numbers refer to the Chinese characters only)
	const enc = new TextEncoder();
	const cjkInfo = $derived.by(() => {
		const inCjk = new Uint8Array(enc.encode(sentence).length);
		let off = 0;
		let chars = 0;
		for (const ch of sentence) {
			const n = enc.encode(ch).length;
			if (isCJK(ch.codePointAt(0) ?? 0)) {
				chars++;
				inCjk.fill(1, off, off + n);
			}
			off += n;
		}
		const single = flat.filter((f) => f.end - f.start === 1 && f.id < 256 && inCjk[f.start] === 1).length;
		return { chars, single };
	});
	const cjk = $derived(cjkInfo.chars);
	const rawBytes = $derived(cjkInfo.single);
	const nChars = $derived(Array.from(sentence).length);
	const nBytes = $derived(enc.encode(sentence).length);

	function pickKey(e: KeyboardEvent, i: number) {
		const n = es.pretokens.length;
		let j = i;
		if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = Math.min(n - 1, i + 1);
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = Math.max(0, i - 1);
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = n - 1;
		else return;
		e.preventDefault();
		userPicked = true;
		sel = j;
		(e.currentTarget as HTMLElement).parentElement?.querySelectorAll<HTMLElement>('.pt')[j]?.focus();
	}
</script>

<div class="tg">
	<p class="stats">
		<span><b class="num">{nChars}</b> {t({ en: 'characters', zh: '个字符' })}</span>
		<span><b class="num">{nBytes}</b> {t({ en: 'UTF-8 bytes', zh: '个 UTF-8 字节' })}</span>
		<span><b class="num">{es.pretokens.length}</b> {t({ en: 'pre-tokens', zh: '个预分词片段' })}</span>
		<span><b class="num big">{es.ids.length}</b> {t({ en: 'TinyGPT tokens', zh: '个 TinyGPT 词元' })}</span>
		<ProvenanceBadge mode="live" compact note={{ en: 'TinyGPT’s real tokenizer (1,024 tokens: 256 bytes + 767 merges + <|endoftext|>), running in your browser. Its ids match the Python tokenizer on 949 test strings.', zh: 'TinyGPT 的真实分词器（1,024 个词元：256 个字节 + 767 次合并 + <|endoftext|>），在你的浏览器里运行。它在 949 条测试文本上与 Python 版本的 ID 完全一致。' }} />
	</p>

	<div class="step-head">
		<span class="pill">1</span>
		<h4>{t({ en: 'Pre-tokens: the text is first split into word-like pieces', zh: '预分词：先把文本切成类似单词的片段' })}</h4>
	</div>
	<p class="hint">{t({ en: 'A regular expression splits off words (with their leading space), numbers, punctuation runs and whitespace. Merges never cross these borders. Pick one to look inside.', zh: '一个正则表达式切出单词（连同前面的空格）、数字、连续标点和空白。合并永远不会跨越这些边界。选一个片段看看里面发生了什么。' })}</p>
	<div class="pts" role="radiogroup" aria-label={t({ en: 'Pre-tokens', zh: '预分词片段' })}>
		{#each es.pretokens as p, i (i)}
			<button
				type="button"
				class="pt"
				role="radio"
				aria-checked={sel === i}
				tabindex={sel === i ? 0 : -1}
				class:on={sel === i}
				onclick={() => {
					userPicked = true;
					sel = i;
				}}
				onkeydown={(e) => pickKey(e, i)}
			>
				<span class="ptext" lang={/[一-鿿]/.test(p.text) ? 'zh-CN' : undefined}>{p.special ? p.text : visibleWhitespace(p.text)}</span>
				<span class="ptoks">{#each p.ids as id, j (j)}<Tok text={disp(id)} small kind={p.special ? 'special' : undefined} />{/each}</span>
				<span class="pmeta num">{p.bytes.length || '—'} B → {p.ids.length}</span>
			</button>
		{/each}
	</div>

	{#if pt}
		<div class="step-head">
			<span class="pill">2</span>
			<h4>{t({ en: 'Inside one pre-token: merges by rank', zh: '一个片段内部：按排名合并' })} <Tok text={pt.text} /></h4>
		</div>
		<p class="hint">{t({ en: 'Start from the bytes. Look at every adjacent pair; if TinyGPT learned that pair, it has a rank (its place in the merge list). Merge the pair with the lowest rank, then repeat. When no pair has a rank, stop.', zh: '从字节开始。查看每一对相邻的词元；如果 TinyGPT 学过这一对，它就有一个排名（在合并列表中的位置）。合并排名最小的那一对，然后重复。没有任何一对有排名时就停止。' })}</p>
		{#if pt.special}
			<p>{t({ en: 'This is the special token <|endoftext|>: it is matched as a whole and never split.', zh: '这是特殊词元 <|endoftext|>：整体匹配，从不拆分。' })}</p>
		{:else}
			<StepPlayer count={pt.steps.length + 1} bind:step {labels} interval={900} label={{ en: 'Merges inside this pre-token', zh: '这个片段内部的合并' }}>
				{#snippet children(s0)}
					{@const s = clampStep(s0)}
					{@const ids = stateAt(s)}
					{@const made = s > 0 ? pt.steps[s - 1].newId : -1}
					<div class="stage">
						<div class="row-toks">
							{#each ids as id, j (j)}
								<span class="tk">
									<Tok text={disp(id)} mark={id === made ? 'new' : null} />
									<span class="num tid">{id}</span>
								</span>
							{/each}
						</div>
						{#if s > 0}
							{@const st = pt.steps[s - 1]}
							<p class="hint">
								{t({ en: 'Merged', zh: '合并了' })} <Tok text={disp(st.pair[0])} small /> + <Tok text={disp(st.pair[1])} small /> → <Tok text={disp(st.newId)} small mark="new" />
								{t({ en: 'because its rank', zh: '因为它的排名' })} <b class="num">{st.rank}</b> {t({ en: 'was the lowest here.', zh: '在这里最小。' })}
								{#if counts[st.rank] !== undefined}
									<span class="pre">{t({ en: 'TinyGPT learned it when the pair occurred', zh: 'TinyGPT 学到它时，这个对在 TinyStories 里出现了' })} <b class="num">{fmtInt(counts[st.rank])}</b> {t({ en: 'times in TinyStories.', zh: '次。' })}</span>
									<ProvenanceBadge mode="precomputed" compact note={{ en: 'Pair count recorded when the tokenizer was trained (tokenizer.json, mergeCounts).', zh: '训练分词器时记录下的相邻对计数（tokenizer.json 中的 mergeCounts）。' }} />
								{/if}
							</p>
						{/if}
						<div class="cands">
							<span class="muted small">{t({ en: 'Pairs now and their ranks:', zh: '当前的相邻对及其排名：' })}</span>
							{#each candidates as c, j (j)}
								<span class="cand" class:best={c.r !== undefined && c.r === minRank}>
									<Tok text={disp(c.a)} small /><Tok text={disp(c.b)} small />
									<span class="num">{c.r === undefined ? '—' : `#${c.r}`}</span>
								</span>
							{/each}
							{#if !candidates.length}<span class="small">{t({ en: 'one token left', zh: '只剩一个词元' })}</span>{/if}
							{#if candidates.length && minRank === Infinity}<span class="small done">{t({ en: 'no pair was ever learned: done', zh: '没有学过的对：结束' })}</span>{/if}
						</div>
					</div>
				{/snippet}
			</StepPlayer>
		{/if}
	{/if}

	<div class="step-head">
		<span class="pill">3</span>
		<h4>{t({ en: 'What the model receives: token ids', zh: '模型收到的：词元 ID' })}</h4>
	</div>
	<Chips items={chipItems} marks={chipMarks} showIds limit={600} label={{ en: 'TinyGPT tokens of your sentence', zh: '你的句子的 TinyGPT 词元' }} onhover={(i) => (hover = i)} onselect={(i) => { userPicked = true; sel = flat[i]?.pt ?? sel; }} />
	<p class="ids num" aria-label={t({ en: 'Token ids', zh: '词元 ID' })}>[{es.ids.join(', ')}]</p>

	<div class="step-head">
		<span class="pill">4</span>
		<h4>{t({ en: 'The bytes underneath', zh: '底层的字节' })}</h4>
	</div>
	<p class="hint">{t({ en: 'Each character is 1 to 4 UTF-8 bytes. Bytes with the same tint belong to the same token; hover a byte or a token to link them.', zh: '每个字符占 1 到 4 个 UTF-8 字节。同一种底色的字节属于同一个词元；把鼠标移到字节或词元上可以看到它们的对应关系。' })}</p>
	<ByteView text={sentence} ranges={flat} bind:hover label={{ en: 'UTF-8 bytes of your sentence', zh: '你的句子的 UTF-8 字节' }} />

	{#if cjk > 0}
		<div class="callout">
			<p>
				<b>{t({ en: 'Why does Chinese turn into byte pieces?', zh: '为什么中文变成了字节碎片？' })}</b>
				{t({ en: 'Your sentence has', zh: '你的句子有' })} <b class="num">{cjk}</b> {t({ en: 'Chinese characters, and TinyGPT split them into', zh: '个汉字，TinyGPT 把它们切成了' })} <b class="num">{rawBytes}</b> {t({ en: 'single-byte tokens.', zh: '个单字节词元。' })}
				{t({ en: 'A Chinese character is 3 UTF-8 bytes. TinyGPT’s merges were learned from English children’s stories, where these bytes almost never appear, so no merge joins them. Each byte stays its own token: 我爱自然语言处理 (8 characters) costs 24 tokens.', zh: '一个汉字占 3 个 UTF-8 字节。TinyGPT 的合并规则是从英文儿童故事里学来的，这些字节几乎从不出现，所以没有任何合并能把它们拼起来，每个字节各自成为一个词元：“我爱自然语言处理”（8 个字）要 24 个词元。' })}
			</p>
			<p>
				{t({ en: 'Byte-level BPE never fails: any text can be written with the 256 bytes, so there is no [UNK]. But the model never saw these ids during training, so its predictions after Chinese text are meaningless.', zh: '字节级 BPE 永远不会失败：任何文本都能用 256 个字节写出来，所以没有 [UNK]。但模型在训练中从没见过这些 ID，所以它对中文后面的预测毫无意义。' })}
			</p>
		</div>
	{/if}
</div>

<style>
	.tg {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s3);
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
	.big {
		font-size: var(--fs-lg);
		color: var(--ink);
	}
	.step-head {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-top: var(--s3);
	}
	.pill {
		display: inline-grid;
		place-items: center;
		width: 22px;
		height: 22px;
		border-radius: 50%;
		background: var(--accent);
		color: var(--accent-ink);
		font-size: var(--fs-xs);
		font-weight: 700;
		flex: none;
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
		max-width: var(--measure);
	}
	.pts {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.pt {
		display: grid;
		gap: 3px;
		justify-items: start;
		padding: 5px 8px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		color: inherit;
		text-align: left;
		max-width: 100%;
		min-width: 0;
	}
	.pt:hover {
		background: var(--surface-2);
	}
	.pt.on {
		border-color: var(--accent);
		box-shadow: 0 0 0 1px var(--accent);
	}
	.ptext {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--muted);
		white-space: pre;
		max-width: 100%;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.ptoks {
		display: flex;
		flex-wrap: wrap;
		gap: 2px;
	}
	.pmeta {
		font-size: var(--t-xs);
		color: var(--muted);
	}
	.stage {
		display: grid;
		gap: 8px;
		padding: 10px 12px;
		border-radius: var(--r2);
		background: var(--surface-2);
		min-width: 0;
	}
	.row-toks {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.tk {
		display: inline-flex;
		flex-direction: column;
		align-items: center;
	}
	.tid {
		font-size: var(--t-xs);
		color: var(--muted);
	}
	.cands {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 10px;
	}
	.cand {
		display: inline-flex;
		align-items: center;
		gap: 2px;
		padding: 1px 4px;
		border-radius: 4px;
		font-size: var(--fs-xs);
	}
	.cand.best {
		outline: 2px solid var(--accent);
	}
	.small {
		font-size: var(--fs-xs);
	}
	.done {
		color: var(--live);
		font-weight: 650;
	}
	.pre {
		color: var(--ink-2);
	}
	.ids {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		overflow-wrap: anywhere;
	}
	.callout {
		border-left: 4px solid var(--cat-1);
		background: var(--surface);
		border-radius: var(--r2);
		padding: 10px 14px;
		display: grid;
		gap: 6px;
	}
	.callout p {
		margin: 0;
	}
</style>

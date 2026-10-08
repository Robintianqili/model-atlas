<!--
  PART 3: the same sentence through six tokenizers: your BPE (Part 1), TinyGPT, GPT-2, BERT,
  GPT-4 (cl100k_base) and Qwen2.5. Real tokenizer files load lazily into the Web Worker when this
  block scrolls into view (sizes shown). Hovering a token anywhere highlights the same bytes in
  every row and in the character strip.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtBytes, fmtInt } from '$lib/kit/format';
	import { bytesToDisplay, type BPETokenizer } from '$lib/models/bpe';
	import Chips from './Chips.svelte';
	import type { ChipItem } from './types';
	import type { RealResult, RealToken } from './realtok';
	import { segment, type TrainTrace } from './train';
	import { REAL, REAL_TOTAL_BYTES, ROW_COLOR, ROW_NAME, PARITY, type RealId, type RowId } from './tokenizers';
	import { real, conn, loadAll, loadReal, readyIds } from './state.svelte';

	interface Props {
		sentence: string;
		tiny: BPETokenizer;
		trace: TrainTrace | null;
		k: number;
	}
	let { sentence, tiny, trace, k }: Props = $props();

	let root: HTMLDivElement | undefined = $state();
	let results = $state.raw<Partial<Record<RealId, RealResult>>>({});
	let encMs = $state<Partial<Record<RealId, number>>>({});
	let seq = 0;
	let hover = $state<{ row: RowId; i: number } | null>(null);
	let charHover = $state<number | null>(null);

	onMount(() => {
		if (!root || typeof IntersectionObserver === 'undefined') {
			loadAll();
			return;
		}
		const io = new IntersectionObserver(
			(es) => {
				if (es.some((e) => e.isIntersecting)) {
					loadAll();
					io.disconnect();
				}
			},
			{ rootMargin: '400px 0px' }
		);
		io.observe(root);
		return () => io.disconnect();
	});

	// re-encode (debounced) when the sentence changes or another tokenizer becomes ready
	$effect(() => {
		const text = sentence;
		void real.version;
		const c = conn.client;
		const ids = readyIds();
		if (!c || !ids.length) return;
		const my = ++seq;
		const id = setTimeout(async () => {
			try {
				const r = await c.encode(text, ids);
				if (my === seq) {
					results = r.results as Partial<Record<RealId, RealResult>>;
					encMs = r.ms as Partial<Record<RealId, number>>;
				}
			} catch {
				/* worker closed */
			}
		}, 120);
		return () => clearTimeout(id);
	});

	const enc = new TextEncoder();
	const display = (bytes: ArrayLike<number>) => bytesToDisplay(bytes);
	const hexOf = (bytes: ArrayLike<number>) => Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join(' ');

	function fromIds(ids: number[], bytesOf: (id: number) => ArrayLike<number>, special?: (id: number) => boolean): RealResult {
		const tokens: RealToken[] = [];
		let off = 0;
		let partial = 0;
		for (const id of ids) {
			const b = bytesOf(id);
			const text = special?.(id) ? 'special' : display(b);
			const isP = text.includes('<0x');
			if (isP) partial++;
			tokens.push({ id, text: special?.(id) ? tiny.tokenToDisplay(id) : text, start: off, end: off + b.length, kind: special?.(id) ? 'special' : isP ? 'partial' : undefined });
			off += b.length;
		}
		return { tokens, count: tokens.length, pretokens: 0, unk: 0, partial, normalized: false };
	}

	const tinyRes = $derived(fromIds(tiny.encode(sentence), (id) => tiny.tokenBytes(id), (id) => tiny.isSpecial(id)));
	const yoursRes = $derived.by(() => {
		if (!trace) return null;
		const ids = segment(sentence, tiny.pattern, trace.merges, Math.min(k, trace.merges.length)).flatMap((s) => s.ids);
		return fromIds(ids, (id) => trace.bytes[id]);
	});

	const BYTE_BPE = { en: 'byte-level BPE', zh: '字节级 BPE' };
	type Row = { id: RowId; vocab: number; algo: { en: string; zh: string } | string; res: RealResult | null; status: string; bytes?: number; ms?: number; note?: string };
	const rows = $derived.by<Row[]>(() => {
		const out: Row[] = [
			{ id: 'yours', vocab: 256 + Math.min(k, trace?.merges.length ?? 0), algo: BYTE_BPE, res: yoursRes, status: trace ? 'ready' : 'loading' },
			{ id: 'tinygpt', vocab: 1024, algo: BYTE_BPE, res: tinyRes, status: 'ready' }
		];
		for (const r of REAL) {
			out.push({
				id: r.id,
				vocab: r.vocab,
				algo: r.algo === 'wordpiece' ? 'WordPiece' : BYTE_BPE,
				res: real.status[r.id] === 'ready' ? (results[r.id] ?? null) : null,
				status: real.status[r.id],
				bytes: r.bytes,
				ms: real.ms[r.id],
				note: real.error[r.id]
			});
		}
		return out;
	});

	// the byte range under the pointer (from a token or a character)
	const charRanges = $derived.by(() => {
		const out: { ch: string; start: number; end: number }[] = [];
		let off = 0;
		for (const ch of Array.from(sentence)) {
			const n = enc.encode(ch).length;
			out.push({ ch, start: off, end: off + n });
			off += n;
		}
		return out;
	});
	const range = $derived.by<[number, number] | null>(() => {
		if (charHover !== null && charRanges[charHover]) return [charRanges[charHover].start, charRanges[charHover].end];
		if (!hover) return null;
		const row = rows.find((r) => r.id === hover!.row);
		const tok = row?.res?.tokens[hover.i];
		if (!tok || tok.end <= tok.start) return null;
		return [tok.start, tok.end];
	});
	const overlaps = (s: number, e: number) => !!range && e > s && s < range[1] && e > range[0];

	// the sentence's UTF-8, encoded once (not once per token: that was quadratic in the sentence length)
	const sentBytes = $derived(enc.encode(sentence));
	function itemsOf(r: Row): ChipItem[] {
		if (!r.res) return [];
		const u8 = sentBytes;
		return r.res.tokens.map((tk) => ({
			text: tk.text,
			id: tk.id,
			kind: tk.kind,
			hex: tk.end > tk.start ? hexOf(u8.subarray(tk.start, tk.end)) : undefined,
			note: tk.kind === 'unk' ? t({ en: 'not in the vocabulary', zh: '不在词表里' }) : tk.kind === 'partial' ? t({ en: 'part of a character', zh: '一个字符的一部分' }) : undefined
		}));
	}
	const charItems = $derived<ChipItem[]>(charRanges.map((c) => ({ text: c.ch, hex: hexOf(enc.encode(c.ch)) })));
	const charMarks = $derived(charRanges.map((c) => (overlaps(c.start, c.end) ? 'link' : null)));
	const maxCount = $derived(Math.max(1, charRanges.length, ...rows.map((r) => r.res?.count ?? 0)));
	const nBytes = $derived(charRanges.length ? charRanges[charRanges.length - 1].end : 0);
	const anyIdle = $derived(REAL.some((r) => real.status[r.id] === 'idle' || real.status[r.id] === 'error'));
</script>

<div class="rc" bind:this={root}>
	<div class="loadbar">
		<span>{t({ en: 'Tokenizer files are served by this site and loaded only when this part is on screen:', zh: '分词器文件由本站提供，只在这一部分出现在屏幕上时才加载：' })} <b class="num">{fmtBytes(REAL_TOTAL_BYTES)}</b> {t({ en: 'in total. Your text never leaves your device.', zh: '（总计）。你的文本不会离开你的设备。' })}</span>
		{#if anyIdle}<button type="button" class="btn" onclick={loadAll}>{t({ en: 'Load all', zh: '全部加载' })} ({fmtBytes(REAL_TOTAL_BYTES)})</button>{/if}
		<ProvenanceBadge mode="live" compact note={PARITY.note} />
	</div>

	<div class="row chars">
		<div class="head">
			<span class="name">{t({ en: 'Characters', zh: '字符' })}</span>
			<span class="count num">{charRanges.length}</span>
			<span class="meta">{nBytes} {t({ en: 'UTF-8 bytes', zh: '个 UTF-8 字节' })}</span>
		</div>
		<Chips items={charItems} marks={charMarks} dense label={{ en: 'Characters of your sentence', zh: '你的句子中的字符' }} onhover={(i) => (charHover = i)} />
	</div>

	{#each rows as r (r.id)}
		{@const color = `var(--cat-${ROW_COLOR[r.id]})`}
		<div class="row" style:--row={color}>
			<div class="head">
				<span class="sw" style:background={color}></span>
				<span class="name">{t(ROW_NAME[r.id])}</span>
				{#if r.res}<span class="count num">{r.res.count}</span>{/if}
				<span class="meta">{t(r.algo)} · {t({ en: 'vocab', zh: '词表' })} <span class="num">{fmtInt(r.vocab)}</span>{#if r.bytes} · <span class="num">{fmtBytes(r.bytes)}</span>{/if}{#if r.ms !== undefined} · <span class="num">{t({ en: 'parsed in', zh: '解析用时' })} {Math.round(r.ms)} ms</span>{/if}</span>
				{#if r.res && (r.res.unk || r.res.partial)}
					<span class="flags">
						{#if r.res.unk}<span class="flag unk">{r.res.unk} × [UNK]</span>{/if}
						{#if r.res.partial}<span class="flag part">{r.res.partial} {t({ en: 'byte pieces', zh: '个字节碎片' })}</span>{/if}
					</span>
				{/if}
				{#if r.id === 'bert' && r.res}<span class="meta">{t({ en: '+ [CLS] and [SEP], not counted', zh: '另加 [CLS] 和 [SEP]，不计入' })}</span>{/if}
			</div>
			{#if r.res}
				<Chips
					items={itemsOf(r)}
					marks={r.res.tokens.map((tk, i) => ((hover?.row === r.id && hover.i === i) || overlaps(tk.start, tk.end) ? 'link' : null))}
					label={ROW_NAME[r.id]}
					limit={400}
					onhover={(i) => (hover = i === null ? null : { row: r.id, i })}
				/>
			{:else if r.status === 'loading'}
				<p class="status">{t({ en: 'Loading…', zh: '加载中…' })}</p>
			{:else if r.status === 'error'}
				<p class="status err">{t({ en: 'Could not load:', zh: '加载失败：' })} {r.note} <button type="button" class="btn ghost" onclick={() => loadReal(r.id as RealId)}>{t({ en: 'Retry', zh: '重试' })}</button></p>
			{:else if r.status === 'idle'}
				<p class="status"><button type="button" class="btn" onclick={() => loadReal(r.id as RealId)}>{t({ en: 'Load', zh: '加载' })} ({fmtBytes(r.bytes ?? 0)})</button></p>
			{:else}
				<p class="status">…</p>
			{/if}
		</div>
	{/each}

	<figure class="bars">
		<figcaption>{t({ en: 'Tokens for your sentence', zh: '你的句子需要的词元数' })}</figcaption>
		{#each rows as r (r.id)}
			<div class="bar-row">
				<span class="bl">{t(ROW_NAME[r.id])}</span>
				<span class="track"><span class="fill" style:width="{r.res ? (100 * r.res.count) / maxCount : 0}%" style:background="var(--cat-{ROW_COLOR[r.id]})"></span></span>
				<span class="bv num">{r.res ? r.res.count : '…'}</span>
			</div>
		{/each}
		<div class="bar-row ref">
			<span class="bl">{t({ en: 'characters', zh: '字符数' })}</span>
			<span class="track"><span class="fill" style:width="{(100 * charRanges.length) / maxCount}%"></span></span>
			<span class="bv num">{charRanges.length}</span>
		</div>
	</figure>
</div>

<style>
	.rc {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s3);
	}
	.loadbar {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 12px;
		align-items: center;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.row {
		display: grid;
		gap: 6px;
		padding: 10px 12px;
		border: 1px solid var(--rule);
		border-left: 4px solid var(--row, var(--rule-strong));
		border-radius: var(--r2);
		background: var(--surface);
		min-width: 0;
	}
	.row.chars {
		background: var(--surface-2);
	}
	.head {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 10px;
	}
	.sw {
		width: 10px;
		height: 10px;
		border-radius: 2px;
		align-self: center;
	}
	.name {
		font-weight: 700;
	}
	.count {
		font-size: var(--fs-lg);
		font-weight: 700;
	}
	.meta {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.flags {
		display: inline-flex;
		gap: 6px;
	}
	.flag {
		font-size: var(--fs-xs);
		padding: 0 6px;
		border-radius: var(--r-pill);
		border: 1px solid currentColor;
	}
	.flag.unk {
		color: var(--k-text);
	}
	.flag.part {
		color: var(--ink-2);
		border-style: dashed;
	}
	.status {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--muted);
	}
	.err {
		color: var(--k-text);
	}
	.bars {
		margin: 0;
		display: grid;
		gap: 4px;
		padding: 10px 12px;
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	figcaption {
		font-weight: 650;
		font-size: var(--fs-sm);
		margin-bottom: 4px;
	}
	.bar-row {
		display: grid;
		grid-template-columns: minmax(7.5em, auto) minmax(0, 1fr) 3em;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-sm);
	}
	.track {
		height: 12px;
		background: var(--surface);
		border-radius: 3px;
		overflow: hidden;
	}
	.fill {
		display: block;
		height: 100%;
		border-radius: 3px;
	}
	.ref .fill {
		background: repeating-linear-gradient(-45deg, var(--rule-strong) 0 4px, transparent 4px 8px);
	}
	.ref {
		color: var(--muted);
	}
	.bv {
		text-align: right;
	}
</style>

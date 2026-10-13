<!--
  Lesson step 7: the same sentence through six tokenizers made the same way at different scales:
  your BPE (all your merges), TinyGPT, GPT-2, BERT (WordPiece), GPT-4 (cl100k_base) and Qwen2.5.
  The four real tokenizer files load into the page's Web Worker when this step is entered (sizes
  shown; served by this site). One row per tokenizer: token count bar and the tokens themselves.
-->
<script lang="ts">
	import { t, i18n } from '$lib/i18n/i18n.svelte';
	import { fmtBytes, fmtInt } from '$lib/kit/format';
	import type { BPETokenizer } from '$lib/models/bpe';
	import Chips from '../Chips.svelte';
	import type { ChipItem } from '../types';
	import type { RealResult } from '../realtok';
	import { REAL, REAL_BY_ID, REAL_TOTAL_BYTES, ROW_COLOR, ROW_NAME, type RealId, type RowId } from '../tokenizers';
	import { real, conn, loadAll, loadReal, readyIds } from '../state.svelte';

	interface Props {
		sentence: string;
		tiny: BPETokenizer;
		yours: BPETokenizer | null;
		/** token counts per row, for the narration (bindable, read-only for the parent) */
		counts?: Partial<Record<RowId, number>>;
	}
	let { sentence, tiny, yours, counts = $bindable({}) }: Props = $props();

	let results = $state.raw<Partial<Record<RealId, RealResult>>>({});
	let seq = 0;
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
				if (my === seq) results = r.results as Partial<Record<RealId, RealResult>>;
			} catch {
				/* worker closed */
			}
		}, 100);
		return () => clearTimeout(id);
	});

	const enc = new TextEncoder();
	const hexOf = (b: ArrayLike<number>) => Array.from(b, (x) => x.toString(16).padStart(2, '0')).join(' ');
	function bpeItems(tok: BPETokenizer): ChipItem[] {
		return tok.encode(sentence).map((id) => {
			const d = tok.tokenToDisplay(id);
			return { text: d, id, hex: hexOf(tok.tokenBytes(id)), kind: tok.isSpecial(id) ? 'special' : d.includes('<0x') ? 'partial' : undefined };
		});
	}
	const sentBytes = $derived(enc.encode(sentence));
	function realItems(r: RealResult): ChipItem[] {
		return r.tokens.map((tk) => ({ text: tk.text, id: tk.id, kind: tk.kind, hex: tk.end > tk.start ? hexOf(sentBytes.subarray(tk.start, tk.end)) : undefined }));
	}

	const HOW: Record<RowId, { en: string; zh: string }> = {
		yours: { en: 'byte-level BPE, merges learned from your training text', zh: '字节级 BPE，合并规则从你的训练文本学来' },
		tinygpt: { en: 'byte-level BPE, 767 merges learned from TinyStories', zh: '字节级 BPE，从 TinyStories 学到 767 次合并' },
		gpt2: { en: 'byte-level BPE, 50,000 merges learned from English web text', zh: '字节级 BPE，从英文网页文本学到 50,000 次合并' },
		bert: { en: 'WordPiece (a cousin of BPE), lower-cased English', zh: 'WordPiece（BPE 的近亲），英文转小写' },
		cl100k: { en: 'byte-level BPE, about 100,000 tokens', zh: '字节级 BPE，约 100,000 个词元' },
		qwen: { en: 'byte-level BPE, about 151,000 tokens, many of them Chinese words', zh: '字节级 BPE，约 151,000 个词元，其中很多是中文词' }
	};

	const howYours = $derived({
		en: `byte-level BPE, ${fmtInt(yours?.merges.length ?? 0)} merges learned from your training text`,
		zh: `字节级 BPE，从你的训练文本学到 ${fmtInt(yours?.merges.length ?? 0)} 次合并`
	});

	type Row = { id: RowId; vocab: number; items: ChipItem[] | null; count: number | null; unk: number; status: string; bytes?: number; err?: string };
	const rows = $derived.by<Row[]>(() => {
		const yi = yours ? bpeItems(yours) : null;
		const ti = bpeItems(tiny);
		const out: Row[] = [
			{ id: 'yours', vocab: yours ? yours.vocabSize : 256, items: yi, count: yi ? yi.length : null, unk: 0, status: yours ? 'ready' : 'loading' },
			{ id: 'tinygpt', vocab: tiny.vocabSize, items: ti, count: ti.length, unk: 0, status: 'ready' }
		];
		for (const r of REAL) {
			const res = real.status[r.id] === 'ready' ? results[r.id] : undefined;
			out.push({ id: r.id, vocab: r.vocab, items: res ? realItems(res) : null, count: res ? res.count : null, unk: res?.unk ?? 0, status: real.status[r.id], bytes: r.bytes, err: real.error[r.id] });
		}
		return out;
	});
	$effect(() => {
		const c: Partial<Record<RowId, number>> = {};
		for (const r of rows) if (r.count !== null) c[r.id] = r.count;
		counts = c;
	});
	const nChars = $derived(Array.from(sentence).length);
	const maxCount = $derived(Math.max(1, nChars, ...rows.map((r) => r.count ?? 0)));
	const anyIdle = $derived(REAL.some((r) => real.status[r.id] === 'idle' || real.status[r.id] === 'error'));
</script>

<div class="ls">
	<h4>{t({ en: 'Your sentence in six tokenizers', zh: '你的句子在六个分词器里' })}</h4>
	<div class="loadbar">
		{#if i18n.lang === 'zh'}<span>四个真实分词器的文件由本站提供，到这一步才加载，总计 <b class="num">{fmtBytes(REAL_TOTAL_BYTES)}</b>。你的文本不会离开你的设备。</span>{:else}<span>The four real tokenizer files come from this site and load only now: <b class="num">{fmtBytes(REAL_TOTAL_BYTES)}</b> in total. Your text never leaves your device.</span>{/if}
		{#if anyIdle}<button type="button" class="btn small" onclick={loadAll}>{t({ en: 'Load all', zh: '全部加载' })} ({fmtBytes(REAL_TOTAL_BYTES)})</button>{/if}
	</div>
	<ul class="rows">
		{#each rows as r (r.id)}
			{@const color = `var(--cat-${ROW_COLOR[r.id]})`}
			<li class="row" style:--row={color}>
				<div class="head">
					<span class="sw" style:background={color}></span>
					<span class="name">{t(ROW_NAME[r.id])}</span>
					<span class="meta">{t(r.id === 'yours' ? howYours : HOW[r.id])}{' · '}{t({ en: 'vocabulary', zh: '词表' })} <span class="num">{fmtInt(r.vocab)}</span>{#if r.bytes}{' · '}<span class="num">{fmtBytes(r.bytes)}</span>{/if}</span>
				</div>
				<div class="cbar">
					<span class="track"><span class="fill" style:width="{r.count !== null ? (100 * r.count) / maxCount : 0}%" style:background={color}></span></span>
					<span class="cnt num">{r.count !== null ? r.count : '…'}</span>
				</div>
				{#if r.items}
					<Chips items={r.items} dense limit={80} label={ROW_NAME[r.id]} />
					{#if r.unk || r.id === 'bert'}
						<p class="flags">
							{#if r.unk}<span class="flag unk">{r.unk} × [UNK]</span>{/if}
							{#if r.id === 'bert'}<span class="muted">{t({ en: '[CLS] and [SEP] are added around every input and not counted.', zh: '[CLS] 和 [SEP] 加在每条输入的两端，不计入词元数。' })}</span>{/if}
						</p>
					{/if}
				{:else if r.status === 'loading'}
					<p class="status">{t({ en: 'Loading', zh: '正在加载' })} {r.bytes ? fmtBytes(r.bytes) : ''}…</p>
				{:else if r.status === 'error'}
					<p class="status err">{t({ en: 'Could not load:', zh: '加载失败：' })} {r.err} <button type="button" class="btn ghost small" onclick={() => loadReal(r.id as RealId)}>{t({ en: 'Retry', zh: '重试' })}</button></p>
				{:else if r.status === 'idle'}
					<p class="status"><button type="button" class="btn small" onclick={() => loadReal(r.id as RealId)}>{t({ en: 'Load', zh: '加载' })} ({fmtBytes(REAL_BY_ID[r.id as RealId]?.bytes ?? 0)})</button></p>
				{:else}
					<p class="status">…</p>
				{/if}
			</li>
		{/each}
		<li class="row ref">
			<div class="cbar">
				<span class="name small">{t({ en: 'For scale: characters in your sentence', zh: '参照：你的句子的字符数' })}</span>
				<span class="track"><span class="fill hatch" style:width="{(100 * nChars) / maxCount}%"></span></span>
				<span class="cnt num">{nChars}</span>
			</div>
		</li>
	</ul>
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
	.loadbar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px 12px;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.rows {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 6px;
	}
	.row {
		display: grid;
		gap: 4px;
		padding: 6px 10px;
		border: 1px solid var(--rule);
		border-left: 4px solid var(--row, var(--rule-strong));
		border-radius: var(--r2);
		background: var(--surface);
		min-width: 0;
	}
	.row.ref {
		border-left-color: var(--rule-strong);
		background: var(--surface-2);
	}
	.head {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 2px 10px;
	}
	.sw {
		width: 10px;
		height: 10px;
		border-radius: 2px;
		align-self: center;
	}
	.name {
		font-weight: 700;
		font-size: var(--fs-sm);
	}
	.name.small {
		font-weight: 400;
		color: var(--ink-2);
		font-size: var(--fs-xs);
	}
	.meta {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.cbar {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 3em;
		align-items: center;
		gap: 8px;
	}
	.ref .cbar .name {
		grid-column: 1 / -1;
	}
	.track {
		height: 10px;
		border-radius: 3px;
		background: var(--surface-2);
		overflow: hidden;
	}
	.fill {
		display: block;
		height: 100%;
		border-radius: 3px;
	}
	.fill.hatch {
		background: repeating-linear-gradient(-45deg, var(--rule-strong) 0 4px, transparent 4px 8px);
	}
	.cnt {
		text-align: right;
		font-weight: 700;
		font-size: var(--fs-lg);
	}
	.flags {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 10px;
		font-size: var(--fs-xs);
	}
	.flag {
		padding: 0 6px;
		border-radius: var(--r-pill);
		border: 1px solid currentColor;
	}
	.flag.unk {
		color: var(--k-text);
	}
	.status {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--muted);
	}
	.err {
		color: var(--k-text);
	}
	.btn.small {
		min-height: 32px;
		padding: 0 12px;
		font-size: var(--fs-sm);
	}
</style>

<!--
  COMPARE: one sentence with the same meaning in 10 languages, counted by every tokenizer (live).
  Cells are tinted by the cost relative to English for that tokenizer (purple = more tokens).
  Select a cell to see how that tokenizer cut that sentence.
-->
<script lang="ts">
	import { t, i18n } from '$lib/i18n/i18n.svelte';
	import { fmtInt } from '$lib/kit/format';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { bytesToDisplay, type BPETokenizer } from '$lib/models/bpe';
	import Chips from './Chips.svelte';
	import type { ChipItem, LangSummary } from './types';
	import type { RealResult } from './realtok';
	import { segment, type TrainTrace } from './train';
	import { PRESETS } from './presets';
	import { REAL, ROW_COLOR, ROW_NAME, type RealId, type RowId } from './tokenizers';
	import { real, conn, readyIds, loadAll } from './state.svelte';

	interface Props {
		tiny: BPETokenizer;
		trace: TrainTrace | null;
		k: number;
		/** called with a sentence to copy it into the page's sentence input */
		onuse?: (text: string) => void;
		/** per tokenizer: English, cheapest and most expensive language (for the pitfalls section) */
		summary?: LangSummary[];
	}
	let { tiny, trace, k, onuse, summary = $bindable([]) }: Props = $props();

	const langs = PRESETS.languages;
	const cols: RowId[] = ['yours', 'tinygpt', 'gpt2', 'bert', 'cl100k', 'qwen'];
	let realCounts = $state.raw<Record<string, [number, number]>[]>([]);
	let mode = $state<'count' | 'ratio'>('count');
	let sel = $state<{ li: number; col: RowId } | null>({ li: 1, col: 'gpt2' });
	let selRes = $state.raw<RealResult | null>(null);

	$effect(() => {
		void real.version;
		const c = conn.client;
		const ids = readyIds();
		if (!c || !ids.length) return;
		c.counts(
			langs.map((l) => l.text),
			ids
		)
			.then((r) => (realCounts = r))
			.catch(() => {});
	});

	const enc = new TextEncoder();
	const yoursCount = (text: string) =>
		trace ? segment(text, tiny.pattern, trace.merges, Math.min(k, trace.merges.length)).reduce((s, p) => s + p.ids.length, 0) : undefined;

	const table = $derived(
		langs.map((l, li) => {
			const row: Partial<Record<RowId, number>> = {
				yours: yoursCount(l.text),
				tinygpt: tiny.encode(l.text, { allowSpecial: false }).length
			};
			const unk: Partial<Record<RowId, number>> = {};
			for (const r of REAL)
				if (real.status[r.id] === 'ready' && realCounts[li]?.[r.id] !== undefined) {
					row[r.id] = realCounts[li][r.id][0];
					unk[r.id] = realCounts[li][r.id][1];
				}
			return { l, row, unk, chars: Array.from(l.text).length, bytes: enc.encode(l.text).length };
		})
	);
	const ratio = (li: number, col: RowId) => {
		const v = table[li].row[col];
		const en = table[0].row[col];
		return v !== undefined && en ? v / en : undefined;
	};
	const scale = $derived(makeScale('sequential', [0, 4], themeState.resolved));
	const worst = $derived(
		cols
			.map((c) => {
				let best = { li: -1, r: 0 };
				langs.forEach((_, li) => {
					const r = ratio(li, c);
					// counts that contain [UNK] lost text, so they cannot be "the most expensive"
					if (r !== undefined && !table[li].unk[c] && r > best.r) best = { li, r };
				});
				return { c, ...best };
			})
			.filter((w) => w.li >= 0)
	);

	$effect(() => {
		const out: LangSummary[] = [];
		for (const c of cols) {
			const en = table[0].row[c];
			if (en === undefined) continue;
			let min = { li: 0, v: en };
			let max = { li: 0, v: en };
			table.forEach((r, li) => {
				const v = r.row[c];
				if (v === undefined || r.unk[c]) return; // skip counts that contain [UNK]
				if (v < min.v) min = { li, v };
				if (v > max.v) max = { li, v };
			});
			out.push({ col: c, en, min, max });
		}
		summary = out;
	});

	// tokens of the selected cell
	$effect(() => {
		const s = sel;
		void real.version;
		selRes = null;
		if (!s) return;
		const text = langs[s.li].text;
		if (s.col === 'yours' || s.col === 'tinygpt') return;
		const c = conn.client;
		if (!c || real.status[s.col as RealId] !== 'ready') return;
		c.encode(text, [s.col as RealId])
			.then((r) => {
				if (sel === s) selRes = r.results[s.col] ?? null;
			})
			.catch(() => {});
	});
	const selItems = $derived.by<ChipItem[]>(() => {
		if (!sel) return [];
		const text = langs[sel.li].text;
		if (sel.col === 'tinygpt') return tiny.encode(text, { allowSpecial: false }).map((id) => ({ text: tiny.tokenToDisplay(id), id }));
		if (sel.col === 'yours') {
			if (!trace) return [];
			return segment(text, tiny.pattern, trace.merges, Math.min(k, trace.merges.length))
				.flatMap((p) => p.ids)
				.map((id) => ({ text: bytesToDisplay(trace.bytes[id]), id }));
		}
		return selRes ? selRes.tokens.map((tk) => ({ text: tk.text, id: tk.id, kind: tk.kind })) : [];
	});
	const anyReal = $derived(REAL.some((r) => real.status[r.id] === 'ready'));
	// on narrow screens the table scrolls inside its box; say so, since no column is cut in half
	let boxW = $state(0);
	let tableW = $state(0);
</script>

<div class="lt">
	<div class="tools">
		<span class="segmented" role="group" aria-label={t({ en: 'Show', zh: '显示' })}>
			<button type="button" aria-pressed={mode === 'count'} onclick={() => (mode = 'count')}>{t({ en: 'Tokens', zh: '词元数' })}</button>
			<button type="button" aria-pressed={mode === 'ratio'} onclick={() => (mode = 'ratio')}>{t({ en: '× English', zh: '× 英语' })}</button>
		</span>
		{#if !anyReal}<button type="button" class="btn" onclick={loadAll}>{t({ en: 'Load the real tokenizers', zh: '加载真实分词器' })}</button>{/if}
		{#if tableW > boxW + 2}<span class="swipe">{t({ en: 'Scroll the table sideways to see all six tokenizers →', zh: '左右滑动表格，查看全部六个分词器 →' })}</span>{/if}
	</div>
	<!-- svelte-ignore a11y_no_noninteractive_tabindex (a scrollable region must be reachable by keyboard) -->
	<div class="scroll" role="region" aria-label={t({ en: 'Token counts by language', zh: '各语言的词元数' })} tabindex="0" bind:clientWidth={boxW}>
		<table bind:clientWidth={tableW}>
			<thead>
				<tr>
					<th scope="col">{t({ en: 'Language', zh: '语言' })}</th>
					<th scope="col" class="numc">{t({ en: 'chars', zh: '字符' })}</th>
					<th scope="col" class="numc">{t({ en: 'bytes', zh: '字节' })}</th>
					{#each cols as c (c)}
						<th scope="col" class="numc"><span class="sw" style:background="var(--cat-{ROW_COLOR[c]})"></span>{t(ROW_NAME[c])}</th>
					{/each}
				</tr>
			</thead>
			<tbody>
				{#each table as r, li (r.l.code)}
					<tr>
						<th scope="row">
							<span class="lname">{t(r.l.name)}</span>
							<span class="ltext" lang={r.l.code}>{r.l.text}</span>
						</th>
						<td class="numc num muted">{r.chars}</td>
						<td class="numc num muted">{r.bytes}</td>
						{#each cols as c (c)}
							{@const v = r.row[c]}
							{@const q = ratio(li, c)}
							<td class="numc">
								{#if v !== undefined}
									<button
										type="button"
										class="cell num"
										class:on={sel?.li === li && sel?.col === c}
										style:background={q !== undefined ? scale.css(Math.min(4, q)) : undefined}
										style:color={q !== undefined ? scale.ink(Math.min(4, q)) : undefined}
										aria-label="{t(r.l.name)}, {t(ROW_NAME[c])}: {v} {t({ en: 'tokens', zh: '个词元' })}{q !== undefined ? `, ×${q.toFixed(2)}` : ''}"
										onclick={() => (sel = { li, col: c })}
									>{mode === 'count' ? v : q !== undefined ? `×${q.toFixed(1)}` : v}{#if r.unk[c]}<span class="unk" title="[UNK]">{r.unk[c]}?</span>{/if}</button>
								{:else}
									<span class="muted">…</span>
								{/if}
							</td>
						{/each}
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
	<p class="legend small">
		<span class="ramp" style:background="linear-gradient(90deg, {scale.css(0)}, {scale.css(1)}, {scale.css(2)}, {scale.css(3)}, {scale.css(4)})"></span>
		{t({ en: 'tint = tokens relative to English for the same tokenizer (×1 to ×4 or more)', zh: '底色 = 与同一分词器下英语相比的词元倍数（×1 到 ×4 及以上）' })}
		<span><span class="unk">n?</span> {t({ en: '= n of the tokens are [UNK]: a low count that lost information', zh: '= 其中 n 个词元是 [UNK]：数字小，但信息丢了' })}</span>
	</p>

	{#if worst.length}
		<ul class="worst">
			{#each worst as w (w.c)}
				<li>
					<span class="sw" style:background="var(--cat-{ROW_COLOR[w.c]})"></span><b>{t(ROW_NAME[w.c])}</b>{#if i18n.lang === 'zh'}：最贵的是{t(langs[w.li].name)}，词元数是英语的 <b class="num">{w.r.toFixed(1)}</b> 倍{:else}: the most expensive language is {t(langs[w.li].name)}, at <b class="num">{w.r.toFixed(1)}×</b> the tokens of English{/if}
				</li>
			{/each}
		</ul>
	{/if}

	{#if sel}
		<div class="panel zoom">
			<p class="zh-head">
				<span class="sw" style:background="var(--cat-{ROW_COLOR[sel.col]})"></span>
				<b>{t(ROW_NAME[sel.col])}</b> · {t(langs[sel.li].name)} · <span class="num">{selItems.length}</span> {t({ en: 'tokens', zh: '个词元' })}
				{#if onuse}<button type="button" class="btn ghost" onclick={() => onuse?.(langs[sel!.li].text)}>{t({ en: 'Use this sentence above', zh: '在上方使用这句话' })}</button>{/if}
			</p>
			{#if selItems.length}
				<Chips items={selItems} label={{ en: 'Tokens of the selected sentence', zh: '所选句子的词元' }} />
			{:else}
				<p class="muted">{t({ en: 'Loading…', zh: '加载中…' })}</p>
			{/if}
		</div>
	{/if}
</div>

<style>
	.lt {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s3);
	}
	.tools {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
	}
	.swipe {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.scroll {
		overflow-x: auto;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
	}
	table {
		border-collapse: collapse;
		width: 100%;
		font-size: var(--fs-sm);
	}
	th,
	td {
		padding: 4px 6px;
		border-bottom: 1px solid var(--rule);
		text-align: left;
		vertical-align: middle;
	}
	thead th {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		white-space: nowrap;
		position: sticky;
		top: 0;
		background: var(--surface);
	}
	th[scope='row'] {
		font-weight: 400;
		min-width: 11em;
		max-width: 22em;
		position: sticky;
		left: 0;
		z-index: 1;
		background: var(--surface);
	}
	thead th:first-child {
		position: sticky;
		left: 0;
		z-index: 2;
	}
	@media (max-width: 640px) {
		th[scope='row'] {
			min-width: 6.5em;
			max-width: 8.5em;
			box-shadow: 1px 0 0 var(--rule);
		}
		.ltext {
			max-width: 8em;
		}
	}
	.lname {
		display: block;
		font-weight: 650;
	}
	.ltext {
		display: block;
		font-size: var(--fs-xs);
		color: var(--muted);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		max-width: 22em;
	}
	.numc {
		text-align: right;
	}
	.cell {
		min-width: 3.2em;
		min-height: 30px;
		padding: 2px 6px;
		border: 1px solid transparent;
		border-radius: 4px;
		font-size: var(--fs-sm);
		text-align: right;
	}
	.cell.on {
		outline: 2px solid var(--accent);
		outline-offset: 1px;
	}
	.sw {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 2px;
		margin-right: 4px;
	}
	.legend {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 8px;
		color: var(--ink-2);
	}
	.unk {
		display: inline-block;
		margin-left: 3px;
		padding: 0 3px;
		border-radius: 3px;
		font-size: 10px;
		font-weight: 700;
		color: var(--surface);
		background: var(--k-text);
		vertical-align: 1px;
	}
	.small {
		font-size: var(--fs-xs);
	}
	.ramp {
		display: inline-block;
		width: 90px;
		height: 10px;
		border-radius: 3px;
		border: 1px solid var(--rule);
		flex: none;
	}
	.worst {
		margin: 0;
		padding-left: 0;
		list-style: none;
		font-size: var(--fs-sm);
		display: grid;
		gap: 2px;
	}
	.zoom {
		display: grid;
		gap: 8px;
	}
	.zh-head {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px;
	}
</style>

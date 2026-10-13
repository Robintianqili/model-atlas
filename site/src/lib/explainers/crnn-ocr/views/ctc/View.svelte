<!--
  crnn-ocr/ctc: the strip of writing (ink on the sequential ramp), under it one cell per column with the symbol of the
  path shown (cell colour = that symbol's chance in the column), under that what CTC does with each cell (a new
  character, a merged repeat, a dropped blank) and the text. Modes (data.ts): decode follows the decode process up to the
  step's column; paths lets the learner pick any symbol for any column; target shows the best path that spells the
  text the learner wrote. Nothing animates by itself (the lesson's scrubber moves decode).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmtPct } from '../../../../atlas/core/format.ts';
	import Notice from '../../../../atlas/ui/views/kit.parts/Notice.svelte';
	import { SYMBOLS } from '../../glyphs.ts';
	import { chance, pathFacts, stripOf } from './data.ts';

	let P: StageProps = $props();
	const S = $derived(stripOf(P));
	const seq = $derived(makeScale('sequential', [0, 1], themeState.resolved));
	let edit = $state<number[] | null>(null);
	let col = $state(0);
	$effect(() => {
		void S?.q.p;
		edit = null;
	});
	const path = $derived(!S ? [] : S.mode === 'paths' ? (edit ?? S.best) : S.best);
	const last = $derived(S ? (S.mode === 'decode' ? S.upTo : S.q.T - 1) : -1);
	const acts = $derived(path.map((k, i) => (k === 0 ? 'blank' : i > 0 && path[i - 1] === k ? 'repeat' : 'emit')));
	const facts = $derived(S ? pathFacts(S, path) : null);
	const soFar = $derived(path.filter((_, i) => i <= last && acts[i] === 'emit').map((k) => SYMBOLS[k]).join(''));
	const sym = (k: number) => SYMBOLS[k];
	const pct = (lp: number) => fmtPct(Math.exp(lp), 2);
	function pick(k: number) {
		if (!S) return;
		const p = (edit ?? S.best).slice();
		p[col] = k;
		edit = p;
	}

	let canvas = $state<HTMLCanvasElement | null>(null);
	$effect(() => {
		const cv = canvas;
		const s = S;
		if (!cv || !s) return;
		const [, h, w] = s.ink.shape;
		const ctx = cv.getContext('2d');
		if (!ctx) return;
		const img = ctx.createImageData(w, h);
		for (let i = 0; i < h * w; i++) seq.write(s.ink.data[i], img.data, i * 4);
		ctx.putImageData(img, 0, 0);
	});
	const L = {
		strip: { en: 'Your writing as the model sees it', zh: '模型看到的你写的字' },
		row: { en: 'Symbol of each column (– = blank)', zh: '每一列的符号（– = 空白）' },
		ctc: { en: 'What CTC keeps', zh: 'CTC 留下的' },
		grid: { en: 'Chance of each symbol (rows) in each column (left to right); the stronger the color, the likelier', zh: '每一列（从左到右）中每个符号（各行）的概率；颜色越浓越可能' },
		picker: { en: 'Pick a symbol for column', zh: '给这一列选符号：第' }
	};
</script>

{#if !S}
	<Notice text={{ en: 'Nothing written yet: write a number on the pad.', zh: '还没有写字：在画板上写一个数。' }} />
{:else}
	<div class="ctc" use:keyScroll>
		<div class="inner" style:--n={S.q.T}>
			<p class="lab muted">{t(L.strip)}</p>
			<canvas bind:this={canvas} width={S.ink.shape[2]} height={S.ink.shape[1]} aria-hidden="true"></canvas>
			{#if S.mode === 'chances'}
				<p class="lab muted">{t(L.grid)}</p>
				{#each SYMBOLS as s, k (k)}
					<div class="cells" aria-hidden="true">
						{#each path as _, i (i)}
							{@const p = chance(S, i, k)}
							<span class="cell" style:background={seq.css(p)} style:color={seq.ink(p)}>{s}</span>
						{/each}
					</div>
				{/each}
			{:else}
				<p class="lab muted">{t(L.row)}</p>
				<div class="cells" role="group" aria-label={t(L.row)}>
					{#each path as k, i (i)}
						{@const p = chance(S, i, k)}
						{#if i > last}
							<span class="cell todo" aria-hidden="true"></span>
						{:else if S.mode === 'paths'}
							<button
								type="button"
								class="cell"
								class:sel={i === col}
								aria-pressed={i === col}
								aria-label={`${t({ en: 'column', zh: '第' })} ${i}${t({ en: ':', zh: ' 列：' })} ${sym(k)}, ${fmtPct(p)}`}
								style:background={seq.css(p)}
								style:color={seq.ink(p)}
								onclick={() => (col = i)}>{sym(k)}</button
							>
						{:else}
							<span class="cell" style:background={seq.css(p)} style:color={seq.ink(p)} title={fmtPct(p)}>{sym(k)}</span>
						{/if}
					{/each}
				</div>
				<p class="lab muted">{t(L.ctc)}</p>
				<div class="cells acts" aria-hidden="true">
					{#each acts as a, i (i)}
						<span class="act {i > last ? '' : a}">{i > last ? '' : a === 'emit' ? sym(path[i]) : a === 'repeat' ? '=' : '·'}</span>
					{/each}
				</div>
			{/if}
		</div>
	</div>
	{#if S.mode !== 'chances'}
		<p class="read">
			{t({ en: 'Text', zh: '文字' })}: <strong class="txt">“{soFar}”</strong>
			{#if S.wrote && last === S.q.T - 1}
				<span class="muted">· {t({ en: 'you wrote', zh: '你写的是' })} “{S.wrote}”</span>
			{/if}
		</p>
		{#if facts && last === S.q.T - 1}
			<p class="facts">
				{t({ en: 'Chance of this path', zh: '这条路径的概率' })}: <strong>{pct(facts.logP)}</strong> ·
				{t({ en: `paths that spell “${facts.text}”: ${facts.paths.toLocaleString('en')}, together`, zh: `拼出“${facts.text}”的路径：${facts.paths.toLocaleString('en')} 条，合计` })}
				<strong>{pct(facts.logText)}</strong>
			</p>
		{/if}
	{/if}
	{#if S.mode === 'paths'}
		<div class="picker" role="group" aria-label={`${t(L.picker)} ${col}`}>
			<span class="muted">{t({ en: `Column ${col} (counting from 0):`, zh: `第 ${col} 列（从 0 数起）：` })}</span>
			{#each SYMBOLS as s, k (k)}
				<button type="button" class="chip" aria-pressed={path[col] === k} onclick={() => pick(k)}>{s} <span class="muted">{fmtPct(chance(S, col, k))}</span></button>
			{/each}
			<button type="button" class="btn small" disabled={!edit} onclick={() => (edit = null)}>{t({ en: 'Back to the top symbols', zh: '恢复成排第一的符号' })}</button>
		</div>
	{/if}
{/if}

<style>
	.ctc {
		overflow-x: auto;
		max-width: 100%;
	}
	.inner {
		min-width: calc(var(--n) * 10px);
		display: grid;
		gap: 4px;
	}
	canvas {
		width: 100%;
		image-rendering: pixelated;
		border: 1px solid var(--rule);
	}
	.lab {
		margin: 4px 0 0;
		font-size: var(--fs-xs);
	}
	.cells {
		display: grid;
		grid-template-columns: repeat(var(--n), 1fr);
		gap: 1px;
	}
	.cell {
		font: inherit;
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		text-align: center;
		min-height: 24px;
		padding: 0;
		border: 1px solid transparent;
		border-radius: 2px;
		line-height: 22px;
	}
	button.cell {
		cursor: pointer;
	}
	.cell.sel {
		outline: 2px solid var(--accent);
		outline-offset: 1px;
	}
	.cell.todo {
		border: 1px dashed var(--rule);
	}
	.act {
		text-align: center;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.act.emit {
		color: var(--ink);
		font-weight: 700;
	}
	.read,
	.facts {
		margin: 6px 0 0;
		font-size: var(--fs-sm);
	}
	.txt {
		font-family: var(--font-mono);
		font-size: var(--fs-md, 1.1em);
	}
	.picker {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 6px;
		align-items: center;
		margin-top: 8px;
		font-size: var(--fs-xs);
	}
	.picker .chip {
		cursor: pointer;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		padding: 2px 8px;
	}
	.picker .chip[aria-pressed='true'] {
		background: var(--accent-soft);
		border-color: var(--accent);
	}
	.small {
		min-height: 32px;
		padding: 0 12px;
		font-size: var(--fs-sm);
	}
</style>

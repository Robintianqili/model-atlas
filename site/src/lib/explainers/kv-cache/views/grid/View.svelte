<!--
  'kv-cache/grid' (meta.ts has the props): the tokens read so far, then the attention grid of one pass (rows: the token
  that looks back, columns: the tokens it looks at) with a key and a value slot per token under it. Purple cells are
  computed for the first time (shade: attention weight); orange cells are computed again (no cache);
  gray cells are not computed because their keys and values are already in the cache. With props.check, both runs' tokens
  pass by pass instead.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical, makeScale } from '$lib/kit/color';
	import { fmt3, fmtInt } from '$lib/kit/format';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { Frame } from '../../../../atlas/types/process.ts';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { cellsOf, tokText } from './meta.ts';

	let { rt, source, at, props }: StageProps = $props();

	let mode = $state<'plain' | 'full' | 'cache'>('cache');
	let layer = $state(0);
	let chosen = $state<number | null>(null);
	let len = $state(0);
	$effect(() => {
		mode = props.mode === 'full' || props.mode === 'plain' ? props.mode : 'cache';
		layer = typeof props.layer === 'number' ? props.layer : 0;
	});
	$effect(() => {
		const tp = at.tape;
		len = tp?.length ?? 0;
		if (!(props.controls || props.check) || !tp) return;
		let alive = true;
		rt.ensure(source, 'last').then(() => {
			if (alive) len = at.tape?.length ?? 0;
		});
		return () => {
			alive = false;
		};
	});
	const idx = $derived(props.controls && chosen !== null && chosen < len ? chosen : at.index);
	const frame = $derived<Frame | undefined>(idx === at.index ? at.frame : at.tape?.frame(idx));
	const st = $derived((idx === at.index ? at.state : at.tape?.state(idx)) as { ids?: { data: ArrayLike<number> }; att?: { data: ArrayLike<number> } } | undefined);
	const m = $derived(frame?.metrics);
	const ids = $derived(Array.from(st?.ids?.data ?? []).filter((v) => v >= 0));
	const C = $derived(m?.C ?? 64);
	const N = $derived(m ? Math.min(ids.length, m.P + m.maxNew - 1) : 0);
	const cells = $derived(cellsOf(m, mode === 'cache' ? 'cache' : 'full'));
	const colors = $derived.by(() => {
		void themeState.resolved;
		return { seq: makeScale('sequential', [0, 1]), redo: categorical(1), kv: categorical(0) };
	});
	const W = 600;
	const cs = $derived(N ? Math.min(12, (W - 40) / N) : 10);
	const gx = 30;
	const gy = 8;
	const attAt = (i: number, j: number) => Number(st?.att?.data[(layer * C + i) * C + j] ?? 0);
	/** 'fresh' | 'redo' | 'skip' | 'later' for cell (i, j), j ≤ i, of the frame's pass. */
	function kind(i: number): 'fresh' | 'redo' | 'skip' | 'later' {
		if (!m || i >= m.n) return 'later';
		if (mode === 'plain' || m.k === 0 || i === m.n - 1) return 'fresh';
		return mode === 'full' ? 'redo' : 'skip';
	}
	const H = $derived(gy + N * cs + 12 + 2 * cs + 4);
	const frames = $derived(props.check ? Array.from({ length: len }, (_, i) => at.tape!.frame(i)) : []);
	const last = $derived(frames.length ? frames[frames.length - 1].metrics : undefined);
	const L = {
		full: { en: 'Without a cache', zh: '不用缓存' },
		cache: { en: 'With the KV cache', zh: '用 KV 缓存' },
		fresh: { en: 'computed for the first time (color strength: attention weight)', zh: '第一次计算（颜色深浅：注意力权重）' },
		plain: { en: 'computed (color strength: attention weight)', zh: '已计算（颜色深浅：注意力权重）' },
		redo: { en: 'computed again in this pass (no cache)', zh: '这一遍又算了一次（没有缓存）' },
		skip: { en: 'not computed: done in an earlier pass', zh: '不用计算：之前某一遍已经算过' }
	};
</script>

{#if m}
	<div class="grid-view">
		{#if props.controls}
			<div class="ctl">
				<div class="seg" role="group" aria-label={t({ en: 'Cache', zh: '缓存' })}>
					<button type="button" class="btn" aria-pressed={mode === 'full'} onclick={() => (mode = 'full')}>{t(L.full)}</button>
					<button type="button" class="btn" aria-pressed={mode === 'cache'} onclick={() => (mode = 'cache')}>{t(L.cache)}</button>
				</div>
				<div class="seg" role="group" aria-label={t({ en: 'Layer whose attention colors the cells', zh: '用哪一层的注意力给单元上色' })}>
					{#each Array.from({ length: m.L }, (_, l) => l) as l (l)}<button type="button" class="btn" aria-pressed={layer === l} onclick={() => (layer = l)}>{t({ en: `Layer ${l}`, zh: `第 ${l} 层` })}</button>{/each}
				</div>
				<label class="pass">
					<span>{t({ en: `Pass ${m.k} of ${m.maxNew - 1}`, zh: `第 ${m.k} 遍（共 0 到 ${m.maxNew - 1} 遍）` })}</span>
					{#key len}<input type="range" min="0" max={Math.max(0, len - 1)} step="1" value={idx} oninput={(e) => (chosen = Number((e.currentTarget as HTMLInputElement).value))} />{/key}
				</label>
			</div>
		{/if}
		<p class="text" aria-label={t({ en: 'The tokens read in this pass', zh: '这一遍读入的词元' })}>
			{#each ids.slice(0, m.n) as id, i (i)}<span class="tk" class:gen={i >= m.P} class:cur={i === m.n - 1}>{tokText(id)}</span>{/each}
			<span class="arrow">→</span><span class="tk next">{ids[m.n] !== undefined ? tokText(ids[m.n]) : ''}</span>
		</p>
		{#if props.check}
			<p class="sum">
				{t({ en: `Same token in ${last?.sameAll ?? '…'} of ${m.maxNew} passes. Largest logit difference between the two runs: ${fmt3(last?.dLogitMax ?? NaN)}; against the kit’s own forward pass: ${fmt3(last?.dKit ?? NaN)}.`, zh: `${m.maxNew} 遍中有 ${last?.sameAll ?? '…'} 遍选出相同的词元。两次运行的 logit 最大差异：${fmt3(last?.dLogitMax ?? NaN)}；与本站自带的前向计算相比：${fmt3(last?.dKit ?? NaN)}。` })}
			</p>
			<div class="tbl" use:keyScroll>
				<table>
					<thead><tr><th scope="col">{t({ en: 'Pass', zh: '遍' })}</th><th scope="col">{t(L.full)}</th><th scope="col">{t(L.cache)}</th><th scope="col">{t({ en: 'Largest |Δ logit|', zh: '最大 |Δ logit|' })}</th></tr></thead>
					<tbody>
						{#each frames as f (f.step)}
							<tr><th scope="row">{f.step}</th><td class="mono">{tokText(f.metrics?.tokFull ?? -1)}</td><td class="mono">{tokText(f.metrics?.token ?? -1)}</td><td class="num">{fmt3(f.metrics?.dLogit ?? NaN)}</td></tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<figure>
				<svg viewBox="0 0 {W} {H}" role="img" aria-label={t({ en: `Attention grid of pass ${m.k}: ${fmtInt(cells.fresh)} new cells, ${fmtInt(mode === 'full' ? cells.redo : cells.skip)} ${mode === 'full' ? 'computed again' : 'not computed'}`, zh: `第 ${m.k} 遍的注意力网格：${fmtInt(cells.fresh)} 个新单元，${fmtInt(mode === 'full' ? cells.redo : cells.skip)} 个${mode === 'full' ? '重新计算' : '不用计算'}` })}>
					{#each Array.from({ length: N }, (_, i) => i) as i (i)}
						{#each Array.from({ length: i + 1 }, (_, j) => j) as j (j)}
							{@const k = kind(i)}
							{#if k === 'later'}
								<rect class="later" x={gx + j * cs + 0.5} y={gy + i * cs + 0.5} width={cs - 1} height={cs - 1} />
							{:else}
								<rect x={gx + j * cs} y={gy + i * cs} width={cs - 0.6} height={cs - 0.6} style:fill={k === 'fresh' ? colors.seq.css(attAt(i, j)) : k === 'redo' ? colors.redo : 'var(--surface-2)'} style:stroke={k === 'fresh' ? colors.seq.css(1) : k === 'skip' ? 'var(--rule-strong)' : 'none'} stroke-width="0.6" />
							{/if}
						{/each}
					{/each}
					<text class="tick" x={gx - 4} y={gy + (m.n - 0.5) * cs + 3} text-anchor="end">{m.n - 1}</text>
					<text class="tick" x={gx - 4} y={gy + 0.5 * cs + 3} text-anchor="end">0</text>
					{#each ['K', 'V'] as lab, r (lab)}
						{@const y = gy + N * cs + 10 + r * cs}
						<text class="tick" x={gx - 4} y={y + cs * 0.7} text-anchor="end">{lab}</text>
						{#each Array.from({ length: m.n }, (_, j) => j) as j (j)}
							{@const isNew = mode === 'plain' || m.k === 0 || j === m.n - 1}
							<rect x={gx + j * cs} y={y} width={cs - 0.6} height={cs - 0.6} style:fill={isNew ? colors.seq.css(0.85) : mode === 'full' ? colors.redo : colors.kv} />
						{/each}
					{/each}
				</svg>
				<figcaption>{t({ en: 'Columns: the tokens looked at (their keys and values), counting from 0. Rows: the token looking back. Under the grid: each token’s key (K) and value (V).', zh: '列：被查看的词元（它们的键和值），从 0 数起。行：回头查看的词元。网格下方：每个词元的键（K）和值（V）。' })}</figcaption>
			</figure>
			<div class="legend">
				<span class="key"><span class="sw" style:background={colors.seq.css(0.6)}></span>{t(mode === 'plain' ? L.plain : L.fresh)}</span>
				{#if mode === 'full'}<span class="key"><span class="sw" style:background={colors.redo}></span>{t(L.redo)}</span>{:else if mode === 'cache'}<span class="key"><span class="sw gray"></span>{t(L.skip)}</span><span class="key"><span class="sw" style:background={colors.kv}></span>{t({ en: 'key or value read from the cache', zh: '从缓存读出的键或值' })}</span>{/if}
			</div>
			<p class="sum">
				{#if mode === 'plain'}{t({ en: `Pass ${m.k} reads ${m.n} tokens; row ${m.n - 1}, the newest token, looks back at all of them (attention of layer ${layer}, averaged over the heads).`, zh: `第 ${m.k} 遍读入 ${m.n} 个词元；第 ${m.n - 1} 行是最新的词元，它回头查看所有这些词元（第 ${layer} 层的注意力，各头平均）。` })}{:else}<strong>{t(mode === 'full' ? L.full : L.cache)}</strong>
				{t({ en: `· pass ${m.k} reads ${m.n} tokens · ${fmtInt(cells.fresh)} new cells · ${fmtInt(mode === 'full' ? cells.redo : cells.skip)} ${mode === 'full' ? 'computed again' : 'not computed'} · ${fmtInt(mode === 'full' ? m.macsFull : m.macsCache)} multiply-adds`, zh: `· 第 ${m.k} 遍读入 ${m.n} 个词元 · ${fmtInt(cells.fresh)} 个新单元 · ${fmtInt(mode === 'full' ? cells.redo : cells.skip)} 个${mode === 'full' ? '重新计算' : '不用计算'} · ${fmtInt(mode === 'full' ? m.macsFull : m.macsCache)} 次乘加` })}{/if}
			</p>
		{/if}
	</div>
{/if}

<style>
	.grid-view {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.ctl {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 14px;
		align-items: center;
	}
	.seg {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.btn {
		min-height: 32px;
		padding: 2px 10px;
		font-size: var(--fs-xs);
	}
	.btn[aria-pressed='true'] {
		border-color: var(--accent);
		background: var(--accent-soft);
		font-weight: 600;
	}
	.pass {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-xs);
		flex: 1 1 14rem;
		min-width: 0;
	}
	.pass input {
		flex: 1;
		min-width: 6rem;
	}
	.text {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 2px;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
	}
	.tk {
		padding: 1px 3px;
		border: 1px solid var(--rule);
		border-radius: 3px;
		white-space: pre;
	}
	.tk.gen {
		border-color: var(--accent);
		border-style: dashed;
	}
	.tk.cur {
		outline: 2px solid var(--accent);
		font-weight: 600;
	}
	.tk.next {
		border-style: dotted;
		color: var(--ink-2);
	}
	.arrow {
		color: var(--ink-2);
		padding: 0 2px;
	}
	figure {
		margin: 0;
		min-width: 0;
		max-width: 640px;
	}
	svg {
		width: 100%;
		height: auto;
		display: block;
	}
	.later {
		fill: none;
		stroke: var(--rule);
		stroke-width: 0.5;
		stroke-dasharray: 1 1;
	}
	.tick {
		font-size: 10px;
		fill: var(--ink-2);
		font-family: var(--font-mono);
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.key {
		display: inline-flex;
		align-items: center;
		gap: 4px;
	}
	.sw {
		width: 12px;
		height: 12px;
		border-radius: 2px;
		display: inline-block;
		flex: none;
	}
	.sw.gray {
		background: var(--surface-2);
		border: 1px solid var(--rule-strong);
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		margin-top: 4px;
	}
	.sum {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.tbl {
		overflow: auto;
		max-height: 320px;
		min-width: 0;
	}
	table {
		border-collapse: collapse;
		font-size: var(--fs-sm);
		width: 100%;
	}
	th,
	td {
		padding: 2px 8px;
		text-align: right;
		border-bottom: 1px solid var(--rule);
		white-space: nowrap;
	}
	thead th {
		color: var(--ink-2);
		font-weight: 500;
	}
	.mono {
		font-family: var(--font-mono);
		text-align: left;
		white-space: pre;
	}
</style>

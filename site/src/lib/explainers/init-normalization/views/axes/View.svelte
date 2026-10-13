<!--
  'init-normalization/axes': BatchNorm and LayerNorm on the same tensor (meta.ts has the props). Toy modes: two small
  tables, the tensor with the statistics of the axis being normalized (column statistics for BatchNorm, row statistics for
  LayerNorm, that column or row outlined) and the result, then the chosen cell worked out. Mode 'gpt': TinyGPT's
  tokens × 76 dimensions as a heatmap with the followed token's row and the chosen dimension's column outlined, and both
  normalizations of their shared cell.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { PROBE_TOKENS } from '../../data.ts';
	import { axesOf, cellOf } from './meta.ts';

	let { at, props }: StageProps = $props();
	const pic = $derived(axesOf({ at, props }));
	const cell = $derived(pic ? cellOf(pic) : null);
	const theme = $derived(themeState.resolved);
	const minus = (s: string) => s.replace(/-/g, '−');
	const f = (v: number) => minus(fmt3(v));
	const pf = (v: number) => (v < 0 ? `(${f(v)})` : f(v));
	const scaleOf = (a: number[]) => makeScale('diverging', [-Math.max(...a.map(Math.abs), 1e-12), Math.max(...a.map(Math.abs), 1e-12)], theme);
	const sx = $derived(pic ? scaleOf(pic.x) : null);
	const sy = $derived(pic ? scaleOf([...(pic.mode === 'layer' ? pic.ln : pic.bn), ...(pic.mode === 'one' ? pic.ln : [])]) : null);
	const idx = (n: number) => Array.from({ length: n }, (_, i) => i);
	const tok = (s: string) => s.replace(/^ /, '␣');

	const L = {
		x: { en: 'Layer 1’s weighted sums', zh: '第 1 层的加权和' },
		bn: { en: 'After BatchNorm (each column)', zh: 'BatchNorm 之后（按列）' },
		ln: { en: 'After LayerNorm (each row)', zh: 'LayerNorm 之后（按行）' },
		bn1: { en: 'BatchNorm on a batch of one', zh: '批次只有一个样例时的 BatchNorm' },
		ln1: { en: 'LayerNorm of the same example', zh: '同一个样例的 LayerNorm' },
		one: { en: 'One example alone', zh: '单独一个样例' },
		corner: { en: 'example ↓ unit →', zh: '样例 ↓ 单元 →' },
		mean: { en: 'mean', zh: '均值' },
		spread: { en: 'spread', zh: '标准差' },
		gpt: { en: 'TinyGPT, block 1’s output: 9 tokens (rows, in order) × 76 dimensions', zh: 'TinyGPT 第 1 块的输出：9 个词元（按顺序成行）× 76 个维度' },
		tokens: { en: 'tokens', zh: '词元' },
		token: { en: 'token', zh: '词元' },
		dim: { en: 'dimension', zh: '维度' },
		lnRow: { en: 'LayerNorm uses the token’s row', zh: 'LayerNorm 用这个词元的一行' },
		bnCol: { en: 'BatchNorm would use the dimension’s column', zh: 'BatchNorm 会用这个维度的一列' }
	};
</script>

{#snippet table(vals: number[], rows: number, cols: number, title: { en: string; zh: string }, stats: 'col' | 'row' | null, sc: ReturnType<typeof makeScale>)}
	<figure class="tbl">
		<figcaption>{t(title)}</figcaption>
		<div class="scroll" use:keyScroll>
			<table>
				<thead>
					<tr>
						<th scope="col" class="corner">{t(L.corner)}</th>
						{#each idx(cols) as j (j)}<th scope="col" class:hl={stats === 'col' && j === pic!.c}>{j}</th>{/each}
						{#if stats === 'row'}<th scope="col">{t(L.mean)}</th><th scope="col">{t(L.spread)}</th>{/if}
					</tr>
				</thead>
				<tbody>
					{#each idx(rows) as i (i)}
						<tr class:hlrow={stats === 'row' && i === pic!.r}>
							<th scope="row">{pic!.mode === 'one' ? pic!.r : i}</th>
							{#each idx(cols) as j (j)}
								{@const v = vals[i * cols + j]}
								<td class:hl={stats === 'col' && j === pic!.c} class:pick={i === pic!.r && j === pic!.c} style:background={sc.css(v)} style:color={sc.ink(v)}>{f(v)}</td>
							{/each}
							{#if stats === 'row'}<td class="st">{f(pic!.rowStats[i].mean)}</td><td class="st">{f(pic!.rowStats[i].std)}</td>{/if}
						</tr>
					{/each}
					{#if stats === 'col'}
						<tr class="strow">
							<th scope="row">{t(L.mean)}</th>
							{#each idx(cols) as j (j)}<td class="st" class:hl={j === pic!.c}>{f(pic!.colStats[j].mean)}</td>{/each}
						</tr>
						<tr class="strow">
							<th scope="row">{t(L.spread)}</th>
							{#each idx(cols) as j (j)}<td class="st" class:hl={j === pic!.c}>{f(pic!.colStats[j].std)}</td>{/each}
						</tr>
					{/if}
				</tbody>
			</table>
		</div>
	</figure>
{/snippet}

{#if pic && cell && sx && sy}
	{#if pic.mode === 'gpt'}
		<div class="gpt">
			<p class="cap">{t(L.gpt)}</p>
			<div class="heat">
				<div class="grid">
					<TensorGrid
						data={pic.x}
						rows={pic.rows}
						cols={pic.cols}
						kind="diverging"
						label={t(L.gpt)}
						minCell={4}
						maxCell={14}
						labels={false}
						readout={false}
						highlights={[
							{ row: pic.r, col: 0, h: 1, w: pic.cols },
							{ row: 0, col: pic.c, h: pic.rows, w: 1, style: 'dashed' }
						]}
					/>
				</div>
			</div>
			<div class="cards">
				<div class="card">
					<p class="k">{t(L.lnRow)}: “{tok(PROBE_TOKENS[pic.r] ?? '')}”</p>
					<p class="v">{t(L.mean)} {f(cell.row.mean)}, {t(L.spread)} {f(cell.row.std)} → ({f(cell.x)} − {pf(cell.row.mean)}) / {f(cell.row.std)} ≈ <b>{f(cell.ln)}</b></p>
				</div>
				<div class="card">
					<p class="k">{t(L.bnCol)}: {t(L.dim)} {pic.c}</p>
					<p class="v">{t(L.mean)} {f(cell.col.mean)}, {t(L.spread)} {f(cell.col.std)} → ({f(cell.x)} − {pf(cell.col.mean)}) / {f(cell.col.std)} ≈ <b>{f(cell.bn)}</b></p>
				</div>
			</div>
		</div>
	{:else}
		<div class="axes">
			<div class="pair">
				{#if pic.mode === 'one'}
					{@render table(pic.x, 1, pic.cols, L.one, null, sx)}
					{@render table(pic.bn, 1, pic.cols, L.bn1, 'col', sy)}
					{@render table(pic.ln, 1, pic.cols, L.ln1, 'row', sy)}
				{:else if pic.mode === 'layer'}
					{@render table(pic.x, pic.rows, pic.cols, L.x, 'row', sx)}
					{@render table(pic.ln, pic.rows, pic.cols, L.ln, null, sy)}
				{:else}
					{@render table(pic.x, pic.rows, pic.cols, L.x, 'col', sx)}
					{@render table(pic.bn, pic.rows, pic.cols, L.bn, null, sy)}
				{/if}
			</div>
			<p class="work num">
				{#if pic.mode === 'layer'}
					LayerNorm: ({f(cell.x)} − {pf(cell.row.mean)}) / √({f(cell.row.std)}² + ε) = <b>{f(cell.ln)}</b>
				{:else if pic.mode === 'one'}
					BatchNorm: ({f(cell.x)} − {pf(cell.col.mean)}) / √(0² + ε) = <b>{f(cell.bn)}</b>; LayerNorm: <b>{f(cell.ln)}</b>
				{:else}
					BatchNorm: ({f(cell.x)} − {pf(cell.col.mean)}) / √({f(cell.col.std)}² + ε) = <b>{f(cell.bn)}</b>
				{/if}
			</p>
		</div>
	{/if}
{/if}

<style>
	.axes,
	.gpt {
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.pair {
		display: grid;
		gap: 12px;
		min-width: 0;
	}
	figure {
		margin: 0;
		min-width: 0;
	}
	figcaption,
	.cap {
		margin: 0 0 4px;
		font-size: var(--fs-sm);
		font-weight: 600;
		color: var(--ink);
	}
	.scroll {
		overflow-x: auto;
		max-width: 100%;
	}
	table {
		border-collapse: collapse;
		font-size: var(--fs-xs);
		font-variant-numeric: tabular-nums;
	}
	th.corner {
		font-size: 10px;
	}
	th {
		font-weight: 500;
		color: var(--ink-2);
		padding: 2px 5px;
		white-space: nowrap;
		text-align: right;
	}
	td {
		padding: 3px 6px;
		text-align: right;
		border: 1px solid var(--surface);
		white-space: nowrap;
	}
	td.st {
		color: var(--ink);
		background: var(--surface-2);
	}
	th.hl,
	td.st.hl {
		color: var(--ink);
		font-weight: 700;
	}
	td.hl {
		border-left: 2px solid var(--highlight);
		border-right: 2px solid var(--highlight);
	}
	tr.hlrow td,
	tr.hlrow th {
		border-top: 2px solid var(--highlight);
		border-bottom: 2px solid var(--highlight);
	}
	td.pick {
		outline: 2px dashed var(--highlight);
		outline-offset: -4px;
		font-weight: 700;
	}
	.work {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink);
	}
	.heat {
		min-width: 0;
	}
	.grid {
		min-width: 0;
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr));
		gap: 8px 16px;
		margin: 0;
	}
	.card {
		border: 1px solid var(--rule);
		border-radius: 6px;
		padding: 6px 8px;
	}
	.k {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.v {
		margin: 2px 0 0;
		font-size: var(--fs-sm);
		color: var(--ink);
		font-variant-numeric: tabular-nums;
	}
</style>

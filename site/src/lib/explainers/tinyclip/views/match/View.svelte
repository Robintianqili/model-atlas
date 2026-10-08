<!-- tinyclip 'match': drawings and captions side by side (props.mode, see data.ts). Thumbnails and vector strips use the
     kit TensorGrid; the batch table is a plain grid so its rows can carry the drawings and its columns the captions. -->
<script module lang="ts">
	/** rows or columns (the softmax step): one choice for both tables of a compare step (before and after training) */
	const tog = $state<{ phase: string; show: 'P' | 'Q' | null }>({ phase: '', show: null });
</script>

<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtFull, fmtPct } from '../../../../atlas/core/format.ts';
	import { PRESETS } from '../../scenes.ts';
	import { rawOf } from '../../adapters/shapes-captions/adapter.ts';
	import { captionsData, matrixData, pairsData, pct, vectorsData, wordsData, type Show } from './data.ts';
	import { SHOW } from './meta.ts';

	let P: StageProps = $props();
	const mode = $derived(String(P.props.mode ?? 'matrix'));
	const phase = $derived(P.at.frame?.phase ?? '');
	const show = $derived<Show>((tog.phase === phase && tog.show) || (P.props.show as Show) || 'L');
	const choose = (s: 'P' | 'Q') => ((tog.phase = phase), (tog.show = s));
	const mx = $derived(mode === 'matrix' ? matrixData(P, show) : null);
	const shares = $derived(show === 'P' || show === 'Q');
	/** one colour scale for every batch table of this step (before and after training side by side) */
	const scale = $derived.by(() => {
		const all = ['nce0', 'nce'].flatMap((s) => {
			const tp = P.rt.sources[s]?.tape;
			const i = tp && P.at.frame ? tp.find({ phase: P.at.frame.phase }) : -1;
			const M = tp && i >= 0 ? (tp.state(i) as Record<string, { data: ArrayLike<number> } | undefined>)[show] : undefined;
			return M ? Array.from(M.data) : [];
		});
		const m = Math.max(1e-9, ...all.map(Math.abs), ...(mx?.M ?? []).map(Math.abs));
		return shares ? makeScale('sequential', [0, 1], themeState.resolved) : makeScale('diverging', [-m, m], themeState.resolved);
	});
	/** shares with one decimal, as the narration prints them; gradients with 2 significant digits */
	const cell = (v: number) => (!shares ? fmt3(v, show === 'G' && Math.abs(v) < 1e-3 ? 1 : 2) : v >= 0.9995 ? '>99.9%' : fmtPct(v));
	const vec = $derived(mode === 'vectors' || mode === 'pair' ? vectorsData(P.rt) : null);
	const wd = $derived(mode === 'words' ? wordsData(P.rt) : null);
	const rows = $derived(mode === 'captions' ? captionsData(P.rt, P.props.texts as string[] | undefined) : []);
	const img = $derived(P.rt.pre?.tensors.image?.data ?? new Float32Array(784));
	const dom = (...xs: number[][]) => ((m: number): [number, number] => [-m, m])(Math.max(1e-9, ...xs.flat().map(Math.abs)));
	$effect(() => void (P.autoplay && P.onend()));
	const L = (en: string, zh: string) => t({ en, zh });
	const shareText = (v: number) => ((x) => (typeof x === 'number' ? '…' : t(x)))(pct(v));
</script>

{#snippet thumb(data: ArrayLike<number>, px: number, label: string)}
	<TensorGrid {data} rows={28} cols={28} kind="sequential" domain={[0, 1]} cell={px} interactive={false} gridLines={false} labels={false} label={{ en: label, zh: label }} />
{/snippet}
{#snippet strip(data: number[], label: string, domain: [number, number])}
	<div class="strip"><span class="sl">{label}</span><TensorGrid {data} rows={1} cols={data.length} kind="diverging" {domain} maxCell={16} minCell={4} labels={false} readout={false} label={{ en: label, zh: label }} /></div>
{/snippet}

<div class="mt">
	{#if mode === 'pairs'}
		<ul class="pairs">
			{#each pairsData(P) as x, i (i)}<li>{@render thumb(x.image, 2, x.caption)}<span>{x.caption}</span></li>{/each}
		</ul>
	{:else if mode === 'words' && wd}
		{@const d = dom(...wd.words.map((w) => w.row))}
		<p class="cap">“{wd.caption}”</p>
		{#each wd.words as w, i (i)}{@render strip(w.row, w.w, d)}{/each}
		<div class="sep">{@render strip(wd.avg, L('average of the word rows', '词行的平均'), d)}</div>
		{@render strip(wd.vec, L('→ linear layer → caption vector', '→ 线性层 → 描述向量'), dom(wd.vec))}
		<p class="muted">{L(`Each strip is ${wd.width} numbers: orange positive, blue negative. Swap two words and the average stays the same.`, `每一条是 ${wd.width} 个数：橙色为正，蓝色为负。对调两个词，平均值不变。`)}</p>
	{:else if vec && mode === 'vectors'}
		{@const da = dom(vec.a, vec.an)}
		{@const db = dom(vec.b, vec.bn)}
		<div class="row">{@render thumb(img, 2, L('your drawing', '你的画'))}<p class="cap">“{vec.caption}”</p></div>
		{@render strip(vec.a, L(`image vector, length ${fmt3(vec.lenA)}`, `图像向量，长度 ${fmt3(vec.lenA)}`), da)}
		{@render strip(vec.an, L(`÷ ${fmt3(vec.lenA)} → length 1`, `÷ ${fmt3(vec.lenA)} → 长度 1`), da)}
		{@render strip(vec.b, L(`caption vector, length ${fmt3(vec.lenB)}`, `描述向量，长度 ${fmt3(vec.lenB)}`), db)}
		{@render strip(vec.bn, L(`÷ ${fmt3(vec.lenB)} → length 1`, `÷ ${fmt3(vec.lenB)} → 长度 1`), db)}
		<p class="muted">{L('Same colours on each pair of strips: the pattern stays, only the size of the numbers changes.', '每两条用同一个色标：图案不变，只是数的大小变了。')}</p>
	{:else if vec && mode === 'pair'}
		{@const d = dom(vec.an, vec.bn)}
		{@render strip(vec.an, L('image vector a (length 1)', '图像向量 a（长度 1）'), d)}
		{@render strip(vec.bn, L('caption vector b (length 1)', '描述向量 b（长度 1）'), d)}
		{@render strip(vec.prods, L('products a × b, number by number', '逐个数的乘积 a × b'), dom(vec.prods))}
		<p class="num sum">Σ a·b = cos = <strong>{fmt3(vec.cos)}</strong> → × s {fmt3(vec.scale)} = <strong>{fmt3(vec.score)}</strong></p>
	{:else if mx}
		{#if shares}
			<div class="tog" role="group" aria-label={L('Softmax direction', 'softmax 的方向')}>
				<button type="button" class="chip" aria-pressed={show === 'P'} onclick={() => choose('P')}>{L('along rows', '沿行')}</button>
				<button type="button" class="chip" aria-pressed={show === 'Q'} onclick={() => choose('Q')}>{L('along columns', '沿列')}</button>
			</div>
		{/if}
		<p class="muted">{t(SHOW[show])}</p>
		{#if mx.M}
			{@const N = mx.N}
			<!-- ARIA table: cells sit in role="row" boxes (a subgrid each), so the grid layout stays the same -->
			<div class="mx" style:--n={N} role="table" aria-label={t(SHOW[show])}>
				<div class="r" role="row"><span role="cell"></span>{#each Array.from({ length: N }, (_, j) => j) as j (j)}<span class="hd num" role="columnheader">{j}</span>{/each}</div>
				{#each Array.from({ length: N }, (_, i) => i) as i (i)}
					<div class="r" role="row">
						<span class="rh" role="rowheader">{@render thumb(mx.images[i], 1, mx.captions[i])}</span>
						{#each Array.from({ length: N }, (_, j) => j) as j (j)}
							{@const v = mx.M[i * N + j]}
							<span class="c num" class:dg={i === j} role="cell" style:background={scale.css(v)} style:color={scale.ink(v)} title={fmtFull(v)}>{cell(v)}</span>
						{/each}
					</div>
				{/each}
			</div>
			<ol class="legend" start="0">{#each mx.captions.slice(0, N) as c, j (j)}<li>{j === 0 ? L('yours: ', '你的：') : ''}“{c}”</li>{/each}</ol>
		{/if}
	{:else if mode === 'captions'}
		<div class="row">
			{@render thumb(img, 3, L('your drawing', '你的画'))}
			<ul class="caps">
				{#each rows as r, k (k)}
					<li><span class="ct">“{r.text}”</span><span class="bar"><i style:width="{Math.max(0, r.share) * 100}%"></i></span><span class="num">{fmt3(r.score)} · {shareText(r.share)}</span></li>
				{/each}
			</ul>
		</div>
		<p class="muted">{L('Score s · cos, then the share after a softmax over these captions.', '分数 s · cos，然后是在这几句描述之间做 softmax 后的份额。')}</p>
		{#if P.props.texts && (P.rt.input as { preset?: string } | null)?.preset !== PRESETS[5].id}
			<button type="button" class="btn small" onclick={() => P.rt.setInput(rawOf(PRESETS[5]))}>{L('Try it on the drawing “circle above square”', '在“圆在正方形上面”这幅画上试试')}</button>
		{/if}
	{/if}
</div>

<style>
	.mt { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.pairs { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(108px, 1fr)); gap: 10px; }
	.pairs li { display: grid; gap: 4px; justify-items: start; font-size: var(--fs-xs); }
	.strip { display: grid; gap: 2px; min-width: 0; }
	.sl { font-size: var(--fs-xs); }
	.sep { border-top: 1px solid var(--line); padding-top: 6px; }
	.cap { margin: 0; font-style: italic; }
	.row { display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-start; }
	.sum, .muted { margin: 0; }
	.tog { display: flex; gap: 6px; }
	.chip[aria-pressed='true'] { background: var(--accent-soft); border-color: var(--accent); }
	.mx { display: grid; grid-template-columns: 30px repeat(var(--n), minmax(0, 46px)); gap: 2px; align-items: center; justify-content: start; }
	.r { display: grid; grid-template-columns: subgrid; grid-column: 1 / -1; align-items: center; }
	.hd { text-align: center; font-size: var(--fs-xs); }
	.rh { display: flex; justify-content: center; }
	.c { display: grid; place-items: center; height: 30px; font-size: 10px; border-radius: 3px; overflow: hidden; }
	.c.dg { outline: 2px solid var(--ink); outline-offset: -2px; }
	.legend { margin: 0; padding-left: 2.2em; font-size: var(--fs-xs); }
	.caps { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; min-width: 0; flex: 1 1 200px; }
	.caps li { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 2px 8px; }
	.ct { grid-column: 1 / -1; overflow-wrap: anywhere; }
	.bar { height: 10px; background: var(--line); border-radius: 5px; overflow: hidden; align-self: center; }
	.bar i { display: block; height: 100%; background: var(--accent); }
	.small { min-height: 32px; padding: 0 12px; justify-self: start; }
</style>

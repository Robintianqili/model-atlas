<!-- two-tower 'movies': the catalog, your choices, the scores, the shortlist, the ranker, the cost and the twins
     (props.mode; meta.ts lists them). A score bar grows right from 0 (orange) or left (blue). -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
	import { GENRES, ITEM_F_NAMES, MOVIES, TWINS, genreColor, toPrefs, type Prefs } from '../../world.ts';
	import { K, T, costs, likesOf, rankOne, ranked, reranked, rowOf, towerOf } from '../../recs.ts';
	import Catalog from '../../parts/Catalog.svelte';
	import Picker from '../../parts/Picker.svelte';
	import Cells from '../../parts/Cells.svelte';

	let P: StageProps = $props();
	const mode = $derived(String(P.props.mode ?? 'catalog'));
	const picks = $derived(toPrefs(P.rt.input).picks);
	const s = $derived(T(P.rt, 'scores')?.data);
	const rows = $derived.by(() => {
		if (!s) return [];
		const all = mode === 'scores' ? MOVIES.map((_, i) => ({ i, s: s[i] })).sort((a, b) => b.s - a.s || a.i - b.i) : ranked(P.rt);
		return all.map((x, k) => ({ ...x, k }));
	});
	const shown = $derived(mode === 'scores' ? [...rows.slice(0, 12), ...rows.slice(-3)] : rows.slice(0, K + 3));
	const max = $derived(Math.max(1e-9, ...rows.map((x) => Math.abs(x.s))));
	const rr = $derived(mode === 'rank' ? reranked(P.rt) : []);
	const c = $derived(costs(P.rt)), big = $derived(costs(P.rt, 1e6));
	const ex = $derived(towerOf(P.rt, 'item', TWINS[0]));
	const setInput = (r: Prefs) => (P.setInput ? P.setInput(r) : P.rt.setInput(r));
	const ttl = (i: number) => t(MOVIES[i].title);
	const n = (v: number) => v.toLocaleString('en-US');
</script>

{#snippet dot(i: number)}<i class="d" style:background={genreColor(MOVIES[i].genre)} title={t(GENRES[MOVIES[i].genre])}></i>{/snippet}

<div class="mv">
	{#if mode === 'catalog'}
		<Catalog mark={[TWINS[0]]} />
		{#if ex}<p class="hd">{t({ en: `What the model reads of ${MOVIES[TWINS[0]].title.en}:`, zh: `模型从《${MOVIES[TWINS[0]].title.zh}》读到的：` })}</p>
			<div class="feats">{#each Array.from(ex.x) as v, k (k)}<span class="f" class:on={v !== 0}><span>{t(ITEM_F_NAMES[k])}</span><b class="num">{fmt3(v)}</b></span>{/each}</div>{/if}
	{:else if mode === 'pick'}
		<Picker raw={P.rt.input} onchange={setInput} />
	{:else if mode === 'scores' || mode === 'shortlist'}
		<p class="hd">{mode === 'scores' ? t({ en: 'Your score for every movie, best first (one dot product each)', zh: '你对每部电影的分数，从高到低（每部一个点积）' }) : t({ en: `Retrieval: the best ${K} of the ${rows.length} movies you did not pick`, zh: `召回：你没选过的 ${rows.length} 部电影里最好的 ${K} 部` })}</p>
		<ol class="ls answer">
			{#each shown as x, j (x.i)}
				{#if j > 0 && x.k !== shown[j - 1].k + 1}<li class="gap">… {t({ en: `${x.k - shown[j - 1].k - 1} more`, zh: `还有 ${x.k - shown[j - 1].k - 1} 部` })}</li>{/if}
				{#if mode === 'shortlist' && x.k === K}<li class="cut">{t({ en: `— the shortlist ends here (${K}) —`, zh: `—— 候选到此为止（${K} 部）——` })}</li>{/if}
				<li class="r" class:in={mode === 'shortlist' && x.k < K} class:pk={picks.includes(x.i)}>
					<span class="k num">{x.k + 1}</span>{@render dot(x.i)}<span class="ti">{ttl(x.i)}{#if picks.includes(x.i)}{' · '}{t({ en: 'picked', zh: '已选' })}{/if}</span>
					<span class="bar"><i style:left="{x.s < 0 ? 50 - (50 * -x.s) / max : 50}%" style:width="{(50 * Math.abs(x.s)) / max}%" class:neg={x.s < 0}></i></span><b class="num">{fmt3(x.s)}</b>
				</li>
			{/each}
		</ol>
	{:else if mode === 'rank'}
		<p class="hd">{t({ en: 'The ranker reorders the shortlist: chance you like it, from the score and the likes in the logs', zh: '排序器重排候选：由分数和日志里的点赞数估计你喜欢它的概率' })}</p>
		<div class="sc" use:keyScroll><table class="tb">
			<thead><tr><th>#</th><th>{t({ en: 'movie', zh: '电影' })}</th><th>{t({ en: 'score', zh: '分数' })}</th><th>{t({ en: 'likes', zh: '点赞' })}</th><th>{t({ en: 'chance', zh: '概率' })}</th><th>{t({ en: 'was', zh: '原来' })}</th></tr></thead>
			<tbody>{#each rr as x, j (x.i)}<tr><td class="num">{j + 1}</td><td>{@render dot(x.i)}{ttl(x.i)}</td><td class="num">{fmt3(x.s)}</td><td class="num">{n(likesOf(P.rt, x.i))}</td><td class="num">{fmtPct(x.p)}</td><td class="num" class:up={j < x.k} class:down={j > x.k}>{x.k + 1}</td></tr>{/each}</tbody>
		</table></div>
	{:else if mode === 'cost'}
		<div class="sc" use:keyScroll><table class="tb">
			<thead><tr><th></th><th>{t({ en: `${c.N} movies`, zh: `${c.N} 部电影` })}</th><th>{t({ en: 'a million movies', zh: '一百万部电影' })}</th></tr></thead>
			<tbody>
				<tr><td>{t({ en: 'Two towers, per request: your tower once + one dot product per movie', zh: '双塔，每次请求：你的塔一次 + 每部电影一个点积' })}</td><td class="num">{n(c.perRequest)}</td><td class="num">{n(big.perRequest)}</td></tr>
				<tr><td>{t({ en: 'Item tower for every movie: once, in advance', zh: '物品塔算遍每部电影：提前做一次' })}</td><td class="num">{n(c.offline)}</td><td class="num">{n(big.offline)}</td></tr>
				<tr><td>{t({ en: 'A network reading you and each movie together, per request', zh: '把你和每部电影放在一起读的网络，每次请求' })}</td><td class="num">{n(c.joint)}</td><td class="num">{n(big.joint)}</td></tr>
			</tbody>
		</table></div>
		<p class="small">{t({ en: 'Counts of multiplications; the additions are about as many.', zh: '表里是乘法次数；加法次数也差不多一样多。' })}</p>
	{:else if mode === 'twins'}
		<div class="tw">
			{#each TWINS as i (i)}
				{@const x = s ? rankOne(P.rt, { i, s: s[i] }) : null}
				<div class="card"><p class="hd">{@render dot(i)}{ttl(i)}</p><p class="num small">{t(GENRES[MOVIES[i].genre])} · {MOVIES[i].year} · {MOVIES[i].length} min</p>
					<Cells data={rowOf(P.rt, i)} label={t({ en: 'its vector', zh: '它的向量' })} />
					{#if x}<p class="num small">{t({ en: `score for you ${fmt3(x.s)} · likes in the logs ${n(likesOf(P.rt, i))} · ranker ${fmtPct(x.p)}`, zh: `对你的分数 ${fmt3(x.s)} · 日志里的点赞 ${n(likesOf(P.rt, i))} · 排序器 ${fmtPct(x.p)}` })}</p>{/if}</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.mv { display: grid; gap: 6px; min-width: 0; font-size: var(--fs-sm); }
	.hd { margin: 2px 0; font-weight: 600; } .small { margin: 0; font-size: var(--fs-xs); }
	.d { display: inline-block; flex: none; width: 9px; height: 9px; margin-right: 5px; border-radius: 50%; }
	.feats { display: flex; flex-wrap: wrap; gap: 3px; } .f { display: inline-grid; padding: 1px 6px; border: 1px solid var(--rule); border-radius: 4px; font-size: 11px; } .f.on { border-color: var(--accent); background: var(--accent-soft); }
	.ls { display: grid; gap: 1px; margin: 0; padding: 0; list-style: none; }
	.r { display: grid; grid-template-columns: 22px auto minmax(0, 1fr) minmax(60px, 30%) 52px; align-items: center; gap: 4px; padding: 1px 4px; border-radius: 3px; }
	/* a movie you picked: muted text in a dashed outline, never opacity (text keeps 4.5:1, KIT.md rule 16) */
	.r.in { background: var(--accent-soft); } .r.pk { color: var(--muted); outline: 1px dashed var(--muted); outline-offset: -1px; }
	.k { color: var(--muted); font-size: 11px; } .ti { overflow: hidden; font-size: var(--fs-xs); text-overflow: ellipsis; white-space: nowrap; }
	.bar { position: relative; height: 10px; border-left: 0; background: linear-gradient(to right, transparent calc(50% - 0.5px), var(--rule) calc(50% - 0.5px), var(--rule) calc(50% + 0.5px), transparent calc(50% + 0.5px)); }
	.bar i { position: absolute; top: 1px; bottom: 1px; background: var(--div-pos); } .bar i.neg { background: var(--div-neg); }
	b.num { font-size: 11px; text-align: right; }
	.gap, .cut { color: var(--muted); font-size: 11px; text-align: center; } .cut { color: var(--accent-strong); font-weight: 600; }
	.sc { overflow-x: auto; max-width: 100%; }
	.tb { width: 100%; border-collapse: collapse; font-size: var(--fs-xs); } .tb th, .tb td { padding: 2px 4px; border-bottom: 1px solid var(--rule); text-align: left; } .tb td.num { text-align: right; }
	.up { color: var(--div-pos-text); font-weight: 600; } .down { color: var(--muted); }
	.tw { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr)); gap: 8px; } .card { display: grid; gap: 4px; padding: 6px 8px; border: 1px solid var(--rule); border-radius: 6px; } .card p { margin: 0; }
	/* the scores answer step 6's predict question: hidden while the kit's predict gate makes the stage inert */
	:global([inert]) .answer { visibility: hidden; }
</style>

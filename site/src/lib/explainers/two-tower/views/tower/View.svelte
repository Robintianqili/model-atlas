<!-- two-tower 'tower': one tower at work, top to bottom: its input numbers with their names, the 16 numbers of the
     hidden layer after the ReLU, and the 8 numbers of the vector. props.tower 'user' (your choices) or 'item' (your new
     movie, or with props.movie a catalog movie); props.table adds the item table (every movie's vector, that movie's
     card outlined); props.edit adds the new-movie editor and the new movie's score and place. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { ITEM_F_NAMES, MOVIES, USER_F_NAMES, type Prefs } from '../../world.ts';
	import { newMovie, rowOf, sameAs, towerOf } from '../../recs.ts';
	import Cells from '../../parts/Cells.svelte';
	import Catalog from '../../parts/Catalog.svelte';
	import NewMovie from '../../parts/NewMovie.svelte';
	import { towerText } from './meta.ts';

	let P: StageProps = $props();
	const user = $derived(P.props.tower === 'user');
	const movie = $derived(typeof P.props.movie === 'number' ? P.props.movie : undefined);
	const L = $derived(towerOf(P.rt, user ? 'user' : 'item', movie));
	const names = $derived((user ? USER_F_NAMES : ITEM_F_NAMES).map((n) => t(n)));
	const txt = $derived(towerText(P));
	const nm = $derived(P.props.edit ? newMovie(P.rt) : null);
	const twin = $derived(P.props.edit ? sameAs(P.rt) : []);
	const setInput = (r: Prefs) => (P.setInput ? P.setInput(r) : P.rt.setInput(r));
</script>

<div class="tw">
	{#if P.props.edit}<NewMovie raw={P.rt.input} onchange={setInput} />{/if}
	{#if L}
		<p class="hd">{t(txt.head)}</p>
		<div class="feats" role="list" aria-label={t({ en: 'Input numbers', zh: '输入的数' })}>
			{#each Array.from(L.x) as v, k (k)}<span role="listitem" class="f" class:on={v !== 0}><span>{names[k]}</span><b class="num">{fmt3(v)}</b></span>{/each}
		</div>
		<p class="ar">↓ {t(txt.layer1)}</p>
		<Cells data={L.h} values={false} label={t({ en: 'hidden layer', zh: '隐藏层' })} />
		<p class="ar">↓ {t(txt.layer2)}</p>
		<div class="out"><Cells data={L.y} label={t(txt.vec)} /><span class="vl">{t(txt.vec)}</span></div>
	{/if}
	{#if nm}<p class="num res">{t(txt.score)}</p>{#if twin.length}<p class="small">{t({ en: `Same features as ${twin.map((i) => MOVIES[i].title.en).join(' and ')}: the same vector, the same score.`, zh: `特征和${twin.map((i) => `《${MOVIES[i].title.zh}》`).join('、')}完全相同：向量相同，分数也相同。` })}</p>{/if}{/if}
	{#if P.props.table}
		<p class="hd">{t({ en: 'The item table: the vector of every movie, computed once in advance', zh: '物品表：每部电影的向量，提前算好一次' })}</p>
		<Catalog mark={movie === undefined ? [] : [movie]} vec={(i) => rowOf(P.rt, i)} />
	{/if}
</div>

<style>
	.tw { display: grid; gap: 6px; min-width: 0; font-size: var(--fs-sm); }
	.hd { margin: 4px 0 0; font-weight: 600; } .ar { margin: 0; color: var(--ink-2); font-size: var(--fs-xs); } .small, .res { margin: 0; font-size: var(--fs-xs); }
	.feats { display: flex; flex-wrap: wrap; gap: 3px; }
	.f { display: inline-grid; padding: 1px 6px; border: 1px solid var(--rule); border-radius: 4px; font-size: 11px; } .f.on { border-color: var(--accent); background: var(--accent-soft); }
	.out { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; } .vl { font-weight: 600; font-size: var(--fs-xs); }
	.res { font-weight: 600; }
</style>

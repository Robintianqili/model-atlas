<!-- two-tower: the learner's choices: genre switches, then the movie cards as toggles (at most 12 picks). onchange gets
     the new raw input. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { GENRES, MAX_PICKS, genreColor, toPrefs, type Prefs } from '../world.ts';
	import Catalog from './Catalog.svelte';

	let { raw, onchange }: { raw: unknown; onchange: (r: Prefs) => void } = $props();
	const p = $derived(toPrefs(raw));
	const flip = (a: number[], x: number) => (a.includes(x) ? a.filter((y) => y !== x) : [...a, x]);
	const pick = (i: number) => (p.picks.includes(i) || p.picks.length < MAX_PICKS) && onchange({ ...p, picks: flip(p.picks, i) });
</script>

<div class="pk">
	<div class="row" role="group" aria-label={t({ en: 'Genres you like', zh: '你喜欢的类型' })}>
		<span>{t({ en: 'I like:', zh: '我喜欢：' })}</span>
		{#each GENRES as g, k (k)}<button type="button" class="chip" aria-pressed={p.genres.includes(k)} onclick={() => onchange({ ...p, genres: flip(p.genres, k) })}><i style:background={genreColor(k)}></i>{t(g)}</button>{/each}
	</div>
	<p class="num">{t({ en: `Movies you liked: ${p.picks.length} picked, at most ${MAX_PICKS} (press a card to pick or unpick it).`, zh: `你喜欢的电影：已选 ${p.picks.length} 部，最多 ${MAX_PICKS} 部（点卡片选中或取消）。` })}</p>
	<Catalog picks={p.picks} onpick={pick} />
</div>

<style>
	.pk { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; align-items: center; gap: 4px; }
	.chip { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); } .chip i { display: inline-block; width: 9px; height: 9px; margin-right: 5px; border-radius: 50%; }
	.chip[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	p { margin: 0; }
</style>

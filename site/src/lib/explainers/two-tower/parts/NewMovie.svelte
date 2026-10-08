<!-- two-tower: the learner's new movie: a genre and two sliders (year, length). onchange gets the new raw input. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { GENRES, NEW_RANGE, genreColor, toPrefs, type Prefs } from '../world.ts';

	let { raw, onchange }: { raw: unknown; onchange: (r: Prefs) => void } = $props();
	const p = $derived(toPrefs(raw));
	const set = (k: 'genre' | 'year' | 'length', v: number) => onchange({ ...p, item: { ...p.item, [k]: v } });
</script>

<fieldset class="nm">
	<legend>{t({ en: 'Your new movie', zh: '你的新电影' })}</legend>
	<div class="row" role="group" aria-label={t({ en: 'Its genre', zh: '它的类型' })}>
		{#each GENRES as g, k (k)}<button type="button" class="chip" aria-pressed={p.item.genre === k} onclick={() => set('genre', k)}><i style:background={genreColor(k)}></i>{t(g)}</button>{/each}
	</div>
	<label>{t({ en: 'Year', zh: '年份' })} <b class="num">{p.item.year}</b><input type="range" min={NEW_RANGE.year[0]} max={NEW_RANGE.year[1]} value={p.item.year} onchange={(e) => set('year', Number(e.currentTarget.value))} /></label>
	<label>{t({ en: 'Length (min)', zh: '片长（分钟）' })} <b class="num">{p.item.length}</b><input type="range" min={NEW_RANGE.length[0]} max={NEW_RANGE.length[1]} value={p.item.length} onchange={(e) => set('length', Number(e.currentTarget.value))} /></label>
</fieldset>

<style>
	.nm { display: grid; gap: 6px; min-width: 0; margin: 0; padding: 6px 10px; border: 1px solid var(--rule); border-radius: 6px; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.chip { min-height: 30px; padding: 2px 8px; font-size: var(--fs-xs); } .chip i { display: inline-block; width: 9px; height: 9px; margin-right: 4px; border-radius: 50%; }
	.chip[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	label { display: grid; grid-template-columns: auto auto minmax(0, 1fr); align-items: center; gap: 8px; } input { width: 100%; min-width: 0; }
</style>

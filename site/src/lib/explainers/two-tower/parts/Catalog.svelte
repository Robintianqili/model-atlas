<!-- two-tower: the 36 movies in six genre columns. Each card: the title, the year and the length; with `vec` also the
     movie's 8 numbers from the item table; `mark` outlines movies; with `onpick` every card is a toggle button (your
     picks are pressed). -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { GENRES, MOVIES, genreColor } from '../world.ts';
	import Cells from './Cells.svelte';

	let { mark = [], picks = [], onpick, vec }: { mark?: number[]; picks?: number[]; onpick?: (i: number) => void; vec?: (i: number) => ArrayLike<number> | null } = $props();
	const cols = GENRES.map((g, k) => ({ g, k, ids: MOVIES.flatMap((m, i) => (m.genre === k ? [i] : [])) }));
	const max = $derived(vec ? Math.max(1e-9, ...MOVIES.flatMap((_, i) => Array.from(vec(i) ?? [], Math.abs))) : 1);
</script>

<div class="cat">
	{#each cols as c (c.k)}
		<div class="col" role="list" aria-label={t(c.g)}>
			<p class="gh"><i style:background={genreColor(c.k)}></i>{t(c.g)}</p>
			{#each c.ids as i (i)}
				{@const m = MOVIES[i]}
				{@const v = vec?.(i)}
				{#snippet body()}
					<span class="ti">{t(m.title)}</span><span class="meta">{m.year} · {m.length} min{#if !onpick && picks.includes(i)}{' · '}{t({ en: 'picked', zh: '已选' })}{/if}</span>
					{#if v}<Cells data={v} {max} values={false} label={t(m.title)} />{/if}
				{/snippet}
				<div role="listitem" class="li">
					{#if onpick}<button type="button" class="card" aria-pressed={picks.includes(i)} onclick={() => onpick(i)}>{@render body()}</button>
					{:else}<div class="card" class:mark={mark.includes(i)}>{@render body()}</div>{/if}
				</div>
			{/each}
		</div>
	{/each}
</div>

<style>
	.cat { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 128px), 1fr)); gap: 8px; min-width: 0; }
	.col { display: grid; gap: 3px; align-content: start; min-width: 0; }
	.li { display: grid; min-width: 0; }
	.gh { margin: 0; font-size: var(--fs-sm); font-weight: 600; } .gh i { display: inline-block; width: 10px; height: 10px; margin-right: 5px; border-radius: 50%; }
	.card { display: grid; gap: 1px; min-width: 0; padding: 3px 6px; border: 1px solid var(--rule); border-radius: 4px; background: var(--surface); color: var(--ink); text-align: left; font: inherit; }
	.card.mark { border-color: var(--ink); box-shadow: 0 0 0 1px var(--ink); }
	button.card[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	.ti { overflow: hidden; font-size: var(--fs-xs); text-overflow: ellipsis; white-space: nowrap; } .meta { color: var(--muted); font: 11px var(--font-mono); }
</style>

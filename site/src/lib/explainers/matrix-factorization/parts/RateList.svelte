<!-- matrix-factorization: the learner rates movies. One row per movie: its kind's color, its title and five buttons from
     −2 (hated it) to +2 (loved it); pressing the chosen button again clears the rating. onchange gets the new raw input. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { KINDS, MOVIES, SCALE, SCALE_NAMES, kindColor, toRaw, type RatingsRaw } from '../world.ts';

	let { raw, onchange }: { raw: unknown; onchange: (r: RatingsRaw) => void } = $props();
	const r = $derived(toRaw(raw).ratings);
	const set = (j: number, v: number) => onchange({ ratings: r.map((x, k) => (k === j ? (x === v ? null : v) : x)) });
	const sign = (v: number) => (v > 0 ? `+${v}` : v < 0 ? `−${-v}` : '0');
</script>

<ul class="rl" aria-label={t({ en: 'Rate the movies', zh: '给电影打分' })}>
	{#each MOVIES as m, j (j)}
		<li>
			<span class="title"><i style:background={kindColor(m.kind)} title={t(KINDS[m.kind])}></i>{t(m.title)}</span>
			<span class="btns" role="group" aria-label={t(m.title)}>
				{#each SCALE as v, k (v)}
					<button type="button" class:on={r[j] === v} aria-pressed={r[j] === v} title={t(SCALE_NAMES[k])} onclick={() => set(j, v)}>{sign(v)}</button>
				{/each}
			</span>
		</li>
	{/each}
</ul>

<style>
	.rl { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr)); gap: 2px 16px; margin: 0; padding: 0; list-style: none; font-size: var(--fs-sm); }
	li { display: flex; align-items: center; justify-content: space-between; gap: 8px; min-width: 0; padding: 2px 0; border-bottom: 1px solid var(--rule); }
	.title { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
	.title i { display: inline-block; width: 10px; height: 10px; margin-right: 6px; border-radius: 50%; }
	.btns { display: flex; gap: 2px; flex: none; }
	/* the site's own colors (the browser's dark-theme button gray is below 4.5:1 with our ink) */
	button { min-width: 32px; min-height: 30px; padding: 0 4px; font: var(--fs-xs) var(--font-mono); color: var(--ink); background: var(--surface); border: 1px solid var(--rule-strong); border-radius: var(--r1); }
	button:hover { background: var(--surface-2); }
	button.on { border-color: var(--accent); background: var(--accent); color: var(--accent-ink); }
</style>

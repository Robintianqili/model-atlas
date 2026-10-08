<!-- matrix-factorization Capture (local adapter 'ratings'): start profiles, how many movies you rated, and (not in the
     compact copy beside the lesson) the rating buttons for every movie. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { MOVIES, PRESETS, toRaw, type RatingsRaw } from '../../world.ts';
	import RateList from '../../parts/RateList.svelte';

	let { raw, onchange, compact = false }: { raw: unknown; onchange: (raw: RatingsRaw) => void; compact?: boolean } = $props();
	const n = $derived(toRaw(raw).ratings.filter((x) => x !== null).length);
	const same = (a: unknown, b: unknown) => JSON.stringify(toRaw(a)) === JSON.stringify(toRaw(b));
</script>

<div class="cap">
	<div class="row" role="group" aria-label={t({ en: 'Start profiles', zh: '起始口味' })}>
		{#each PRESETS as p (p.id)}<button type="button" class="btn" aria-pressed={same(p.raw, raw)} onclick={() => onchange(toRaw(p.raw))}>{t(p.label)}</button>{/each}
	</div>
	<p class="num">{t({ en: `You rated ${n} of ${MOVIES.length} movies.`, zh: `${MOVIES.length} 部电影你评了 ${n} 部。` })}{#if compact}{t({ en: ' Rate more in the stage of step 2, or below the lesson.', zh: '在第 2 步的舞台上或课程下方可以继续评分。' })}{/if}</p>
	{#if !compact}<RateList {raw} {onchange} />{/if}
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	p { margin: 0; }
</style>

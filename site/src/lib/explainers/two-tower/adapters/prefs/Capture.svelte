<!-- two-tower Capture (local adapter 'prefs'): start profiles and a summary; outside the compact copy beside the lesson
     also the genre switches, the movie cards to pick and the new-movie editor. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { GENRES, PRESETS, toPrefs, type Prefs } from '../../world.ts';
	import Picker from '../../parts/Picker.svelte';
	import NewMovie from '../../parts/NewMovie.svelte';

	let { raw, onchange, compact = false }: { raw: unknown; onchange: (raw: Prefs) => void; compact?: boolean } = $props();
	const p = $derived(toPrefs(raw));
	const same = (a: unknown) => JSON.stringify(toPrefs(a)) === JSON.stringify(p);
</script>

<div class="cap">
	<div class="row" role="group" aria-label={t({ en: 'Start profiles', zh: '起始口味' })}>
		{#each PRESETS as x (x.id)}<button type="button" class="btn" aria-pressed={same(x.raw)} onclick={() => onchange(toPrefs(x.raw))}>{t(x.label)}</button>{/each}
	</div>
	{#if compact}
		<p class="num">{t({ en: `You like ${p.genres.map((g) => GENRES[g].en).join(', ') || 'no genre'}; ${p.picks.length} movies picked. Change them in the stage of step 3, or below the lesson.`, zh: `你喜欢的类型：${p.genres.map((g) => GENRES[g].zh).join('、') || '还没选'}；选了 ${p.picks.length} 部电影。可以在第 3 步的舞台上或课程下方修改。` })}</p>
	{:else}
		<Picker {raw} {onchange} />
		<NewMovie {raw} {onchange} />
	{/if}
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); } .btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	p { margin: 0; }
</style>

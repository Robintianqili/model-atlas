<!-- two-tower 'batch': the inbatch run as one table. Row i is viewer i of the batch, column j is the movie viewer j liked,
     so the diagonal holds the true pairs and every other cell is an in-batch negative. Three readings (buttons):
     the scores, the row softmax (shares) and the gradient of the loss on each score. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
	import { GENRES, MOVIES, genreColor } from '../../world.ts';
	import { batchPairs } from '../../recs.ts';
	import { SHOW, fieldAt } from './meta.ts';

	let P: StageProps = $props();
	let pick = $state<keyof typeof SHOW | null>(null);
	const show = $derived(pick ?? (P.props.show as keyof typeof SHOW) ?? 'P');
	const pairs = batchPairs();
	const M = $derived(fieldAt(P, show));
	const B = $derived(M?.shape[0] ?? 0);
	const sc = $derived(M ? makeScale(show === 'P' ? 'sequential' : 'diverging', show === 'P' ? [0, 1] : [-1, 1].map((x) => x * Math.max(1e-9, ...Array.from(M.data, Math.abs))) as [number, number], themeState.resolved) : null);
	const f = (v: number) => (show === 'P' ? fmtPct(v) : fmt3(v));
</script>

<div class="bt">
	<div class="row" role="group" aria-label={t({ en: 'What the table shows', zh: '表里显示什么' })}>
		{#each Object.entries(SHOW) as [k, l] (k)}<button type="button" class="btn" aria-pressed={show === k} onclick={() => (pick = k as keyof typeof SHOW)}>{t(l)}</button>{/each}
	</div>
	{#if M && sc}
		<div class="sc" use:keyScroll><table aria-label={t(SHOW[show])}>
			<thead><tr><th class="cn">{t({ en: 'viewer ╲ movie', zh: '观众 ╲ 电影' })}</th>{#each { length: B } as _, j (j)}<th><i class="d" style:background={genreColor(MOVIES[pairs[j].movie].genre)}></i>{j + 1}</th>{/each}</tr></thead>
			<tbody>{#each { length: B } as _, i (i)}<tr><th>{i + 1}</th>{#each { length: B } as _, j (j)}{@const v = M.data[i * B + j]}<td class:diag={i === j} style:background={sc.css(v)} style:color={sc.ink(v)}>{f(v)}</td>{/each}</tr>{/each}</tbody>
		</table></div>
		<ol class="lg">{#each pairs as p, j (j)}<li><i class="d" style:background={genreColor(MOVIES[p.movie].genre)}></i>{t({ en: `${MOVIES[p.movie].title.en} (${GENRES[MOVIES[p.movie].genre].en}), liked by viewer ${j + 1}`, zh: `《${MOVIES[p.movie].title.zh}》（${GENRES[MOVIES[p.movie].genre].zh}），观众 ${j + 1} 喜欢的` })}</li>{/each}</ol>
	{/if}
</div>

<style>
	.bt { display: grid; gap: 6px; min-width: 0; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; } .btn { min-height: 30px; padding: 2px 10px; font-size: var(--fs-xs); } .btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	.sc { overflow-x: auto; max-width: 100%; }
	table { border-collapse: separate; border-spacing: 2px; font: 11px var(--font-mono); } th { padding: 0 3px; font-weight: 600; white-space: nowrap; } th.cn { max-width: 56px; font: 10px/1.1 var(--font-body); color: var(--muted); white-space: normal; }
	td { min-width: 40px; padding: 4px 3px; border-radius: 3px; text-align: center; } td.diag { outline: 2px solid var(--ink); outline-offset: -2px; }
	.d { display: inline-block; width: 8px; height: 8px; margin-right: 3px; border-radius: 50%; }
	.lg { display: grid; gap: 1px; margin: 0; padding-left: 20px; font-size: var(--fs-xs); }
</style>

<!-- transe 'kg': the knowledge graph (see meta.ts). Each fact is an arrow from its head to its tail in its relation's
     colour; the relation names sit next to their colours under the drawing. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { ENT, hasFact, inputOf, REL, START_FACTS, type Fact } from '../../kg.ts';

	let P: StageProps = $props();
	const g = $derived((void P.at, inputOf(P.rt.input)));
	const mark = $derived((P.props.mark as number[] | undefined) ?? []);
	const theme = $derived(themeState.resolved);
	const relColor = (k: number) => categorical(4 + k, theme);
	/** column (0 capitals, 1 countries, 2 continents and languages) and row position of an entity */
	const col = (i: number) => [1, 0, 2, 2][ENT[i].type];
	const X = [112, 200, 290];
	const Y = (i: number) => (ENT[i].type < 2 ? 14 + (i % 11) * 24 : ENT[i].type === 2 ? 14 + (i - 22) * 24 : 112 + (i - 26) * 21);
	/** where an arrow leaves or reaches an entity: beside its name, on the side that faces the other end */
	const side = (i: number, right: boolean) => (col(i) === 0 ? 118 : col(i) === 2 ? 283 : right ? 244 : 156);
	function path(f: Fact) {
		const [h, , tl] = f, a = col(h), b = col(tl), y0 = Y(h), y1 = Y(tl);
		if (a !== b) {
			const x0 = side(h, a < b), x1 = side(tl, a > b);
			return `M${x0},${y0} C${(x0 + x1) / 2},${y0} ${(x0 + x1) / 2},${y1} ${x1},${y1}`;
		}
		const x = side(h, a !== 0), out = a === 0 ? -40 : 40;
		return `M${x},${y0} C${x + out},${y0} ${x + out},${y1} ${x},${y1}`;
	}
	const anchor = (i: number) => (col(i) === 0 ? 'end' : col(i) === 1 ? 'middle' : 'start');
	const tx = (i: number) => (col(i) === 2 ? X[2] : X[col(i)]);
</script>

<div class="kg" style:min-height="340px">
	<svg viewBox="0 0 400 284" role="img" aria-label={t({ en: 'The knowledge graph: entities and facts', zh: '知识图谱：实体和事实' })}>
		<defs>{#each REL as _, k (k)}<marker id="transe-kg-a{k}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill={relColor(k)} /></marker>{/each}</defs>
		{#each g.facts as f (f.join())}
			{@const hot = mark.includes(f[0]) && mark.includes(f[2])}
			<path d={path(f)} fill="none" stroke={relColor(f[1])} stroke-width={hot ? 2.6 : 1.2} stroke-dasharray={hasFact(START_FACTS, f) ? undefined : '4 3'} marker-end="url(#transe-kg-a{f[1]})" opacity={hot || !mark.length ? 1 : 0.55} />
		{/each}
		{#each ENT as e, i (i)}
			<text x={tx(i)} y={Y(i) + 3.5} text-anchor={anchor(i)} class:on={mark.includes(i)}>{t(e.name)}</text>
		{/each}
	</svg>
	<p class="legend small">
		{#each REL as r, k (k)}<span><i style:background={relColor(k)}></i>{t(r)}</span>{/each}
		<span class="muted">{t({ en: 'columns: capitals · countries · continents, then languages', zh: '各列：首都 · 国家 · 大洲，然后是语言' })}</span>
		{#if g.facts.some((f) => !hasFact(START_FACTS, f))}<span>{t({ en: 'dashed: facts you added', zh: '虚线：你加的事实' })}</span>{/if}
	</p>
	<p class="small num">{t({ en: `${ENT.length} entities, ${g.facts.length} facts`, zh: `${ENT.length} 个实体，${g.facts.length} 个事实` })}</p>
</div>

<style>
	.kg { display: grid; gap: 4px; min-width: 0; align-content: start; } svg { width: 100%; max-width: 560px; height: auto; display: block; }
	text { font: 500 10px var(--font-body); fill: var(--ink); } text.on { font-weight: 700; text-decoration: underline; }
	p { margin: 0; } .small { font-size: var(--fs-sm); } .legend { display: flex; flex-wrap: wrap; gap: 4px 12px; }
	.legend i { display: inline-block; width: 14px; height: 4px; margin-right: 5px; vertical-align: middle; border-radius: 2px; }
</style>

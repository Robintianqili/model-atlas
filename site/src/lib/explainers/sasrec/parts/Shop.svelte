<!-- sasrec: the shop. Your history as a strip of clicks in order (position counted from 0, the product id under the name),
     with "remove the last click" and "clear"; then the 31 products in six category columns, each card a button that adds
     a click (a product is clicked at most once; at most 12 clicks). The new arrival is marked. onchange gets the raw input. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { CATS, ITEMS, MAX_CLICKS, NEW_ITEM, catColor, itemsOf, toClicks, type Clicks } from '../world.ts';

	let { raw, onchange, grid = true }: { raw: unknown; onchange: (r: Clicks) => void; grid?: boolean } = $props();
	const c = $derived(toClicks(raw).clicks);
	const add = (i: number) => !c.includes(i) && c.length < MAX_CLICKS && onchange({ clicks: [...c, i] });
	const cols = CATS.map((g, k) => ({ g, k, ids: [...itemsOf(k), ...(k === ITEMS[NEW_ITEM].cat ? [NEW_ITEM] : [])] }));
</script>

<div class="shop">
	<div class="hist" role="group" aria-label={t({ en: 'Your clicks, in order', zh: '你的点击，按顺序' })}>
		<span class="lbl">{t({ en: `Your clicks (${c.length} of at most ${MAX_CLICKS}):`, zh: `你的点击（${c.length} 次，最多 ${MAX_CLICKS} 次）：` })}</span>
		{#each c as i, k (i)}<span class="chip"><i style:background={catColor(ITEMS[i].cat)}></i><span class="num pos">{k}</span>{t(ITEMS[i].name)}<span class="num id">#{i}</span></span>{#if k < c.length - 1}<span class="arr" aria-hidden="true">→</span>{/if}{/each}
		{#if !c.length}<span class="muted">{t({ en: 'none yet', zh: '还没有' })}</span>{/if}
		<button type="button" class="btn" disabled={!c.length} onclick={() => onchange({ clicks: c.slice(0, -1) })}>{t({ en: 'Remove the last', zh: '去掉最后一次' })}</button>
		<button type="button" class="btn" disabled={!c.length} onclick={() => onchange({ clicks: [] })}>{t({ en: 'Clear', zh: '清空' })}</button>
	</div>
	{#if grid}
		<div class="cat">
			{#each cols as col (col.k)}
				<div class="col" role="list" aria-label={t(col.g)}>
					<p class="gh"><i style:background={catColor(col.k)}></i>{t(col.g)}</p>
					{#each col.ids as i (i)}
						<div role="listitem" class="li">
							<button type="button" class="card" aria-pressed={c.includes(i)} disabled={!c.includes(i) && c.length >= MAX_CLICKS} onclick={() => add(i)}>
								<span class="ti">{t(ITEMS[i].name)}</span><span class="meta num">#{i}{#if c.includes(i)}{' · '}{t({ en: `click ${c.indexOf(i)}`, zh: `位置 ${c.indexOf(i)}` })}{/if}{#if i === NEW_ITEM}{' · '}{t({ en: 'new', zh: '新品' })}{/if}</span>
							</button>
						</div>
					{/each}
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.shop { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.hist { display: flex; flex-wrap: wrap; align-items: center; gap: 4px; }
	.lbl { font-weight: 600; } .muted { color: var(--muted); }
	.chip { display: inline-flex; align-items: center; gap: 4px; padding: 1px 6px; border: 1px solid var(--rule); border-radius: 4px; font-size: var(--fs-xs); }
	.chip i, .gh i { display: inline-block; flex: none; width: 9px; height: 9px; border-radius: 50%; }
	.pos { color: var(--muted); font-size: 11px; } .id { color: var(--muted); font-size: 11px; } .arr { color: var(--muted); }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.cat { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 100px), 1fr)); gap: 8px; min-width: 0; }
	.col { display: grid; gap: 3px; align-content: start; min-width: 0; } .li { display: grid; min-width: 0; }
	.gh { display: flex; align-items: center; gap: 5px; margin: 0; font-weight: 600; }
	.card { display: grid; gap: 1px; min-width: 0; min-height: 32px; padding: 3px 6px; border: 1px solid var(--rule); border-radius: 4px; background: var(--surface); color: var(--ink); text-align: left; font: inherit; }
	.card[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	.ti { overflow: hidden; font-size: var(--fs-xs); text-overflow: ellipsis; white-space: nowrap; } .meta { color: var(--muted); font-size: 11px; }
</style>

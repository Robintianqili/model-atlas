<!--
  'optimizers/league' (meta.ts): a table, one row per landscape and one column per optimizer, with the loss after the
  frame's number of updates; each row's lowest loss is bold, outlined and followed by the word "lowest".
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { RACERS } from '../../data.ts';
	import { leaguePic } from './meta.ts';

	let stage: StageProps = $props();
	const pic = $derived(leaguePic(stage));
	const L = {
		cap: { en: 'Loss after {n} updates (lower is better)', zh: '更新 {n} 次之后的损失（越低越好）' },
		land: { en: 'Landscape', zh: '地形' },
		low: { en: 'lowest', zh: '最低' }
	};
</script>

{#if pic}
	<div class="wrap" use:keyScroll>
		<table>
			<caption>{t(L.cap).replace('{n}', String(pic.step))}</caption>
			<thead>
				<tr><th scope="col">{t(L.land)}</th>{#each RACERS as r (r.key)}<th scope="col">{t(r.name)}</th>{/each}</tr>
			</thead>
			<tbody>
				{#each pic.rows as row (row.id)}
					<tr>
						<th scope="row">{t(row.name)}</th>
						{#each row.losses as v, j (j)}
							<td class="num" class:best={j === row.best}>{fmt3(v)}{#if j === row.best}<span class="tag">{` ${t(L.low)}`}</span>{/if}</td>
						{/each}
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
{/if}

<style>
	.wrap {
		max-width: 100%;
		overflow-x: auto;
	}
	table {
		border-collapse: collapse;
		font-size: var(--fs-sm);
		min-width: 0;
	}
	caption {
		text-align: left;
		color: var(--ink-2);
		padding-bottom: 6px;
	}
	th,
	td {
		padding: 6px 10px;
		border-bottom: 1px solid var(--rule);
		text-align: right;
		white-space: nowrap;
	}
	th[scope='row'],
	thead th:first-child {
		text-align: left;
	}
	td.best {
		font-weight: 700;
		outline: 2px solid var(--highlight);
		outline-offset: -3px;
	}
	.tag {
		margin-left: 0.15em;
		font-weight: 400;
		font-size: var(--fs-xs);
	}
</style>

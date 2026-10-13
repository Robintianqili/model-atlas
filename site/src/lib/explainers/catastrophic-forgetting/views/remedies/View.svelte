<!--
  'catastrophic-forgetting/remedies': one table row per way of training on task B, with the share of task A and of task
  B the network gets right as bars from 0 to 100% and as numbers (meta.ts has the props). The two tasks are told apart by
  their names in the header and the legend, and by two colors of the kit's categorical palette.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtPct } from '$lib/kit/format';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { rowsOf } from './meta.ts';

	let { at, props }: StageProps = $props();
	const rows = $derived(rowsOf({ at, props }));
	const theme = $derived(themeState.resolved);
	const colA = $derived(categorical(2, theme));
	const colB = $derived(categorical(3, theme));
	const L = {
		cap: { en: 'Share right on each task after training on task B', zh: '在任务 B 上训练之后，每个任务答对的比例' },
		how: { en: 'How it trained on B', zh: '在 B 上的训练方式' },
		a: { en: 'Task A (negative x)', zh: '任务 A（x 为负）' },
		b: { en: 'Task B (positive x)', zh: '任务 B（x 为正）' },
		wait: { en: 'Training the copies…', zh: '正在训练这几个副本…' }
	};
	const w = (v: number) => `${Math.max(0, Math.min(1, v)) * 100}%`;
</script>

{#if rows.length}
	<figure class="rem">
		<table>
			<caption>{t(L.cap)}</caption>
			<thead>
				<tr>
					<th scope="col">{t(L.how)}</th>
					<th scope="col"><span class="sw" style:background={colA}></span>{t(L.a)}</th>
					<th scope="col"><span class="sw" style:background={colB}></span>{t(L.b)}</th>
				</tr>
			</thead>
			<tbody>
				{#each rows as r (r.id)}
					<tr>
						<th scope="row">{t(r.name)}</th>
						<td><span class="bar"><span class="fill" style:width={w(r.accA)} style:background={colA}></span></span><span class="num">{fmtPct(r.accA)}</span></td>
						<td><span class="bar"><span class="fill" style:width={w(r.accB)} style:background={colB}></span></span><span class="num">{fmtPct(r.accB)}</span></td>
					</tr>
				{/each}
			</tbody>
		</table>
	</figure>
{:else}
	<p class="wait">{t(L.wait)}</p>
{/if}

<style>
	.rem {
		margin: 0;
		min-width: 0;
		max-width: 640px;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		table-layout: fixed;
		font-size: var(--fs-sm);
	}
	caption {
		text-align: left;
		font-size: var(--fs-xs);
		color: var(--muted);
		padding-bottom: 6px;
	}
	th,
	td {
		padding: 5px 6px;
		border-bottom: 1px solid var(--rule);
		vertical-align: middle;
		text-align: left;
	}
	thead th {
		font-weight: 600;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	thead th:first-child {
		width: 34%;
	}
	tbody th {
		font-weight: 500;
		color: var(--ink);
		overflow-wrap: anywhere;
	}
	td {
		display: table-cell;
	}
	.bar {
		display: inline-block;
		width: calc(100% - 4.1em);
		height: 12px;
		vertical-align: middle;
		background: var(--surface-2);
		border: 1px solid var(--rule);
		border-radius: 2px;
		overflow: hidden;
	}
	.fill {
		display: block;
		height: 100%;
	}
	.num {
		display: inline-block;
		width: 3.9em;
		text-align: right;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
	}
	.sw {
		display: inline-block;
		width: 10px;
		height: 10px;
		margin-right: 5px;
		border-radius: 2px;
		vertical-align: -1px;
	}
	.wait {
		color: var(--muted);
		font-size: var(--fs-sm);
	}
	@media (max-width: 420px) {
		thead th:first-child {
			width: 30%;
		}
		.bar {
			width: calc(100% - 3.9em);
		}
		.num {
			width: 3.7em;
		}
	}
</style>

<!--
  backprop Capture for the local 'leaves' adapter: the four leaves of the toy graph (sliders, 0.1 apart, with the exact
  value), the loss L they give (process.ts lossOf: the kit's autograd, forward only), recomputed on every move, and three
  starting settings as buttons (short names in the compact copy beside the lesson). Reports { x, w, b, y }.
-->
<script lang="ts">
	import { colon, t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import type { CaptureProps } from '../../../../atlas/types/explainer.ts';
	import { LEAVES, leavesOf, type LeafId, type Leaves } from '../../data.ts';
	import { lossOf } from '../../process.ts';
	import { STARTS } from './adapter.ts';

	let { raw, onchange, compact = false }: CaptureProps<Leaves> = $props();

	const l = $derived(leavesOf(raw));
	const loss = $derived(lossOf(l));
	const round = (v: number) => Math.round(v * 100) / 100;
	function set(k: LeafId, v: number, live = false) {
		onchange?.({ ...l, [k]: round(v) }, { live });
	}
	const ROLE = { data: { en: 'data', zh: '数据' }, param: { en: 'parameter', zh: '参数' } };
</script>

<div class="leaves" class:compact>
	{#each LEAVES as d (d.id)}
		<label class="row">
			<span class="name">{#if compact}{d.id}{:else}{t(d.name)} <span class="role">({t(ROLE[d.role])})</span>{/if}</span>
			<input
				type="range"
				min={d.range[0]}
				max={d.range[1]}
				step="0.1"
				value={l[d.id]}
				oninput={(e) => set(d.id, Number((e.currentTarget as HTMLInputElement).value), true)}
				onchange={(e) => set(d.id, Number((e.currentTarget as HTMLInputElement).value))}
			/>
			<output class="val num">{fmt3(l[d.id])}</output>
		</label>
	{/each}
	<p class="loss">
		{t({ en: 'Loss L = (tanh(w·x + b) − y)²', zh: '损失 L = (tanh(w·x + b) − y)²' })}{colon()} <strong class="num">{fmt3(loss)}</strong>
	</p>
	<div class="starts" role="group" aria-label={t({ en: 'Start again from', zh: '重新开始的位置' })}>
		{#each STARTS as s (s.id)}
			<button type="button" class="btn" onclick={() => onchange?.({ ...s.raw })}>{t(compact ? s.short : s.label)}</button>
		{/each}
	</div>
</div>

<style>
	.leaves {
		display: grid;
		gap: 8px;
		min-width: 0;
		max-width: 560px;
	}
	.row {
		display: grid;
		grid-template-columns: minmax(0, 12rem) minmax(0, 1fr) 4rem;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-sm);
	}
	.compact .row {
		grid-template-columns: 1.2rem minmax(0, 1fr) 3.4rem;
	}
	.name {
		color: var(--ink-2);
	}
	.role {
		color: var(--muted);
		font-size: var(--fs-xs);
	}
	input[type='range'] {
		width: 100%;
		min-width: 0;
		accent-color: var(--accent);
	}
	.val {
		text-align: right;
	}
	.loss {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.starts {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.starts .btn {
		font-size: var(--fs-xs);
	}
	@media (max-width: 420px) {
		.leaves:not(.compact) .row {
			grid-template-columns: minmax(0, 1fr) 4rem;
		}
		.leaves:not(.compact) .row .name {
			grid-column: 1 / -1;
		}
	}
</style>
